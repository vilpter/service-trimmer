"""Shared audio/DSP helpers for service-trimmer.

Deliberately dependency-light: numpy + the ffmpeg/ffprobe binaries. Nothing here
imports faster_whisper, so `sync`, `mix`, `schedule` and `reel` run on
a bare `python3 -m venv` with only numpy installed.
"""

from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

FFMPEG = shutil.which("ffmpeg") or "ffmpeg"
FFPROBE = shutil.which("ffprobe") or "ffprobe"


# --------------------------------------------------------------------------- #
# ffmpeg / ffprobe
# --------------------------------------------------------------------------- #

def require_ffmpeg() -> None:
    for binary in (FFMPEG, FFPROBE):
        if shutil.which(binary) is None and not Path(binary).exists():
            die(f"{binary} not found on PATH. Install ffmpeg (Debian/Pi OS: apt install ffmpeg).")


def probe(path: str | Path) -> dict:
    """Return duration/rate/channels/codec for the first audio stream."""
    out = subprocess.run(
        [FFPROBE, "-v", "error", "-print_format", "json",
         "-show_format", "-show_streams", "-select_streams", "a:0", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    info = json.loads(out)
    if not info.get("streams"):
        die(f"{path}: no audio stream found")
    stream = info["streams"][0]
    duration = stream.get("duration") or info.get("format", {}).get("duration")
    return {
        "path": str(path),
        "duration": float(duration) if duration else None,
        "sample_rate": int(stream["sample_rate"]),
        "channels": int(stream["channels"]),
        "codec": stream.get("codec_name"),
        "bit_depth": stream.get("bits_per_raw_sample") or stream.get("bits_per_sample"),
    }


def decode_mono(
    path: str | Path,
    rate: int,
    start: float | None = None,
    duration: float | None = None,
    channel: int | None = None,
) -> np.ndarray:
    """Decode a slice of `path` to mono float32 in [-1, 1] at `rate` Hz.

    `channel` selects one channel of a multi-channel file; None downmixes.
    Decoded as s16 on the pipe (half the bytes of f32) then scaled.
    """
    cmd = [FFMPEG, "-v", "error", "-nostdin"]
    if start is not None:
        cmd += ["-ss", f"{max(0.0, start):.6f}"]
    cmd += ["-i", str(path)]
    if duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    if channel is not None:
        cmd += ["-af", f"pan=mono|c0=c{channel}"]
    else:
        cmd += ["-ac", "1"]
    cmd += ["-ar", str(rate), "-f", "s16le", "-acodec", "pcm_s16le", "-"]

    proc = subprocess.run(cmd, capture_output=True)
    if proc.returncode != 0:
        die(f"ffmpeg decode failed for {path}:\n{proc.stderr.decode(errors='replace')}")
    return np.frombuffer(proc.stdout, dtype="<i2").astype(np.float32) / 32768.0


def measure_peak_db(filter_complex: str, inputs: list[str], out_label: str) -> float:
    """Run a filtergraph to /dev/null with volumedetect and return peak dBFS.

    Used to pick a safe pre-mix attenuation without a limiter touching the audio.
    """
    cmd = [FFMPEG, "-v", "info", "-nostdin"]
    for src in inputs:
        cmd += ["-i", str(src)]
    cmd += ["-filter_complex", f"{filter_complex};[{out_label}]volumedetect[vd]",
            "-map", "[vd]", "-f", "null", "-"]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    peak = None
    for line in proc.stderr.splitlines():
        if "max_volume:" in line:
            try:
                peak = float(line.split("max_volume:")[1].strip().split()[0])
            except (IndexError, ValueError):
                pass
    if peak is None:
        die("volumedetect produced no max_volume; cannot compute safe gain.\n"
            + proc.stderr[-2000:])
    return peak


def run_ffmpeg(cmd: list[str], what: str) -> None:
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"{what} failed:\n{' '.join(cmd)}\n\n{proc.stderr[-4000:]}")


# --------------------------------------------------------------------------- #
# GCC-PHAT alignment
# --------------------------------------------------------------------------- #

def gcc_phat(
    a: np.ndarray,
    b: np.ndarray,
    fs: int,
    max_tau: float | None = None,
    band: tuple[float, float] | None = (200.0, 4000.0),
) -> tuple[float, float, bool]:
    """Generalised cross-correlation with phase transform.

    Returns (tau, sharpness, polarity_inverted).

    `tau` is the delay of `a` relative to `b` in seconds: positive means the
    content in `a` arrives later. PHAT divides out the cross-spectrum magnitude,
    keeping only phase, which is what makes this work between a dry close mic and
    a reverberant room mic -- plain correlation would be dominated by their very
    different spectra and give a broad, mushy peak.

    `sharpness` is peak / RMS of the correlation. Anything above ~8 on speech is
    a confident lock; below ~4 the window is probably silence or pure noise.
    """
    if a.size == 0 or b.size == 0:
        return 0.0, 0.0, False

    n = a.size + b.size
    nfft = 1 << (n - 1).bit_length()

    A = np.fft.rfft(a, n=nfft)
    B = np.fft.rfft(b, n=nfft)
    R = A * np.conj(B)

    mag = np.abs(R)
    np.maximum(mag, 1e-12, out=mag)
    R /= mag

    # PHAT whitening amplifies bands that contain nothing but noise. Restricting
    # to the speech band keeps HVAC rumble and ultrasonic hiss out of the peak.
    if band is not None:
        freqs = np.fft.rfftfreq(nfft, 1.0 / fs)
        R[(freqs < band[0]) | (freqs > band[1])] = 0.0

    cc = np.fft.irfft(R, n=nfft)

    max_shift = nfft // 2 if max_tau is None else min(nfft // 2, int(fs * max_tau))
    if max_shift < 1:
        return 0.0, 0.0, False
    cc = np.concatenate((cc[-max_shift:], cc[: max_shift + 1]))

    k_pos = int(np.argmax(cc))
    k_neg = int(np.argmax(-cc))
    inverted = abs(cc[k_neg]) > abs(cc[k_pos])
    work = -cc if inverted else cc
    k = k_neg if inverted else k_pos

    # Sub-sample refinement. A parabola through the three samples around the peak
    # gets us well under a tenth of a sample, which beats zero-padding the IFFT by
    # 16x and costs nothing -- that matters on a Pi.
    delta = 0.0
    if 0 < k < work.size - 1:
        y0, y1, y2 = float(work[k - 1]), float(work[k]), float(work[k + 1])
        denom = y0 - 2.0 * y1 + y2
        if denom != 0.0:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -1.0, 1.0))

    tau = (k - max_shift + delta) / fs
    rms = float(np.sqrt(np.mean(work.astype(np.float64) ** 2))) or 1e-12
    return float(tau), float(abs(work[k]) / rms), bool(inverted)


def select_dominant_cluster(measurements: list[dict], tol_s: float = 0.025) -> tuple[list, list]:
    """Keep the offset cluster the confident windows agree on; return (kept, dropped).

    A sanctuary has two acoustic paths between these mics, and which one wins
    depends on who is making noise. While the pastor preaches, sound travels
    pulpit -> loft and the offset is positive. While the choir and congregation
    sing, the loft is the source and its sound bleeds into the pulpit mics, so the
    offset flips to roughly minus the same value. Both are real; they are just
    different paths.

    Fitting across both produces a number that describes neither -- observed on
    2026-02-08 as +83 ms with a fabricated -17 ppm of drift, when the confident
    windows all agreed on +51.3 ms. Outlier rejection alone does not save it,
    because neither cluster is an "outlier": they are comparable in size.

    The tell is sharpness (correlation peak / RMS). The pulpit path produces tight,
    confident locks; the bleed path produces weak, scattered ones. So group by tau
    and keep the group carrying the most total confidence.
    """
    if len(measurements) < 3:
        return measurements, []

    order = sorted(measurements, key=lambda m: m["tau"])
    clusters, cur = [], [order[0]]
    for m in order[1:]:
        if m["tau"] - cur[-1]["tau"] <= tol_s:
            cur.append(m)
        else:
            clusters.append(cur)
            cur = [m]
    clusters.append(cur)

    if len(clusters) == 1:
        return measurements, []

    best = max(clusters, key=lambda c: sum(x["sharpness"] for x in c))
    dropped = [m for c in clusters if c is not best for m in c]
    return best, dropped


def fit_drift(times: list[float], taus: list[float], mad_k: float = 3.0) -> dict:
    """Least-squares fit of offset(t) = a + b*t with iterative outlier rejection.

    `b` is the relative clock drift between the two recorders. Two independent
    devices typically drift 10-100 ppm; over a 90-minute service 50 ppm is 270 ms
    of slip, which is very audible as progressive flamming. This is the single
    thing a fixed-offset implementation gets wrong.
    """
    t = np.asarray(times, dtype=np.float64)
    y = np.asarray(taus, dtype=np.float64)
    used = np.ones(t.size, dtype=bool)

    if t.size < 3:
        offset = float(np.median(y)) if y.size else 0.0
        return {
            "offset_s": offset, "drift_ppm": 0.0, "residual_rms_ms": 0.0,
            "n_used": int(t.size), "n_total": int(t.size),
            "used_mask": used.tolist(),
            "note": "fewer than 3 usable windows; drift not estimated",
        }

    slope = intercept = 0.0
    for _ in range(5):
        slope, intercept = np.polyfit(t[used], y[used], 1)
        resid = y - (intercept + slope * t)
        centre = np.median(resid[used])
        mad = np.median(np.abs(resid[used] - centre))
        if mad <= 0:
            break
        keep = np.abs(resid - centre) <= mad_k * 1.4826 * mad
        keep &= used | keep  # never resurrect a window we already rejected twice
        if keep.sum() < 3 or np.array_equal(keep, used):
            break
        used = keep

    resid = y - (intercept + slope * t)
    return {
        "offset_s": float(intercept),
        "drift_ppm": float(slope * 1e6),
        "residual_rms_ms": float(np.sqrt(np.mean(resid[used] ** 2)) * 1000.0),
        "n_used": int(used.sum()),
        "n_total": int(t.size),
        "used_mask": used.tolist(),
        "note": "",
    }


# --------------------------------------------------------------------------- #
# Framing / level analysis
# --------------------------------------------------------------------------- #

def frame_rms_db(x: np.ndarray, fs: int, hop_s: float, win_s: float | None = None,
                 chunk_frames: int = 512) -> np.ndarray:
    """RMS level in dBFS per frame. Silence floors at -120 dB rather than -inf.

    Chunked deliberately. Building the whole frame matrix at once costs
    n_frames x win entries -- for a 2.5-hour service that is over a gigabyte of
    index plus another gigabyte of gathered samples, which is fine on a 72-core
    PC and fatal on a Pi. Chunking caps the working set at a few tens of MB
    regardless of how long the recording is.
    """
    win_s = win_s or hop_s
    hop = max(1, int(round(hop_s * fs)))
    win = max(hop, int(round(win_s * fs)))
    if x.size < win:
        return np.array([], dtype=np.float64)

    n_frames = 1 + (x.size - win) // hop
    windows = np.lib.stride_tricks.sliding_window_view(x, win)
    power = np.empty(n_frames, dtype=np.float64)
    for i in range(0, n_frames, chunk_frames):
        j = min(i + chunk_frames, n_frames)
        block = windows[i * hop: (j - 1) * hop + 1: hop]
        power[i:j] = np.mean(block.astype(np.float64) ** 2, axis=1)
    return 10.0 * np.log10(np.maximum(power, 1e-12))


def otsu_threshold(values: np.ndarray, bins: int = 256) -> float:
    """Threshold that maximises between-class variance (Otsu's method).

    The ambient/sanctuary ratio is strongly bimodal -- one cluster where the PA
    feed carries a voice and the loft only hears the room, another where the
    congregation and organ light up the loft while the PA feed is nearly silent.
    A percentile midpoint assumes the two classes are balanced; they are not
    (preaching is a minority of a service), so it drifts into the larger cluster.
    Otsu finds the split itself and does not care about class sizes.
    """
    v = np.asarray(values, dtype=np.float64)
    v = v[np.isfinite(v)]
    if v.size < 8:
        return float(np.median(v)) if v.size else 0.0

    hist, edges = np.histogram(v, bins=bins)
    centers = 0.5 * (edges[:-1] + edges[1:])
    p = hist.astype(np.float64)
    total = p.sum()
    if total <= 0:
        return float(np.median(v))
    p /= total

    omega = np.cumsum(p)
    mu = np.cumsum(p * centers)
    mu_t = mu[-1]
    denom = omega * (1.0 - omega)
    with np.errstate(divide="ignore", invalid="ignore"):
        sigma_b = (mu_t * omega - mu) ** 2 / denom
    sigma_b[~np.isfinite(sigma_b)] = -1.0
    return float(centers[int(np.argmax(sigma_b))])


def median_smooth(x: np.ndarray, k: int) -> np.ndarray:
    """Odd-length running median. Edge frames use the shortened window."""
    if k <= 1 or x.size == 0:
        return x
    k = k if k % 2 == 1 else k + 1
    half = k // 2
    padded = np.pad(x, half, mode="edge")
    view = np.lib.stride_tricks.sliding_window_view(padded, k)
    return np.median(view, axis=-1)


# --------------------------------------------------------------------------- #
# Misc
# --------------------------------------------------------------------------- #

def write_json(path: str | Path, obj: dict) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, indent=2)
        fh.write("\n")


def hms(seconds: float | None) -> str:
    if seconds is None or not math.isfinite(seconds):
        return "--:--:--"
    sign = "-" if seconds < 0 else ""
    seconds = abs(seconds)
    h, rem = divmod(int(seconds), 3600)
    m, s = divmod(rem, 60)
    return f"{sign}{h:02d}:{m:02d}:{s:02d}.{int((seconds % 1) * 1000):03d}"


def die(msg: str, code: int = 1) -> None:
    print(f"error: {msg}", file=sys.stderr)
    raise SystemExit(code)


def info(msg: str) -> None:
    print(msg, file=sys.stderr, flush=True)


def normalized_mp3(src: str | Path, dst: str | Path, target_lufs: float = -16.0,
                   bitrate: str = "128k", tags: dict | None = None) -> dict:
    """Two-pass EBU R128 loudness normalisation to MP3.

    Sermon extracts come off the desk around -33 LUFS, which is unlistenable on a
    phone. Two passes (measure, then apply the measured values) instead of one so
    the result actually lands on target. Level only -- the sanctuary/ambient
    balance is untouched.
    """
    probe_cmd = [FFMPEG, "-hide_banner", "-nostdin", "-i", str(src), "-af",
                 f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11:print_format=json",
                 "-f", "null", "-"]
    out = subprocess.run(probe_cmd, capture_output=True, text=True).stderr
    block = out[out.rfind("{"):out.rfind("}") + 1] if "{" in out else ""
    try:
        m = json.loads(block)
    except json.JSONDecodeError:
        die(f"loudnorm measurement failed for {src}:\n{out[-1500:]}")

    filt = (f"loudnorm=I={target_lufs}:TP=-1.5:LRA=11"
            f":measured_I={m['input_i']}:measured_TP={m['input_tp']}"
            f":measured_LRA={m['input_lra']}:measured_thresh={m['input_thresh']}"
            f":linear=true")
    meta = []
    for k, v in (tags or {}).items():
        meta += ["-metadata", f"{k}={v}"]
    run_ffmpeg([FFMPEG, "-v", "error", "-nostdin", "-y", "-i", str(src),
                "-af", filt, "-c:a", "libmp3lame", "-b:a", bitrate, *meta, str(dst)],
               "mp3 render")
    return {"input_lufs": float(m["input_i"]), "target_lufs": target_lufs}


# --------------------------------------------------------------------------- #
# Runs of a boolean mask
# --------------------------------------------------------------------------- #

def find_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    """[start, end) index pairs of every True run."""
    if mask.size == 0 or not mask.any():
        return []
    edges = np.diff(mask.astype(np.int8))
    starts = list(np.flatnonzero(edges == 1) + 1)
    ends = list(np.flatnonzero(edges == -1) + 1)
    if mask[0]:
        starts.insert(0, 0)
    if mask[-1]:
        ends.append(mask.size)
    return list(zip(starts, ends))


def merge_runs(runs: list[tuple[int, int]], gap: int) -> list[tuple[int, int]]:
    """Join runs separated by at most `gap` frames."""
    if not runs:
        return []
    merged = [list(runs[0])]
    for a, b in runs[1:]:
        if a - merged[-1][1] <= gap:
            merged[-1][1] = b
        else:
            merged.append([a, b])
    return [tuple(r) for r in merged]


def channel_level_stats(path: str | Path, channel: int | None = None,
                        rate: int = 2000, hop_s: float = 10.0) -> dict:
    """Cheap whole-file level survey: percentiles of hop-sized RMS frames.

    Used to triage recordings before spending an hour transcribing them -- a dead
    feed or a silent recorder shows up here in a few seconds.
    """
    x = decode_mono(path, rate, channel=channel)
    db = frame_rms_db(x, rate, hop_s)
    if db.size == 0:
        return {"frames": 0, "p05": -120.0, "p50": -120.0, "p95": -120.0, "db": db}
    return {
        "frames": int(db.size),
        "p05": float(np.percentile(db, 5)),
        "p50": float(np.percentile(db, 50)),
        "p95": float(np.percentile(db, 95)),
        "db": db,
    }
