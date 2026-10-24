#!/usr/bin/env python3
"""Generate a synthetic two-channel service with known ground truth.

Lets you verify the whole pipeline without a real recording -- and re-verify it
after porting to a Pi, where numpy/CTranslate2 versions differ.

    ./tests/make_fixture.py --outdir tests/fixture

Produces fixture-sanctuary.wav, fixture-ambient.wav, a stand-in transcript, and
fixture-truth.json holding the offset, drift and sermon bounds that were baked in.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path

import numpy as np

FS = 16000

# (start, end, kind) -- kind drives how hot the loft mics are.
PROGRAM = [
    (0, 240, "music"),
    (240, 360, "speech"),
    (360, 420, "music"),
    (420, 1200, "sermon"),
    (1200, 1440, "music"),
]
SERMON_START, SERMON_END = 420.0, 1200.0

# How loud the ambient (choir loft) channel is relative to the sanctuary feed.
# This gap is the feature the sermon stage keys on.
AMBIENT_GAIN = {"music": 0.80, "speech": 0.13, "sermon": 0.045}


def speech_like(n: int, rng: np.random.Generator) -> np.ndarray:
    """Bandpass noise with a syllabic amplitude envelope."""
    x = rng.standard_normal(n).astype(np.float32)
    # cheap 2-pole-ish bandpass via difference + leaky integrator
    x = np.diff(x, prepend=0.0)
    env_rate = 4.0  # syllables/sec
    t = np.arange(n) / FS
    env = 0.5 + 0.5 * np.sin(2 * np.pi * env_rate * t + rng.uniform(0, 6.28))
    pauses = (np.sin(2 * np.pi * 0.13 * t) > -0.85).astype(np.float32)
    return (x * env * pauses).astype(np.float32)


def music_like(n: int, rng: np.random.Generator) -> np.ndarray:
    t = np.arange(n) / FS
    out = np.zeros(n, dtype=np.float32)
    for f in (110.0, 165.0, 220.0, 330.0, 440.0):
        out += np.sin(2 * np.pi * f * t + rng.uniform(0, 6.28)).astype(np.float32) / 5.0
    out += 0.3 * rng.standard_normal(n).astype(np.float32)
    return out * (0.7 + 0.3 * np.sin(2 * np.pi * 0.25 * t)).astype(np.float32)


def write_wav(path: Path, x: np.ndarray) -> None:
    clipped = np.clip(x, -1.0, 1.0)
    with wave.open(str(path), "wb") as fh:
        fh.setnchannels(1)
        fh.setsampwidth(2)
        fh.setframerate(FS)
        fh.writeframes((clipped * 32767.0).astype("<i2").tobytes())


def make_transcript(path: Path, duration: float, rng: np.random.Generator) -> None:
    """Stand-in for `service-trimmer transcribe` output: same schema, plausible markers."""
    filler = ("and so we come to understand that the calling placed before us "
              "is not one we carry alone but together as a people").split()
    segments, sid, t = [], 0, 5.0
    while t < duration - 6:
        if abs(t - SERMON_START) < 5.0:
            text = "Turn with me in your Bibles to Romans chapter eight."
        elif SERMON_END - 12.0 <= t < SERMON_END - 6.0:
            # Must finish before the music starts at SERMON_END.
            text = "In the name of the Father, and of the Son, and of the Holy Spirit. Amen."
        else:
            k = rng.integers(6, 14)
            start = rng.integers(0, max(1, len(filler) - k))
            text = " ".join(filler[start:start + k]) + "."
        toks = text.split()
        dur = 0.38 * len(toks)
        words = [
            {"start": round(t + i * dur / len(toks), 3),
             "end": round(t + (i + 1) * dur / len(toks), 3),
             "word": w, "probability": 0.95}
            for i, w in enumerate(toks)
        ]
        # Real Whisper output over music is low-confidence (sung text comes
        # back around log-prob -0.7, no-speech 0.85); model that, or the
        # fixture claims confident speech where there is only music.
        in_music = any(a <= t < b and k == "music" for a, b, k in PROGRAM)
        lp, ns = (-0.7, 0.85) if in_music else (-0.2, 0.01)
        segments.append({"id": sid, "start": round(t, 3), "end": round(t + dur, 3),
                         "text": text, "avg_logprob": lp, "no_speech_prob": ns,
                         "words": words})
        sid += 1
        t += dur + float(rng.uniform(0.3, 1.4))

    path.write_text(json.dumps({
        "audio": {"duration": duration}, "model": "fixture", "language": "en",
        "duration": duration, "vad": True, "initial_prompt": None,
        "segments": segments,
    }, indent=2) + "\n", encoding="utf-8")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--outdir", default="tests/fixture")
    ap.add_argument("--offset", type=float, default=0.734,
                    help="true start offset, seconds (ambient lags by this much)")
    ap.add_argument("--drift-ppm", type=float, default=60.0,
                    help="true relative clock drift between the two recorders")
    ap.add_argument("--seed", type=int, default=7)
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    duration = float(PROGRAM[-1][1])
    n = int(duration * FS)

    # --- sanctuary: the reference clock ------------------------------------
    san = np.zeros(n, dtype=np.float32)
    gain_env = np.zeros(n, dtype=np.float32)
    for start, end, kind in PROGRAM:
        a, b = int(start * FS), int(end * FS)
        block = music_like(b - a, rng) if kind == "music" else speech_like(b - a, rng)
        block /= (np.abs(block).max() or 1.0)
        san[a:b] = block * 0.35
        gain_env[a:b] = AMBIENT_GAIN[kind]

    # --- ambient: same content, shifted and stretched -----------------------
    # An event at sanctuary-time t lands in the ambient file at a + t*(1+b),
    # so ambient_file(s) samples the source at (s - a)/(1 + b).
    b_rate = args.drift_ppm / 1e6
    s = np.arange(n, dtype=np.float64) / FS
    src_t = (s - args.offset) / (1.0 + b_rate)
    base_t = np.arange(n, dtype=np.float64) / FS
    amb = np.interp(src_t, base_t, san.astype(np.float64), left=0.0, right=0.0)
    amb *= np.interp(src_t, base_t, gain_env.astype(np.float64), left=0.0, right=0.0)

    # Room character: a couple of early reflections plus its own noise floor, so
    # the two channels are genuinely different signals rather than a pure copy.
    for delay_ms, g in ((23.0, 0.35), (41.0, 0.22), (67.0, 0.14)):
        d = int(delay_ms * FS / 1000.0)
        amb[d:] += g * amb[:-d]
    amb += 0.0015 * rng.standard_normal(n)
    amb *= 0.9 / (np.abs(amb).max() or 1.0)

    san_path = outdir / "fixture-sanctuary.wav"
    amb_path = outdir / "fixture-ambient.wav"
    write_wav(san_path, san)
    write_wav(amb_path, amb.astype(np.float32))
    make_transcript(outdir / "fixture-transcript.json", duration, rng)

    truth = {
        "offset_s": args.offset, "drift_ppm": args.drift_ppm,
        "duration_s": duration, "sample_rate": FS,
        "sermon_start_s": SERMON_START, "sermon_end_s": SERMON_END,
        "program": [{"start": a, "end": b, "kind": k} for a, b, k in PROGRAM],
        "ambient_gain": AMBIENT_GAIN,
        "sanctuary": str(san_path), "ambient": str(amb_path),
        "transcript": str(outdir / "fixture-transcript.json"),
    }
    (outdir / "fixture-truth.json").write_text(json.dumps(truth, indent=2) + "\n")

    print(f"wrote {outdir}/  ({duration / 60:.0f} min @ {FS} Hz)")
    print(f"  true offset  {args.offset * 1000:+.1f} ms")
    print(f"  true drift   {args.drift_ppm:+.1f} ppm "
          f"({abs(args.drift_ppm) * duration / 1000:.0f} ms end-to-end slip)")
    print(f"  true sermon  {SERMON_START:.0f}s -> {SERMON_END:.0f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
