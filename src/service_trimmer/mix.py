"""Apply the sync report and produce the aligned stems plus a mix.

    service-trimmer mix --sync sync-report.json --outdir out/ --basename 2026-08-09

Always writes a 2-channel *stems* file (L = sanctuary, R = ambient, time-aligned).
Re-mixing from stems is free once alignment is solved, so keeping them means you
never have to re-run sync to re-balance.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .audio import (
    FFMPEG, decode_mono, die, info, measure_peak_db, require_ffmpeg, run_ffmpeg,
)
from .config import load_config


# Intermediate rate used only as a unit for the drift resample ratio. Sets the
# correction granularity to 1 ppm (worst-case 0.5 ppm error = 2.7 ms over a
# 90-minute service). Nothing is actually sampled at this rate.
RESAMPLE_BASE = 1_000_000


def db_to_lin(db: float) -> float:
    return 10.0 ** (db / 20.0)


def verify_stems_channels(stems: str, san_path: str, amb_path: str,
                          duration: float | None) -> None:
    """Confirm stems ch0 really is the sanctuary feed and ch1 the ambient.

    ffmpeg's `join` has to guess a mapping when every input is mono (they all
    report channel layout FC), and it guessed *reversed* here -- producing a
    stems file whose channels were silently swapped. Everything downstream
    (the sermon stage reads ch0 as sanctuary) would then be quietly wrong, with
    no error anywhere. The map is explicit now; this asserts it stayed that way.
    """
    import numpy as np

    if not duration or duration < 60:
        return
    # Pick the liveliest probe point -- a silent stretch correlates with nothing.
    best_t, best_e = None, -1.0
    for frac in (0.35, 0.45, 0.55, 0.65, 0.75):
        t = duration * frac
        e = float(np.mean(decode_mono(san_path, 8000, start=t, duration=8)
                          .astype(np.float64) ** 2))
        if e > best_e:
            best_t, best_e = t, e
    if best_e < 1e-8:
        info("  channel check: no signal found to test against, skipped")
        return

    def corr(a: np.ndarray, b: np.ndarray) -> float:
        n = min(a.size, b.size)
        if n < 1000:
            return 0.0
        a, b = a[:n].astype(np.float64), b[:n].astype(np.float64)
        a -= a.mean(); b -= b.mean()
        d = float(np.sqrt((a * a).sum() * (b * b).sum()))
        return float((a * b).sum() / d) if d > 0 else 0.0

    src = decode_mono(san_path, 8000, start=best_t, duration=8)
    ch0 = decode_mono(stems, 8000, start=best_t, duration=8, channel=0)
    ch1 = decode_mono(stems, 8000, start=best_t, duration=8, channel=1)
    c0, c1 = corr(src, ch0), corr(src, ch1)

    if c1 > c0:
        die(f"stems channels are SWAPPED: sanctuary correlates {c1:.3f} with ch1 "
            f"but only {c0:.3f} with ch0. Refusing to continue -- "
            "the sermon stage reads ch0 as the sanctuary feed.")
    if c0 < 0.5:
        info(f"  channel check: weak match (ch0 corr {c0:.3f}); "
             "verify the stems by ear before trusting the sermon pass")
    else:
        info(f"  channel check: ch0=sanctuary ok (corr {c0:.3f} vs ch1 {c1:.3f})")


def ambient_chain(rec: dict, rate: int, src_label: str, out_label: str,
                  preserve_delay_ms: float, force_invert: bool | None) -> str:
    """Filter chain that puts the ambient channel onto the sanctuary's timebase."""
    parts = [f"[{src_label}]"]
    steps = ["aformat=sample_fmts=fltp"]

    steps.append(f"aresample={rate}")

    scale = rec["ambient_rate_scale"]
    if abs(scale - 1.0) > 1e-9:
        # Drift correction, done as a genuine polyphase resample.
        #
        # The obvious `asetrate=rate*scale` does not work: asetrate takes an
        # *integer* rate, so at 48 kHz the smallest expressible change is
        # 1/48000 = 20.8 ppm -- coarser than the drift being removed. And atempo,
        # which does take a float, is WSOLA: measured here it left 5 ppm of drift
        # and 4.3 ms RMS of splice jitter.
        #
        # Instead: relabel the stream to a high rate (no resampling, asetrate is
        # just a reinterpretation), resample by the ratio there, then relabel back.
        # Granularity becomes 1/RESAMPLE_BASE = 1 ppm, the filter is smooth, and
        # the sample count is unchanged so it costs no more than a normal resample.
        # Measured residual: 0.8 ppm drift, 0.00 ms RMS jitter.
        q = int(round(RESAMPLE_BASE / scale))
        steps.append(f"asetrate={RESAMPLE_BASE}")
        steps.append(f"aresample={q}")
        steps.append(f"asetrate={rate}")

    invert = rec["invert_ambient_polarity"] if force_invert is None else force_invert
    if invert:
        steps.append("volume=-1")

    # Net shift = what sync measured, MINUS the delay we deliberately keep.
    #
    # The measured offset is start error + acoustic propagation from the pulpit to
    # the loft mics. Only the start error should be removed: the propagation delay
    # is the real arrival time of the room and removing it collapses the sense of
    # depth and invites comb filtering against the dry PA feed. Subtracting keeps
    # it, and still removes a genuinely late recorder start on any week that has one.
    trim = rec["ambient_trim_s"] - preserve_delay_ms / 1000.0
    if trim > 1e-6:
        # Ambient lags: cut the head off so it lines up.
        steps.append(f"atrim=start={trim:.6f}")
        steps.append("asetpts=PTS-STARTPTS")
    elif trim < -1e-6:
        # Ambient leads: pad the head with silence.
        steps.append(f"adelay={abs(trim) * 1000.0:.3f}:all=1")

    parts.append(",".join(steps))
    parts.append(f"[{out_label}]")
    return "".join(parts)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="service-trimmer mix", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sync", required=True, help="sync report from `service-trimmer sync`")
    ap.add_argument("--config", default=None)
    ap.add_argument("--outdir", default=".")
    ap.add_argument("--basename", default=None, help="defaults to the sanctuary filename stem")
    ap.add_argument("--layout", choices=["mono", "stereo-wide", "stems-only"], default=None,
                    help="mix layout (default from config)")
    ap.add_argument("--gain-sanctuary-db", type=float, default=None)
    ap.add_argument("--gain-ambient-db", type=float, default=None)
    ap.add_argument("--preserve-delay-ms", type=float, default=None,
                    help="ms of ambient delay to KEEP (the pulpit->loft acoustic "
                         "propagation time for this room). Default from config.")
    ap.add_argument("--rate", type=int, default=48000, help="output sample rate")
    ap.add_argument("--codec", choices=["flac", "wav"], default="flac")
    ap.add_argument("--width-ms", type=float, default=14.0,
                    help="stereo-wide: L/R ambient delay difference")
    ap.add_argument("--invert-ambient", dest="invert", action="store_true", default=None)
    ap.add_argument("--no-invert-ambient", dest="invert", action="store_false")
    ap.add_argument("--no-auto-gain", action="store_true",
                    help="skip the pre-render peak check that prevents clipping")
    ap.add_argument("--no-stems", action="store_true", help="do not write the stems file")
    args = ap.parse_args(argv)

    require_ffmpeg()
    cfg = load_config(args.config).get("mix", {})
    layout = args.layout or cfg.get("layout", "mono")
    g_san_db = args.gain_sanctuary_db if args.gain_sanctuary_db is not None \
        else cfg.get("gain_sanctuary_db", 0.0)
    g_amb_db = args.gain_ambient_db if args.gain_ambient_db is not None \
        else cfg.get("gain_ambient_db", -12.0)
    setting = args.preserve_delay_ms if args.preserve_delay_ms is not None \
        else cfg.get("preserve_acoustic_delay_ms", 0.0)
    measured_ms = None
    if setting == "auto":
        # These recorders are sample-locked (0 ppm across 2.5 h), so a small
        # measured offset is entirely pulpit->loft propagation and should be kept
        # whole. It is not a constant: 52.85 ms on 2026-08-09, 51.3 ms on
        # 2026-02-08 -- the preacher does not stand in exactly the same spot.
        # Only an implausibly large offset indicates a genuinely late start.
        with open(args.sync, encoding="utf-8") as fh:
            measured_ms = json.load(fh)["recommendation"]["ambient_trim_s"] * 1000.0
        cap = float(cfg.get("acoustic_max_ms", 150.0))
        if abs(measured_ms) <= cap:
            preserve_ms = measured_ms
        else:
            preserve_ms = cap if measured_ms > 0 else -cap
    else:
        preserve_ms = float(setting)

    with open(args.sync, encoding="utf-8") as fh:
        report = json.load(fh)
    rec = report["recommendation"]
    san_path = report["sanctuary"]["path"]
    amb_path = report["ambient"]["path"]
    for p in (san_path, amb_path):
        if not Path(p).exists():
            die(f"input from sync report no longer exists: {p}")

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    base = args.basename or Path(san_path).stem
    ext = "flac" if args.codec == "flac" else "wav"
    enc = ["-c:a", "flac", "-compression_level", "5"] if args.codec == "flac" \
        else ["-c:a", "pcm_s24le"]

    info(f"  offset {rec['ambient_trim_s'] * 1000:+.1f} ms   "
         f"rate scale {rec['ambient_rate_scale']:.8f}   mode {rec['mode']}")
    net_ms = rec["ambient_trim_s"] * 1000.0 - preserve_ms
    if preserve_ms:
        info(f"  preserving {preserve_ms:.2f} ms of acoustic delay "
             f"(~{preserve_ms * 0.343:.1f} m pulpit->loft)")
        info(f"  net shift applied to ambient: {net_ms:+.2f} ms")
        if abs(net_ms) > 500.0:
            info(f"  NOTE  {abs(net_ms):.0f} ms of non-acoustic offset -- one recorder "
                 "likely started late this week; it is being removed as intended")

    amb_chain = ambient_chain(rec, args.rate, "1:a", "amb",
                              preserve_ms, args.invert)
    san_chain = f"[0:a]aformat=sample_fmts=fltp,aresample={args.rate}[san]"
    inputs = [san_path, amb_path]

    # ---- stems -------------------------------------------------------------
    if not args.no_stems:
        stems_path = outdir / f"{base}-stems.{ext}"
        graph = (f"{san_chain};{amb_chain};"
                 f"[san][amb]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[out]")
        run_ffmpeg(
            [FFMPEG, "-v", "error", "-nostdin", "-y",
             "-i", san_path, "-i", amb_path,
             "-filter_complex", graph, "-map", "[out]", *enc, str(stems_path)],
            "stems render",
        )
        info(f"  wrote {stems_path}   (L = sanctuary, R = ambient, aligned)")
        verify_stems_channels(str(stems_path), san_path, amb_path,
                              report.get("sanctuary", {}).get("duration"))

    if layout == "stems-only":
        info("\nlayout=stems-only; no mix rendered.")
        return 0

    # ---- mix ---------------------------------------------------------------
    g_san, g_amb = db_to_lin(g_san_db), db_to_lin(g_amb_db)

    if layout == "mono":
        mix_tail = (f"[san]volume={g_san:.6f}[sg];"
                    f"[amb]volume={g_amb:.6f}[ag];"
                    f"[sg][ag]amix=inputs=2:weights=1 1:normalize=0:duration=shortest[premix]")
    else:
        # The ambient feed is already a mono sum of the two loft mics, so there is
        # no real stereo image to recover. This synthesises width by offsetting the
        # ambient a few ms between L and R -- a genuine effect, not a reconstruction.
        mix_tail = (
            f"[san]volume={g_san:.6f},pan=stereo|c0=c0|c1=c0[sg];"
            f"[amb]volume={g_amb:.6f},asplit=2[al][ar];"
            f"[ar]adelay={args.width_ms:.3f}:all=1[ard];"
            f"[al][ard]join=inputs=2:channel_layout=stereo:map=0.0-FL|1.0-FR[ag];"
            f"[sg][ag]amix=inputs=2:weights=1 1:normalize=0:duration=shortest[premix]"
        )

    def build_graph(trim_db: float) -> str:
        return (f"{san_chain};{amb_chain};{mix_tail};"
                f"[premix]volume={db_to_lin(trim_db):.6f}[out]")

    trim_db = 0.0
    if not args.no_auto_gain:
        info("  checking peak level (one analysis pass, no audio written)...")
        peak = measure_peak_db(build_graph(0.0), inputs, "out")
        info(f"  peak would be {peak:+.2f} dBFS")
        if peak > -0.3:
            trim_db = -(peak + 0.3)
            info(f"  applying {trim_db:.2f} dB to stay below clipping "
                 f"(pass --no-auto-gain to disable)")

    graph = build_graph(trim_db)
    mix_path = outdir / f"{base}-mix.{ext}"
    run_ffmpeg(
        [FFMPEG, "-v", "error", "-nostdin", "-y",
         "-i", san_path, "-i", amb_path,
         "-filter_complex", graph, "-map", "[out]", *enc, str(mix_path)],
        "mix render",
    )
    info(f"  wrote {mix_path}   (layout={layout}, sanctuary {g_san_db:+.1f} dB, "
         f"ambient {g_amb_db:+.1f} dB)")
    return 0

