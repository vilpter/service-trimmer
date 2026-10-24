"""Measure the time alignment between the sanctuary and ambient recordings.

Emits a sync report (JSON) describing the start offset, the relative clock drift
between the two recorders, and how confident the measurement is. `service-trimmer mix`
consumes that report.

    service-trimmer sync --sanctuary A.wav --ambient B.wav -o sync-report.json
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from .audio import (
    decode_mono, die, fit_drift, gcc_phat, hms, info, probe, require_ffmpeg,
    select_dominant_cluster, write_json,
)


def parse_band(text: str) -> tuple[float, float] | None:
    if text.lower() in ("none", "off", ""):
        return None
    lo, _, hi = text.partition(":")
    return (float(lo), float(hi))


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="service-trimmer sync", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--sanctuary", required=True, help="mic-mix / PA feed (the reference clock)")
    ap.add_argument("--ambient", required=True, help="choir-loft mics")
    ap.add_argument("--sanctuary-channel", type=int, default=None,
                    help="channel index if the file is multi-channel")
    ap.add_argument("--ambient-channel", type=int, default=None)
    ap.add_argument("-o", "--output", default="sync-report.json")
    ap.add_argument("--rate", type=int, default=16000, help="analysis sample rate (default 16000)")
    ap.add_argument("--window", type=float, default=30.0, help="correlation window seconds")
    ap.add_argument("--step", type=float, default=240.0, help="seconds between windows")
    ap.add_argument("--max-offset", type=float, default=10.0,
                    help="largest offset to search, seconds (default 10)")
    ap.add_argument("--band", default="200:4000", help="speech band for PHAT, Hz, or 'none'")
    ap.add_argument("--min-sharpness", type=float, default=6.0,
                    help="reject windows whose correlation peak is weaker than this")
    ap.add_argument("--quiet-db", type=float, default=30.0,
                    help="reject windows this far below the channel's own 90th-percentile "
                         "window level (relative, so low-gain recordings still sync)")
    ap.add_argument("--floor-db", type=float, default=-85.0,
                    help="absolute floor: windows below this are always rejected")
    ap.add_argument("--cluster-tolerance-ms", type=float, default=25.0,
                    help="offsets within this of each other count as one acoustic path")
    args = ap.parse_args(argv)

    require_ffmpeg()
    band = parse_band(args.band)

    san = probe(args.sanctuary)
    amb = probe(args.ambient)
    for label, meta in (("sanctuary", san), ("ambient", amb)):
        info(f"{label:>9}: {Path(meta['path']).name}  {hms(meta['duration'])}  "
             f"{meta['sample_rate']} Hz  {meta['channels']} ch  {meta['codec']}")

    if san["duration"] is None or amb["duration"] is None:
        die("could not determine duration of both inputs")

    span = min(san["duration"], amb["duration"])
    if span < args.window * 2:
        die(f"recordings are only {hms(span)} long; need at least {args.window * 2:.0f}s")

    # Leave a max-offset guard band at each end so a window never runs off the
    # front or back of the shorter file while it is being shifted.
    guard = args.max_offset + 1.0
    starts = np.arange(guard, span - args.window - guard, args.step)
    if starts.size == 0:
        starts = np.array([max(0.0, (span - args.window) / 2.0)])

    info(f"\nprobing {starts.size} windows of {args.window:.0f}s "
         f"across {hms(span)} (band {args.band}, search +/-{args.max_offset:.0f}s)\n")

    measurements: list[dict] = []
    for start in starts:
        # The ambient slice is widened by max_offset on both sides so the true
        # match is inside it no matter which way the offset runs.
        a = decode_mono(args.ambient, args.rate,
                        start=start - args.max_offset,
                        duration=args.window + 2 * args.max_offset,
                        channel=args.ambient_channel)
        b = decode_mono(args.sanctuary, args.rate,
                        start=start, duration=args.window,
                        channel=args.sanctuary_channel)
        if a.size == 0 or b.size == 0:
            continue

        a_db = 10.0 * np.log10(max(float(np.mean(a.astype(np.float64) ** 2)), 1e-12))
        b_db = 10.0 * np.log10(max(float(np.mean(b.astype(np.float64) ** 2)), 1e-12))

        # The ambient slice starts max_offset earlier than the sanctuary slice, so
        # a true offset of T shows up in the arrays as T + max_offset. That lands
        # in [0, 2*max_offset], which is what the correlator has to search.
        raw_tau, sharpness, inverted = gcc_phat(a, b, args.rate,
                                                max_tau=2.0 * args.max_offset, band=band)
        tau = raw_tau - args.max_offset

        measurements.append({
            "t": float(start), "tau": float(tau), "sharpness": float(sharpness),
            "sanctuary_db": round(b_db, 1), "ambient_db": round(a_db, 1),
            "polarity_inverted": bool(inverted),
        })

    # Quiet gate, relative to each channel's own level. An absolute threshold
    # (-50 dBFS) rejected every window of services recorded with low input gain;
    # what matters is whether a window has program in it relative to the rest.
    if measurements:
        gate_san = max(args.floor_db, float(np.percentile(
            [m["sanctuary_db"] for m in measurements], 90)) - args.quiet_db)
        gate_amb = max(args.floor_db, float(np.percentile(
            [m["ambient_db"] for m in measurements], 90)) - args.quiet_db)
    for m in measurements:
        quiet = m["sanctuary_db"] < gate_san or m["ambient_db"] < gate_amb
        m["usable"] = bool((not quiet) and m["sharpness"] >= args.min_sharpness
                           and abs(m["tau"]) <= args.max_offset)
        flag = "  " if m["usable"] else ("quiet" if quiet else " weak")
        info(f"  {hms(m['t'])}  tau={m['tau'] * 1000:+9.2f} ms  sharp={m['sharpness']:6.1f}  "
             f"san={m['sanctuary_db']:6.1f} dB  amb={m['ambient_db']:6.1f} dB  {flag}")

    usable = [m for m in measurements if m["usable"]]
    if len(usable) < 2:
        quiet = sum(1 for m in measurements
                    if min(m["sanctuary_db"], m["ambient_db"]) < args.floor_db)
        if quiet == len(measurements) and measurements:
            die(f"all {len(measurements)} windows are below {args.floor_db:g} dBFS in at "
                "least one channel -- one feed appears to be silent. Check the "
                "source files before re-running.")
        die(f"only {len(usable)} usable windows out of {len(measurements)}. "
            "Try --min-sharpness 4, a wider --max-offset, or --band none.")

    # Separate the two acoustic paths before fitting; see select_dominant_cluster.
    usable, dropped = select_dominant_cluster(usable, args.cluster_tolerance_ms / 1000.0)
    for m in dropped:
        m["used_in_fit"] = False
        m["dropped_reason"] = "competing acoustic path"
    if dropped:
        info(f"\n  discarded {len(dropped)} window(s) on a competing acoustic path "
             f"(median {1000 * sorted(x['tau'] for x in dropped)[len(dropped) // 2]:+.1f} ms, "
             f"mean sharpness {sum(x['sharpness'] for x in dropped) / len(dropped):.1f}) "
             f"in favour of {len(usable)} at "
             f"{1000 * sorted(x['tau'] for x in usable)[len(usable) // 2]:+.1f} ms "
             f"(mean sharpness {sum(x['sharpness'] for x in usable) / len(usable):.1f})")
        if len(dropped) > len(usable):
            info("  WARNING  the discarded group was the larger one -- if this recording "
                 "has real clock drift, clustering may have split it. Check 'measurements'.")

    fit = fit_drift([m["t"] for m in usable], [m["tau"] for m in usable])

    # Map the used-mask from the fit back onto the full measurement list.
    for m, kept in zip(usable, fit.pop("used_mask")):
        m["used_in_fit"] = bool(kept)
    for m in measurements:
        m.setdefault("used_in_fit", False)

    drift_ppm = fit["drift_ppm"]
    # Total slip the drift causes end-to-end. Under ~25 ms is inaudible in a mix
    # of a close mic against a room mic, so a constant shift is fine there.
    total_slip_ms = abs(drift_ppm) * span / 1000.0
    mode = "resample" if total_slip_ms > 25.0 else "constant"

    rate_scale = 1.0 + drift_ppm / 1e6 if mode == "resample" else 1.0
    # Positive offset means ambient lags, so ambient gets trimmed at the front.
    trim_s = fit["offset_s"] / rate_scale

    inverted_votes = sum(1 for m in usable if m["polarity_inverted"])
    polarity_warning = inverted_votes > len(usable) / 2

    report = {
        "sanctuary": san,
        "ambient": amb,
        "analysis": {
            "rate": args.rate, "window_s": args.window, "step_s": args.step,
            "max_offset_s": args.max_offset, "band_hz": args.band,
            "min_sharpness": args.min_sharpness, "quiet_db": args.quiet_db,
            "floor_db": args.floor_db,
        },
        "measurements": measurements,
        "fit": fit,
        "overlap_s": float(span),
        "total_drift_slip_ms": float(total_slip_ms),
        "polarity_inverted": bool(polarity_warning),
        "recommendation": {
            "mode": mode,
            "ambient_rate_scale": float(rate_scale),
            "ambient_trim_s": float(trim_s),
            "invert_ambient_polarity": bool(polarity_warning),
        },
    }
    write_json(args.output, report)

    info("")
    info(f"  start offset      {fit['offset_s'] * 1000:+.1f} ms  "
         f"(ambient {'lags' if fit['offset_s'] > 0 else 'leads'} sanctuary)")
    info(f"  clock drift       {drift_ppm:+.1f} ppm  "
         f"-> {total_slip_ms:.0f} ms of slip over {hms(span)}")
    info(f"  fit residual      {fit['residual_rms_ms']:.2f} ms RMS "
         f"({fit['n_used']}/{len(measurements)} windows)")
    info(f"  correction mode   {mode}"
         + ("  (constant shift is enough)" if mode == "constant"
            else f"  (resample ambient by {rate_scale:.8f})"))
    if polarity_warning:
        info("  WARNING           ambient appears polarity-inverted vs sanctuary; "
             "the mix stage will flip it")
    if fit["residual_rms_ms"] > 15.0:
        info("  WARNING           high residual -- the linear drift model may not fit. "
             "Inspect 'measurements' in the report before trusting the mix.")
    info(f"\nwrote {args.output}")
    return 0

