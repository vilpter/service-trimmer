#!/usr/bin/env python3
"""End-to-end self-test against a synthetic service with known ground truth.

    ./venv/bin/python tests/selftest.py

Exercises sync -> mix -> sermon through the real `service-trimmer` CLI and checks each
result against what make_fixture.py baked in. Run it after installing on a new
machine (a Pi especially): it turns "it didn't crash" into a pass/fail.

Transcription is not covered -- the fixture holds no real speech. Edge
re-transcription is disabled for the same reason; tests/test_edges.py covers
that logic with synthetic envelopes.
"""

from __future__ import annotations

import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
CFG = HERE / "fixture-config.json"
CHECKS: list[tuple] = []


def check(label, got, want, tol, unit=""):
    CHECKS.append((label, got, want, tol, unit, abs(got - want) <= tol))


def run(*argv: str) -> None:
    proc = subprocess.run([sys.executable, "-m", "service_trimmer", *argv], capture_output=True, text=True)
    if proc.returncode != 0:
        print(f"\n!! service-trimmer {argv[0]} failed:\n{proc.stderr[-3000:]}", file=sys.stderr)
        raise SystemExit(2)


def main() -> int:
    work = Path(tempfile.mkdtemp(prefix="ilc-selftest-"))
    try:
        print(f"workdir {work}\n\n[1/4] generating fixture")
        subprocess.run([sys.executable, str(HERE / "make_fixture.py"), "--outdir", str(work / "fx")],
                       check=True, capture_output=True)
        truth = json.loads((work / "fx" / "fixture-truth.json").read_text())

        print("[2/4] sync")
        sync = work / "sync.json"
        run("sync", "--sanctuary", truth["sanctuary"], "--ambient", truth["ambient"],
            "-o", str(sync), "--step", "90", "--max-offset", "3")
        s = json.loads(sync.read_text())
        check("sync offset", s["fit"]["offset_s"] * 1000, truth["offset_s"] * 1000, 15.0, "ms")
        check("sync drift", s["fit"]["drift_ppm"], truth["drift_ppm"], 5.0, "ppm")
        check("sync residual", s["fit"]["residual_rms_ms"], 0.0, 5.0, "ms")

        print("[3/4] mix")
        run("mix", "--sync", str(sync), "--config", str(CFG), "--outdir", str(work),
            "--basename", "fx")
        stems = work / "fx-stems.flac"
        if not stems.exists() or stems.stat().st_size == 0:
            print("!! stems file missing or empty", file=sys.stderr)
            return 2

        print("[4/4] sermon")
        rep_path = work / "fx-sermon.json"
        run("sermon", "--stems", str(stems), "--transcript", truth["transcript"],
            "-o", str(rep_path), "--config", str(CFG), "--key", "fixture", "--no-edges")
        rep = json.loads(rep_path.read_text())["sermon"]
        check("sermon start", rep["start"], truth["sermon_start_s"], 6.0, "s")
        check("sermon end", rep["end"], truth["sermon_end_s"], 6.0, "s")
        CHECKS.append(("start via opening", rep["start_method"], "opening:*", None, "",
                       rep["start_method"].startswith("opening:")))
        CHECKS.append(("end via closing", rep["end_method"], "closing:*", None, "",
                       rep["end_method"].startswith("closing:")))
    finally:
        shutil.rmtree(work, ignore_errors=True)

    print()
    failed = 0
    for label, got, want, tol, unit, ok in CHECKS:
        failed += not ok
        if tol is None:
            print(f"  {'PASS' if ok else 'FAIL'}  {label:<20} {got}")
        else:
            print(f"  {'PASS' if ok else 'FAIL'}  {label:<20} got {got:9.2f} {unit:<3} "
                  f"want {want:9.2f} +/- {tol:g}")
    print(f"\n{len(CHECKS) - failed}/{len(CHECKS)} checks passed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
