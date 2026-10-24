"""Process every recording in an ilc-Audio-Recorder upload directory.

    service-trimmer batch --audio-dir ~/recordings --workdir ~/service-trimmer-work --extract

Per service: triage -> sync -> mix -> transcribe -> sermon (-> extract). Every
stage writes its artefact into --workdir and is skipped when that artefact
already exists, so the batch is safe to re-run as new recordings arrive (e.g.
nightly from cron). One failing service never stops the others.

Triage routes each recording before anything expensive runs:
  ok              both channels live          -> full pipeline
  degraded        one channel dead or missing -> single-channel transcript mode
  silent / dead   nothing recorded            -> reported, not processed
  test-recording  short or outside service hours -> reported, not processed
"""

from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

from .audio import hms, info, write_json
from .config import load_config
from .naming import Service, discover
from .schedule import triage

STAGES = ["sync", "mix", "transcribe", "sermon"]


class Runner:
    def __init__(self, workdir: Path, config: str | None):
        self.workdir = workdir
        self.cfg_args = ["--config", config] if config else []

    def __call__(self, key: str, stage: str, argv: list[str]) -> bool:
        cmd = [sys.executable, "-m", "service_trimmer", stage, *argv]
        log = self.workdir / "logs" / f"{key}-{stage}.log"
        log.parent.mkdir(parents=True, exist_ok=True)
        started = time.time()
        with open(log, "w", encoding="utf-8") as fh:
            fh.write(f"$ {' '.join(cmd)}\n")
            fh.flush()
            proc = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT)
            fh.write(f"\n[exit {proc.returncode} after {time.time() - started:.0f}s]\n")
        return proc.returncode == 0


def process(svc: Service, args, cfg: dict, run: Runner) -> dict:
    wd = Path(args.workdir)
    key = svc.key
    force = set(args.force or [])
    rec: dict = {"key": key, "job_id": svc.job_id, "start": svc.start.isoformat(timespec="minutes")}

    tri = triage(svc, cfg, wd)
    rec.update({"triage": tri["status"], "why": tri["why"], "mode": tri["mode"]})
    if svc.duplicates:
        rec["duplicates"] = svc.duplicates
    if tri["status"] not in ("ok", "degraded"):
        rec["status"] = f"skipped: {tri['status']}"
        return rec

    mode = tri["mode"]
    p = {name: wd / f"{key}{suffix}" for name, suffix in {
        "sync": "-sync.json", "stems": "-stems.flac", "mix": "-mix.flac",
        "sermon": "-sermon.json"}.items()}

    def needed(stage: str, artefact: Path) -> bool:
        return stage in force or "all" in force or not artefact.exists()

    if mode == "dual":
        scfg = cfg.get("sync", {})
        if needed("sync", p["sync"]):
            ok = run(key, "sync", ["--sanctuary", str(svc.sanctuary), "--ambient", str(svc.ambient),
                                   "-o", str(p["sync"]), "--step", str(scfg.get("step_s", 300)),
                                   "--max-offset", str(scfg.get("max_offset_s", 5)),
                                   "--quiet-db", str(scfg.get("quiet_below_channel_p90_db", 30))])
            if not ok:
                # Channels that will not align cannot be mixed, but each still
                # holds the service: carry on with the PA feed alone.
                mode = "sanctuary-only"
                rec["notes"] = ["sync failed; fell back to sanctuary-only"]
        if mode == "dual" and needed("mix", p["stems"]):
            if not run(key, "mix", ["--sync", str(p["sync"]), "--outdir", str(wd),
                                    "--basename", key, *run.cfg_args]):
                rec["status"] = "failed: mix"
                return rec
    rec["mode"] = mode

    # Transcript of whichever channel carries the preacher.
    if mode == "ambient-only":
        audio, tr_stem = svc.ambient, wd / f"{key}-loft"
    else:
        audio, tr_stem = svc.sanctuary, wd / key
    tr_json = Path(f"{tr_stem}.json")
    if needed("transcribe", tr_json):
        argv = ["--audio", str(audio), "-o", str(tr_stem), *run.cfg_args]
        if mode == "ambient-only":
            # Silero VAD rejects distant, reverberant loft speech almost entirely
            # (2026-07-26: 561 words in 2.5 h with VAD; ~2 words/s without).
            argv.append("--no-vad")
        if args.threads:
            argv += ["--threads", str(args.threads)]
        if not run(key, "transcribe", argv):
            rec["status"] = "failed: transcribe"
            return rec

    if needed("sermon", p["sermon"]) or (args.extract and not (wd / f"{key}-sermon.flac").exists()):
        argv = ["--transcript", str(tr_json), "-o", str(p["sermon"]), "--key", key,
                "--job-id", svc.job_id, *run.cfg_args]
        argv += ["--stems", str(p["stems"])] if mode == "dual" else \
            ["--audio", str(audio), "--mode", mode]
        if args.threads:
            argv += ["--threads", str(args.threads)]
        if args.recorder_db:
            argv += ["--recorder-db", args.recorder_db]
        if args.no_edges:
            argv.append("--no-edges")
        if args.extract:
            argv += ["--extract", "--mp3"]
        if not run(key, "sermon", argv):
            rec["status"] = "failed: sermon"
            return rec

    rec.update(describe(p["sermon"], rec.get("notes", [])))
    if mode == "dual" and p["sync"].exists():
        sync = json.loads(p["sync"].read_text())
        rec["offset_ms"] = round(sync["fit"]["offset_s"] * 1000, 1)
        rec["drift_ppm"] = round(sync["fit"]["drift_ppm"], 2)
    if args.prune and mode == "dual":
        p["stems"].unlink(missing_ok=True)
    return rec


def describe(sermon_json: Path, notes: list[str] | None = None) -> dict:
    rep = json.loads(sermon_json.read_text())
    s = rep["sermon"]
    return {
        "status": "ok",
        "sermon_start": s["start"], "sermon_end": s["end"],
        "minutes": round(s["duration"] / 60.0, 1),
        "start_method": s["start_method"], "end_method": s["end_method"],
        "confidence": s["confidence"], "flags": s["flags"],
        "notes": (notes or []) + s.get("notes", []),
        "clock": s.get("clock"), "mode": rep.get("mode"),
    }


def existing_record(svc: Service, wd: Path) -> dict:
    """Summary row for a service this run did not touch (e.g. outside --only)."""
    rec: dict = {"key": svc.key, "job_id": svc.job_id,
                 "start": svc.start.isoformat(timespec="minutes")}
    tri = wd / f"{svc.key}-triage.json"
    if tri.exists():
        t = json.loads(tri.read_text())
        rec.update({"triage": t["status"], "why": t["why"], "mode": t["mode"]})
        if t["status"] not in ("ok", "degraded"):
            rec["status"] = f"skipped: {t['status']}"
    if svc.duplicates:
        rec["duplicates"] = svc.duplicates
    sj = wd / f"{svc.key}-sermon.json"
    if sj.exists():
        rec.update(describe(sj))
    rec.setdefault("status", "not processed yet")
    return rec


def write_summary(wd: Path, results: list[dict], warnings: list[str], elapsed: float) -> Path:
    results = sorted(results, key=lambda r: r["key"])
    write_json(wd / "summary.json", {"results": results, "warnings": warnings})
    cols = ["key", "job_id", "status", "mode", "confidence", "minutes", "sermon_start",
            "sermon_end", "start_method", "end_method", "flags"]
    with open(wd / "summary.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols + ["clock_start", "clock_end"])
        for r in results:
            clock = r.get("clock") or {}
            w.writerow([("; ".join(r.get(c) or []) if c == "flags" else r.get(c, ""))
                        for c in cols] + [clock.get("sermon_start", ""), clock.get("sermon_end", "")])

    ok = [r for r in results if r.get("status") == "ok"]
    by = {c: sum(1 for r in ok if r.get("confidence") == c) for c in ("high", "medium", "low")}
    L = ["Sermon detection summary", "=" * 100, "",
         f"{len(ok)} sermons found ({by['high']} high, {by['medium']} medium, {by['low']} low "
         f"confidence); {len(results) - len(ok)} recordings not processed. "
         f"{elapsed / 60:.0f} min.", "",
         "Each sermon runs from the preacher's lead-in (invocation, acclamation, greeting or",
         "prayer, when there is one) to the end of the closing formula plus the",
         "congregation's Amen. Review low/medium first; the flags say why.", "",
         f"{'service':<16} {'conf':<6} {'len':>6}  {'start':>9} {'end':>9}  {'clock':<11}  "
         f"{'start via':<28} {'end via':<34} flags",
         "-" * 140]
    for r in results:
        if r.get("status") != "ok":
            L.append(f"{r['key']:<16} {r.get('status')}  ({r.get('why', '')})")
            continue
        clock = r.get("clock") or {}
        cl = f"{clock.get('sermon_start', '')[:5]}-{clock.get('sermon_end', '')[:5]}"
        L.append(f"{r['key']:<16} {r['confidence']:<6} {r['minutes']:>5.1f}m  "
                 f"{hms(r['sermon_start'])[:8]:>9} {hms(r['sermon_end'])[:8]:>9}  {cl:<11}  "
                 f"{r['start_method']:<28} {r['end_method']:<34} {'; '.join(r['flags'])}")
    quiet_amen = [r["key"] for r in ok
                  if any(n.startswith("no congregational response") for n in r.get("notes", []))]
    noted = [(r["key"], n) for r in ok for n in r.get("notes", [])
             if not n.startswith("no congregational response")]
    if noted or quiet_amen:
        L += ["", "Notes (informational):"]
        for key, n in noted:
            L.append(f"  {key}: {n}")
        if quiet_amen:
            L.append(f"  {len(quiet_amen)} of {len(ok)} services: congregation's Amen not "
                     "distinct on the loft mics; a fixed allowance after the closing formula "
                     "was kept instead (bounded by the next speaker).")
    dups = [r for r in results if r.get("duplicates")]
    if dups or warnings:
        L += ["", "Files:"]
        for r in dups:
            L.append(f"  {r['key']}: duplicate copies ignored: {', '.join(r['duplicates'])}")
        for w in warnings:
            L.append(f"  {w}")
    path = wd / "SUMMARY.txt"
    path.write_text("\n".join(L) + "\n", encoding="utf-8")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="service-trimmer batch", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", required=True, help="where the recorder's WAVs land")
    ap.add_argument("--workdir", required=True, help="artefacts, logs and the summary")
    ap.add_argument("--config", default=None)
    ap.add_argument("--jobs", type=int, default=3, help="services processed in parallel")
    ap.add_argument("--threads", type=int, default=None, help="CPU threads per transcription")
    ap.add_argument("--min-age", type=float, default=300.0,
                    help="ignore files modified this recently (still uploading)")
    ap.add_argument("--only", nargs="+", default=None, help="service keys to process")
    ap.add_argument("--force", nargs="+", choices=STAGES + ["all"], default=None,
                    help="re-run these stages even if their output exists")
    ap.add_argument("--extract", action="store_true", help="write sermon FLAC + MP3")
    ap.add_argument("--recorder-db", default=None, help="recorder schedule.db (section hints)")
    ap.add_argument("--no-edges", action="store_true")
    ap.add_argument("--prune", action="store_true", help="delete stems after success")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    wd = Path(args.workdir).expanduser()
    wd.mkdir(parents=True, exist_ok=True)
    args.workdir = str(wd)
    audio_dir = Path(args.audio_dir).expanduser()
    services, warnings = discover(audio_dir, cfg["recorder"]["channel_suffixes"], args.min_age)
    untouched = [s for s in services if args.only and s.key not in args.only]
    if args.only:
        services = [s for s in services if s.key in args.only]
    info(f"{len(services)} recording(s) in {audio_dir}")

    run = Runner(wd, args.config)
    started = time.time()
    results: list[dict] = []
    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futs = {pool.submit(process, s, args, cfg, run): s for s in services}
        for fut in as_completed(futs):
            svc = futs[fut]
            try:
                rec = fut.result()
            except Exception as exc:  # one bad service must not stop the batch
                rec = {"key": svc.key, "job_id": svc.job_id, "status": f"failed: {exc}"}
            results.append(rec)
            line = rec.get("status", "?")
            if line == "ok":
                line = (f"{rec['minutes']:.1f} min  {rec['confidence']}  "
                        f"{rec['start_method']} / {rec['end_method']}")
            info(f"  [{len(results)}/{len(services)}] {svc.key}  {line}")

    # The summary always covers the whole archive, not just this run.
    results += [existing_record(s, wd) for s in untouched]
    path = write_summary(wd, results, warnings, time.time() - started)
    info(f"\nsummary: {path}")
    return 0
