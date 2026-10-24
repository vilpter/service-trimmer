"""Integration with ilc-Audio-Recorder's database (~/.audio-recorder/schedule.db).

Read side (always safe, opened read-only):
  * recording_sections -- the "Mark Beginning / Mark Ending" taps an operator
    makes during a live recording. When one exists it is the best possible hint
    for where the sermon is, and `service-trimmer sermon --recorder-db` uses it to choose
    between candidate windows.
  * scheduled_jobs     -- the recurring schedule, for `service-trimmer schedule`.

Write side (explicit opt-in): `service-trimmer recorder push-sections` inserts each
detected sermon as a section, labelled "Sermon (auto)", so it shows up in the
recorder's Recordings page next to the audio. Dry-run unless --apply. It never
touches rows it did not create, and re-running replaces its own rows rather than
duplicating them.

Schema as of ilc-Audio-Recorder v1.10.1 (ARCHITECTURE.md):
    recording_sections(id, job_id, label, start_offset, end_offset, created_at)
    scheduled_jobs(id, name, start_time, duration, ..., is_recurring, recurrence_pattern, ...)
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from datetime import datetime
from pathlib import Path

from .audio import die, hms, info

AUTO_LABEL = "Sermon (auto)"


def connect_ro(db: str | Path) -> sqlite3.Connection:
    p = Path(db).expanduser()
    if not p.exists():
        die(f"recorder database not found: {p}")
    return sqlite3.connect(f"file:{p}?mode=ro", uri=True)


def has_table(con: sqlite3.Connection, name: str) -> bool:
    return con.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                       (name,)).fetchone() is not None


def read_sections(db: str | Path, job_id: str) -> list[dict]:
    """Sections tagged on one recording, oldest first. [] if none or no table."""
    with connect_ro(db) as con:
        if not has_table(con, "recording_sections"):
            return []
        rows = con.execute(
            "SELECT id, label, start_offset, end_offset, created_at FROM recording_sections "
            "WHERE job_id = ? ORDER BY start_offset", (job_id,)).fetchall()
    return [{"id": r[0], "label": r[1], "start": r[2], "end": r[3], "created_at": r[4]}
            for r in rows]


def manual_sermon_hint(sections: list[dict]) -> dict | None:
    """The operator's section most likely to be the sermon.

    Prefer one labelled like a sermon; otherwise the longest closed section that
    we did not write ourselves.
    """
    human = [s for s in sections if s.get("label") != AUTO_LABEL and s.get("end") is not None
             and s["end"] > s["start"]]
    if not human:
        return None
    named = [s for s in human if "sermon" in (s.get("label") or "").lower()
             or "homily" in (s.get("label") or "").lower()]
    pool = named or human
    best = max(pool, key=lambda s: s["end"] - s["start"])
    return {"start": float(best["start"]), "end": float(best["end"]),
            "label": best.get("label"), "labelled_sermon": bool(named)}


def read_jobs(db: str | Path) -> list[dict]:
    with connect_ro(db) as con:
        if not has_table(con, "scheduled_jobs"):
            return []
        cols = [c[1] for c in con.execute("PRAGMA table_info(scheduled_jobs)")]
        rows = con.execute(f"SELECT {', '.join(cols)} FROM scheduled_jobs").fetchall()
    jobs = []
    for r in rows:
        job = dict(zip(cols, r))
        pat = job.get("recurrence_pattern")
        if isinstance(pat, str) and pat.strip().startswith("{"):
            try:
                job["recurrence_pattern"] = json.loads(pat)
            except json.JSONDecodeError:
                pass
        jobs.append(job)
    return jobs


def push_sections(db: str | Path, reports: list[dict], apply: bool) -> list[str]:
    """Write detected sermons into recording_sections. Returns a log of actions."""
    p = Path(db).expanduser()
    if not p.exists():
        die(f"recorder database not found: {p}")
    log: list[str] = []
    con = sqlite3.connect(str(p))
    try:
        if not has_table(con, "recording_sections"):
            die("this recorder database has no recording_sections table "
                "(section tagging arrived in ilc-Audio-Recorder v1.10.0)")
        now = datetime.now().isoformat(timespec="seconds")
        for rep in reports:
            job_id = rep.get("job_id")
            s = rep.get("sermon") or {}
            if not job_id or s.get("start") is None:
                continue
            old = con.execute("SELECT COUNT(*) FROM recording_sections WHERE job_id=? AND label=?",
                              (job_id, AUTO_LABEL)).fetchone()[0]
            verb = "replace" if old else "add"
            log.append(f"{verb:<7} {job_id}  {hms(s['start'])} -> {hms(s['end'])}  "
                       f"[{s.get('confidence', '?')}]")
            if apply:
                con.execute("DELETE FROM recording_sections WHERE job_id=? AND label=?",
                            (job_id, AUTO_LABEL))
                con.execute("INSERT INTO recording_sections (job_id, label, start_offset, "
                            "end_offset, created_at) VALUES (?, ?, ?, ?, ?)",
                            (job_id, AUTO_LABEL, float(s["start"]), float(s["end"]), now))
        if apply:
            con.commit()
    finally:
        con.close()
    return log


# --------------------------------------------------------------------------- #

def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="service-trimmer recorder", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)

    s1 = sub.add_parser("sections", help="list section tags for a recording")
    s1.add_argument("--db", required=True)
    s1.add_argument("--job-id", required=True, help="e.g. 2026_Feb_08_09:40")

    s2 = sub.add_parser("jobs", help="list the recorder's scheduled jobs")
    s2.add_argument("--db", required=True)

    s3 = sub.add_parser("push-sections", help="write detected sermons as section tags")
    s3.add_argument("--db", required=True)
    s3.add_argument("--workdir", required=True, help="directory holding *-sermon.json")
    s3.add_argument("--min-confidence", choices=["low", "medium", "high"], default="medium")
    s3.add_argument("--apply", action="store_true", help="actually write (default: dry run)")
    args = ap.parse_args(argv)

    if args.cmd == "sections":
        for s in read_sections(args.db, args.job_id):
            info(f"  {s['label'] or '(unlabelled)':<20} {hms(s['start'])} -> {hms(s['end'])}")
        return 0
    if args.cmd == "jobs":
        for j in read_jobs(args.db):
            info(f"  {j.get('name')!s:<28} start {j.get('start_time')}  "
                 f"{int(j.get('duration') or 0) / 60:.0f} min  recurring={j.get('is_recurring')}  "
                 f"{j.get('recurrence_pattern')}")
        return 0

    rank = {"low": 0, "medium": 1, "high": 2}
    reports = []
    for p in sorted(Path(args.workdir).glob("*-sermon.json")):
        rep = json.loads(p.read_text())
        conf = (rep.get("sermon") or {}).get("confidence", "low")
        if rank.get(conf, 0) >= rank[args.min_confidence]:
            reports.append(rep)
    for line in push_sections(args.db, reports, args.apply):
        info("  " + line)
    if not args.apply:
        info("\n  dry run -- nothing written. Re-run with --apply. Stop the recorder service "
             "first if it is running (sudo systemctl stop audio-recorder).")
    return 0
