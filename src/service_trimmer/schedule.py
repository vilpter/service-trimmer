"""Check recordings against the church's service schedule.

    service-trimmer schedule --audio-dir ~/recordings --workdir work/ [-o SCHEDULE.txt]

For every recording: when program audio actually starts and stops (from channel
levels, so it works without transcription), how much dead air precedes it,
whether the recording was still running program when it stopped (truncated),
and which of the services in force that day the recording window covers.

Then, for the current schedule era, a recommended recorder schedule built from
what the recordings show (typical program length, lead-in), ready to enter in
ilc-Audio-Recorder's Schedule page.

Why this exists: on 2026-09-13 the church moved from one 10:00 service to
8:00 + 10:30, and the recorder kept its 09:40-12:10 window. Every recording
since then misses the early service, opens with ~45 minutes of silence, and
runs the late service right up to the moment recording stops.
"""

from __future__ import annotations

import argparse
import json
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np

from .audio import channel_level_stats, die, find_runs, info, median_smooth, probe, write_json
from .config import load_config
from .naming import Service, discover

WEEKDAYS = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
FRAME_S = 10.0


# --------------------------------------------------------------------------- #
# Triage: what is on each channel, and where is the program
# --------------------------------------------------------------------------- #

def activity_window(db: np.ndarray, frame_s: float = FRAME_S) -> dict | None:
    """First and last sustained program audio in a 10 s level series.

    Active = more than 12 dB over the recording's own noise floor (5th
    percentile), median-smoothed over 70 s so a door or a cough does not count,
    and kept only in runs of at least a minute.
    """
    if db.size < 6:
        return None
    floor = float(np.percentile(db, 5))
    active = median_smooth((db > floor + 12.0).astype(float), 7) > 0.5
    runs = [(a, b) for a, b in find_runs(active) if (b - a) * frame_s >= 60.0]
    if not runs:
        return None
    a, b = runs[0][0] * frame_s, runs[-1][1] * frame_s
    total = db.size * frame_s
    return {"start": a, "end": min(b, total), "noise_floor_db": round(floor, 1),
            "tail_s": round(total - min(b, total), 1),
            "truncated": bool(total - min(b, total) < 60.0)}


def triage(svc: Service, cfg: dict, cache_dir: Path | None = None) -> dict:
    """Classify a recording before any expensive stage runs."""
    cache = cache_dir / f"{svc.key}-triage.json" if cache_dir else None
    if cache and cache.exists():
        old = json.loads(cache.read_text())
        if old.get("inputs") == svc.to_dict():
            return old

    rcfg = cfg.get("recorder", {})
    out: dict = {"key": svc.key, "job_id": svc.job_id, "inputs": svc.to_dict(), "channels": {}}
    duration = None
    levels: dict[str, np.ndarray] = {}
    for role in ("sanctuary", "ambient"):
        path = getattr(svc, role)
        if path is None:
            continue
        meta = probe(path)
        duration = max(duration or 0.0, meta["duration"] or 0.0)
        st = channel_level_stats(path)
        levels[role] = st.pop("db")
        st["silent"] = bool(st["p95"] < float(rcfg.get("silent_dbfs", -80.0)))
        out["channels"][role] = st
    out["duration"] = duration

    # Where is the program? Take the loudest channel at each moment.
    combined = None
    for role, db in levels.items():
        if not out["channels"][role]["silent"]:
            combined = db if combined is None else np.maximum(combined[:db.size], db[:combined.size])
    act = activity_window(combined) if combined is not None else None

    # A channel is dead when it carries signal for too little of the program.
    # Percentiles over the whole file are not enough: on 2026-07-26 the PA feed
    # was off except for four short bursts, which lifted its 95th percentile to
    # a healthy-looking -38 dBFS while it was live for ~10% of the service.
    min_share = float(rcfg.get("min_live_share", 0.35))
    for role, db in levels.items():
        st = out["channels"][role]
        share = 0.0
        if act is not None:
            a, b = int(act["start"] // FRAME_S), int(act["end"] // FRAME_S)
            seg = db[a:b]
            if seg.size:
                # Relative to the channel's own loud level: a dead feed's hiss can
                # sit well above the digital floor (2026-07-26: -80..-70 dBFS).
                share = float(np.mean(seg > st["p95"] - 25.0))
        st["live_share"] = share
        st["dead"] = bool(st["silent"] or st["p95"] < float(rcfg.get("dead_channel_dbfs", -60.0))
                          or share < min_share)
        out["channels"][role] = {k: (round(v, 2) if isinstance(v, float) else v)
                                 for k, v in st.items()}

    live = [r for r, c in out["channels"].items() if not c["dead"]]
    first_hour, last_hour = rcfg.get("service_hours", [6, 21])
    if duration is not None and duration < 60.0 * float(rcfg.get("min_service_minutes", 20)):
        status, mode = "test-recording", None
        why = f"only {duration / 60:.0f} min long"
    elif not first_hour <= svc.start.hour < last_hour:
        status, mode = "test-recording", None
        why = f"starts at {svc.start:%H:%M}, outside service hours"
    elif not out["channels"]:
        status, mode, why = "no-audio", None, "no recognised channel files"
    elif all(c["silent"] for c in out["channels"].values()):
        status, mode, why = "silent", None, "every channel is silent (recorder captured nothing)"
    elif not live:
        status, mode, why = "dead", None, "no channel carries program audio"
    elif live == ["sanctuary", "ambient"]:
        status, mode, why = "ok", "dual", "both channels live"
    elif live == ["ambient"]:
        status, mode = "degraded", "ambient-only"
        why = ("sanctuary feed dead" if "sanctuary" in out["channels"] else "no sanctuary file")
    else:
        status, mode = "degraded", "sanctuary-only"
        why = ("loft feed dead" if "ambient" in out["channels"] else "no loft file")
    out.update({"status": status, "mode": mode, "why": why})
    out["activity"] = act
    if cache:
        write_json(cache, out)
    return out


# --------------------------------------------------------------------------- #
# Schedule eras
# --------------------------------------------------------------------------- #

def era_for(day: date, eras: list[dict]) -> dict | None:
    current = None
    for era in sorted(eras, key=lambda e: e["effective"]):
        if date.fromisoformat(era["effective"]) <= day:
            current = era
    return current


def expected_services(day: date, eras: list[dict]) -> list[dict]:
    era = era_for(day, eras)
    if not era:
        return []
    wd = WEEKDAYS[day.weekday()]
    out = []
    for s in era.get("services", []):
        if s.get("weekday", "Sun") == wd:
            hh, mm = map(int, s["time"].split(":"))
            out.append({**s, "at": datetime.combine(day, time(hh, mm))})
    return out


def coverage(rec_start: datetime, rec_end: datetime, svc_at: datetime,
             svc_minutes: float, truncated: bool | None = None) -> str:
    """Does the recording window hold this service?

    `truncated` (program audio still running when the recording stopped) is
    better evidence than an assumed service length when it is known.
    """
    svc_end = svc_at + timedelta(minutes=svc_minutes)
    if rec_end <= svc_at or rec_start >= svc_end:
        return "missed"
    if rec_start > svc_at:
        return "started late"
    if truncated is True:
        return "cut off"
    if truncated is False or rec_end >= svc_end:
        return "covered"
    return "ends early"


def fmt_clock(dt: datetime | None) -> str:
    return dt.strftime("%H:%M") if dt else "--:--"


def round_up(minutes: float, step: int = 5) -> int:
    return int(-(-minutes // step) * step)


# --------------------------------------------------------------------------- #

def audit(services: list[Service], cfg: dict, workdir: Path | None) -> dict:
    scfg = cfg.get("schedule", {})
    eras = scfg.get("eras", [])
    default_len = float(scfg.get("default_service_minutes", 100))
    rows = []
    for svc in services:
        tri = triage(svc, cfg, workdir)
        dur = tri.get("duration") or 0.0
        rec_start, rec_end = svc.start, svc.start + timedelta(seconds=dur)
        act = tri.get("activity")
        row = {
            "key": svc.key, "job_id": svc.job_id, "status": tri["status"], "why": tri["why"],
            "mode": tri["mode"], "recording": [rec_start.isoformat(timespec="minutes"),
                                               rec_end.isoformat(timespec="minutes")],
            "minutes": round(dur / 60.0, 1),
        }
        if act:
            row["program"] = [(rec_start + timedelta(seconds=act["start"])).strftime("%H:%M"),
                              (rec_start + timedelta(seconds=act["end"])).strftime("%H:%M")]
            row["program_minutes"] = round((act["end"] - act["start"]) / 60.0, 1)
            row["dead_air_minutes"] = round(act["start"] / 60.0, 1)
            row["truncated"] = act["truncated"]
        if workdir:
            sj = workdir / f"{svc.key}-sermon.json"
            if sj.exists():
                clock = (json.loads(sj.read_text()).get("sermon") or {}).get("clock")
                if clock:
                    row["sermon"] = [clock["sermon_start"][:5], clock["sermon_end"][:5]]
        exp = expected_services(svc.start.date(), eras) if tri["status"] in ("ok", "degraded",
                                                                              "silent") else []
        row["era"] = (era_for(svc.start.date(), eras) or {}).get("label")
        # Only the LAST service in the window can be cut off by the stop time.
        trunc = act["truncated"] if act else None
        row["services"] = []
        for n, e in enumerate(exp):
            later = any(x["at"] > e["at"] and x["at"] < rec_end for x in exp[n + 1:])
            cov = coverage(rec_start, rec_end, e["at"], default_len,
                           None if later or tri["status"] == "silent" else trunc)
            if tri["status"] == "silent" and cov != "missed":
                cov = "nothing recorded"
            row["services"].append({"time": e["time"], "name": e.get("name", ""), "coverage": cov})
        rows.append(row)

    # Typical program length from complete (untruncated, two-channel) services.
    lengths = [r["program_minutes"] for r in rows
               if r.get("program_minutes") and not r.get("truncated") and r["status"] == "ok"
               and 50 <= r["program_minutes"] <= 180]
    typical = float(np.median(lengths)) if lengths else default_len

    # Recommendation for the latest era.
    rec = []
    if eras:
        latest = sorted(eras, key=lambda e: e["effective"])[-1]
        lead = float(scfg.get("lead_minutes", 10))
        tail = float(scfg.get("tail_minutes", 15))
        for s in latest.get("services", []):
            hh, mm = map(int, s["time"].split(":"))
            start = datetime.combine(date(2000, 1, 1), time(hh, mm)) - timedelta(minutes=lead)
            minutes = round_up(lead + typical + tail)
            rec.append({"weekday": s.get("weekday", "Sun"), "service": s["time"],
                        "name": s.get("name", ""), "record_start": start.strftime("%H:%M"),
                        "duration_minutes": minutes,
                        "record_end": (start + timedelta(minutes=minutes)).strftime("%H:%M")})
        rec_era = {"effective": latest["effective"], "label": latest.get("label"),
                   "source": latest.get("source"), "recommended_jobs": rec}
    else:
        rec_era = None
    return {"rows": rows, "typical_program_minutes": round(typical, 1),
            "typical_from_n": len(lengths), "recommendation": rec_era}


def render(result: dict, recorder_jobs: list[dict] | None = None) -> str:
    L = ["Recording schedule audit", "=" * 100, ""]
    L.append(f"{'service':<16} {'recorded':<13} {'program':<13} {'dead air':>8}  {'trunc':<5} "
             f"{'sermon':<13} {'status':<14} services in force that day")
    L.append("-" * 130)
    for r in result["rows"]:
        recd = f"{r['recording'][0][11:]}-{r['recording'][1][11:]}"
        prog = f"{r['program'][0]}-{r['program'][1]}" if r.get("program") else "-"
        dead = f"{r['dead_air_minutes']:.0f} min" if r.get("dead_air_minutes") is not None else "-"
        trunc = "YES" if r.get("truncated") else ""
        serm = f"{r['sermon'][0]}-{r['sermon'][1]}" if r.get("sermon") else "-"
        svcs = ", ".join(f"{s['time']} {s['coverage']}" for s in r["services"]) or "(none listed)"
        status = r["status"] if r["status"] == "ok" else f"{r['status']}"
        L.append(f"{r['key']:<16} {recd:<13} {prog:<13} {dead:>8}  {trunc:<5} {serm:<13} "
                 f"{status:<14} {svcs}")
        if r["status"] != "ok":
            L.append(f"{'':<16} -> {r['why']}")
    L += ["", f"Typical program length (complete two-channel recordings): "
          f"{result['typical_program_minutes']:.0f} min (n={result['typical_from_n']})."]
    rec = result.get("recommendation")
    if rec:
        L += ["", f"Recommended recorder schedule for the current era "
              f"({rec['label']}, effective {rec['effective']}):"]
        for j in rec["recommended_jobs"]:
            L.append(f"  {j['weekday']} weekly  start {j['record_start']}  duration "
                     f"{j['duration_minutes']} min (until {j['record_end']})   "
                     f"-> {j['service']} {j['name']}")
        L.append(f"  source: {rec.get('source')}")
    if recorder_jobs:
        L += ["", "Current jobs in the recorder database:"]
        for j in recorder_jobs:
            L.append(f"  {j.get('name')!s:<28} start {j.get('start_time')}  "
                     f"{int(j.get('duration') or 0) / 60:.0f} min  recurrence "
                     f"{j.get('recurrence_pattern')}")
    return "\n".join(L) + "\n"


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="service-trimmer schedule", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio-dir", required=True)
    ap.add_argument("--workdir", default=None,
                    help="batch workdir: caches triage and adds sermon clock times")
    ap.add_argument("--config", default=None)
    ap.add_argument("--recorder-db", default=None, help="show the recorder's current jobs")
    ap.add_argument("-o", "--output", default=None, help="write the report here as well")
    ap.add_argument("--json", default=None, help="write machine-readable results here")
    args = ap.parse_args(argv)

    cfg = load_config(args.config)
    audio_dir = Path(args.audio_dir).expanduser()
    if not audio_dir.is_dir():
        die(f"not a directory: {audio_dir}")
    workdir = Path(args.workdir) if args.workdir else None
    if workdir:
        workdir.mkdir(parents=True, exist_ok=True)
    services, warnings = discover(audio_dir, cfg["recorder"]["channel_suffixes"])
    for w in warnings:
        info(f"  note: {w}")
    result = audit(services, cfg, workdir)
    jobs = None
    if args.recorder_db:
        from .recorder import read_jobs
        jobs = read_jobs(args.recorder_db)
    text = render(result, jobs)
    print(text)
    if args.output:
        Path(args.output).write_text(text, encoding="utf-8")
    if args.json:
        write_json(args.json, result)
    return 0
