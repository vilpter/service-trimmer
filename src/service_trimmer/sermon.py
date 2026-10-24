"""Locate the sermon -- including its lead-in and the congregation's closing Amen.

    service-trimmer sermon --stems out/KEY-stems.flac --transcript out/KEY.json -o out/KEY-sermon.json
    service-trimmer sermon --audio loft.wav --mode ambient-only --transcript out/KEY-loft.json ...

Stages:

1. Candidates. With both channels (`dual`), the loft-minus-sanctuary level ratio
   drops and stays low while one voice preaches into the PA; every long low run is
   a candidate. With one channel, transcript speech density stands in for it.
2. Scoring. Each candidate (and each pair of adjacent candidates, merged) is
   scored on liturgical evidence in the full transcript -- an opening formula
   near its start, a closing formula and a post-sermon act near its end -- on
   plausible length, and on overlap with an operator's "Mark Beginning/Ending"
   section from the recorder database, when one exists.
3. Edge refinement. A few minutes around each edge are re-transcribed without
   VAD, and the edges are resolved word-accurately (see edges.py).
4. Grading. Each boundary reports how it was decided; the pair yields a
   high / medium / low confidence and a list of flags for the review reel.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

from . import edges as E
from .audio import (
    FFMPEG, decode_mono, die, find_runs, frame_rms_db, hms, info, median_smooth,
    merge_runs, normalized_mp3, otsu_threshold, probe, require_ffmpeg, write_json,
)
from .config import asr_model, asr_threads, load_config
from .liturgy import WordIndex, find_all

HOP_S = 0.5
WIN_S = 1.0


# --------------------------------------------------------------------------- #
# Candidates
# --------------------------------------------------------------------------- #

def acoustic_runs(stems: str, cfg: dict, rate: int = 8000) -> dict:
    """Low loft/sanctuary-ratio runs. Ported from the v1 acoustic pass."""
    san = decode_mono(stems, rate, channel=0)
    amb = decode_mono(stems, rate, channel=1)
    n = min(san.size, amb.size)
    if n == 0:
        die(f"{stems}: could not decode two channels (expected stems from `service-trimmer mix`)")
    san_db = frame_rms_db(san[:n], rate, HOP_S, WIN_S)
    amb_db = frame_rms_db(amb[:n], rate, HOP_S, WIN_S)
    del san, amb
    m = min(san_db.size, amb_db.size)
    san_db, amb_db = san_db[:m], amb_db[:m]
    if m < 10:
        die("recording too short to analyse")

    smooth_frames = int(round(float(cfg.get("smooth_seconds", 21.0)) / HOP_S)) | 1
    ratio_s = median_smooth(amb_db - san_db, smooth_frames)
    san_s = median_smooth(san_db, smooth_frames)

    # Activity gate: keeps dead air out of the classifier. It must not try to
    # tell speech from music -- the sermon is usually the *quietest* active part
    # of the service (one voice, versus organ plus a singing congregation).
    loud = float(np.percentile(san_s, 95))
    gate = max(float(cfg.get("silence_floor_db", -65.0)),
               loud - float(cfg.get("silence_range_db", 40.0)))
    active = san_s > gate

    setting = cfg.get("ratio_threshold_db", "auto")
    pool = ratio_s[active] if active.sum() >= 10 else ratio_s
    if setting in ("auto", "otsu"):
        # Bimodal by construction; Otsu does not assume equal class sizes, and
        # preaching is a minority of a service.
        threshold = otsu_threshold(pool)
        below = float(np.mean(pool < threshold))
        if not (0.02 <= below <= 0.90):
            threshold = float((np.percentile(pool, 25) + np.percentile(pool, 75)) / 2.0)
    elif setting == "midpoint":
        threshold = float((np.percentile(pool, 25) + np.percentile(pool, 75)) / 2.0)
    else:
        threshold = float(setting)

    gap = int(round(float(cfg.get("merge_gap_seconds", 30.0)) / HOP_S))
    runs = merge_runs(find_runs(active & (ratio_s < threshold)), gap)
    min_frames = int(round(float(cfg.get("min_sermon_seconds", 300.0)) / HOP_S))
    runs = [(a * HOP_S, b * HOP_S) for a, b in runs if (b - a) >= min_frames]
    return {
        "runs": runs, "threshold_db": threshold, "activity_gate_db": gate,
        "ratio_percentiles": {str(p): round(float(np.percentile(ratio_s, p)), 2)
                              for p in (5, 25, 50, 75, 95)},
    }


def density_runs(segments: list[dict], duration: float, cfg: dict) -> dict:
    """Single-channel stand-in for the ratio: where the transcript is dense speech.

    Preaching runs at ~2.3 words/s for 10-25 minutes; readings are dense but
    short, and music yields few or no words. Coarser than the ratio, which is why
    single-channel results are always graded at most 'medium'.
    """
    bin_s = 10.0
    nb = int(np.ceil(duration / bin_s)) + 1
    counts = np.zeros(nb)
    for seg in segments:
        for w in seg.get("words") or []:
            counts[min(nb - 1, int(w["start"] // bin_s))] += 1
    density = median_smooth(counts / bin_s, 7)
    mask = density >= float(cfg.get("density_words_per_second", 1.2))
    runs = merge_runs(find_runs(mask), 3)
    min_bins = int(np.ceil(float(cfg.get("min_sermon_seconds", 300.0)) / bin_s))
    runs = [(a * bin_s, b * bin_s) for a, b in runs if b - a >= min_bins]
    return {"runs": runs, "density_threshold": cfg.get("density_words_per_second", 1.2)}


def score_candidates(runs: list[tuple[float, float]], longform: WordIndex, cfg: dict,
                     liturgy: dict, hint: dict | None) -> list[dict]:
    """Rank candidate windows on liturgical evidence and liturgical ORDER.

    Formulas are weighted by how specific they are to the sermon: an invocation
    or the Pauline benediction is strong; "let us pray" or a Trinitarian formula
    also opens and closes prayers and rites (2026-06-28: the Prayers of the
    Church plus a reception rite outscored the sermon on generic phrases alone).
    Order matters too: the sermon follows the Gospel and precedes the Creed, so
    a window that begins just after the Creed is the Prayers, not the sermon.
    """
    scfg = cfg.get("sermon", {})
    radius = float(scfg.get("marker_radius_seconds", 240.0))
    lo_min, hi_min = scfg.get("typical_minutes", [6, 35])
    gap_max = float(scfg.get("candidate_merge_gap_seconds", 150.0))
    strong_open = set(liturgy.get("strong_opening", []))
    strong_close = set(liturgy.get("strong_closing", []))
    creed = [x for x in liturgy.get("after_sermon", []) if x.get("kind") == "creed"]

    cands = [{"start": a, "end": b, "merged": False} for a, b in runs]
    for (a1, b1), (a2, b2) in zip(runs, runs[1:]):
        if a2 - b1 <= gap_max:
            cands.append({"start": a1, "end": b2, "merged": True})

    for c in cands:
        s, e = c["start"], c["end"]
        ev: dict = {}
        score = 0.0
        opens = find_all(longform, liturgy.get("opening", []), s - radius, s + 60.0, 1.5)
        closes = find_all(longform, liturgy.get("closing", []), e - radius, e + radius, 1.5)
        afters = find_all(longform, liturgy.get("after_sermon", []) + liturgy.get("rites", []),
                          e - 60.0, e + radius, 1.5)
        gospel = find_all(longform, liturgy.get("before_sermon", []), s - 900.0, s + 30.0, 1.5)
        creed_before = find_all(longform, creed, s - 300.0, s + 30.0, 1.5)
        if opens:
            strong = [m for m in opens if m.kind in strong_open]
            score += 1.0 if strong else 0.3
            ev["opening"] = (strong or opens)[-1].to_dict()
        if closes:
            strong = [m for m in closes if m.kind in strong_close]
            score += 2.0 if strong else 0.5
            ev["closing"] = min(strong or closes, key=lambda m: abs(m.end - e)).to_dict()
        if afters:
            score += 0.5
            ev["after_sermon"] = afters[0].to_dict()
        if gospel:
            score += 1.5
            ev["gospel_before"] = gospel[-1].to_dict()
        if creed_before:
            score -= 2.0
            ev["creed_before"] = creed_before[-1].to_dict()
        minutes = (e - s) / 60.0
        score += 1.0 if lo_min <= minutes <= hi_min else -1.0
        if c["merged"]:
            score -= 0.5
        if hint:
            inter = max(0.0, min(e, hint["end"]) - max(s, hint["start"]))
            union = max(e, hint["end"]) - min(s, hint["start"])
            overlap = inter / union if union > 0 else 0.0
            score += 3.0 * overlap
            ev["manual_overlap"] = round(overlap, 2)
        c["score"] = round(score, 2)
        c["evidence"] = ev
        c["minutes"] = round(minutes, 1)
    cands.sort(key=lambda c: (c["score"], c["end"] - c["start"]), reverse=True)
    return cands


def trim_at_rite(cand: dict, longform: WordIndex, liturgy: dict,
                 min_seconds: float) -> tuple[float, dict | None]:
    """End of the sermon when a rite follows it inside the same acoustic run.

    Confirmation, reception of members, installation of officers: the pastor
    keeps talking into the PA with no hymn in between, so the level-ratio run
    carries straight on. The rite announces itself ("I'd like to invite forward
    the newly elected officers", "Beloved in the Lord", "Do you renounce..."),
    and the sermon's closing formula sits just before that.
    """
    s, e = cand["start"], cand["end"]
    rites = find_all(longform, liturgy.get("rites", []), s + 180.0, e, 1.5)
    if not rites:
        return e, None
    first = rites[0]
    strong = set(liturgy.get("strong_closing", []))
    closes = [m for m in find_all(longform, liturgy.get("closing", []), s + 180.0,
                                  first.start + 5.0, 1.5) if m.kind in strong]
    new_end = (max(m.end for m in closes) + 3.0) if closes else first.start - 3.0
    if new_end - s < min_seconds:
        return e, None
    return new_end, {"rite": first.to_dict(), "moved_from": round(e, 1),
                     "moved_to": round(new_end, 1),
                     "anchor": "closing formula before the rite" if closes else "rite start"}


# --------------------------------------------------------------------------- #
# Edge transcription cache
# --------------------------------------------------------------------------- #

def edge_segments(cache_path: Path, src: str, channel: int | None, a: float, b: float,
                  cfg: dict, threads: int) -> list[dict]:
    """Re-transcribe [a, b) without VAD, reusing a cached window that covers it."""
    ecfg = cfg.get("edges", {})
    model_name = ecfg.get("model") or asr_model(cfg)
    prompt = ecfg.get("prompt")
    cache = {"windows": []}
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text())
        except json.JSONDecodeError:
            pass
    for w in cache.get("windows", []):
        if (w["model"] == model_name and w["prompt"] == prompt and w["channel"] == channel
                and w["a"] <= a + 0.01 and w["b"] >= b - 0.01):
            return [s for s in w["segments"] if s["end"] > a and s["start"] < b]

    from .transcribe import load_model, transcribe_window
    model = load_model(model_name, cfg, threads=threads)
    info(f"  transcribing edge window {hms(a)} -> {hms(b)} ({model_name}, no VAD)")
    segs = transcribe_window(model, src, a, b, channel, prompt)
    cache.setdefault("windows", []).append({"a": a, "b": b, "channel": channel,
                                           "model": model_name, "prompt": prompt,
                                           "segments": segs})
    write_json(cache_path, cache)
    return segs


# --------------------------------------------------------------------------- #

def grade(mode: str, start: dict, end: dict, ambiguous: bool, minutes: float,
          cfg: dict) -> tuple[str, list[str], list[str]]:
    """Confidence plus review flags (things a person should check) and notes
    (facts about the service that do not make the cut doubtful)."""
    flags = list(end.get("flags", []))
    notes: list[str] = []
    after_music = "music" in start.get("lead_in_basis", "")
    # Starting on the first words after the hymn's music ends is as solid as a
    # formula: the edge transcript has no VAD, so any lead-in would be in it.
    start_strong = start["method"].startswith("opening:") or (
        after_music and start["method"] in ("lead_in_speech", "first_word"))
    end_strong = end["method"].startswith("closing:")
    if not start_strong:
        flags.append(f"start decided by {start['method']} ({start.get('lead_in_basis', '')})")
    elif start["method"] == "first_word":
        notes.append("no lead-in before the sermon (starts directly after the music)")
    if end["method"].startswith("closing:") and "+response" not in end["method"] \
            and mode != "sanctuary-only":
        notes.append("no congregational response audible; kept "
                     f"{cfg.get('edges', {}).get('amen_allowance_seconds', 2.0):.1f} s after "
                     "the closing formula (or up to the next speaker)")
    lo, hi = cfg.get("sermon", {}).get("typical_minutes", [6, 35])
    if not lo <= minutes <= hi:
        flags.append(f"unusual length ({minutes:.1f} min)")
    if ambiguous:
        flags.append("another candidate window scored nearly as well")
    if mode != "dual":
        flags.append(f"single-channel mode ({mode})")

    if start_strong and end_strong and mode == "dual" and not ambiguous:
        conf = "high"
    elif (start_strong or end_strong) and not (ambiguous and mode != "dual"):
        conf = "medium"
    else:
        conf = "low"
    if not lo <= minutes <= hi and conf == "high":
        conf = "medium"
    return conf, flags, notes


def recording_start(key: str | None, explicit: str | None) -> datetime | None:
    if explicit:
        return datetime.fromisoformat(explicit)
    if key:
        try:
            return datetime.strptime(key, "%Y-%m-%d_%H%M")
        except ValueError:
            return None
    return None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="service-trimmer sermon", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--stems", help="2-channel aligned stems from `service-trimmer mix` (dual mode)")
    src.add_argument("--audio", help="a single channel, for sanctuary-only / ambient-only mode")
    ap.add_argument("--mode", choices=["sanctuary-only", "ambient-only"], default=None,
                    help="required with --audio: which channel it is")
    ap.add_argument("--transcript", required=True, help="JSON from `service-trimmer transcribe`")
    ap.add_argument("-o", "--output", required=True, help="sermon report JSON")
    ap.add_argument("--config", default=None)
    ap.add_argument("--key", default=None, help="service key, e.g. 2026-02-08_0940")
    ap.add_argument("--job-id", default=None, help="recorder job id, e.g. 2026_Feb_08_09:40")
    ap.add_argument("--recording-start", default=None, help="ISO datetime of the recording start")
    ap.add_argument("--recorder-db", default=None,
                    help="ilc-Audio-Recorder schedule.db; manual section tags guide detection")
    ap.add_argument("--no-edges", action="store_true", help="skip edge re-transcription")
    ap.add_argument("--threads", type=int, default=None)
    ap.add_argument("--ratio-threshold-db", default=None)
    ap.add_argument("--extract", action="store_true", help="also cut the sermon audio (FLAC)")
    ap.add_argument("--mp3", action="store_true", help="also write a -16 LUFS MP3")
    ap.add_argument("--extract-from", default=None,
                    help="file to cut (default: the stems' sibling -mix.flac, or --audio)")
    ap.add_argument("--pad", type=float, default=0.0,
                    help="extra seconds of context on both sides of the extract")
    args = ap.parse_args(argv)

    require_ffmpeg()
    cfg = load_config(args.config)
    scfg = dict(cfg.get("sermon", {}))
    if args.ratio_threshold_db is not None:
        scfg["ratio_threshold_db"] = args.ratio_threshold_db
    ecfg = cfg.get("edges", {})
    liturgy = cfg.get("liturgy", {})
    threads = args.threads if args.threads is not None else asr_threads(cfg)

    if args.audio and not args.mode:
        die("--audio needs --mode sanctuary-only or --mode ambient-only")
    mode = "dual" if args.stems else args.mode
    media = args.stems or args.audio
    meta = probe(media)
    duration = meta["duration"] or 0.0

    with open(args.transcript, encoding="utf-8") as fh:
        tr = json.load(fh)
    segments = tr.get("segments", [])
    if not segments:
        die(f"{args.transcript}: no segments")
    longform = WordIndex.from_segments(segments)

    key = args.key or Path(args.output).name.replace("-sermon.json", "")
    from .naming import key_to_job_id
    try:
        job_id = args.job_id or key_to_job_id(key)
    except ValueError:
        job_id = args.job_id
    rec_start = recording_start(key, args.recording_start)

    hint = None
    if args.recorder_db and job_id:
        from .recorder import manual_sermon_hint, read_sections
        hint = manual_sermon_hint(read_sections(args.recorder_db, job_id))
        if hint:
            info(f"recorder section '{hint['label']}': {hms(hint['start'])} -> {hms(hint['end'])}")

    # ---- 1. candidates ------------------------------------------------------
    info(f"mode: {mode}")
    if mode == "dual":
        info("pass 1: acoustic (loft/sanctuary level ratio)")
        acoustic = acoustic_runs(media, scfg)
        info(f"  threshold {acoustic['threshold_db']:.1f} dB  ratio pct {acoustic['ratio_percentiles']}")
    else:
        info("pass 1: transcript speech density (single channel)")
        acoustic = density_runs(segments, duration, scfg)
    runs = acoustic["runs"]
    if hint and not runs:
        runs = [(hint["start"], hint["end"])]
    if not runs:
        die("no sermon-length candidate found (no long preaching run). "
            "Check the recording, or try --ratio-threshold-db.")

    # ---- 2. scoring ---------------------------------------------------------
    cands = score_candidates(runs, longform, cfg, liturgy, hint)
    for c in cands[:5]:
        info(f"  cand {hms(c['start'])} -> {hms(c['end'])}  {c['minutes']:5.1f} min  "
             f"score {c['score']:+.1f}{'  (merged)' if c['merged'] else ''}  "
             f"{','.join(k for k in c['evidence'] if k != 'manual_overlap')}")
    best = cands[0]

    def overlap(a: dict, b: dict) -> float:
        inter = max(0.0, min(a["end"], b["end"]) - max(a["start"], b["start"]))
        return inter / max(1.0, min(a["end"] - a["start"], b["end"] - b["start"]))

    # Ambiguous only when a *different* region scores close. A merged or nested
    # variant of the same window is not a rival: edge refinement settles its end.
    rivals = [c for c in cands[1:] if c["score"] >= best["score"] - 0.5 and overlap(c, best) < 0.5]
    ambiguous = bool(rivals)
    A, B = best["start"], best["end"]
    B, rite = trim_at_rite(best, longform, liturgy, float(scfg.get("min_sermon_seconds", 300.0)))
    if rite:
        info(f"  a rite follows the sermon inside this run (\"{rite['rite']['text'][:50]}\" at "
             f"{hms(rite['rite']['start'])}); end moved {hms(rite['moved_from'])} -> "
             f"{hms(rite['moved_to'])}")

    # ---- 3. edges -----------------------------------------------------------
    edge_src, edge_ch = (media, 0) if mode == "dual" else (media, None)
    cache_path = Path(args.output).with_name(Path(args.output).name.replace(
        "-sermon.json", "-edges.json"))
    edge_notes: list[str] = []
    if args.no_edges or not ecfg.get("enabled", True):
        start_segs, end_segs = segments, segments
        edge_notes.append("edge re-transcription disabled; using the full transcript")
    else:
        info("\npass 3: edge refinement")
        sw, ew = ecfg.get("start_window", [-150, 60]), ecfg.get("end_window", [-150, 120])
        start_segs = edge_segments(cache_path, edge_src, edge_ch, A + sw[0], A + sw[1],
                                   cfg, threads)
        end_segs = edge_segments(cache_path, edge_src, edge_ch, B + ew[0], min(duration, B + ew[1]),
                                 cfg, threads)

    sw, ew = ecfg.get("start_window", [-150, 60]), ecfg.get("end_window", [-150, 120])
    env_s = E.load_envelope(media, A + sw[0], A + sw[1], mode)
    env_e = E.load_envelope(media, B + ew[0], min(duration, B + ew[1]), mode)
    ctx_s = E.build_context(env_s, start_segs, (A + 10.0, A + 55.0), ecfg)
    ctx_e = E.build_context(env_e, end_segs, (B - 70.0, B - 15.0), ecfg)
    start = E.resolve_start(ctx_s, A, ecfg, liturgy)
    # Rites are post-sermon acts too: a closing formula must precede them.
    end_liturgy = dict(liturgy, after_sermon=liturgy.get("after_sermon", []) + liturgy.get("rites", []))
    end = E.resolve_end(ctx_e, B, ecfg, end_liturgy)
    if end["cut"] <= start["time"]:
        die(f"resolved end ({hms(end['cut'])}) is not after start ({hms(start['time'])})")

    s_time, e_time = start["time"], end["cut"]
    minutes = (e_time - s_time) / 60.0
    confidence, flags, notes = grade(mode, start, end, ambiguous, minutes, cfg)
    if abs((B - A) - (e_time - s_time)) > 180:
        flags.append("refined length differs from the candidate window by "
                     f"{abs((B - A) - (e_time - s_time)) / 60:.1f} min")

    clock = None
    if rec_start:
        clock = {
            "recording_start": rec_start.isoformat(timespec="minutes"),
            "sermon_start": (rec_start + timedelta(seconds=s_time)).strftime("%H:%M:%S"),
            "sermon_end": (rec_start + timedelta(seconds=e_time)).strftime("%H:%M:%S"),
        }

    report = {
        "tool": "service-trimmer sermon",
        "key": key, "job_id": job_id, "mode": mode,
        "inputs": {"media": media, "transcript": args.transcript, "duration": duration},
        "candidates": cands[:6],
        "detector": {k: v for k, v in acoustic.items() if k != "runs"},
        "manual_hint": hint,
        "rite_after_sermon": rite,
        "edges": {"start": start, "end": end,
                  "notes": edge_notes + ctx_s.notes + ctx_e.notes},
        "sermon": {
            "start": round(s_time, 3), "end": round(e_time, 3),
            "duration": round(e_time - s_time, 3),
            "cut_start": start["cut"], "cut_end": end["cut"],
            "spoken_end": end["time"],
            "start_method": start["method"], "end_method": end["method"],
            "confidence": confidence, "flags": flags, "notes": notes,
            "clock": clock,
        },
    }
    write_json(args.output, report)

    info(f"\n  START {hms(s_time)}  {start['method']}"
         + (f"  \"{start['evidence']['text'][:60]}\"" if start.get("evidence") else ""))
    info(f"  END   {hms(e_time)}  {end['method']}"
         + (f"  \"{end['evidence']['text'][:60]}\"" if end.get("evidence") else ""))
    info(f"  {minutes:.1f} min   confidence {confidence.upper()}"
         + (f"   flags: {'; '.join(flags)}" if flags else ""))
    if clock:
        info(f"  clock {clock['sermon_start']} -> {clock['sermon_end']}")
    info(f"  wrote {args.output}")

    if args.extract:
        extract(args, report, mode, cfg)
    return 0


def extract(args, report: dict, mode: str, cfg: dict) -> None:
    source = args.extract_from
    if source is None:
        if mode == "dual":
            guess = Path(args.stems).with_name(Path(args.stems).name.replace("-stems", "-mix"))
            source = str(guess) if guess.exists() else args.stems
        else:
            source = args.audio
    s = report["sermon"]
    a = max(0.0, s["cut_start"] - args.pad)
    b = s["cut_end"] + args.pad
    out = Path(args.output).with_name(Path(args.output).name.replace("-sermon.json",
                                                                       "-sermon.flac"))
    cmd = [FFMPEG, "-v", "error", "-nostdin", "-y", "-ss", f"{a:.3f}", "-i", source,
           "-t", f"{b - a:.3f}", "-c:a", "flac", str(out)]
    proc = subprocess.run(cmd, capture_output=True, text=True)
    if proc.returncode != 0:
        die(f"extract failed:\n{proc.stderr[-2000:]}")
    info(f"  wrote {out}  ({(b - a) / 60:.1f} min)")
    if args.mp3:
        date = report["key"][:10] if report.get("key") else ""
        ecfg = cfg.get("extract", {})
        tags = {"title": f"Sermon {date}".strip(), "album": cfg.get("church_name", ""),
                "date": date[:4], "comment": f"service-trimmer; confidence {s['confidence']}"}
        n = normalized_mp3(out, out.with_suffix(".mp3"), float(ecfg.get("target_lufs", -16.0)),
                           ecfg.get("mp3_bitrate", "128k"), tags)
        info(f"  wrote {out.with_suffix('.mp3')}  ({n['input_lufs']:.1f} -> "
             f"{n['target_lufs']:.0f} LUFS)")
