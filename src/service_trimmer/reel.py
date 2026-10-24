"""Boundary-review reel: hear every detected sermon edge in one file.

    service-trimmer reel --workdir work/ -o sermon-boundary-review.mp3

For each service, chronologically:

    [voice] "February 8th. Starting clip follows."
    10 s BEFORE the cut   ->  BEEP  ->  first 15 s of the sermon
    [voice] "February 8th. Ending clip follows."
    last 15 s of the sermon  ->  BEEP  ->  10 s AFTER the cut

The beep (500 ms at 1 kHz, the pitch of an alpine-racing start gate) sits
exactly on the cut. So: if you hear the sermon's lead-in BEFORE the start beep,
the start is late; if you hear the closing Amen AFTER the end beep, the end is
early. Services graded medium/low confidence are announced as such.

Also writes <output>.cues.txt (the time of every clip in the reel, with how each
boundary was decided) and embeds the same as MP3 chapters, so a podcast player
can skip service to service.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import tempfile
from datetime import datetime
from pathlib import Path

from .audio import FFMPEG, die, hms, info, probe, require_ffmpeg, run_ffmpeg
from .config import load_config

AR, CH = 48000, 2
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August",
          "September", "October", "November", "December"]


def ordinal(n: int) -> str:
    suf = "th" if 11 <= n % 100 <= 13 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")
    return f"{n}{suf}"


def spoken_label(key: str) -> str:
    dt = datetime.strptime(key, "%Y-%m-%d_%H%M")
    label = f"{MONTHS[dt.month - 1]} {ordinal(dt.day)}"
    if dt.hour >= 17:
        label += ", evening service"
    return label


class Builder:
    def __init__(self, tmp: Path, cfg: dict, voice: str):
        self.tmp = tmp
        self.cfg = cfg.get("reel", {})
        self.voice = voice
        self.n = 0
        self.parts: list[Path] = []
        self.cursor = 0.0

    def _path(self, tag: str) -> Path:
        self.n += 1
        return self.tmp / f"{self.n:05d}-{tag}.wav"

    def _add(self, p: Path) -> float:
        d = probe(p)["duration"] or 0.0
        self.parts.append(p)
        self.cursor += d
        return d

    def silence(self, seconds: float) -> None:
        p = self._path("sil")
        run_ffmpeg([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"anullsrc=r={AR}:cl=stereo", "-t", f"{seconds:.3f}",
                    "-c:a", "pcm_s16le", str(p)], "silence")
        self._add(p)

    def tone(self, hz: float, seconds: float, amp: float = 0.5) -> None:
        # aevalsrc at a stated amplitude: the `sine` source is not full-scale,
        # which once left the beep 30 dB under -16 LUFS program audio.
        p = self._path("tone")
        fade = min(0.02, seconds / 4)
        run_ffmpeg([FFMPEG, "-v", "error", "-y", "-f", "lavfi", "-i",
                    f"aevalsrc={amp}*sin(2*PI*{hz}*t):d={seconds}:s={AR}",
                    "-af", f"afade=t=in:st=0:d={fade},afade=t=out:st={seconds - fade}:d={fade}",
                    "-ar", str(AR), "-ac", str(CH), "-c:a", "pcm_s16le", str(p)], "tone")
        self._add(p)

    def beep(self) -> None:
        self.tone(float(self.cfg.get("beep_hz", 1000.0)), float(self.cfg.get("beep_seconds", 0.5)))

    def say(self, text: str, cue: str) -> None:
        if self.voice == "gtts":
            try:
                from gtts import gTTS
                mp3 = self.tmp / f"tts-{self.n}.mp3"
                gTTS(text, lang="en", tld="com").save(str(mp3))
                p = self._path("voice")
                run_ffmpeg([FFMPEG, "-v", "error", "-y", "-i", str(mp3),
                            "-af", "loudnorm=I=-16:TP=-1.5:LRA=11",
                            "-ar", str(AR), "-ac", str(CH), "-c:a", "pcm_s16le", str(p)], "voice")
                self._add(p)
                return
            except Exception as exc:  # network down, gTTS missing: fall back to tones
                info(f"  voice unavailable ({exc.__class__.__name__}); using tone cues")
                self.voice = "none"
        # Tone cue: rising pair = start clip, falling pair = end clip.
        lo, hi = 660.0, 880.0
        first, second = (lo, hi) if cue == "start" else (hi, lo)
        self.tone(first, 0.12, 0.3)
        self.silence(0.06)
        self.tone(second, 0.12, 0.3)

    def clip(self, src: str, start: float, length: float, gain_db: float) -> None:
        if length <= 0.05:
            return
        p = self._path("clip")
        run_ffmpeg([FFMPEG, "-v", "error", "-y", "-ss", f"{max(0.0, start):.3f}", "-i", src,
                    "-t", f"{length:.3f}",
                    "-af", f"volume={gain_db:.2f}dB,alimiter=limit=0.89:level=false",
                    "-ar", str(AR), "-ac", str(CH), "-c:a", "pcm_s16le", str(p)], "clip")
        self._add(p)


def loudness_gain(src: str, start: float, length: float, target: float = -16.0) -> float:
    """Gain that brings a stretch of sermon to `target` LUFS.

    Measured once per service on its middle and applied to all four clips, so
    run-in, sermon and run-out keep their real relative levels -- normalising
    each clip separately would make a quiet run-in sound as loud as the sermon.
    """
    proc = subprocess.run(
        [FFMPEG, "-hide_banner", "-nostdin", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
         "-i", src, "-af", f"loudnorm=I={target}:print_format=json", "-f", "null", "-"],
        capture_output=True, text=True)
    err = proc.stderr
    try:
        m = json.loads(err[err.rfind("{"): err.rfind("}") + 1])
        measured = float(m["input_i"])
    except (ValueError, KeyError):
        return 0.0
    if measured < -70:
        return 0.0
    return max(-20.0, min(30.0, target - measured))


def source_for(rep: dict, workdir: Path) -> str | None:
    key = rep["key"]
    if rep.get("mode") == "dual":
        mix = workdir / f"{key}-mix.flac"
        if mix.exists():
            return str(mix)
    media = rep.get("inputs", {}).get("media")
    return media if media and Path(media).exists() else None


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="service-trimmer reel", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--workdir", required=True, help="directory holding *-sermon.json")
    ap.add_argument("-o", "--output", required=True, help="reel MP3")
    ap.add_argument("--config", default=None)
    ap.add_argument("--only", nargs="+", default=None, help="service keys")
    ap.add_argument("--min-confidence", choices=["low", "medium", "high"], default="low",
                    help="include only services at or above this confidence")
    ap.add_argument("--voice", choices=["gtts", "none"], default=None)
    args = ap.parse_args(argv)

    require_ffmpeg()
    cfg = load_config(args.config)
    rcfg = cfg.get("reel", {})
    sermon_s = float(rcfg.get("sermon_seconds", 15.0))
    ctx_s = float(rcfg.get("context_seconds", 10.0))
    voice = args.voice or rcfg.get("voice", "gtts")
    wd = Path(args.workdir)
    rank = {"low": 0, "medium": 1, "high": 2}

    reports = []
    for p in sorted(wd.glob("*-sermon.json")):
        rep = json.loads(p.read_text())
        if args.only and rep.get("key") not in args.only:
            continue
        if rank.get(rep["sermon"].get("confidence", "low"), 0) < rank[args.min_confidence]:
            continue
        reports.append(rep)
    if not reports:
        die(f"no sermon reports in {wd}")

    tmp = Path(tempfile.mkdtemp(prefix="ilc-reel-"))
    cues: list[dict] = []
    try:
        b = Builder(tmp, cfg, voice)
        b.silence(0.5)
        for i, rep in enumerate(reports, 1):
            key, s = rep["key"], rep["sermon"]
            src = source_for(rep, wd)
            if src is None:
                info(f"  skip {key}: source audio not found")
                continue
            total = probe(src)["duration"] or 0.0
            a, z = float(s["cut_start"]), float(s["cut_end"])
            mid = (a + z) / 2.0
            gain = loudness_gain(src, max(a, mid - 30.0), min(60.0, z - a))
            label = spoken_label(key)
            conf = s.get("confidence", "low")
            warn = "" if conf == "high" else f" {conf.capitalize()} confidence."
            info(f"  [{i}/{len(reports)}] {key}  {hms(a)} / {hms(z)}  gain {gain:+.1f} dB  {conf}")

            t_voice = b.cursor
            b.say(f"{label}. Starting clip follows.{warn}", "start")
            b.silence(0.25)
            run_in = min(ctx_s, a)
            b.clip(src, a - run_in, run_in, gain)
            t_beep_start = b.cursor
            b.beep()
            b.clip(src, a, min(sermon_s, z - a), gain)
            b.silence(0.6)

            t_voice_end = b.cursor
            b.say(f"{label}. Ending clip follows.", "end")
            b.silence(0.25)
            tail_from = max(a, z - sermon_s)
            b.clip(src, tail_from, z - tail_from, gain)
            t_beep_end = b.cursor
            b.beep()
            b.clip(src, z, min(ctx_s, max(0.0, total - z)), gain)
            b.silence(1.0)

            ev_s = (rep["edges"]["start"].get("evidence") or {}).get("text", "")
            ev_e = (rep["edges"]["end"].get("evidence") or {}).get("text", "")
            cues.append({
                "key": key, "label": label, "confidence": conf,
                "voice_start": t_voice, "start_beep": t_beep_start,
                "voice_end": t_voice_end, "end_beep": t_beep_end,
                "sermon_start": a, "sermon_end": z,
                "start_method": s["start_method"], "end_method": s["end_method"],
                "start_evidence": ev_s, "end_evidence": ev_e, "flags": s.get("flags", []),
                "clock": s.get("clock"),
            })

        concat = tmp / "concat.txt"
        concat.write_text("".join(f"file '{p}'\n" for p in b.parts))
        master = tmp / "master.wav"
        run_ffmpeg([FFMPEG, "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", str(concat),
                    "-c:a", "pcm_s16le", str(master)], "concat")

        # Chapters, one per service, so a player can skip between them.
        meta = tmp / "chapters.txt"
        lines = [";FFMETADATA1", "title=Sermon boundary review", f"album={cfg.get('church_name', '')}"]
        for j, c in enumerate(cues):
            end = cues[j + 1]["voice_start"] if j + 1 < len(cues) else b.cursor
            lines += ["[CHAPTER]", "TIMEBASE=1/1000", f"START={int(c['voice_start'] * 1000)}",
                      f"END={int(end * 1000)}", f"title={c['key']} ({c['confidence']})"]
        meta.write_text("\n".join(lines) + "\n")
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        run_ffmpeg([FFMPEG, "-v", "error", "-y", "-i", str(master), "-i", str(meta),
                    "-map_metadata", "1", "-map_chapters", "1", "-c:a", "libmp3lame",
                    "-b:a", "128k", "-id3v2_version", "3", str(out)], "mp3 encode")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    write_cues(out, cues, b.voice)
    info(f"\nwrote {out}  ({b.cursor / 60:.1f} min, {len(cues)} services)")
    info(f"wrote {out.with_suffix('.cues.txt')}")
    return 0


def write_cues(out: Path, cues: list[dict], voice: str) -> None:
    def mmss(t: float) -> str:
        return f"{int(t // 60):02d}:{t % 60:04.1f}"

    L = [f"Cue sheet for {out.name}", "=" * 100, "",
         "Each service: a START clip (10 s before the cut, BEEP at the cut, then the first",
         "15 s of the sermon) and an END clip (last 15 s, BEEP at the cut, then 10 s after).",
         "Start beep: the lead-in (invocation/acclamation/prayer) should come right AFTER it.",
         "End beep: the closing formula and the congregation's Amen should come right BEFORE it.",
         "" if voice == "gtts" else "Spoken labels were unavailable; rising tones = start clip, "
                                    "falling tones = end clip.", "",
         f"{'reel time':<10} {'service':<16} {'conf':<6} {'boundary':<8} {'recording time':<14} "
         f"{'decided by':<36} evidence"]
    L.append("-" * 140)
    for c in cues:
        L.append(f"{mmss(c['start_beep']):<10} {c['key']:<16} {c['confidence']:<6} {'START':<8} "
                 f"{hms(c['sermon_start'])[:11]:<14} {c['start_method']:<36} "
                 f"{c['start_evidence'][:60]}")
        L.append(f"{mmss(c['end_beep']):<10} {'':<16} {'':<6} {'END':<8} "
                 f"{hms(c['sermon_end'])[:11]:<14} {c['end_method']:<36} {c['end_evidence'][:60]}")
        for f in c["flags"]:
            L.append(f"{'':<34}  ! {f}")
    out.with_suffix(".cues.txt").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out.with_suffix(".cues.json")).write_text(json.dumps(cues, indent=2) + "\n", encoding="utf-8")
