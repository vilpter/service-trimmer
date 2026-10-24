"""Transcribe a service channel with faster-whisper.

    service-trimmer transcribe --audio sanctuary.wav -o out/2026-08-09

Writes <out>.json (segments + word-level timestamps), <out>.srt and <out>.txt.
The JSON is what `service-trimmer sermon` consumes.

Normally the sanctuary (PA feed) channel. When that feed is dead -- as on
2026-07-26 -- transcribe the loft channel instead: it hears the preacher through
the room's PA speakers, which is noisier but entirely usable.

Runs on CPU via CTranslate2 -- no torch, which is both ~4x faster than
the PyTorch-based whisper package and the reason this ports to a Raspberry Pi.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import tempfile
import time
from pathlib import Path

# onnxruntime (the Silero VAD runtime) probes /sys/class/drm and /sys/class/accel
# for GPUs at import time. On a shared server those are unreadable and it prints
# two alarming-looking permission warnings. There is no GPU either way.
os.environ.setdefault("ORT_LOGGING_LEVEL", "3")

import numpy as np

from .audio import FFMPEG, decode_mono, die, hms, info, probe, require_ffmpeg, write_json
from .config import asr_model, asr_threads, load_config, model_dir as default_model_dir

MODELS = ["tiny.en", "base.en", "small.en", "medium.en", "distil-medium.en",
          "large-v3", "distil-large-v3"]


def srt_time(seconds: float) -> str:
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def predecode(src: str, rate: int, channel: int | None = None) -> str:
    """16 kHz mono WAV in a temp file.

    Streaming a WAV keeps peak RAM low, which matters on a Pi -- handing
    faster-whisper a numpy array would hold ~350 MB for a 90-minute service.
    """
    fd, path = tempfile.mkstemp(prefix="church-asr-", suffix=".wav")
    os.close(fd)
    proc = subprocess.run(
        [FFMPEG, "-v", "error", "-nostdin", "-y", "-i", src,
         *(["-af", f"pan=mono|c0=c{channel}"] if channel is not None else ["-ac", "1"]),
         "-ar", str(rate), "-c:a", "pcm_s16le", path],
        capture_output=True, text=True,
    )
    if proc.returncode != 0:
        os.unlink(path)
        die(f"pre-decode failed:\n{proc.stderr[-2000:]}")
    return path


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(prog="service-trimmer transcribe", description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--audio", required=True, help="sanctuary / PA-feed channel (or loft)")
    ap.add_argument("--channel", type=int, default=None,
                    help="channel index when --audio is multi-channel (stems: 0=sanctuary 1=loft)")
    ap.add_argument("-o", "--output", required=True, help="output path stem (no extension)")
    ap.add_argument("--config", default=None)
    ap.add_argument("--model", default=None, choices=MODELS + [None],
                    help="whisper model (default from config)")
    ap.add_argument("--model-dir", default=None,
                    help="where to cache model weights (default ./models next to this script)")
    ap.add_argument("--compute-type", default="int8",
                    choices=["int8", "int8_float32", "float32"],
                    help="int8 is the right choice on CPU; float32 for a quality check")
    ap.add_argument("--threads", type=int, default=None,
                    help="CPU threads (default from config; 0 = let CTranslate2 decide)")
    ap.add_argument("--beam-size", type=int, default=5)
    ap.add_argument("--language", default="en")
    ap.add_argument("--no-vad", action="store_true",
                    help="disable voice-activity filtering (VAD skips music and silence)")
    ap.add_argument("--no-predecode", action="store_true")
    args = ap.parse_args(argv)

    require_ffmpeg()
    cfg = load_config(args.config)
    model_name = args.model or asr_model(cfg)
    model_dir = args.model_dir or default_model_dir(cfg)
    threads = args.threads if args.threads is not None else asr_threads(cfg)

    # Conditioning the decoder on church-specific vocabulary is the single biggest
    # accuracy win here -- it is what gets scripture references, the pastor's name
    # and denominational terms spelled correctly.
    prompt = cfg.get("vocabulary_prompt") or None

    try:
        from faster_whisper import WhisperModel
    except ImportError:
        die("faster_whisper not installed. Activate the venv:\n"
            "  source venv/bin/activate\n"
            "or run ./install.sh")

    meta = probe(args.audio)
    info(f"  input   {Path(args.audio).name}  {hms(meta['duration'])}  "
         f"{meta['sample_rate']} Hz  {meta['channels']} ch")
    info(f"  model   {model_name}  ({args.compute_type}, cache {model_dir})")
    if prompt:
        info(f"  prompt  {prompt[:70]}{'...' if len(prompt) > 70 else ''}")

    audio_path = args.audio
    tmp = None
    if not args.no_predecode:
        tmp = predecode(args.audio, 16000, args.channel)
        audio_path = tmp

    try:
        Path(model_dir).mkdir(parents=True, exist_ok=True)
        model = WhisperModel(
            model_name, device="cpu", compute_type=args.compute_type,
            cpu_threads=threads, download_root=model_dir,
        )

        started = time.time()
        segments, tr_info = model.transcribe(
            audio_path,
            language=args.language,
            beam_size=args.beam_size,
            word_timestamps=True,
            initial_prompt=prompt,
            vad_filter=not args.no_vad,
            vad_parameters={"min_silence_duration_ms": 700},
            # A 90-minute service has long stretches of music and silence. Carrying
            # previous text across them is the classic trigger for Whisper looping
            # on a hallucinated phrase, so it stays off.
            condition_on_previous_text=False,
        )

        out_segments = []
        total = meta["duration"] or 0.0
        for seg in segments:
            out_segments.append({
                "id": seg.id, "start": round(seg.start, 3), "end": round(seg.end, 3),
                "text": seg.text.strip(),
                "avg_logprob": round(seg.avg_logprob, 4),
                "no_speech_prob": round(seg.no_speech_prob, 4),
                "words": [
                    {"start": round(w.start, 3), "end": round(w.end, 3),
                     "word": w.word, "probability": round(w.probability, 4)}
                    for w in (seg.words or [])
                ],
            })
            if total and len(out_segments) % 25 == 0:
                pct = 100.0 * seg.end / total
                elapsed = time.time() - started
                info(f"  {hms(seg.end)} / {hms(total)}  ({pct:5.1f}%)  "
                     f"{seg.end / max(elapsed, 1e-6):.2f}x realtime")
    finally:
        if tmp:
            os.unlink(tmp)

    elapsed = time.time() - started
    stem = Path(args.output)
    stem.parent.mkdir(parents=True, exist_ok=True)

    write_json(f"{stem}.json", {
        "audio": meta,
        "channel": args.channel,
        "model": model_name,
        "compute_type": args.compute_type,
        "language": tr_info.language,
        "duration": tr_info.duration,
        "vad": not args.no_vad,
        "initial_prompt": prompt,
        "transcribe_seconds": round(elapsed, 1),
        "segments": out_segments,
    })

    with open(f"{stem}.srt", "w", encoding="utf-8") as fh:
        for i, seg in enumerate(out_segments, 1):
            fh.write(f"{i}\n{srt_time(seg['start'])} --> {srt_time(seg['end'])}\n"
                     f"{seg['text']}\n\n")

    with open(f"{stem}.txt", "w", encoding="utf-8") as fh:
        for seg in out_segments:
            fh.write(f"[{hms(seg['start'])}] {seg['text']}\n")

    words = sum(len(s["words"]) for s in out_segments)
    speed = (meta["duration"] or 0) / max(elapsed, 1e-6)
    info(f"\n  {len(out_segments)} segments, {words} words in {elapsed / 60:.1f} min "
         f"({speed:.2f}x realtime)")
    info(f"  wrote {stem}.json / .srt / .txt")
    return 0



# --------------------------------------------------------------------------- #
# Window transcription (edge refinement)
# --------------------------------------------------------------------------- #

_MODELS: dict[tuple, object] = {}


def load_model(name: str, cfg: dict, threads: int = 0, compute_type: str = "int8"):
    """Load (once per process) a faster-whisper model."""
    key = (name, threads, compute_type)
    if key not in _MODELS:
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            die("faster_whisper not installed; run ./install.sh")
        d = default_model_dir(cfg)
        Path(d).mkdir(parents=True, exist_ok=True)
        _MODELS[key] = WhisperModel(name, device="cpu", compute_type=compute_type,
                                    cpu_threads=threads, download_root=d)
    return _MODELS[key]


def transcribe_window(model, path: str, start: float, end: float, channel: int | None,
                      prompt: str | None, language: str = "en", beam_size: int = 5) -> list[dict]:
    """Transcribe [start, end) of `path` with NO voice-activity filter.

    The full-service pass uses VAD to skip music, and VAD is what loses the short
    liturgical formulas at the sermon edges: an invocation said into the tail of a
    hymn, or a benediction followed straight away by the congregation's Amen,
    lands inside a region VAD discarded. Here the window is only a few minutes, so
    transcribing all of it is cheap.

    Returns segments in the same schema as the full transcript, with times on the
    recording's own timeline.
    """
    start = max(0.0, start)
    x = decode_mono(path, 16000, start=start, duration=max(0.0, end - start), channel=channel)
    if x.size < 16000:
        return []
    segments, _ = model.transcribe(
        np.ascontiguousarray(x, dtype=np.float32), language=language, beam_size=beam_size,
        word_timestamps=True, vad_filter=False, initial_prompt=prompt,
        condition_on_previous_text=False,
    )
    out = []
    for seg in segments:
        out.append({
            "start": round(start + seg.start, 3), "end": round(start + seg.end, 3),
            "text": seg.text.strip(),
            "avg_logprob": round(seg.avg_logprob, 4),
            "no_speech_prob": round(seg.no_speech_prob, 4),
            "words": [
                {"start": round(start + w.start, 3), "end": round(start + w.end, 3),
                 "word": w.word, "probability": round(w.probability, 4)}
                for w in (seg.words or [])
            ],
        })
    return out
