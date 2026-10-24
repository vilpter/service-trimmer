"""service-trimmer: one entry point for every stage.

    service-trimmer batch     --audio-dir DIR --workdir DIR    process every service (resumable)
    service-trimmer reel      --workdir DIR -o review.mp3      boundary-review reel for QC
    service-trimmer schedule  --audio-dir DIR                  check recordings against service times

    service-trimmer sync | mix | transcribe | sermon           single stages, for one service
    service-trimmer recorder  sections | jobs | push-sections  ilc-Audio-Recorder database
"""

from __future__ import annotations

import importlib
import sys

from . import __version__

COMMANDS = {
    "sync": "service_trimmer.sync",
    "mix": "service_trimmer.mix",
    "transcribe": "service_trimmer.transcribe",
    "sermon": "service_trimmer.sermon",
    "batch": "service_trimmer.batch",
    "reel": "service_trimmer.reel",
    "schedule": "service_trimmer.schedule",
    "recorder": "service_trimmer.recorder",
}


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help", "help"):
        print(__doc__)
        return 0
    if argv[0] in ("-V", "--version", "version"):
        print(f"service-trimmer {__version__}")
        return 0
    cmd = argv.pop(0)
    if cmd not in COMMANDS:
        print(f"unknown command {cmd!r}\n{__doc__}", file=sys.stderr)
        return 2
    return int(importlib.import_module(COMMANDS[cmd]).main(argv) or 0)
