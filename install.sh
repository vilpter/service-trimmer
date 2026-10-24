#!/usr/bin/env bash
# Install service-trimmer. Works on Raspberry Pi OS Bookworm/Trixie 64-bit
# (the same Pi that runs ilc-Audio-Recorder, if you want to process there) and
# on any 64-bit Debian/Ubuntu or other Linux PC with Python 3.11+.
#
#   ./install.sh                  everything needed to run (asks for sudo only
#                                 if ffmpeg or python3-venv must be installed)
#   ./install.sh --reel           also spoken labels in review reels (gTTS)
#   ./install.sh --model NAME     speech model to download (default: small.en on
#                                 a Pi, medium.en elsewhere)
#   ./install.sh --no-model       skip the model download
#   ./install.sh --lock           reproduce the reference x86_64/Python 3.12 build
#
# Re-running is safe: it reuses the existing venv and upgrades in place.
set -euo pipefail

cd "$(dirname "$0")"
VENV="venv"
WITH_REEL=0
USE_LOCK=0
MODEL=""
NO_MODEL=0

while [[ $# -gt 0 ]]; do
    case "$1" in
        --reel)     WITH_REEL=1; shift ;;
        --lock)     USE_LOCK=1; shift ;;
        --model)    MODEL="$2"; shift 2 ;;
        --no-model) NO_MODEL=1; shift ;;
        --venv)     VENV="$2"; shift 2 ;;
        -h|--help)  sed -n '2,14p' "$0"; exit 0 ;;
        *) echo "unknown option: $1" >&2; exit 1 ;;
    esac
done

say() { printf '\033[0;32m[install]\033[0m %s\n' "$*"; }
warn() { printf '\033[1;33m[warn]\033[0m %s\n' "$*" >&2; }
fail() { printf '\033[0;31m[fail]\033[0m %s\n' "$*" >&2; exit 1; }

# ── Preflight ────────────────────────────────────────────────────────────────
ARCH="$(uname -m)"
say "architecture: $ARCH"
case "$ARCH" in
    x86_64|aarch64|arm64) ;;
    armv7l|armv6l)
        fail "32-bit ARM detected. The speech recogniser (CTranslate2) publishes no
       packages for armv7l/armv6l. Reflash with the 64-bit Raspberry Pi OS image
       (aarch64) -- it runs on any Pi 3 or newer." ;;
    *) warn "untested architecture; Python packages may not be available" ;;
esac

PY="${PYTHON:-python3}"

# System packages: ffmpeg (decode/encode) and the Python venv module.
NEED=()
command -v ffmpeg >/dev/null && command -v ffprobe >/dev/null || NEED+=(ffmpeg)
command -v "$PY" >/dev/null || NEED+=(python3)
"$PY" -c 'import venv, ensurepip' 2>/dev/null || NEED+=(python3-venv)
if [[ ${#NEED[@]} -gt 0 ]]; then
    if command -v apt-get >/dev/null; then
        say "installing system packages: ${NEED[*]}"
        sudo apt-get update
        sudo apt-get install -y "${NEED[@]}"
    else
        fail "missing: ${NEED[*]}. Install them with your package manager and re-run."
    fi
fi
say "ffmpeg: $(ffmpeg -version | head -1 | cut -d' ' -f1-3)"

PYVER="$($PY -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
$PY -c 'import sys; sys.exit(0 if sys.version_info >= (3, 11) else 1)' \
    || fail "Python 3.11+ required, found $PYVER"
say "python: $PYVER ($PY)"

if [[ $USE_LOCK -eq 1 && "$PYVER" != "3.12" ]]; then
    fail "--lock pins numpy 2.5.1, which has no package for Python $PYVER.
       Drop --lock to use the portable ranges in requirements.txt."
fi

# ── Build ────────────────────────────────────────────────────────────────────
if [[ ! -d "$VENV" ]]; then
    say "creating $VENV"
    "$PY" -m venv "$VENV"
else
    say "reusing existing $VENV"
fi

PIP="$VENV/bin/pip"
"$PIP" install --upgrade pip --quiet

REQ="requirements.txt"
[[ $USE_LOCK -eq 1 ]] && REQ="requirements-lock.txt"
say "installing $REQ"
"$PIP" install -r "$REQ"

if [[ $WITH_REEL -eq 1 ]]; then
    say "installing gTTS (spoken reel labels; needs internet when a reel is built)"
    # --no-deps: gTTS pins click<8.2 for its command-line tool only; the library
    # never imports click, and that pin conflicts with huggingface-hub.
    "$PIP" install "requests>=2.27,<3"
    "$PIP" install --no-deps "gTTS==2.5.4"
fi

say "installing the service-trimmer command"
"$PIP" install --no-deps -e .

# ── Verify ───────────────────────────────────────────────────────────────────
say "verifying imports"
"$VENV/bin/python" - <<'PY'
import numpy, faster_whisper, ctranslate2
print(f"  numpy           {numpy.__version__}")
print(f"  faster-whisper  {faster_whisper.__version__}")
print(f"  ctranslate2     {ctranslate2.__version__}")
PY

# ── Speech model ─────────────────────────────────────────────────────────────
if [[ $NO_MODEL -eq 0 ]]; then
    if [[ -z "$MODEL" ]]; then
        case "$ARCH" in aarch64|arm64) MODEL="small.en" ;; *) MODEL="medium.en" ;; esac
    fi
    say "downloading speech model: $MODEL (one time, into ./models)"
    "$VENV/bin/python" - "$MODEL" <<'PY'
import sys, pathlib
from faster_whisper import WhisperModel
dest = pathlib.Path("models")
dest.mkdir(exist_ok=True)
WhisperModel(sys.argv[1], device="cpu", compute_type="int8", download_root=str(dest))
print(f"  stored in {dest.resolve()}")
PY
fi

cat <<EOF

$(say "done")

  source $VENV/bin/activate
  service-trimmer batch --audio-dir ~/recordings --workdir ~/service-trimmer-work --extract
  service-trimmer reel  --workdir ~/service-trimmer-work -o ~/sermon-review.mp3

Check the install any time with:  python tests/selftest.py
EOF
