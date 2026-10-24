# Service Trimmer

**Sermon Finder and Post-Production for ilc-Audio-Recorder Church Recordings**

Service Trimmer takes the dual-mono WAV files recorded by
[ilc-Audio-Recorder](https://github.com/vilpter/ilc-Audio-Recorder) and turns
each Sunday's recording into a finished sermon file. It lines up and mixes the
two channels, transcribes the service, finds where the sermon starts and ends,
and cuts it out as FLAC and MP3. It also builds a short review file so a person
can check every cut by ear in a few minutes, and it checks the recorder's
schedule against the church's service times.

Everything runs locally on a Raspberry Pi or any 64-bit Linux PC. No accounts,
subscriptions or cloud services are needed.

## Features

### Sermon Detection
- Finds the sermon automatically in a full service recording
- Starts on the preacher's lead-in: invocation ("In the name of the Father..."),
  Easter acclamation, greeting ("Grace to you and peace...") or prayer
- Ends after the closing formula (benediction, "In the name of Jesus", doxology,
  acclamation) **and the congregation's Amen or response**
- Stops before whatever follows: Creed, offertory, announcement, hymn, or a rite
  such as confirmation or reception of members
- Cut points are set from the audio itself, to a fraction of a second
- Confidence grade (high / medium / low) for every service, with the reason

### Audio Processing
- Aligns the sanctuary (PA) and choir-loft channels, correcting both the offset
  and the clock drift between them
- Mixes the channels and writes a time-aligned two-channel stems file
- Exports each sermon as FLAC (archive) and loudness-normalised MP3 (-16 LUFS,
  ready for a podcast feed or website)
- Handles problem recordings: silent files, a dead or missing channel, test
  recordings, and duplicate copies are detected and reported

### Quality Review
- Builds one review MP3 with a beep exactly at every cut, plus a few seconds of
  audio on each side, for all services at once
- Spoken labels ("February 8th. Starting clip follows."), MP3 chapters per
  service, and a cue sheet listing where every beep falls
- `SUMMARY.txt` lists every recording, its sermon times, and anything to check

### Recorder Integration
- Reads the recorder's file naming and channel suffixes (Settings -> Filename)
- Uses an operator's **Mark Beginning / Mark Ending** section tags as hints
- Can write detected sermons back to the recorder database so they appear on
  its Recordings page (dry run by default)
- Schedule audit: compares each recording window with the service times in force
  that day and recommends recorder schedule settings

### System
- One-step installer for Raspberry Pi OS and Debian/Ubuntu
- Resumable batch processing: re-run any time, finished work is skipped
- Nightly automation with cron or a systemd timer
- Self-test that checks an installation end to end

## Hardware Requirements

- **Raspberry Pi:** Pi 4 (4GB+ RAM) or Pi 5 recommended; 64-bit OS required.
  This can be the same Pi that runs ilc-Audio-Recorder.
- **Or any PC:** 64-bit Linux (x86_64), 4GB+ RAM. A PC is much faster.
- **Storage:** about 1 GB for the software and speech model (2 GB on a PC),
  plus about 1.3 GB of working space per service processed (stems, mix and
  sermon files; see `--prune` below)

## Software Requirements

- **OS:** Raspberry Pi OS Lite (64-bit) - Trixie (Debian 13) or Bookworm
  (Debian 12); or any 64-bit Debian, Ubuntu or similar Linux
- **Python:** 3.11 or newer (included with Pi OS)
- **FFmpeg:** installed automatically by `install.sh`

## Software Stack

| Component | Used for |
|-----------|----------|
| **Python 3.11+** | All program logic |
| **NumPy** | Signal processing: channel alignment, level analysis, sermon edge detection |
| **FFmpeg / FFprobe** | Reading WAV files, resampling, mixing, FLAC and MP3 encoding |
| **faster-whisper + CTranslate2** | Offline speech-to-text, run on the CPU. The speech model downloads once at install; after that no network is needed |
| **SQLite** (Python standard library) | Reading and writing the recorder's `schedule.db` |
| **gTTS** (optional) | Spoken labels in the review MP3. Uses Google's text-to-speech over the internet; without it, tones are used instead |
| **cron or systemd** | Optional nightly automation |

No GPU is required. All Python packages install as prebuilt packages on both
Raspberry Pi (aarch64) and PC (x86_64), so nothing needs to be compiled.

## Quick Start

### New Installation
```bash
cd ~
git clone https://github.com/vilpter/service-trimmer.git service-trimmer
cd service-trimmer
./install.sh
```

### Upgrading Existing Installation
```bash
cd ~/service-trimmer
git pull
./install.sh
```

The installer handles:
- System dependencies (FFmpeg, Python venv) via `apt`, asking for `sudo` only if
  something is missing
- A self-contained Python environment in `venv/`
- Python packages
- The speech model (`small.en` on a Pi, `medium.en` on a PC)
- The `service-trimmer` command

Installer options:

```bash
./install.sh --reel             # also spoken labels in review files (gTTS)
./install.sh --model base.en    # choose a different speech model
./install.sh --no-model         # skip the model download
```

### Check the Installation
```bash
cd ~/service-trimmer
source venv/bin/activate
python tests/selftest.py
```
This builds a synthetic service recording and runs the whole pipeline on it. It
should end with `7/7 checks passed`.

## Installation (Manual)

If you prefer manual installation or the installer encounters issues:

### Step 1: System Updates

```bash
sudo apt update && sudo apt upgrade -y
```

### Step 2: Install Dependencies

```bash
sudo apt install -y python3-venv ffmpeg git
```

### Step 3: Get the Code

```bash
cd ~
git clone https://github.com/vilpter/service-trimmer.git service-trimmer
cd service-trimmer
```

### Step 4: Create the Python Environment

```bash
python3 -m venv venv
venv/bin/pip install --upgrade pip
venv/bin/pip install -r requirements.txt
venv/bin/pip install --no-deps -e .
```

### Step 5: Download the Speech Model

```bash
venv/bin/python -c "from faster_whisper import WhisperModel; WhisperModel('small.en', device='cpu', compute_type='int8', download_root='models')"
```
Use `medium.en` instead of `small.en` on a PC.

### Step 6: Verify

```bash
source venv/bin/activate
python tests/selftest.py
```

## Usage

Activate the environment first (once per terminal session):

```bash
cd ~/service-trimmer
source venv/bin/activate
```

### Process All Recordings

```bash
service-trimmer batch --audio-dir ~/recordings --workdir ~/service-trimmer-work --extract
```
- `--audio-dir` is where the recorder's WAV files are (`~/recordings` on the
  recorder Pi, or wherever you copy them)
- `--workdir` is where results go
- `--extract` writes each sermon as FLAC and MP3
- Safe to re-run: only new recordings are processed
- Files modified in the last 5 minutes are skipped (still being copied)

When it finishes, open `~/service-trimmer-work/SUMMARY.txt`.

### Review the Cuts

```bash
service-trimmer reel --workdir ~/service-trimmer-work -o ~/sermon-review.mp3
```

For each service you hear:

> *"February 8th. Starting clip follows."* - 10 s before the cut - **beep** - first 15 s of the sermon
>
> *"February 8th. Ending clip follows."* - last 15 s of the sermon - **beep** - 10 s after the cut

The beep sits exactly on the cut. The lead-in should come right **after** the
start beep; the closing words and Amen should come right **before** the end
beep. `sermon-review.cues.txt` lists where each beep falls. Review only some
services with `--only 2026-04-05_0940 2026-05-03_0940`.

### Check the Recording Schedule

```bash
service-trimmer schedule --audio-dir ~/recordings --workdir ~/service-trimmer-work -o ~/service-trimmer-work/SCHEDULE-AUDIT.txt
```
Shows, for every recording, when the service audio actually starts and stops,
how much silence comes first, whether recording stopped before the service
ended, and recommended start times and durations to enter on the recorder's
Schedule page.

### Use the Recorder's Section Tags

If an operator taps **Mark Beginning / Mark Ending** during recordings, point
the batch at the recorder's database:

```bash
service-trimmer batch --audio-dir ~/recordings --workdir ~/service-trimmer-work --extract \
    --recorder-db ~/.audio-recorder/schedule.db
```

### Send Detected Sermons to the Recorder

```bash
# Preview (changes nothing)
service-trimmer recorder push-sections --db ~/.audio-recorder/schedule.db --workdir ~/service-trimmer-work

# Write them (stop the recorder service first)
sudo systemctl stop audio-recorder
service-trimmer recorder push-sections --db ~/.audio-recorder/schedule.db --workdir ~/service-trimmer-work --apply
sudo systemctl start audio-recorder
```
Sermons appear as sections labelled `Sermon (auto)`. Re-running replaces them;
sections an operator created are never touched. Only high and medium confidence
sermons are sent unless you add `--min-confidence low`.

### Single Steps (one service)

```bash
service-trimmer sync       --sanctuary S.wav --ambient L.wav -o work/K-sync.json
service-trimmer mix        --sync work/K-sync.json --outdir work --basename K
service-trimmer transcribe --audio S.wav -o work/K
service-trimmer sermon     --stems work/K-stems.flac --transcript work/K.json -o work/K-sermon.json --extract --mp3
```
Every command has `--help`.

## Output Files

Each service is named by its recording start, `YYYY-MM-DD_HHMM` (the `KEY`
below). The recorder's own job id (`YYYY_Mon_DD_HH:MM`) is kept in every report.

| File | Contents |
|------|----------|
| `KEY-sermon.mp3` | The sermon, loudness-normalised MP3 (with `--extract`) |
| `KEY-sermon.flac` | The sermon, lossless FLAC (with `--extract`) |
| `KEY-sermon.json` | Sermon times, how each cut was decided, confidence, flags |
| `KEY-mix.flac` | The full service, both channels mixed |
| `KEY-stems.flac` | The full service, 2 channels: L = sanctuary, R = loft, aligned |
| `KEY.txt` / `.srt` / `.json` | Transcript (plain text, subtitles, word timings) |
| `KEY-sync.json` | Measured offset and drift between the channels |
| `KEY-triage.json` | Channel health and how the recording was handled |
| `KEY-edges.json` | Cached detail transcription around the sermon edges |
| `SUMMARY.txt` / `summary.csv` / `summary.json` | Every recording at a glance |
| `logs/` | One log per service and step |

**Disk use:** stems and mix are about 1.1 GB per 2.5-hour recording. Add
`--prune` to the batch to delete the stems once a service is done.

## How the Sermon Is Found

1. **Triage.** Channel levels decide the route: both channels live -> full
   processing; one dead or missing -> single-channel mode; silent -> reported;
   short or outside service hours -> treated as a test recording.
2. **Candidates.** While one voice preaches into the PA, the loft microphones
   hear mostly the room, so the loft-to-sanctuary level drops and stays low.
   Every long low stretch is a candidate.
3. **Scoring.** Each candidate is scored on the liturgy around it: an opening
   formula near its start, a closing formula and a following act (Creed,
   offertory, rite) near its end, the Gospel reading shortly before, a plausible
   length, and any operator section tag.
4. **Rites.** A confirmation or reception can follow the sermon with no hymn
   between. Phrases such as "I'd like to invite forward..." or "Do you
   renounce..." cut the candidate back to the closing formula before them.
5. **Edges.** A few minutes around each edge are transcribed again in detail
   and matched against the sound levels of both channels, 20 times a second:
   - **start** = the first word of the lead-in after the hymn ends, moved back
     to where the voice actually begins;
   - **end** = the end of the closing formula, plus the congregation's response
     when the loft microphones hear one, and never past the moment the next
     speaker begins.
6. **Grade.** High means both edges were anchored on liturgy with both channels
   available.

Word timings from speech recognition can be off by seconds right at these
boundaries, so they are never trusted on their own: the audio levels set the
final cut. Details are in [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md).

## File Structure

```
service-trimmer/
├── install.sh                 # Automated installer
├── requirements.txt           # Python dependencies (Pi and PC)
├── requirements-lock.txt      # Exact versions of the reference x86_64 build
├── pyproject.toml             # Package definition (the service-trimmer command)
├── src/service_trimmer/
│   ├── cli.py                 # service-trimmer command, dispatches subcommands
│   ├── config.py              # Default settings + site overrides
│   ├── default-config.json    # All settings, with defaults
│   ├── naming.py              # Recorder file names, job ids, channel pairing
│   ├── audio.py               # FFmpeg helpers, alignment math, level analysis
│   ├── sync.py                # Offset and drift between the two channels
│   ├── mix.py                 # Align, drift-correct and mix
│   ├── transcribe.py          # Speech-to-text
│   ├── liturgy.py             # Liturgical phrase matching with word timings
│   ├── edges.py               # Exact start/end from words + sound levels
│   ├── sermon.py              # Candidates, scoring, grading, extraction
│   ├── schedule.py            # Triage and schedule audit
│   ├── batch.py               # Whole-directory processing (resumable)
│   ├── reel.py                # Review MP3, cue sheet, chapters
│   └── recorder.py            # ilc-Audio-Recorder database (sections, jobs)
├── tests/
│   ├── selftest.py            # End-to-end check on a synthetic service
│   └── test_*.py              # Unit tests
├── examples/
│   ├── nightly.sh             # Nightly processing script
│   ├── service-trimmer-nightly.service   # systemd unit
│   ├── service-trimmer-nightly.timer     # systemd timer
│   └── site-config.example.json          # Example site settings
└── docs/
    ├── ARCHITECTURE.md        # Technical notes
    └── INTEGRATION.md         # Recorder integration, schedule, automation
```

## Configuration

All settings and their defaults are in `src/service_trimmer/default-config.json`.
To change any of them, create `service-trimmer.json` in the folder you run from
(or pass `--config FILE`, or set `$SERVICE_TRIMMER_CONFIG`) containing only what
you change. It is merged on top of the defaults:

```json
{
  "transcribe": {"model": "small.en", "threads": 4},
  "recorder": {"channel_suffixes": {"sanctuary": ["L"], "ambient": ["R"]}}
}
```

| Section | What to adjust |
|---------|----------------|
| `recorder` | Channel suffixes (must match the recorder's Settings -> Filename), service hours, silence thresholds |
| `schedule.eras` | Service times by effective date - **add a new era when the schedule changes** |
| `transcribe` | Speech model (`auto`, `tiny.en`, `base.en`, `small.en`, `medium.en`) and CPU threads |
| `vocabulary_prompt` | Church name, pastor's name and terms, to improve spelling in transcripts |
| `liturgy` | Opening, closing, after-sermon and rite phrases |
| `edges` | Fine-tuning of the cut points (margins, Amen allowance) |
| `extract` | MP3 bitrate and loudness |
| `reel` | Review file layout, beep, voice or tones |

See [examples/site-config.example.json](examples/site-config.example.json) and
[docs/INTEGRATION.md](docs/INTEGRATION.md) for the recorder channel suffixes and
the schedule eras.

### Speech Models

| Model | Size | Notes |
|-------|------|-------|
| `tiny.en` | 75 MB | Fastest, least accurate |
| `base.en` | ~145 MB | For a slow Pi |
| `small.en` | 464 MB | **Default on a Pi** |
| `medium.en` | 1.5 GB | **Default on a PC**; most accurate |

## Automation

Process new recordings every night:

```bash
cp examples/nightly.sh ~/service-trimmer-nightly.sh   # edit the paths at the top
crontab -e
# add:
15 3 * * * $HOME/service-trimmer-nightly.sh >> $HOME/service-trimmer-work/nightly.log 2>&1
```

Or with a systemd timer:

```bash
mkdir -p ~/.config/systemd/user
cp examples/service-trimmer-nightly.{service,timer} ~/.config/systemd/user/
systemctl --user daemon-reload
systemctl --user enable --now service-trimmer-nightly.timer
```

On the recorder Pi itself, schedule processing for a time when nothing is being
recorded, and use `--jobs 1`.

## Troubleshooting

### `service-trimmer: command not found`

```bash
cd ~/service-trimmer
source venv/bin/activate
```

### Installer Fails on 32-bit Pi OS

The speech recogniser needs 64-bit Raspberry Pi OS. Check with `uname -m`: it
must say `aarch64`. If it says `armv7l`, reflash the SD card with
"Raspberry Pi OS Lite (64-bit)".

### A Recording Is Skipped

Check the top of `SUMMARY.txt`. Common reasons:
- `silent` - nothing was recorded (check the recorder's input levels)
- `test-recording` - shorter than 20 minutes, or outside service hours
- `unrecognised channel suffix` - the channel suffixes don't match; set
  `recorder.channel_suffixes` to match the recorder's Settings -> Filename

### A Cut Is in the Wrong Place

1. Listen to that service in the review file
2. Look at its line in `SUMMARY.txt` and the flags and notes in `KEY-sermon.json`
3. If the preacher used a phrase the software doesn't know, add it to
   `liturgy.opening` or `liturgy.closing` in your site config
4. Re-run just that service:
   ```bash
   service-trimmer batch --audio-dir ~/recordings --workdir ~/service-trimmer-work --extract \
       --only 2026-04-05_0940 --force sermon
   ```

### Processing Is Slow on a Pi

- Use a smaller model: `"transcribe": {"model": "base.en"}`
- Process one service at a time: `--jobs 1`
- Or copy the recordings to a PC and process them there

### Check the Logs

```bash
ls ~/service-trimmer-work/logs/
less ~/service-trimmer-work/logs/2026-04-05_0940-sermon.log
```

## Built With

- **Language:** Python 3.11+
- **Signal processing:** NumPy
- **Audio:** FFmpeg
- **Speech-to-text:** faster-whisper, CTranslate2 (offline, CPU)
- **Database:** SQLite3
- **Testing:** Python `unittest`

## Development Notes

### Running the Tests

```bash
source venv/bin/activate
python -m unittest discover -s tests
python tests/selftest.py
```

### Post-Work Checklist
After completing development work, ensure the following are updated:

1. **Version Numbers**
   - Update `__version__` in `src/service_trimmer/__init__.py`
   - Update `version` in `pyproject.toml`
   - Update version at bottom of `README.md`
   - Add entry to `CHANGELOG.md`

2. **Documentation**
   - Update `README.md` features if applicable
   - Update `docs/ARCHITECTURE.md` if detection logic changed
   - Update `docs/INTEGRATION.md` if the recorder interface changed

3. **Testing**
   - Unit tests and self-test pass
   - Re-run affected services and listen to the review file

## Support

- **Issues:** [GitHub Issues](https://github.com/vilpter/service-trimmer/issues)
- **Recorder:** [ilc-Audio-Recorder](https://github.com/vilpter/ilc-Audio-Recorder)

---

**Version:** 1.0.0
**Last Updated:** 2026-10-24
