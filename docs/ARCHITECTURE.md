# Architecture and technical notes

The signal-processing core (sync, mix) is carried over unchanged in behaviour from the original church pipeline it grew out of; the notes below explain the non-obvious decisions.

## Sync, mix, transcribe

**Sync.** GCC-PHAT cross-correlation in 30 s windows every 4 minutes. PHAT divides
out the cross-spectrum magnitude and keeps only phase, which is what makes a dry
close mic correlate cleanly against a reverberant room mic — plain correlation is
dominated by their very different spectra and gives a broad, useless peak.

It then least-squares fits `offset(t) = a + b·t` with outlier rejection. `b` is the
relative **clock drift** between the two recorders, and it is the thing a
fixed-offset implementation gets wrong: two independent devices drift 10–100 ppm,
and at 50 ppm a 90-minute service accumulates 270 ms of slip — tight at the start,
audibly flammed by the sermon. Under 25 ms of total slip it applies a constant
shift; over that it resamples.

**Mix.** Applies the shift and the drift correction, then writes stems and a mix.
Auto-gain measures the peak in an analysis pass and attenuates just enough to avoid
clipping — no limiter touches the audio.

**Transcribe.** faster-whisper (CTranslate2, int8) on the sanctuary channel, with
VAD, word timestamps, and a vocabulary prompt from the config.

**Sermon.** Two passes. The acoustic pass uses the two channels as the feature:
during singing and organ the loft mics are hot, during preaching one voice goes
into the PA and the loft mics get mostly room reverb, so ambient-minus-sanctuary
level drops sharply and stays low for the length of the sermon. The transcript pass
then snaps each edge to a liturgical phrase using word timestamps.

---

## If sync fails

`service-trimmer sync` prints every window it probed with a sharpness score. Above ~8 on
speech is a confident lock. If too few windows qualify:

- `--min-sharpness 4` — accept weaker locks
- `--max-offset 30` — one recorder started much later than the other
- `--band none` — disable the 200–4000 Hz speech-band restriction
- `--step 60` — probe more windows

A **high fit residual** warning means the linear drift model doesn't fit — look at
`measurements` in the report before trusting the mix. A recorder that dropped
samples mid-service will show as a step, not a slope.

`service-trimmer sync` also detects and reports **polarity inversion** between the two
channels (miswired preamp or cable); `the mix stage` flips it automatically.

---

In a batch, a service whose channels will not align is not lost: it falls back
to sanctuary-only mode (transcript-based detection on the PA feed) and is graded
accordingly.

The quiet-window gate is relative (`--quiet-db`, default 30 dB below the
channel's own 90th-percentile window level) rather than the original absolute
-50 dBFS, which rejected every window of services recorded at low input gain.

## Verified behaviour

Measured on the reference host (x86_64, ffmpeg 8.1.2, py3.12) against
`tests/make_fixture.py`, which bakes in a known offset, drift, and sermon window.

| Check | Truth | Measured |
|---|---|---|
| Start offset | 734.0 ms | 735.6 ms |
| Clock drift | 60.0 ppm | 59.2 ppm |
| Sync fit residual | — | 0.78 ms RMS, 16/16 windows |
| Residual drift after mix | 0 | 1.0 ppm, 0.00 ms jitter |
| Sermon start | 420.0 s | 420.0 s |
| Sermon end | 1200.0 s | 1201.8 s |

Transcription throughput on real speech (60 s audiobook slice, int8, ~3.4 cores):

| Model | Speed | 90-min service | Disk |
|---|---|---|---|
| `tiny.en` | very fast | — | 75 MB |
| `small.en` | 4.6× realtime | ~20 min | 464 MB |
| `medium.en` | 2.1× realtime | ~43 min | 1.5 GB |

A Pi 4/5 will be several times slower than these; `small.en` is the sensible
ceiling there. Each run prints its own realtime factor.

### A note on drift correction

The obvious `asetrate=rate*scale` does **not** work: asetrate takes an integer
sample rate, so at 48 kHz the smallest expressible change is 20.8 ppm — coarser
than the drift being corrected. `atempo` accepts a float but is WSOLA; measured
here it left 5 ppm of drift and 4.3 ms RMS of splice jitter.

What the mix stage does instead is relabel the stream to 1 MHz (asetrate is a
pure reinterpretation, no resampling), resample by the ratio at that scale, then
relabel back. Granularity becomes 1 ppm, the filter is a smooth polyphase resample
with no splicing, the sample count is unchanged so it costs no more than an
ordinary resample — and it measured 3× faster than atempo as well.

### A note on ffmpeg's `join` filter

`join` has to guess how inputs map to output channels, and when every input is
mono — they all report channel layout `FC` — its guess is not input order. It
silently produced a stems file with the channels **reversed**: no error, no
warning, and every downstream conclusion inverted, because the sermon stage
reads ch0 as the sanctuary feed.

the mix stage now passes an explicit `map=0.0-FL|1.0-FR`, and after rendering
it correlates the stems against the source files and refuses to continue if they
do not line up. If you build filtergraphs of your own, always write the map out.

### A note on the mix layout

The ambient feed is already a mono sum of the two loft mics, so there is no real
stereo image to recover. `--layout mono` (the default) is the honest deliverable.
`--layout stereo-wide` synthesises width by offsetting the ambient a few ms between
L and R; that is a genuine effect, not a reconstruction. If you want to do real
stereo work later, use the stems.

---

## Module map

```
src/service_trimmer/
  cli.py          `service-trimmer` entry point, dispatches subcommands
  config.py       packaged defaults + deep-merged site override
  default-config.json
  naming.py       ilc-Audio-Recorder file names, job ids, pairing, duplicates
  audio.py        ffmpeg wrappers, GCC-PHAT, drift fit, framing, level stats
  sync.py         offset + drift between the two recorders  -> KEY-sync.json
  mix.py          align, drift-correct, mix                 -> KEY-stems/mix.flac
  transcribe.py   faster-whisper full pass + no-VAD window pass
  liturgy.py      transcript normalisation, formula matching with word times
  edges.py        start/end resolution from edge words + 50 ms envelopes (pure)
  sermon.py       candidates, scoring, rite trimming, grading, extraction
  schedule.py     triage + schedule audit
  batch.py        resumable driver over a whole upload directory
  reel.py         boundary-review reel, cue sheet, MP3 chapters
  recorder.py     ilc-Audio-Recorder schedule.db (sections, jobs, push)
tests/            unit tests (stdlib unittest) + synthetic end-to-end selftest
```
