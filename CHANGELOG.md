# Changelog

## [1.0.0] - 2026-10-24

First release, as a companion to
[ilc-Audio-Recorder](https://github.com/vilpter/ilc-Audio-Recorder). Validated
against 34 recordings from February-October 2026 (30 services, every sermon
boundary reviewed by ear).

### Added
- `service-trimmer` command with subcommands: `batch`, `reel`, `schedule`, `sync`,
  `mix`, `transcribe`, `sermon`, `recorder`.
- `install.sh`: one-step install on Raspberry Pi OS (64-bit) or any 64-bit Linux
  PC, including system packages and the speech model.
- **Sermon boundaries include the lead-in and the Amen.** Starts on the
  preacher's invocation, Easter acclamation, greeting or prayer after the Hymn of
  the Day; ends after the closing formula (benediction, "In the name of Jesus",
  doxology, acclamation) plus the congregation's response, and before the next act.
- Edge refinement: each boundary is re-transcribed with no voice-activity filter
  (cached), then resolved against 50 ms level envelopes of both channels. Music
  detection rejects sung or misheard words; congregational responses are detected
  as loft bursts while the PA is quiet; the start is backed up to the voice's
  acoustic onset.
- The next speaker bounds the end even with no pause before them: an offertory
  sentence begun the instant the congregation finishes, or a short, quiet
  announcement ("We stand to confess the Creed", "Continue with the collection of
  our offering"). The transcript says that someone speaks next; the PA feed says
  when.
- Candidate scoring by liturgical evidence and order (Gospel before, Creed after,
  formula specificity, plausible length, operator section marks).
- Rite detection: confirmation / reception / installation following the sermon
  without a hymn between is not merged into it.
- Confidence grading (high / medium / low) with review flags and notes.
- Triage before any expensive stage: silent recordings, dead or missing channels,
  test recordings (short, or outside service hours), duplicate `09_40` copies.
- Single-channel modes (sanctuary-only, loft-only) when a feed is dead or
  missing; a sync failure falls back to sanctuary-only instead of failing the
  service.
- Schedule audit with per-era service times and a recommended recorder schedule.
- ilc-Audio-Recorder database integration: section marks as detection hints;
  optional `push-sections` writes detected sermons back (dry run by default).
- Review reel: run-in / beep on the cut / sermon, spoken labels with confidence,
  per-service loudness matching, MP3 chapters, cue sheet.
- Speech model and thread count default to `auto`: `small.en` on a Pi,
  `medium.en` on a PC.
- Layered config (packaged defaults + deep-merged site file).
- Unit tests (standard library only) and a synthetic end-to-end self-test.
