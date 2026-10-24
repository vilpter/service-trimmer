# Working with ilc-Audio-Recorder

service-trimmer reads what the recorder writes and nothing more, unless you ask
it to write sermon marks back. The two systems share three things.

## 1. File names and channel suffixes

The recorder names each channel file `<job_id>_<suffix>.wav`, where `job_id` is
the recording start (`2026_Feb_08_09:40`) and the suffixes come from the
recorder's **Settings -> Filename** page (defaults `L` / `R`).

Tell service-trimmer which suffix is which role in `recorder.channel_suffixes`. The
shipped config knows this site's `Sanctuary`/`Loft` and the recorder defaults
`L`/`R` (left = sanctuary PA feed, right = loft). If you change the suffixes in
the recorder, add the new ones here; unknown suffixes are reported in
`SUMMARY.txt`, never silently ignored.

Copies that went through a filesystem without `:` support arrive as
`2026_Feb_08_09_40_...`. They are folded into the same service and reported as
duplicates.

## 2. Section tags (`recording_sections`)

Since recorder v1.10 an operator can tap **Mark Beginning / Mark Ending** during
a recording. Those marks are the best possible hint for where the sermon is.

Read them during detection (copy `~/.audio-recorder/schedule.db` from the Pi, or
run service-trimmer on the Pi itself):

```sh
service-trimmer batch ... --recorder-db /path/to/schedule.db
service-trimmer recorder sections --db schedule.db --job-id 2026_Feb_08_09:40
```

A section labelled with "sermon" (or the longest closed section) adds weight to
the candidate it overlaps. It guides the choice of region; the exact edges are
still resolved from the audio, so a late tap does not make a late cut.

Write detected sermons back so they appear in the recorder's Recordings page:

```sh
service-trimmer recorder push-sections --db schedule.db --workdir work/            # dry run
sudo systemctl stop audio-recorder
service-trimmer recorder push-sections --db schedule.db --workdir work/ --apply
sudo systemctl start audio-recorder
```

Rows are labelled `Sermon (auto)`; re-running replaces them and never touches
operator rows. `--min-confidence` (default `medium`) keeps doubtful detections
out of the recorder UI.

## 3. The schedule

`service-trimmer schedule` compares every recording window with the service times in
force that day (`schedule.eras` in the config) and with what the audio shows:
when program audio starts and stops, how much dead air precedes it, and whether
the recorder stopped while the service was still going.

`--recorder-db schedule.db` lists the recorder's current jobs alongside.

### The September 2026 change

Immanuel moved from one Sunday service to two on **2026-09-13**
(8:00 AM and 10:30 AM, with Scripture study / Sunday School at 9:15). The
recorder kept its 09:40 start and 150-minute duration. In the recordings since:

- the 8:00 service is not recorded at all;
- each file opens with ~40-45 minutes of silence (program starts 10:19-10:22);
- the 10:30 service fits, with ~10 minutes to spare at the end.

The sermon is therefore present in every post-change recording, and the
pipeline finds it wherever it falls -- nothing assumes a time of day. But the
early service is being lost. The audit's recommendation, from the measured
program length (median 114 min, prelude to postlude):

| Job (weekly, Sunday) | Start | Duration | Ends |
|---|---|---|---|
| Early Divine Service | 07:45 | 145 min | 10:10 |
| Late Divine Service  | 10:15 | 145 min | 12:40 |

Enter these on the recorder's **Schedule** page as two weekly recurring jobs
(and retire the 09:40 job). If the Sunday-school hour should not be recorded,
the two windows above already leave it out. Re-run `service-trimmer schedule` after a
couple of Sundays to confirm.

When the schedule changes again, add an era to the config:

```json
{ "schedule": { "eras": [
  {"effective": "2026-01-01", "label": "...", "services": [{"weekday": "Sun", "time": "10:00"}]},
  {"effective": "2026-09-13", "label": "...", "services": [{"weekday": "Sun", "time": "08:00"},
                                                          {"weekday": "Sun", "time": "10:30"}]}
]}}
```

(A site config replaces the whole `eras` list, so include the old eras too.)

## Automation

Recordings are uploaded from the Pi to the processing host
(for example `~/recordings`). Run the batch nightly; `--min-age` (default
5 minutes) skips files that are still arriving.

```cron
# m h dom mon dow   -- 03:15 every night
15 3 * * * /path/to/service-trimmer/examples/nightly.sh >> /path/to/work/logs/nightly.log 2>&1
```

See [`examples/nightly.sh`](../examples/nightly.sh) and the systemd units in
[`examples/`](../examples/) for a timer-based alternative. On a Pi, use
`small.en` and `--jobs 1`.
