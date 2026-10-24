"""ilc-Audio-Recorder file conventions.

The recorder writes one mono WAV per input channel:

    <job_id>_<suffix>.wav        job_id = YYYY_Mon_DD_HH:MM   (recording start, local)

`suffix` is configurable in the recorder (Settings -> Filename; defaults L/R).
`job_id` is also the key of the recorder's `recording_sections` table, so it is
the join key between the two systems.

Copies that passed through a filesystem without ':' support come back as
YYYY_Mon_DD_HH_MM_<suffix>.wav. Those are the same recording, so they are
folded into one service and reported as duplicates rather than processed twice.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

MONTHS = "Jan Feb Mar Apr May Jun Jul Aug Sep Oct Nov Dec".split()
MONTH_NUM = {m: i for i, m in enumerate(MONTHS, 1)}
FILE_RE = re.compile(
    r"^(?P<y>\d{4})_(?P<mon>[A-Za-z]{3})_(?P<d>\d{2})_(?P<h>\d{2})(?P<sep>[:_])(?P<mi>\d{2})"
    r"_(?P<suffix>[^/]+?)\.wav$")


@dataclass
class Recording:
    path: Path
    job_id: str          # canonical, with ':' -- matches recorder DB rows
    start: datetime
    suffix: str
    role: str | None     # 'sanctuary' | 'ambient' | None (unknown suffix)
    colon_form: bool


@dataclass
class Service:
    key: str             # 2026-02-08_0940 -- filesystem-safe, sorts chronologically
    job_id: str          # 2026_Feb_08_09:40
    start: datetime
    sanctuary: Path | None = None
    ambient: Path | None = None
    duplicates: list[str] = field(default_factory=list)
    unknown: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def channels(self) -> list[str]:
        return [r for r in ("sanctuary", "ambient") if getattr(self, r)]

    def to_dict(self) -> dict:
        return {
            "key": self.key, "job_id": self.job_id, "start": self.start.isoformat(),
            "sanctuary": str(self.sanctuary) if self.sanctuary else None,
            "ambient": str(self.ambient) if self.ambient else None,
            "duplicates": self.duplicates, "unknown": self.unknown, "notes": self.notes,
        }


def role_for_suffix(suffix: str, suffixes: dict[str, list[str]]) -> str | None:
    low = suffix.lower()
    for role, names in suffixes.items():
        if any(low == n.lower() for n in names):
            return role
    return None


def parse_name(path: Path, suffixes: dict[str, list[str]]) -> Recording | None:
    m = FILE_RE.match(path.name)
    if not m or m["mon"].title() not in MONTH_NUM:
        return None
    mon = m["mon"].title()
    start = datetime(int(m["y"]), MONTH_NUM[mon], int(m["d"]), int(m["h"]), int(m["mi"]))
    job_id = f"{m['y']}_{mon}_{m['d']}_{m['h']}:{m['mi']}"
    return Recording(path=path, job_id=job_id, start=start, suffix=m["suffix"],
                     role=role_for_suffix(m["suffix"], suffixes),
                     colon_form=m["sep"] == ":")


def service_key(start: datetime) -> str:
    return start.strftime("%Y-%m-%d_%H%M")


def job_id_for(start: datetime) -> str:
    return f"{start:%Y}_{MONTHS[start.month - 1]}_{start:%d}_{start:%H}:{start:%M}"


def key_to_job_id(key: str) -> str:
    return job_id_for(datetime.strptime(key, "%Y-%m-%d_%H%M"))


def discover(audio_dir: Path, suffixes: dict[str, list[str]],
             min_age_s: float = 0.0) -> tuple[list[Service], list[str]]:
    """Group recorder files into services. Returns (services, warnings).

    `min_age_s` skips files modified that recently -- an upload still in progress
    is not safe to read.
    """
    services: dict[str, Service] = {}
    warnings: list[str] = []
    now = time.time()

    for path in sorted(audio_dir.glob("*.wav")):
        rec = parse_name(path, suffixes)
        if rec is None:
            warnings.append(f"{path.name}: not a recorder filename, ignored")
            continue
        if min_age_s and now - path.stat().st_mtime < min_age_s:
            warnings.append(f"{path.name}: modified {now - path.stat().st_mtime:.0f}s ago, "
                            "probably still uploading; skipped this run")
            continue
        svc = services.setdefault(service_key(rec.start),
                                  Service(service_key(rec.start), rec.job_id, rec.start))
        if rec.role is None:
            svc.unknown.append(path.name)
            continue

        current: Path | None = getattr(svc, rec.role)
        if current is None:
            setattr(svc, rec.role, path)
            continue
        # Same job, same role, twice: a ':'-vs-'_' copy. Keep the canonical
        # colon form when the copies agree; keep the larger when they do not.
        keep, drop = current, path
        same_size = current.stat().st_size == path.stat().st_size
        if same_size:
            if rec.colon_form and ":" not in current.name:
                keep, drop = path, current
        else:
            if path.stat().st_size > current.stat().st_size:
                keep, drop = path, current
            svc.notes.append(f"{rec.role}: two copies of different size; kept {keep.name}")
        setattr(svc, rec.role, keep)
        svc.duplicates.append(drop.name)

    for svc in services.values():
        if svc.unknown:
            warnings.append(f"{svc.key}: unrecognised channel suffix in {', '.join(svc.unknown)} "
                            "(add it to recorder.channel_suffixes)")
    return sorted(services.values(), key=lambda s: s.start), warnings
