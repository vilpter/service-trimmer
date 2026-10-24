"""Liturgical formula matching over word-timestamped transcripts.

Patterns are matched against a *normalised* text stream -- lowercase, punctuation
removed, apostrophes dropped, single spaces -- so "Jesus' name," and "jesus name"
are the same thing to a regex. Each character of that stream maps back to the
word it came from, so a match yields exact word-level start/end times.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

from .audio import info

_STRIP = re.compile(r"[^a-z0-9 ]+")


def normalise(token: str) -> str:
    t = token.lower().replace("’", "'").replace("'", "")
    t = t.replace("-", " ")
    return " ".join(_STRIP.sub(" ", t).split())


@dataclass
class WordIndex:
    words: list[dict]          # {"start", "end", "word", ...}
    text: str                  # normalised stream
    char_to_word: list[int]

    @classmethod
    def from_segments(cls, segments: list[dict], keep=None) -> "WordIndex":
        """Flatten segments to words. `keep(word)` may veto individual words."""
        words: list[dict] = []
        for seg in segments:
            src = seg.get("words") or (
                [{"start": seg["start"], "end": seg["end"], "word": seg.get("text", "")}]
                if seg.get("text", "").strip() else [])
            for w in src:
                if keep is None or keep(w):
                    words.append(w)
        words.sort(key=lambda w: w["start"])

        parts: list[str] = []
        char_to_word: list[int] = []
        for i, w in enumerate(words):
            token = normalise(w.get("word") or "")
            if not token:
                continue
            if parts:
                parts.append(" ")
                char_to_word.append(i)
            parts.append(token)
            char_to_word.extend([i] * len(token))
        return cls(words, "".join(parts), char_to_word)

    def slice_words(self, a: int, b: int) -> list[dict]:
        return self.words[a:b + 1]


@dataclass
class Match:
    kind: str
    pattern: str
    text: str
    first: int                 # word index
    last: int
    start: float               # first word start
    end: float                 # last word end

    def to_dict(self) -> dict:
        return {"kind": self.kind, "text": self.text, "start": round(self.start, 3),
                "end": round(self.end, 3), "pattern": self.pattern}


def compile_patterns(specs: list) -> list[tuple[str, re.Pattern, str]]:
    out = []
    for spec in specs:
        kind, rx = (spec["kind"], spec["re"]) if isinstance(spec, dict) else ("marker", spec)
        try:
            out.append((kind, re.compile(rx), rx))
        except re.error as exc:
            info(f"  warning: bad liturgy regex {rx!r}: {exc}")
    return out


def find_all(index: WordIndex, specs: list, t_min: float = float("-inf"),
             t_max: float = float("inf"), max_s_per_word: float | None = None) -> list[Match]:
    """Every match whose words lie within [t_min, t_max].

    `max_s_per_word` rejects matches whose words are implausibly stretched.
    Whisper does two things that produce those: it spreads a word across a VAD
    gap (one sentence observed claiming 00:45:20-00:50:08), and, given a prompt,
    it can hallucinate a phrase across a stretch of music (a fake invocation
    spanning 30 s of organ). Real speech runs ~0.3-0.5 s per word.
    """
    found: list[Match] = []
    if not index.text:
        return found
    for kind, rx, raw in compile_patterns(specs):
        for m in rx.finditer(index.text):
            if m.end() <= m.start():
                continue
            a = index.char_to_word[m.start()]
            b = index.char_to_word[min(m.end() - 1, len(index.char_to_word) - 1)]
            start, end = index.words[a]["start"], index.words[b]["end"]
            if start < t_min or end > t_max:
                continue
            n = b - a + 1
            if max_s_per_word and (end - start) / max(n, 1) > max_s_per_word:
                continue
            found.append(Match(kind, raw, m.group(0), a, b, float(start), float(end)))
    found.sort(key=lambda x: (x.start, -x.end))
    return found
