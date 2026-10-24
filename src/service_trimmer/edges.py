"""Resolve the exact sermon start and end from an edge transcript plus level envelopes.

What counts as "the sermon" here (the deliverable a listener wants):

  START  includes any lead-in the preacher says after the Hymn of the Day --
         the Trinitarian invocation and the congregation's Amen, an Easter
         acclamation and its response, a greeting, or a short prayer.
  END    includes the closing formula -- "In the name of Jesus", the Pauline
         benediction ("The peace of God which passes all understanding...") --
         AND the congregation's Amen after it, and stops before the next act
         ("We stand to confess the Creed", the offertory sentence, a hymn).

Two signals:

  * Words, from a no-VAD re-transcription of a few minutes around each edge.
  * Level envelopes of both channels at 50 ms. The loft (ambient) channel hears
    the congregation; the sanctuary channel is the PA feed and barely does. So
    music and congregational responses are where loft-minus-sanctuary jumps up:
    sustained for a hymn, a 0.5-3 s burst for an Amen.

Everything in this module is a pure function of those inputs, so it can be
tested with synthetic words and envelopes (tests/test_edges.py).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .audio import find_runs, frame_rms_db, median_smooth, decode_mono
from .liturgy import Match, WordIndex, find_all, normalise

HOP = 0.05


@dataclass
class Envelope:
    """50 ms level frames over one edge window."""
    t0: float
    san: np.ndarray | None      # dBFS, sanctuary (PA feed)
    amb: np.ndarray | None      # dBFS, loft
    mode: str                   # dual | sanctuary-only | ambient-only

    @property
    def n(self) -> int:
        ref = self.san if self.san is not None else self.amb
        return 0 if ref is None else int(ref.size)

    def idx(self, t: float) -> int:
        return int(np.clip(round((t - self.t0) / HOP), 0, max(self.n - 1, 0)))

    def time(self, i: int) -> float:
        return self.t0 + i * HOP

    def feature(self) -> np.ndarray | None:
        """The 'congregation is louder than the pulpit' signal."""
        if self.mode == "dual":
            return self.amb - self.san
        if self.mode == "ambient-only":
            return self.amb
        return None


def load_envelope(path: str, a: float, b: float, mode: str, rate: int = 8000) -> Envelope:
    """Read [a, b) of a stems file (ch0 sanctuary, ch1 loft) or a single channel."""
    a = max(0.0, a)
    dur = max(0.0, b - a)
    san = amb = None
    if mode == "dual":
        san = frame_rms_db(decode_mono(path, rate, a, dur, channel=0), rate, HOP, 2 * HOP)
        amb = frame_rms_db(decode_mono(path, rate, a, dur, channel=1), rate, HOP, 2 * HOP)
        m = min(san.size, amb.size)
        san, amb = san[:m], amb[:m]
    elif mode == "ambient-only":
        amb = frame_rms_db(decode_mono(path, rate, a, dur), rate, HOP, 2 * HOP)
    else:
        san = frame_rms_db(decode_mono(path, rate, a, dur), rate, HOP, 2 * HOP)
    return Envelope(a, san, amb, mode)


# --------------------------------------------------------------------------- #

@dataclass
class EdgeContext:
    env: Envelope
    words: WordIndex                 # validated words only
    music: np.ndarray | None         # bool per frame
    ref: float | None                # feature level while preaching
    san_ref: float | None = None     # PA-feed level while the preacher talks
    amb_ref: float | None = None     # loft level while the preacher talks (PA in the room)
    rejected_words: int = 0
    notes: list[str] = field(default_factory=list)

    def in_music(self, t: float) -> bool:
        return bool(self.music is not None and self.music[self.env.idx(t)])

    def music_runs(self) -> list[tuple[float, float]]:
        if self.music is None:
            return []
        return [(self.env.time(a), self.env.time(b)) for a, b in find_runs(self.music)]


def build_context(env: Envelope, segments: list[dict], body: tuple[float, float],
                  cfg: dict) -> EdgeContext:
    """Measure the preaching reference level, mark music, and validate words.

    `body` is a stretch known to be sermon (well inside the acoustic run); the
    feature's median there is what "the preacher talking" looks like.
    """
    feat = env.feature()
    music = ref = san_ref = amb_ref = None
    notes: list[str] = []
    a, b = sorted((env.idx(body[0]), env.idx(body[1])))
    # Frames where the pulpit is actually speaking define every reference level.
    speaking = slice(a, b + 1)
    talk = None
    if env.san is not None and env.san[speaking].size:
        san_sl = env.san[speaking]
        talk = san_sl > np.percentile(san_sl, 40)
        san_ref = float(np.median(san_sl[talk])) if talk.any() else None
    if env.amb is not None and env.amb[speaking].size:
        amb_sl = env.amb[speaking]
        amb_ref = float(np.median(amb_sl[talk] if talk is not None and talk.any() else amb_sl))
    if feat is not None and feat.size:
        sl = feat[speaking]
        if talk is not None and talk.size == sl.size:
            sl = sl[talk]
        if sl.size >= 20:
            ref = float(np.median(sl))
            smooth = median_smooth(feat, int(round(1.0 / HOP)) | 1)
            hot = smooth > ref + float(cfg.get("music_margin_db", 8.0))
            # Music must also be *loud* in the loft, not merely loft-heavy silence.
            if env.amb is not None and amb_ref is not None:
                hot &= median_smooth(env.amb, int(round(1.0 / HOP)) | 1) > amb_ref - 3.0
            min_frames = int(round(float(cfg.get("music_min_seconds", 6.0)) / HOP))
            music = np.zeros_like(hot)
            for s, e in find_runs(hot):
                if e - s >= min_frames:
                    music[s:e] = True
        else:
            notes.append("too little preaching in the window to calibrate levels")

    # Whisper's own silence rule: high no-speech probability AND low confidence.
    # no_speech_prob alone is not enough -- distant loft-mic speech comes back
    # correctly transcribed with no_speech_prob ~0.8 (2026-07-26).
    # Loft-only audio is distant and reverberant: correct text routinely comes
    # back at log-prob -0.9, so the bar is lower there.
    ns_max, lp_quiet, lp_floor = (0.9, -1.0, -1.4) if env.mode == "ambient-only" else (0.6, -0.8, -1.2)
    seg_ok = [s for s in segments
              if not (s.get("no_speech_prob", 0.0) > ns_max and s.get("avg_logprob", 0.0) < lp_quiet)
              and s.get("avg_logprob", 0.0) > lp_floor]
    dropped = len(segments) - len(seg_ok)
    rejected = 0
    clean_lp = float(cfg.get("speech_over_music_logprob", -0.45))
    clean_ns = float(cfg.get("speech_over_music_no_speech", 0.3))

    # Words inside music are sung text or hallucinations -- unless Whisper is
    # confident they are speech. A preacher often begins the lead-in over the
    # hymn's last chord (2026-09-20, 09-27: "Grace to you and peace..." under
    # the organ). Sung lyrics come back at log-prob ~-0.7 / no-speech ~0.85;
    # that greeting came back at -0.26 / 0.005.
    for seg in seg_ok:
        clean = seg.get("avg_logprob", -9) > clean_lp and seg.get("no_speech_prob", 1) < clean_ns
        for w in seg.get("words") or []:
            w["_clean"] = clean

    # ...but only in the last few seconds of a music run. Mid-hymn, even a
    # confident transcription is sung text or a hallucination.
    tail_s = float(cfg.get("speech_over_music_tail_seconds", 8.0))
    tails = []
    if music is not None:
        tails = [(env.time(b) - tail_s, env.time(b)) for _, b in find_runs(music)]

    def keep(w: dict) -> bool:
        nonlocal rejected
        mid = 0.5 * (w["start"] + w["end"])
        if music is not None and music[env.idx(mid)]:
            if not (w.get("_clean") and any(a <= mid <= b for a, b in tails)):
                rejected += 1
                return False
        return True

    words = WordIndex.from_segments(seg_ok, keep=keep)
    if dropped:
        notes.append(f"{dropped} low-confidence segment(s) ignored")
    if rejected:
        notes.append(f"{rejected} word(s) inside music ignored (sung text or hallucination)")
    return EdgeContext(env, words, music, ref, san_ref, amb_ref, rejected, notes)


def first_word_at_or_after(words: WordIndex, t: float) -> dict | None:
    return next((w for w in words.words if w["start"] >= t), None)


# --------------------------------------------------------------------------- #
# START
# --------------------------------------------------------------------------- #

def first_voice(ctx: EdgeContext, t_from: float) -> float | None:
    """First sustained (0.2 s) speech-level signal on the PA feed at or after t_from."""
    if ctx.env.san is None or ctx.san_ref is None:
        return None
    talk = median_smooth(ctx.env.san, 3) > ctx.san_ref - 12.0
    need = int(round(0.2 / HOP))
    for i in range(ctx.env.idx(t_from), talk.size - need):
        if talk[i:i + need].all():
            return ctx.env.time(i)
    return None


def voice_on_pa(ctx: EdgeContext, words: list[dict]) -> bool:
    """Is the preacher audible on the PA feed under these words?

    Over a hymn the PA mic carries only bleed (~-42 dBFS here); a lead-in spoken
    over the last chord lifts it toward speech level. Without a PA channel there
    is nothing to check, so the transcript's confidence has to suffice.
    """
    if ctx.env.san is None or ctx.san_ref is None or not words:
        return True
    san = median_smooth(ctx.env.san, 5)
    hits = [san[ctx.env.idx(0.5 * (w["start"] + w["end"]))] > ctx.san_ref - 10.0 for w in words]
    return sum(hits) >= 0.5 * len(hits)


def utterance_onset(ctx: EdgeContext, t: float, floor: float, max_back: float = 1.5) -> float:
    """Back `t` up to where the utterance containing it actually starts on the PA.

    Whisper's first word of an utterance is often late: on 2026-03-15 it
    smeared "In the name of" backwards across the end of the hymn, those words
    were (rightly) dropped as sung, and the remaining "the Father" started half
    a second after the voice did. Reading the PA level recovers the onset.
    Never moves past `floor` (the end of the music) or by more than `max_back`.
    """
    env = ctx.env
    if env.san is None or ctx.san_ref is None:
        return t
    talk = median_smooth(env.san, 3) > ctx.san_ref - 12.0
    lo = max(env.idx(floor), env.idx(t - max_back))
    i = env.idx(t)
    if not talk[i] and not (i + 1 < talk.size and talk[i + 1]):
        return t
    gap, j = 0, i
    while j - 1 >= lo:
        if talk[j - 1]:
            gap = 0
        else:
            gap += 1
            if gap > int(round(0.3 / HOP)):
                break
        j -= 1
    first = j + gap if gap else j
    return min(t, env.time(first))


def resolve_start(ctx: EdgeContext, acoustic_start: float, cfg: dict, liturgy: dict) -> dict:
    lookback = float(cfg.get("max_lead_in_seconds", 90.0))
    pre_roll = float(cfg.get("pre_roll_seconds", 0.35))
    spw = float(cfg.get("max_seconds_per_word", 1.2))
    words = ctx.words.words
    if not words:
        return {"time": acoustic_start, "cut": max(0.0, acoustic_start - pre_roll),
                "method": "acoustic_only", "evidence": None, "lead_in_from": None}

    w0 = first_word_at_or_after(ctx.words, acoustic_start - 3.0) or words[-1]
    anchor = w0["start"]

    # Where can the lead-in begin? After the last music before the sermon.
    music_end = None
    for s, e in ctx.music_runs():
        if e <= anchor + 0.5 and e >= anchor - lookback:
            music_end = e
    if music_end is not None:
        zone = music_end
        zone_why = "end of the music before the sermon"
        # A lead-in that starts over the music's tail: follow the utterance
        # that is still running at music_end back to its first word.
        prior = [w for w in words if w["start"] < zone and w["end"] > zone - 10.0]
        later = [w for w in words if w["start"] >= zone]
        if prior and later and later[0]["start"] - prior[-1]["end"] < 1.5:
            chain = prior[-1]["start"]
            for a, b in zip(reversed(prior[:-1]), reversed(prior[1:])):
                if b["start"] - a["end"] >= 1.5:
                    break
                chain = a["start"]
            if chain < zone and not voice_on_pa(ctx, [w for w in prior if w["start"] >= chain]):
                chain = zone
            if chain < zone:
                zone = chain
                zone_why = "end of the music before the sermon (lead-in begins over its tail)"
    else:
        # No music found in range: walk back through the words and stop at the
        # first long silence -- whatever is before it is a different act.
        zone = anchor
        prev_end = anchor
        for w in reversed([w for w in words if w["start"] < anchor]):
            if prev_end - w["end"] > 12.0 or anchor - w["start"] > lookback:
                break
            zone = w["start"]
            prev_end = w["start"]
        zone_why = "first long pause before the sermon (no music found)"

    openings = find_all(ctx.words, liturgy.get("opening", []), t_min=zone - 0.5,
                        t_max=anchor + 8.0, max_s_per_word=spw)
    lead_words = [w for w in words if zone - 0.5 <= w["start"] < anchor]

    if openings:
        m = openings[0]
        start, method, evidence = m.start, f"opening:{m.kind}", m.to_dict()
    elif lead_words:
        start, method = lead_words[0]["start"], "lead_in_speech"
        evidence = {"text": " ".join(normalise(w["word"]) for w in lead_words)[:160],
                    "start": lead_words[0]["start"]}
    else:
        start, method, evidence = anchor, "first_word", {"text": normalise(w0["word"]),
                                                         "start": anchor}

    floor = min(music_end, zone) if music_end is not None else 0.0
    # Nobody speaks before the pulpit mic hears them. Whisper sometimes times
    # the first words seconds early, into the hymn (2026-09-20: 6.5 s), so a
    # word-based start earlier than the PA's first voice is moved to it.
    pa = first_voice(ctx, floor)
    if pa is not None and start < pa - 0.1:
        start = pa
    start = utterance_onset(ctx, start, floor)
    cut = max(floor, start - pre_roll)
    return {"time": round(float(start), 3), "cut": round(float(cut), 3), "method": method,
            "evidence": evidence, "lead_in_from": round(float(zone), 3),
            "lead_in_basis": zone_why, "first_sermon_word": round(float(anchor), 3)}


# --------------------------------------------------------------------------- #
# END
# --------------------------------------------------------------------------- #

def speech_onset(ctx: EdgeContext, t_from: float, t_to: float | None = None) -> float | None:
    """When the preacher (or the next speaker at the pulpit) starts talking again.

    Read from the PA feed, not from word timestamps: Whisper's word times drift
    by seconds exactly where it matters (on 2026-05-17 it put "Through" at
    4830.7 when it is spoken at 4833.8, which would have cut the congregation's
    "He is risen indeed!" in half). A lone one-word "Amen" from the pastor is not
    a new sentence, so onset needs sustained speech: 60% of the next 1.2 s.
    """
    if ctx.env.san is None or ctx.san_ref is None:
        return None
    san = median_smooth(ctx.env.san, 3)
    talk = san > ctx.san_ref - 6.0
    quiet = san < ctx.san_ref - 12.0
    win = int(round(1.2 / HOP))
    b = ctx.env.n - 1 if t_to is None else ctx.env.idx(t_to)
    b = min(b, talk.size - 1)
    # Whisper often ends the last word a few tenths early. If the PA is still
    # carrying the preacher at t_from, that is the same sentence finishing, not
    # the next one starting: an onset only counts after a real pause.
    a = ctx.env.idx(t_from)
    min_pause = int(round(0.25 / HOP))
    run = 0
    while a <= b:
        run = run + 1 if quiet[a] else 0
        if run >= min_pause:
            break
        a += 1
    else:
        return None
    for i in range(a, b + 1):
        if talk[i] and talk[i:i + win].mean() >= 0.6:
            return ctx.env.time(i)
    return None


def detect_response(ctx: EdgeContext, after: float, until: float | None,
                    cfg: dict) -> tuple[float, float] | None:
    """The congregation's response (Amen, 'He is risen indeed!') just after `after`.

    It is a burst in the LOFT channel while the PULPIT is quiet. Both halves
    matter: the loft-minus-PA ratio alone also goes up whenever the PA simply
    falls silent (on 2026-02-08 the loft *dropped* after the benediction while
    the ratio rose by 20 dB). So: loft at least `response_margin_db` above its
    level in the pause right after the formula, and near or above the level it
    carries while the preacher talks, with the PA feed well below speech.
    """
    env = ctx.env
    if env.amb is None or ctx.amb_ref is None:
        return None
    search = float(cfg.get("response_search_seconds", 3.0))
    max_len = float(cfg.get("response_max_seconds", 5.0))
    margin = float(cfg.get("response_margin_db", 5.0))
    amb = median_smooth(env.amb, 3)
    a = env.idx(after)
    pause = amb[a: env.idx(after + 0.6) + 1]
    floor = float(np.min(pause)) if pause.size else float(amb[a])
    hot = (amb > floor + margin) & (amb > ctx.amb_ref - 3.0)
    if env.san is not None and ctx.san_ref is not None:
        hot &= median_smooth(env.san, 3) < ctx.san_ref - 6.0
    stop = env.idx(until) if until is not None else env.n - 1
    b = min(env.idx(after + search), stop)
    gap_frames = int(round(0.3 / HOP))
    min_frames = int(round(0.3 / HOP))
    # The first burst long enough to be voices. The preacher's last syllable
    # decaying in the loft is a one- or two-frame blip right at `after`; taking
    # it as the onset then rejected the real response as too short (2026-05-03:
    # "He is risen indeed! Alleluia!" ran 4247.6-4250.9 and was missed).
    i = a
    while i <= b:
        if not hot[i]:
            i += 1
            continue
        onset = last_hot = j = i
        limit = min(env.idx(env.time(onset) + max_len), stop)
        while j <= limit and j < hot.size:
            if hot[j]:
                last_hot = j
            elif j - last_hot > gap_frames:
                break
            j += 1
        if last_hot - onset >= min_frames:
            return env.time(onset), env.time(last_hot + 1)
        i = j
    return None


CONGREGATION_WORDS = ("amen", "he", "is", "risen", "indeed", "alleluia", "hallelujah")


def next_speaker_onset(ctx: EdgeContext, t_from: float, t_to: float | None = None) -> float | None:
    """Where the next spoken act (offertory sentence, "We stand to confess the
    Creed", an announcement) begins, even with no pause before it.

    speech_onset() needs a pause and a sustained 1.2 s of speech; it misses a
    pastor who starts the offertory the instant the congregation finishes
    (2026-04-05) and a short, quiet "Continue with the collection of our
    offering" (2026-08-02). Here the transcript says *that* someone speaks next
    and the PA feed says *when*: take the second word of the next utterance
    (the first word's start is unreliable -- Whisper stretches it back over
    the silence before it: "Through" at 4249.4 for speech at 4251.7 on
    2026-05-03), confirm the PA carries a voice there, and walk back to where
    that stretch of voice began.
    """
    env = ctx.env
    if env.san is None or ctx.san_ref is None:
        return None
    later = [w for w in ctx.words.words if w["start"] > t_from + 0.05
             and (t_to is None or w["start"] < t_to)
             and normalise(w["word"]) not in CONGREGATION_WORDS]
    if not later:
        return None
    anchor = later[0]["start"]
    if len(later) > 1 and later[1]["start"] - later[0]["start"] < 4.0:
        anchor = later[1]["start"]
    san = median_smooth(env.san, 3)
    voice = san > ctx.san_ref - 8.0
    i0 = env.idx(anchor)
    hit = next((i for i in range(i0, min(env.idx(anchor + 0.5), env.n - 1) + 1) if voice[i]), None)
    if hit is None:
        return None
    lo = env.idx(t_from)
    if hit <= lo:
        return None
    quiet = san < ctx.san_ref - 12.0
    gap = int(round(0.15 / HOP))
    # Back to the end of the last pause. No pause since t_from means the
    # "next" words run on from the closing itself: not a new speaker.
    k = hit
    while k - gap >= lo and not quiet[k - gap:k].all():
        k -= 1
    if k - gap < lo:
        return None
    # A soft first syllable ("Con-tinue" at -63 dB over a -73 dB gap) sits
    # under the quiet line but is already speech: when real silence lies at
    # most 0.3 s before, start just after it. Congregation bleed on the PA
    # (2026-05-17) never gets that low, so the onset stays put there.
    floor = float(np.min(san[lo:hit]))
    back = range(k - 1, max(lo, k - int(round(0.3 / HOP))) - 1, -1)
    silent = next((m for m in back if san[m] <= floor + 6.0), None)
    if silent is not None:
        k = silent + 1
    onset = env.time(k)
    return onset if onset > t_from else None


def resolve_end(ctx: EdgeContext, acoustic_end: float, cfg: dict, liturgy: dict) -> dict:
    spw = float(cfg.get("max_seconds_per_word", 1.2))
    post_roll = float(cfg.get("post_roll_seconds", 0.5))
    amen = float(cfg.get("amen_allowance_seconds", 2.0))
    lo, hi = cfg.get("after_sermon_search", [-60.0, 120.0])
    before = float(cfg.get("closing_search_before_end", 150.0))
    words = ctx.words.words
    B = acoustic_end
    if not words:
        return {"time": B, "cut": B + post_roll, "method": "acoustic_only", "evidence": None}

    # 1. What comes next? The earliest post-sermon act after the sermon.
    nexts: list[tuple[float, dict]] = []
    for m in find_all(ctx.words, liturgy.get("after_sermon", []),
                      t_min=B + float(lo), t_max=B + float(hi), max_s_per_word=spw):
        nexts.append((m.start, {"kind": m.kind, "text": m.text, "start": round(m.start, 3)}))
    for s, e in ctx.music_runs():
        if B - 30.0 <= s <= B + float(hi):
            nexts.append((s, {"kind": "music", "start": round(s, 3), "end": round(e, 3)}))
    nexts.sort(key=lambda x: x[0])
    U, next_act = (nexts[0][0], nexts[0][1]) if nexts else (None, None)

    # 2. The closing formula: the LAST one before the next act.
    limit = U if U is not None else B + 45.0
    closings = [m for m in find_all(ctx.words, liturgy.get("closing", []),
                                    t_min=B - before, t_max=limit + 0.5, max_s_per_word=spw)
                if m.end <= limit + 0.5]
    closing: Match | None = None
    # The Pauline benediction is this preacher's definitive close: when it is
    # there, it wins over any later formula (rites have their own Trinitarian
    # blessings). Otherwise take the last formula before the next act.
    bened = [m for m in closings if m.kind == "benediction"]
    pool = bened or closings
    if pool:
        closing = max(pool, key=lambda m: (m.end, m.end - m.start))
        if not bened:
            # A formula followed by lots more preaching was mid-sermon, not the close.
            tail = [w for w in words if closing.end + 0.05 < w["start"] < limit
                    and normalise(w["word"]) not in ("amen", "alleluia", "hallelujah")]
            if len(tail) > 25:
                closing = None

    flags: list[str] = []
    gap = float(cfg.get("next_speaker_gap_seconds", 0.15))

    def next_onset(t: float) -> float | None:
        # The next speech on the PA feed bounds everything after the formula.
        found = [o for o in (speech_onset(ctx, t), next_speaker_onset(ctx, t)) if o is not None]
        return min(found) if found else None

    if closing is not None:
        F = closing.end
        onset = next_onset(F)
        resp = detect_response(ctx, F, onset, cfg)
        if resp is not None:
            end, method = resp[1] + post_roll, f"closing:{closing.kind}+response"
        else:
            end, method = F + amen, f"closing:{closing.kind}"
        evidence = closing.to_dict()
        end = max(end, F + 0.3)
        if onset is not None:
            end = min(end, onset - gap)
        elif ctx.env.san is None:
            # Loft-only: no PA feed to read an onset from; fall back to words.
            # Congregational words (Amen, risen indeed) belong to the response.
            after_resp = resp[1] if resp else F
            nxt = next((w for w in words if w["start"] > after_resp + 0.05
                        and normalise(w["word"]) not in ("amen", "he", "is", "risen", "indeed",
                                                         "alleluia", "hallelujah")), None)
            if nxt is not None:
                end = min(end, nxt["start"] - 0.15)
        end = max(end, F + 0.05)
        response = {"start": round(resp[0], 3), "end": round(resp[1], 3)} if resp else None
    else:
        evidence, response = None, None
        bound = U if U is not None else B + 2.0
        last = [w for w in words if w["end"] <= bound]
        if last:
            F = last[-1]["end"]
            method = "last_word_before_next_act" if U is not None else "last_word_near_acoustic_end"
        else:
            F, method = B, "acoustic_only"
        if ctx.env.san is None:
            # Loft-only: no PA feed to find the next speaker on, and Whisper's
            # word ends run late on distant audio (2026-07-26: "Thanks be to
            # God" ends ~4011.4, transcribed to 4012.2; room noise from 4013.3).
            amen = float(cfg.get("amen_allowance_ambient_seconds", 1.0))
        end = F + amen
        onset = next_onset(F)
        if onset is not None:
            end = min(end, onset - gap)
        elif U is not None:
            end = min(end, U - 0.15)
        end = max(end, F + 0.05)
        flags.append("no closing formula found")

    return {"time": round(float(F), 3), "cut": round(float(max(end, F)), 3), "method": method,
            "evidence": evidence, "response": response, "next_act": next_act, "flags": flags}
