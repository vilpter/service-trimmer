"""Edge resolution against synthetic envelopes modelled on real services.

Levels are dBFS per 50 ms frame. 'san' is the PA feed, 'amb' the loft.
Preaching: san -30, amb -38 (the loft hears the PA in the room).
Music:     san -45, amb -20.  Congregation response: san -48, amb -30.
"""
import unittest

import numpy as np

from service_trimmer import edges as E
from service_trimmer.config import default_config

CFG = default_config()
ECFG, LIT = CFG["edges"], CFG["liturgy"]
HOP = E.HOP


def env(spans, total, t0=0.0):
    n = int(round(total / HOP))
    san = np.full(n, -70.0)
    amb = np.full(n, -50.0)
    for a, b, s, m in spans:
        i, j = int(round((a - t0) / HOP)), int(round((b - t0) / HOP))
        san[i:j], amb[i:j] = s, m
    rng = np.random.default_rng(0)
    return E.Envelope(t0, san + rng.normal(0, 0.5, n), amb + rng.normal(0, 0.5, n), "dual")


def words(text, start, step=0.4):
    return [{"start": start + i * step, "end": start + i * step + step - 0.05, "word": w}
            for i, w in enumerate(text.split())]


def seg(ws):
    return {"start": ws[0]["start"], "end": ws[-1]["end"], "text": " ".join(w["word"] for w in ws),
            "words": ws, "avg_logprob": -0.2, "no_speech_prob": 0.01}


PREACH, MUSIC, RESP = (-30.0, -38.0), (-45.0, -20.0), (-48.0, -30.0)
FILLER = ("and so the word of the lord comes to us today in the gospel which we have heard "
          "and it calls us to repentance and to faith ") * 14


class StartTest(unittest.TestCase):
    def test_invocation_after_hymn_is_included(self):
        e = env([(0, 60, *MUSIC), (62, 66, *PREACH), (66, 67, *RESP), (67.5, 200, *PREACH)], 200)
        hallucinated = words("in the name of the father and of the holy spirit", 10.0)
        inv = words("In the name of the Father and of the Son and of the Holy Spirit", 62.0, 0.3)
        sermon = words(FILLER, 67.5)
        ctx = E.build_context(e, [seg(hallucinated), seg(inv), seg(sermon)], (80, 130), ECFG)
        r = E.resolve_start(ctx, 70.0, ECFG, LIT)
        self.assertEqual(r["method"], "opening:invocation")
        self.assertAlmostEqual(r["time"], 62.0, places=2)
        self.assertGreaterEqual(r["cut"], 59.9)       # never reaches back into the hymn
        self.assertGreater(ctx.rejected_words, 0)     # the sung/hallucinated copy was dropped

    def test_cold_start_after_music(self):
        e = env([(0, 60, *MUSIC), (64, 200, *PREACH)], 200)
        ctx = E.build_context(e, [seg(words(FILLER, 64.0))], (80, 130), ECFG)
        r = E.resolve_start(ctx, 66.0, ECFG, LIT)
        self.assertEqual(r["method"], "first_word")
        self.assertIn("music", r["lead_in_basis"])
        self.assertAlmostEqual(r["time"], 64.0, places=2)


class EndTest(unittest.TestCase):
    def test_easter_response_survives_bad_word_timestamps(self):
        # 2026-05-17: Whisper put the pastor's next word ("Through") inside the
        # congregation's response. The end must follow the audio, not the word.
        e = env([(0, 99.0, *PREACH), (99.6, 102.4, *RESP), (103.0, 130, *PREACH)], 130)
        body = words(FILLER, 0.0)
        body = [w for w in body if w["end"] < 95.5]
        close = words("hallelujah christ is risen", 97.4)
        bogus = [{"start": 99.5, "end": 99.9, "word": "through"}]
        nxt = words("christ let us offer the sacrifice of thanksgiving", 103.0)
        ctx = E.build_context(e, [seg(body), seg(close), seg(bogus), seg(nxt)], (10, 80), ECFG)
        r = E.resolve_end(ctx, 100.0, ECFG, LIT)
        self.assertTrue(r["method"].endswith("+response"), r)
        self.assertGreater(r["cut"], 102.3)
        self.assertLessEqual(r["cut"], 102.85)

    def test_silent_pa_is_not_a_response(self):
        # 2026-02-08: after the benediction the PA is silent and the loft drops;
        # the loft/PA ratio rises anyway. That must not read as an Amen.
        e = env([(0, 99.0, *PREACH), (99.0, 105.0, -70.0, -45.0), (105.0, 130, *PREACH)], 130)
        body = [w for w in words(FILLER, 0.0) if w["end"] < 93.0]
        close = words("the peace of god which passes all understanding keep your hearts "
                      "and minds in christ jesus", 93.6, 0.33)
        nxt = words("we stand to confess the creed", 105.0)
        ctx = E.build_context(e, [seg(body), seg(close), seg(nxt)], (10, 80), ECFG)
        r = E.resolve_end(ctx, 100.0, ECFG, LIT)
        self.assertEqual(r["method"], "closing:benediction")
        end_of_formula = close[-1]["end"]
        self.assertAlmostEqual(r["cut"], end_of_formula + ECFG["amen_allowance_seconds"], places=1)
        self.assertEqual(r["next_act"]["kind"], "creed")

    def test_next_speaker_caps_the_allowance(self):
        # 2026-03-29: "In the name of Jesus," then "We stand..." 1.8 s later.
        e = env([(0, 99.0, *PREACH), (100.8, 130, *PREACH)], 130)
        body = [w for w in words(FILLER, 0.0) if w["end"] < 97.0]
        close = words("in the name of jesus", 97.2)
        nxt = words("we stand to confess the creed", 100.8)
        ctx = E.build_context(e, [seg(body), seg(close), seg(nxt)], (10, 80), ECFG)
        r = E.resolve_end(ctx, 100.0, ECFG, LIT)
        self.assertLessEqual(r["cut"], 100.8 - 0.1)
        self.assertGreater(r["cut"], close[-1]["end"])

    def test_offertory_straight_after_the_response(self):
        # 2026-04-05: "He is risen indeed! Hallelujah!" is in the transcript and
        # the pastor starts "Through Christ let us offer..." 0.2 s later. The PA
        # carries the congregation (-38) so there is no 0.25 s pause to find.
        e = env([(0, 99.0, *PREACH), (99.0, 100.95, -38.0, -25.0), (100.95, 101.15, -50.0, -45.0),
                 (101.15, 130, *PREACH)], 130)
        body = [w for w in words(FILLER, 0.0) if w["end"] < 97.0]
        close = words("hallelujah christ is risen he is risen indeed hallelujah", 97.4, 0.4)
        nxt = words("through christ let us offer the sacrifice of thanksgiving", 101.15, 0.3)
        ctx = E.build_context(e, [seg(body), seg(close), seg(nxt)], (10, 80), ECFG)
        r = E.resolve_end(ctx, 100.0, ECFG, LIT)
        self.assertLessEqual(r["cut"], 101.15 - 0.1)
        self.assertGreaterEqual(r["cut"], close[-1]["end"])

    def test_response_after_a_trailing_blip(self):
        # 2026-05-03: the pastor's last syllable decays in the loft for 0.1 s;
        # the congregation answers 0.5 s later and runs 3.3 s. The blip must not
        # be taken for the response (and then rejected as too short).
        e = env([(0, 97.0, *PREACH), (97.0, 97.1, -70.0, -40.0), (97.6, 100.9, *RESP),
                 (101.7, 130, *PREACH)], 130)
        body = [w for w in words(FILLER, 0.0) if w["end"] < 95.0]
        close = words("alleluia christ is risen", 95.45, 0.4)
        nxt = words("through christ let us offer the sacrifice of thanksgiving", 101.7, 0.3)
        ctx = E.build_context(e, [seg(body), seg(close), seg(nxt)], (10, 80), ECFG)
        r = E.resolve_end(ctx, 100.0, ECFG, LIT)
        self.assertTrue(r["method"].endswith("+response"), r)
        self.assertGreaterEqual(r["cut"], 100.9)
        self.assertLessEqual(r["cut"], 101.7 - 0.1)

    def test_short_announcement_is_not_inside_the_sermon(self):
        # 2026-06-28 / 2026-08-02: a short, quiet "We stand to confess the Creed"
        # or "Continue with the collection of our offering" is too brief to count
        # as sustained speech, and its first syllable is soft.
        e = env([(0, 96.8, *PREACH), (98.8, 99.0, -50.0, -48.0), (99.0, 99.5, -33.0, -40.0)], 130)
        body = [w for w in words(FILLER, 0.0) if w["end"] < 95.0]
        close = words("in jesus name amen", 95.2, 0.4)
        nxt = words("continue with the collection of our offering", 98.8, 0.2)
        ctx = E.build_context(e, [seg(body), seg(close), seg(nxt)], (10, 80), ECFG)
        r = E.resolve_end(ctx, 100.0, ECFG, LIT)
        self.assertLessEqual(r["cut"], 98.8 - 0.1)
        self.assertGreater(r["cut"], close[-1]["end"])

    def test_mid_sermon_formula_is_not_the_close(self):
        e = env([(0, 140, *PREACH), (145, 160, *MUSIC)], 160)
        early = words("in the name of jesus", 60.0)
        rest = [w for w in words(FILLER, 62.5) if w["end"] < 139.0]
        ctx = E.build_context(e, [seg(early), seg(rest)], (10, 50), ECFG)
        r = E.resolve_end(ctx, 140.0, ECFG, LIT)
        self.assertNotIn("closing", r["method"])
        self.assertIn("no closing formula found", r["flags"])
        self.assertGreater(r["cut"], 138.0)


if __name__ == "__main__":
    unittest.main()


class OnsetTest(unittest.TestCase):
    def test_late_word_timestamp_does_not_truncate(self):
        # Whisper ends the formula 0.6 s before the preacher actually stops.
        e = env([(0, 99.6, *PREACH), (104.0, 130, *PREACH)], 130)
        body = [w for w in words(FILLER, 0.0) if w["end"] < 96.0]
        close = words("in the name of jesus", 97.0)          # ends 98.95 by Whisper
        nxt = words("we stand to confess the creed", 104.0)
        ctx = E.build_context(e, [seg(body), seg(close), seg(nxt)], (10, 80), ECFG)
        r = E.resolve_end(ctx, 100.0, ECFG, LIT)
        self.assertGreaterEqual(r["cut"], 99.6)               # past the real end of speech
        self.assertLess(r["cut"], 104.0)


class LeadInOverMusicTest(unittest.TestCase):
    def test_greeting_spoken_under_the_last_chord(self):
        # A preacher starts "Grace to you and peace..." while the organ's last
        # chord still rings: the loft carries music, the PA carries his voice.
        e = env([(0, 60, *MUSIC), (60, 64.5, -30.0, -20.0), (64.5, 200, *PREACH)], 200)
        lyrics = dict(seg(words("and let us come back to you", 40.0)), avg_logprob=-0.69,
                      no_speech_prob=0.82)
        greet = dict(seg(words("Grace to you and peace from God our Father and from our Lord "
                               "Jesus Christ", 60.0)), avg_logprob=-0.26, no_speech_prob=0.005)
        sermon = seg(words(FILLER, 67.5))
        ctx = E.build_context(e, [lyrics, greet, sermon], (80, 130), ECFG)
        r = E.resolve_start(ctx, 70.0, ECFG, LIT)
        self.assertEqual(r["method"], "opening:greeting")
        self.assertAlmostEqual(r["time"], 60.0, delta=0.3)
        self.assertIn("tail", r["lead_in_basis"])

    def test_words_smeared_into_the_hymn_do_not_move_the_start(self):
        # 2026-09-20: Whisper timed the greeting 6.5 s early, inside the hymn,
        # but the PA shows the hymn, 2 s of silence, then the voice. The start
        # must stay at the voice.
        e = env([(0, 64.5, *MUSIC), (64.5, 66.5, -70.0, -55.0), (66.5, 200, *PREACH)], 200)
        greet = dict(seg(words("Grace to you and peace from God our Father and from our Lord "
                               "Jesus Christ", 60.0)), avg_logprob=-0.26, no_speech_prob=0.005)
        sermon = seg(words(FILLER, 67.5))
        ctx = E.build_context(e, [greet, sermon], (80, 130), ECFG)
        r = E.resolve_start(ctx, 70.0, ECFG, LIT)
        self.assertGreaterEqual(r["cut"], 64.5)
        self.assertAlmostEqual(r["time"], 66.5, delta=0.6)
