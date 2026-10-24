"""Candidate scoring and rite trimming, on synthetic full-service transcripts."""
import unittest

from service_trimmer.config import default_config
from service_trimmer.liturgy import WordIndex
from service_trimmer.sermon import score_candidates, trim_at_rite

CFG = default_config()
LIT = CFG["liturgy"]


def transcript(*parts):
    """parts: (start_seconds, text) -> WordIndex, 0.4 s per word."""
    segs = []
    for t0, text in parts:
        words = [{"start": t0 + i * 0.4, "end": t0 + i * 0.4 + 0.35, "word": w}
                 for i, w in enumerate(text.split())]
        segs.append({"start": t0, "end": words[-1]["end"], "text": text, "words": words})
    return WordIndex.from_segments(segs)


class CandidateTest(unittest.TestCase):
    def test_prayers_after_creed_lose_to_the_sermon(self):
        # 2026-06-28: the Prayers of the Church plus a reception rite carried
        # generic formulas and briefly outscored the sermon.
        idx = transcript(
            (2400, "this is the gospel of the lord"),
            (2760, "in the name of the father and of the son and of the holy spirit amen"),
            (3700, "the peace of god which passes all understanding keep your hearts and "
                   "minds in christ jesus amen"),
            (3720, "we stand to confess the creed i believe in one god"),
            (3820, "let us pray for the whole church of god in christ jesus"),
            (4250, "in the name of the father and of the son and of the holy spirit amen"),
            (4260, "let us pray"))
        runs = [(2762.0, 3714.0), (3821.0, 4260.0)]
        best = score_candidates(runs, idx, CFG, LIT, None)[0]
        self.assertEqual((best["start"], best["end"]), runs[0])

    def test_rite_inside_run_trims_to_closing(self):
        # 2026-07-12: installation of officers followed the sermon, no hymn between.
        idx = transcript(
            (2930, "in the name of the father and of the son and of the holy spirit"),
            (3456, "in the name of jesus the peace of christ that transcends all human "
                   "understanding keep and guard your hearts and minds in christ jesus "
                   "to life everlasting"),
            (3467, "id like to invite forward the newly elected officers and board members"),
            (3660, "do you accept the offices entrusted to you"))
        cand = {"start": 2926.0, "end": 3938.0}
        end, rite = trim_at_rite(cand, idx, LIT, 300.0)
        self.assertIsNotNone(rite)
        self.assertEqual(rite["rite"]["kind"], "invitation")
        self.assertLess(end, 3470.0)
        self.assertGreater(end, 3466.0)

    def test_no_rite_leaves_run_alone(self):
        idx = transcript((3000, "in the name of jesus amen"))
        self.assertEqual(trim_at_rite({"start": 2000.0, "end": 3100.0}, idx, LIT, 300.0),
                         (3100.0, None))


if __name__ == "__main__":
    unittest.main()
