"""Liturgy patterns against phrasing actually transcribed at Immanuel, 2026."""
import unittest

from service_trimmer.config import default_config
from service_trimmer.liturgy import WordIndex, find_all, normalise

LIT = default_config()["liturgy"]


def index(text, t0=0.0, step=0.4):
    words = [{"start": t0 + i * step, "end": t0 + (i + 1) * step - 0.05, "word": w}
             for i, w in enumerate(text.split())]
    return WordIndex.from_segments([{"start": t0, "end": t0 + len(words) * step,
                                     "text": text, "words": words}])


def kinds(text, which):
    return [m.kind for m in find_all(index(text), LIT[which])]


class LiturgyTest(unittest.TestCase):
    def test_normalise(self):
        self.assertEqual(normalise("Jesus'"), "jesus")
        self.assertEqual(normalise("Self-controlled,"), "self controlled")
        self.assertEqual(normalise("O’er"), "oer")

    def test_invocation_variants(self):
        self.assertIn("invocation", kinds(
            "In the name of the Father, and of the Son, and of the Holy Spirit. Amen.", "opening"))
        # Whisper dropped "the Son" on 2026-04-05; still an invocation.
        self.assertIn("invocation", kinds(
            "In the name of the Father and of the Holy Spirit.", "opening"))

    def test_easter_acclamation(self):
        k = kinds("Alleluia! Christ is risen! He is risen indeed. Alleluia.", "opening")
        self.assertIn("easter_acclamation", k)

    def test_closing_benediction(self):
        txt = ("in the name of Jesus the peace of God that transcends all human "
               "understanding keep and guard your hearts and minds in Christ Jesus "
               "to life everlasting")
        ms = find_all(index(txt), LIT["closing"])
        self.assertIn("benediction", [m.kind for m in ms])
        self.assertIn("name_of_jesus", [m.kind for m in ms])
        last = max(ms, key=lambda m: m.end)
        self.assertTrue(last.text.endswith("to life everlasting"))

    def test_after_sermon_markers(self):
        self.assertIn("creed", kinds("We stand to confess the Creed.", "after_sermon"))
        self.assertIn("offertory", kinds(
            "Through Christ let us offer the sacrifice of thanksgiving", "after_sermon"))

    def test_stretched_match_rejected(self):
        # A hallucinated invocation smeared across 30 s of organ (2026-03-29).
        words = "In the name of the Father and of the Holy Spirit.".split()
        idx = WordIndex.from_segments([{"start": 0, "end": 30, "text": "", "words": [
            {"start": i * 3.0, "end": i * 3.0 + 2.9, "word": w} for i, w in enumerate(words)]}])
        self.assertEqual(find_all(idx, LIT["opening"], max_s_per_word=1.2), [])
        self.assertTrue(find_all(idx, LIT["opening"]))


if __name__ == "__main__":
    unittest.main()
