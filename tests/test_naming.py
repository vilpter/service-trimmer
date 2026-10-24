import os
import tempfile
import unittest
from pathlib import Path

from service_trimmer.naming import discover, key_to_job_id, parse_name

SUFFIXES = {"sanctuary": ["Sanctuary", "L"], "ambient": ["Loft", "R"]}


class NamingTest(unittest.TestCase):
    def setUp(self):
        self.dir = Path(tempfile.mkdtemp())

    def touch(self, name, size=10):
        p = self.dir / name
        p.write_bytes(b"x" * size)
        return p

    def test_parse_colon_and_underscore_forms(self):
        a = parse_name(Path("2026_Feb_08_09:40_Sanctuary.wav"), SUFFIXES)
        b = parse_name(Path("2026_Feb_08_09_40_Sanctuary.wav"), SUFFIXES)
        self.assertEqual(a.job_id, "2026_Feb_08_09:40")
        self.assertEqual(b.job_id, a.job_id)
        self.assertTrue(a.colon_form)
        self.assertFalse(b.colon_form)
        self.assertEqual(a.role, "sanctuary")

    def test_recorder_default_suffixes(self):
        self.assertEqual(parse_name(Path("2026_Jan_20_23:47_L.wav"), SUFFIXES).role, "sanctuary")
        self.assertEqual(parse_name(Path("2026_Jan_20_23:47_R.wav"), SUFFIXES).role, "ambient")

    def test_not_a_recording(self):
        self.assertIsNone(parse_name(Path("notes.wav"), SUFFIXES))
        self.assertIsNone(parse_name(Path("2026_Xyz_08_09:40_L.wav"), SUFFIXES))

    def test_discover_pairs_and_folds_duplicates(self):
        self.touch("2026_Feb_08_09:40_Sanctuary.wav")
        self.touch("2026_Feb_08_09_40_Sanctuary.wav")
        self.touch("2026_Feb_08_09:40_Loft.wav")
        self.touch("2026_Feb_22_09:40_Sanctuary.wav")
        self.touch("2026_Mar_01_09:40_Mystery.wav")
        services, warnings = discover(self.dir, SUFFIXES)
        by = {s.key: s for s in services}
        feb8 = by["2026-02-08_0940"]
        self.assertEqual(feb8.sanctuary.name, "2026_Feb_08_09:40_Sanctuary.wav")
        self.assertEqual(feb8.duplicates, ["2026_Feb_08_09_40_Sanctuary.wav"])
        self.assertEqual(feb8.channels, ["sanctuary", "ambient"])
        self.assertEqual(by["2026-02-22_0940"].channels, ["sanctuary"])
        self.assertTrue(any("Mystery" in w for w in warnings))

    def test_differing_duplicates_keep_larger(self):
        self.touch("2026_Feb_08_09:40_Sanctuary.wav", 10)
        self.touch("2026_Feb_08_09_40_Sanctuary.wav", 99)
        (svc,), _ = discover(self.dir, SUFFIXES)
        self.assertEqual(svc.sanctuary.name, "2026_Feb_08_09_40_Sanctuary.wav")
        self.assertTrue(svc.notes)

    def test_min_age_skips_uploading(self):
        p = self.touch("2026_Feb_08_09:40_Sanctuary.wav")
        os.utime(p, None)
        services, warnings = discover(self.dir, SUFFIXES, min_age_s=3600)
        self.assertEqual(services, [])
        self.assertTrue(any("uploading" in w for w in warnings))

    def test_key_round_trip(self):
        self.assertEqual(key_to_job_id("2026-04-02_1856"), "2026_Apr_02_18:56")


if __name__ == "__main__":
    unittest.main()
