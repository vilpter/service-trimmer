import unittest
from datetime import date, datetime

import numpy as np

from service_trimmer.config import default_config
from service_trimmer.schedule import activity_window, coverage, expected_services

ERAS = default_config()["schedule"]["eras"]


class ScheduleTest(unittest.TestCase):
    def test_eras_switch_on_2026_09_13(self):
        self.assertEqual([s["time"] for s in expected_services(date(2026, 9, 6), ERAS)], ["10:00"])
        self.assertEqual([s["time"] for s in expected_services(date(2026, 9, 20), ERAS)],
                         ["08:00", "10:30"])
        self.assertEqual(expected_services(date(2026, 9, 16), ERAS), [])   # a Wednesday

    def test_coverage(self):
        rs, re_ = datetime(2026, 9, 20, 9, 40), datetime(2026, 9, 20, 12, 10)
        self.assertEqual(coverage(rs, re_, datetime(2026, 9, 20, 8, 0), 100), "missed")
        self.assertEqual(coverage(rs, re_, datetime(2026, 9, 20, 10, 30), 110), "ends early")
        self.assertEqual(coverage(rs, re_, datetime(2026, 9, 20, 10, 30), 110, truncated=False), "covered")
        self.assertEqual(coverage(rs, re_, datetime(2026, 9, 20, 10, 30), 110, truncated=True), "cut off")
        self.assertEqual(coverage(rs, re_, datetime(2026, 9, 20, 10, 0), 100), "covered")

    def test_activity_window_dead_air_and_truncation(self):
        db = np.full(900, -90.0)          # 2.5 h of 10 s frames
        db[270:] = -35.0                  # program from minute 45 to the very end
        w = activity_window(db)
        self.assertAlmostEqual(w["start"], 2700.0, delta=40)
        self.assertTrue(w["truncated"])
        db[860:] = -90.0
        self.assertFalse(activity_window(db)["truncated"])

    def test_silence_has_no_window(self):
        self.assertIsNone(activity_window(np.full(100, -91.0)))


if __name__ == "__main__":
    unittest.main()
