import sqlite3
import tempfile
import unittest
from pathlib import Path

from service_trimmer.recorder import AUTO_LABEL, manual_sermon_hint, push_sections, read_jobs, read_sections

SCHEMA = """
CREATE TABLE scheduled_jobs (id TEXT PRIMARY KEY, name TEXT NOT NULL, start_time TEXT NOT NULL,
  duration INTEGER NOT NULL, created_at TEXT NOT NULL, status TEXT DEFAULT 'pending',
  completed_at TEXT, notes TEXT, is_recurring INTEGER DEFAULT 0, recurrence_pattern TEXT,
  parent_template_id TEXT, allow_override INTEGER DEFAULT 0, capture_video INTEGER DEFAULT 0);
CREATE TABLE recording_sections (id INTEGER PRIMARY KEY AUTOINCREMENT, job_id TEXT NOT NULL,
  label TEXT, start_offset REAL NOT NULL, end_offset REAL, created_at TEXT NOT NULL);
"""


class RecorderDbTest(unittest.TestCase):
    def setUp(self):
        self.db = Path(tempfile.mkdtemp()) / "schedule.db"
        con = sqlite3.connect(self.db)
        con.executescript(SCHEMA)
        con.execute("INSERT INTO scheduled_jobs (id,name,start_time,duration,created_at,"
                    "is_recurring,recurrence_pattern) VALUES ('j1','Sunday','2026-01-04T09:40:00',"
                    "9000,'2026-01-01',1,'{\"type\": \"weekly\", \"days\": [6]}')")
        con.executemany("INSERT INTO recording_sections (job_id,label,start_offset,end_offset,"
                        "created_at) VALUES (?,?,?,?,?)", [
                            ("2026_Feb_08_09:40", "Section 1", 100.0, 400.0, "x"),
                            ("2026_Feb_08_09:40", "Sermon", 2810.0, 3900.0, "x"),
                            ("2026_Feb_08_09:40", "Section 3", 5000.0, None, "x")])
        con.commit()
        con.close()

    def test_read_and_hint(self):
        secs = read_sections(self.db, "2026_Feb_08_09:40")
        self.assertEqual(len(secs), 3)
        hint = manual_sermon_hint(secs)
        self.assertEqual(hint["label"], "Sermon")
        self.assertTrue(hint["labelled_sermon"])
        self.assertEqual(read_sections(self.db, "nope"), [])

    def test_jobs_parse_recurrence(self):
        (job,) = read_jobs(self.db)
        self.assertEqual(job["recurrence_pattern"]["type"], "weekly")

    def test_push_is_dry_run_then_idempotent(self):
        rep = {"job_id": "2026_Feb_08_09:40",
               "sermon": {"start": 2814.0, "end": 3893.0, "confidence": "high"}}
        push_sections(self.db, [rep], apply=False)
        self.assertFalse(any(s["label"] == AUTO_LABEL
                             for s in read_sections(self.db, "2026_Feb_08_09:40")))
        push_sections(self.db, [rep], apply=True)
        push_sections(self.db, [rep], apply=True)
        autos = [s for s in read_sections(self.db, "2026_Feb_08_09:40") if s["label"] == AUTO_LABEL]
        self.assertEqual(len(autos), 1)
        # Our own rows never become the "manual" hint.
        self.assertEqual(manual_sermon_hint(read_sections(self.db, "2026_Feb_08_09:40"))["label"],
                         "Sermon")


if __name__ == "__main__":
    unittest.main()
