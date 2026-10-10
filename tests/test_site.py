"""lodestone/site.py: the installation's time zone from config, TZ, or the machine (docs/LAYERS.md M2)."""

import os
import sys
import unittest
from datetime import datetime
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "StoneSage", "backend"))

from lodestone import site  # noqa: E402


def _zoneinfo_available():
    try:
        from zoneinfo import ZoneInfo
        ZoneInfo("Europe/London")
        return True
    except Exception:      # Windows without the tzdata package
        return False


class TestSite(unittest.TestCase):
    def setUp(self):
        self.env = mock.patch.dict(os.environ, {k: v for k, v in os.environ.items() if k != "TZ"}, clear=True)
        self.env.start()
        self.addCleanup(self.env.stop)

    def test_config_wins_over_tz(self):
        with mock.patch.dict(os.environ, {"TZ": "Asia/Tokyo"}):
            self.assertEqual(site.tz_name({"server": {"timezone": "Europe/London"}}), "Europe/London")
            self.assertEqual(site.tz_name({}), "Asia/Tokyo")
        self.assertEqual(site.tz_name({}), "")

    @unittest.skipUnless(_zoneinfo_available(), "no time zone database on this machine")
    def test_named_zone_follows_daylight_saving(self):
        tz = site.local_tz({"server": {"timezone": "Europe/London"}})
        self.assertEqual(datetime(2026, 1, 15, 12, tzinfo=tz).utcoffset().total_seconds(), 0)
        self.assertEqual(datetime(2026, 7, 15, 12, tzinfo=tz).utcoffset().total_seconds(), 3600)
        self.assertEqual(site.describe({"server": {"timezone": "Europe/London"}}).split(" (")[0], "Europe/London")

    def test_unknown_name_falls_back_to_the_machine(self):
        tz = site.local_tz({"server": {"timezone": "Mars/Olympus_Mons"}})
        self.assertIsNotNone(datetime.now(tz).utcoffset())

    def test_no_zone_named_in_code(self):
        src = open(site.__file__, encoding="utf-8").read()
        self.assertNotIn("America/", src)                                  # nor in its docs: they are public


if __name__ == "__main__":
    unittest.main()
