import importlib.util
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

import pandas as pd

spec = importlib.util.spec_from_file_location("rainfall_demo", Path(__file__).resolve().parents[1] / "src/run_demo.py")
demo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(demo)


class RainfallDemoTest(unittest.TestCase):
    def test_does_not_match_future_scan(self):
        t = pd.Timestamp("2025-01-01T00:10Z")
        labels = pd.DataFrame({"timestamp": [t]})
        scans = pd.DataFrame({"scan_time": [t - pd.Timedelta(minutes=4), t + pd.Timedelta(minutes=1)],
                              "path": ["past", "future"]})
        self.assertEqual(demo.pair(labels, scans, 10)["path"].tolist(), ["past"])
        self.assertTrue(demo.pair(labels, scans, 2).empty)

    def test_shared_scan_purged_at_boundary(self):
        times = pd.date_range("2025-01-01", periods=12, freq="10min", tz="UTC")
        pairs = pd.DataFrame({"timestamp": times, "scan_time": times})
        pairs.loc[9, "scan_time"] = times[8]
        train, test = demo.chronological_split(pairs)
        self.assertEqual(len(train), 8)
        self.assertLess(train["scan_time"].max(), test["scan_time"].min())

    def test_synthetic_pipeline(self):
        with TemporaryDirectory() as tmp:
            root = Path(tmp)
            radar, csv, station = demo.synthetic_inputs(root / "inputs")
            self.assertTrue(pd.isna(demo.read_dbzh(demo.radar_index(radar)["path"].iloc[0])[0, 0]))
            report = demo.run(radar, csv, station, root, 10, synthetic=True)
            self.assertEqual(report["usable_pairs"], 120)
            self.assertEqual(report["test"]["rows"], 30)
            self.assertLess(report["linear_regression"]["mae_mm"], report["training_median"]["mae_mm"])
            self.assertTrue((root / "holdout.png").is_file())
