import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import collect_dataset_parquets as cdp
import create_interaction_parquet as cip


def write_minimal_default_parquet(path):
    if cip.pd is None:
        raise unittest.SkipTest("pandas is not installed")
    df = cip.pd.DataFrame([{
        "timestamp_ms": 0.0,
        "p1_root_valid": True,
        "p1_root_x": 0.0,
        "p1_root_y": 0.0,
        "p1_root_z": 0.0,
        "p2_root_valid": True,
        "p2_root_x": 1.0,
        "p2_root_y": 0.0,
        "p2_root_z": 0.0,
        "p1_gaze_observed": False,
        "p1_gaze_interpolated": False,
        "p2_gaze_observed": False,
        "p2_gaze_interpolated": False,
    }])
    df.to_parquet(path, engine="pyarrow", index=False)


def write_calibration(path):
    np.savez(
        path,
        ref_camera=np.array(1),
        num_cameras=np.array(1),
        image_size=np.array([1280, 720]),
        K1=np.eye(3),
        dist1=np.zeros((1, 5)),
        R_1_to_ref=np.eye(3),
        t_1_to_ref=np.zeros(3),
    )


class CollectDatasetParquetsTests(unittest.TestCase):
    def test_discover_sessions_orders_each_direct_parent_by_timestamp(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = root / "amildak_ramazan" / "recording_group"
            (group / "session_20260101_120000").mkdir(parents=True)
            (group / "session_20260101_110000").mkdir(parents=True)

            sessions = cdp.discover_sessions(root)

        self.assertEqual([s.session_stamp for s in sessions], ["20260101_110000", "20260101_120000"])
        self.assertEqual([s.order for s in sessions], [1, 2])
        self.assertEqual(cdp.output_filename(sessions[0]), "20260101_110000_order1.parquet")

    def test_collect_creates_default_copy_and_final_interaction_parquet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = root / "amildak_ramazan" / "recording_group"
            session = group / "session_20260101_110000"
            session.mkdir(parents=True)
            write_minimal_default_parquet(session / cdp.DEFAULT_PARQUET_NAME)
            write_calibration(group / "multicam_calibration.npz")

            results = cdp.collect_dataset_parquets(root)

            default_path = root / cdp.DEFAULT_OUTPUT_DIRNAME / "20260101_110000_order1.parquet"
            final_path = root / cdp.FINAL_OUTPUT_DIRNAME / "20260101_110000_order1.parquet"
            self.assertEqual(len(results), 1)
            self.assertTrue(default_path.exists())
            self.assertTrue(final_path.exists())
            final = cip.pd.read_parquet(final_path)

        self.assertEqual(final.loc[0, "schema_version"], cip.SCHEMA_VERSION)
        self.assertEqual(final.shape[0], 1)

    def test_dry_run_raw_session_reports_pipeline_would_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = root / "group"
            session = group / "session_20260101_110000"
            session.mkdir(parents=True)
            write_calibration(group / "multicam_calibration.npz")

            results = cdp.collect_dataset_parquets(root, dry_run=True)

        self.assertEqual(results[0]["process_status"], "would_run")
        self.assertEqual(results[0]["default_dest"].name, "20260101_110000_order1.parquet")

    def test_collect_runs_pipeline_when_default_parquet_is_missing(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = root / "group"
            session = group / "session_20260101_110000"
            session.mkdir(parents=True)
            write_minimal_default_parquet(session / "template_default.parquet")
            write_calibration(group / "multicam_calibration.npz")
            pipeline = root / "fake_pipeline.sh"
            pipeline.write_text(
                "#!/usr/bin/env bash\n"
                "set -euo pipefail\n"
                'cp "$1/template_default.parquet" "$1/session_ml_dataset.parquet"\n'
            )
            pipeline.chmod(0o755)

            results = cdp.collect_dataset_parquets(root, pipeline_script=pipeline)

            default_path = root / cdp.DEFAULT_OUTPUT_DIRNAME / "20260101_110000_order1.parquet"
            final_path = root / cdp.FINAL_OUTPUT_DIRNAME / "20260101_110000_order1.parquet"
            self.assertEqual(results[0]["process_status"], "ran")
            self.assertTrue((session / cdp.DEFAULT_PARQUET_NAME).exists())
            self.assertTrue(default_path.exists())
            self.assertTrue(final_path.exists())

    def test_duplicate_output_names_fail_without_group_prefix(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for group_name in ("group_a", "group_b"):
                session = root / group_name / "session_20260101_110000"
                session.mkdir(parents=True)

            sessions = cdp.discover_sessions(root)

        with self.assertRaises(ValueError):
            cdp.check_duplicate_outputs(sessions, include_group_name=False)

        cdp.check_duplicate_outputs(sessions, include_group_name=True)

    def test_require_four_validates_each_parent_group(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            group = root / "group"
            (group / "session_20260101_110000").mkdir(parents=True)
            sessions = cdp.discover_sessions(root)

        with self.assertRaises(ValueError):
            cdp.check_group_counts(sessions, require_four=True)


if __name__ == "__main__":
    unittest.main()
