import csv
import tempfile
import unittest
from pathlib import Path

try:
    import numpy as np
    import pandas as pd
except ImportError as exc:
    raise unittest.SkipTest(f"runtime dependency missing: {exc.name}")

import resample_and_transform as rt


class ResampleAndTransformTests(unittest.TestCase):
    def test_calibration_npz_ref_to_cam_is_inverted(self):
        angle = np.deg2rad(90.0)
        r_ref_to_cam = np.array([
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ])
        t_ref_to_cam = np.array([1.0, 2.0, 3.0])
        p_ref = np.array([[0.25, -0.5, 2.0]])
        p_cam = p_ref @ r_ref_to_cam.T + t_ref_to_cam

        calib = {"R_2_to_ref": r_ref_to_cam, "t_2_to_ref": t_ref_to_cam}
        r_cam_to_ref, t_cam_to_ref = rt.load_camera_to_ref_transform(calib, 2)
        recovered = rt.transform_points(p_cam, r_cam_to_ref, t_cam_to_ref)

        np.testing.assert_allclose(recovered, p_ref, atol=1e-9)

    def test_short_series_interpolates_without_smoothing_crash(self):
        source_times = np.array([0.0, rt.TARGET_STEP_MS])
        target_times = np.array([0.0, rt.TARGET_STEP_MS / 2.0, rt.TARGET_STEP_MS])
        values, observed, interpolated = rt.interpolate_series(
            source_times,
            np.array([0.0, 2.0]),
            target_times,
            smooth=True,
        )

        np.testing.assert_allclose(values, [0.0, 1.0, 2.0])
        self.assertTrue(observed[0])
        self.assertTrue(observed[2])
        self.assertTrue(interpolated[1])

    def test_resolve_calib_path_handles_trailing_session_slash(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            session_dir = root / "session_001"
            session_dir.mkdir()
            calib_path = root / "multicam_calibration.npz"
            np.savez(calib_path, R_2_to_ref=np.eye(3), t_2_to_ref=np.zeros(3))

            resolved = rt.resolve_calib_path(str(session_dir) + "/", None)

            self.assertEqual(Path(resolved), calib_path)

    def test_mapping_hip_root_provenance_and_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            for cam_id in (1, 2, 3, 4):
                (session_dir / f"cam{cam_id}").mkdir()

            times = [0.0, rt.TARGET_STEP_MS, rt.TARGET_STEP_MS * 2]
            self._write_pose_csv(session_dir / "cam1" / "cam1_pose_raw.csv", times, root_start=100.0, right_hip_missing=True)
            self._write_pose_csv(session_dir / "cam2" / "cam2_pose_raw.csv", times, root_start=10.0, left_hip_missing_middle=True)
            self._write_gaze_csv(session_dir / "cam3" / "cam3_gaze_raw.csv", times, [1.0, 0.0, 0.0], yaw=0.3, pitch=0.1)
            self._write_gaze_csv(session_dir / "cam4" / "cam4_gaze_raw.csv", times, [0.0, 0.0, -1.0], yaw=0.4, pitch=0.2)

            calib_path = session_dir / "multicam_calibration.npz"
            np.savez(
                calib_path,
                R_2_to_ref=np.eye(3),
                t_2_to_ref=np.zeros(3),
                R_3_to_ref=np.eye(3),
                t_3_to_ref=np.zeros(3),
                R_4_to_ref=np.eye(3),
                t_4_to_ref=np.zeros(3),
            )

            df = rt.build_dataset(str(session_dir), str(calib_path))

            self.assertEqual(set(df["p1_pose_cam_id"].unique()), {2})
            self.assertEqual(set(df["p1_gaze_cam_id"].unique()), {4})
            self.assertEqual(set(df["p2_pose_cam_id"].unique()), {1})
            self.assertEqual(set(df["p2_gaze_cam_id"].unique()), {3})

            self.assertTrue(df.loc[0, "p1_root_valid"])
            self.assertAlmostEqual(df.loc[0, "p1_root_x"], 11.0)
            self.assertAlmostEqual(df.loc[0, "p1_kpt0_rel_x"], 3.0)
            self.assertAlmostEqual(df.loc[0, "p1_kpt0_rel_y"], 3.0)

            self.assertTrue(df.loc[1, "p1_root_valid"])
            self.assertFalse(df.loc[1, "p1_left_hip_observed"])
            self.assertTrue(df.loc[1, "p1_left_hip_interpolated"])
            self.assertAlmostEqual(df.loc[1, "p1_root_x"], 12.0)

            self.assertFalse(df.loc[0, "p2_root_valid"])
            self.assertTrue(np.isnan(df.loc[0, "p2_root_x"]))
            self.assertTrue(np.isnan(df.loc[0, "p2_kpt0_rel_x"]))

            np.testing.assert_allclose(df.loc[0, ["p1_gaze_dir_x", "p1_gaze_dir_y", "p1_gaze_dir_z"]].to_numpy(dtype=float), [0.0, 0.0, -1.0])
            np.testing.assert_allclose(df.loc[0, ["p2_gaze_dir_x", "p2_gaze_dir_y", "p2_gaze_dir_z"]].to_numpy(dtype=float), [1.0, 0.0, 0.0])

            expected_columns = [
                "timestamp_ms",
                "p1_root_valid",
                "p1_kpt0_score",
                "p1_kpt0_observed",
                "p1_kpt0_interpolated",
                "p1_face_detected",
                "p1_gaze_observed",
                "p1_gaze_interpolated",
                "p1_gaze_yaw",
                "p1_gaze_pitch",
            ]
            for column in expected_columns:
                self.assertIn(column, df.columns)

            try:
                import pyarrow  # noqa: F401
            except ImportError:
                self.skipTest("pyarrow is not installed")
            out_path = session_dir / "session_ml_dataset.parquet"
            df.to_parquet(out_path, engine="pyarrow", index=False)
            roundtrip = pd.read_parquet(out_path)
            self.assertIn("p2_gaze_cam_id", roundtrip.columns)

    def _write_pose_csv(self, path, times, root_start, right_hip_missing=False, left_hip_missing_middle=False):
        fieldnames = ["frame_idx", "hw_timestamp_ms"]
        for kpt_i in (0, 11, 12):
            fieldnames.extend([f"kpt{kpt_i}_x", f"kpt{kpt_i}_y", f"kpt{kpt_i}_z", f"kpt{kpt_i}_score"])

        rows = []
        for idx, timestamp in enumerate(times):
            left_x = root_start + idx
            right_x = root_start + idx + 2.0
            row = {"frame_idx": idx, "hw_timestamp_ms": timestamp}
            row.update(self._kpt_fields(0, root_start + idx + 4.0, 3.0, 0.0, 0.9))
            if left_hip_missing_middle and idx == 1:
                row.update(self._kpt_fields(11, "", "", "", 0.9))
            else:
                row.update(self._kpt_fields(11, left_x, 0.0, 0.0, 0.9))
            if right_hip_missing:
                row.update(self._kpt_fields(12, "", "", "", 0.9))
            else:
                row.update(self._kpt_fields(12, right_x, 0.0, 0.0, 0.9))
            rows.append(row)

        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def _kpt_fields(self, kpt_i, x, y, z, score):
        return {
            f"kpt{kpt_i}_x": x,
            f"kpt{kpt_i}_y": y,
            f"kpt{kpt_i}_z": z,
            f"kpt{kpt_i}_score": score,
        }

    def _write_gaze_csv(self, path, times, vector, yaw, pitch):
        fieldnames = ["frame_idx", "hw_timestamp_ms", "gaze_yaw", "gaze_pitch", "gaze_x", "gaze_y", "gaze_z", "face_detected"]
        rows = []
        for idx, timestamp in enumerate(times):
            rows.append({
                "frame_idx": idx,
                "hw_timestamp_ms": timestamp,
                "gaze_yaw": yaw,
                "gaze_pitch": pitch,
                "gaze_x": vector[0],
                "gaze_y": vector[1],
                "gaze_z": vector[2],
                "face_detected": 1,
            })
        with path.open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
