import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import create_interaction_parquet as cip


class CreateInteractionParquetTests(unittest.TestCase):
    def test_rotate_points_matches_rviz_coordinate_convention(self):
        points = np.array([[1.0, 2.0, 3.0]])

        np.testing.assert_allclose(cip.rotate_points(points), [[1.0, 3.0, -2.0]])

    def test_build_dataset_reconstructs_world_person_and_dyad_relative_points(self):
        if cip.pd is None:
            self.skipTest("pandas is not installed")

        source = cip.pd.DataFrame([{
            "timestamp_ms": 0.0,
            "p1_root_valid": True,
            "p1_root_x": 1.0,
            "p1_root_y": 2.0,
            "p1_root_z": 3.0,
            "p1_root_source": 2,
            "p1_root_observed": True,
            "p1_root_interpolated": False,
            "p1_pose_cam_id": 2,
            "p1_gaze_cam_id": 4,
            "p1_kpt0_rel_x": 0.5,
            "p1_kpt0_rel_y": 0.0,
            "p1_kpt0_rel_z": 1.0,
            "p1_kpt0_score": 0.9,
            "p1_kpt0_observed": True,
            "p1_kpt0_interpolated": False,
            "p1_gaze_dir_x": 1.0,
            "p1_gaze_dir_y": 0.0,
            "p1_gaze_dir_z": 0.0,
            "p1_gaze_observed": True,
            "p1_gaze_interpolated": False,
            "p1_face_detected": True,
            "p2_root_valid": True,
            "p2_root_x": 3.0,
            "p2_root_y": 2.0,
            "p2_root_z": 3.0,
            "p2_root_source": 2,
            "p2_root_observed": True,
            "p2_root_interpolated": False,
            "p2_pose_cam_id": 1,
            "p2_gaze_cam_id": 3,
            "p2_kpt0_rel_x": -0.5,
            "p2_kpt0_rel_y": 0.0,
            "p2_kpt0_rel_z": 1.0,
            "p2_kpt0_score": 0.8,
            "p2_kpt0_observed": True,
            "p2_kpt0_interpolated": False,
            "p2_gaze_dir_x": -1.0,
            "p2_gaze_dir_y": 0.0,
            "p2_gaze_dir_z": 0.0,
            "p2_gaze_observed": True,
            "p2_gaze_interpolated": False,
            "p2_face_detected": True,
        }])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "session_ml_dataset.parquet"
            calib_path = tmp_path / "multicam_calibration.npz"
            source.to_parquet(source_path, engine="pyarrow", index=False)
            np.savez(
                calib_path,
                ref_camera=np.array(1),
                num_cameras=np.array(1),
                image_size=np.array([1280, 720]),
                K1=np.eye(3),
                dist1=np.zeros((1, 5)),
                R_1_to_ref=np.eye(3),
                t_1_to_ref=np.zeros(3),
            )

            out = cip.build_interaction_dataset(str(source_path), str(calib_path), session_dir=str(tmp_path))

        self.assertEqual(out.loc[0, "schema_version"], cip.SCHEMA_VERSION)
        np.testing.assert_allclose(
            out.loc[0, ["p1_ref_x", "p1_ref_y", "p1_ref_z"]].to_numpy(float),
            [1.0, 3.0, -2.0],
        )
        np.testing.assert_allclose(
            out.loc[0, ["p1_kpt0_world_x", "p1_kpt0_world_y", "p1_kpt0_world_z"]].to_numpy(float),
            [1.5, 4.0, -2.0],
        )
        np.testing.assert_allclose(
            out.loc[0, ["p1_kpt0_person_rel_x", "p1_kpt0_person_rel_y", "p1_kpt0_person_rel_z"]].to_numpy(float),
            [0.5, 1.0, 0.0],
        )
        np.testing.assert_allclose(
            out.loc[0, ["dyad_ref_x", "dyad_ref_y", "dyad_ref_z"]].to_numpy(float),
            [2.0, 3.0, -2.0],
        )
        np.testing.assert_allclose(
            out.loc[0, ["p1_kpt0_dyad_rel_x", "p1_kpt0_dyad_rel_y", "p1_kpt0_dyad_rel_z"]].to_numpy(float),
            [-0.5, 1.0, 0.0],
        )
        self.assertEqual(out.loc[0, "calib_sha256"].__class__, str)

    def test_missing_roots_produce_invalid_dyad_and_nan_relative_columns(self):
        if cip.pd is None:
            self.skipTest("pandas is not installed")

        source = cip.pd.DataFrame([{
            "timestamp_ms": 0.0,
            "p1_root_valid": True,
            "p1_root_x": 0.0,
            "p1_root_y": 0.0,
            "p1_root_z": 0.0,
            "p2_root_valid": False,
            "p2_root_x": np.nan,
            "p2_root_y": np.nan,
            "p2_root_z": np.nan,
            "p1_kpt0_rel_x": 1.0,
            "p1_kpt0_rel_y": 0.0,
            "p1_kpt0_rel_z": 0.0,
        }])

        with tempfile.TemporaryDirectory() as tmp:
            tmp_path = Path(tmp)
            source_path = tmp_path / "session_ml_dataset.parquet"
            calib_path = tmp_path / "multicam_calibration.npz"
            source.to_parquet(source_path, engine="pyarrow", index=False)
            np.savez(calib_path, ref_camera=np.array(1), num_cameras=np.array(0), image_size=np.array([1, 1]))

            out = cip.build_interaction_dataset(str(source_path), str(calib_path), session_dir=str(tmp_path))

        self.assertFalse(bool(out.loc[0, "dyad_ref_valid"]))
        self.assertTrue(np.isnan(out.loc[0, "p1_kpt0_dyad_rel_x"]))
        self.assertFalse(bool(out.loc[0, "frame_interaction_valid"]))

    def test_default_output_path_uses_session_directory(self):
        path = cip.default_output_path("/tmp/session_001/session_ml_dataset.parquet")

        self.assertEqual(path, "/tmp/session_001/session_interaction_dataset.parquet")


if __name__ == "__main__":
    unittest.main()
