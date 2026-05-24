import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import revised_visualize_rviz as rv


class RevisedVisualizeRvizTests(unittest.TestCase):
    def row_with_root(self, person_id, root):
        prefix = f"p{person_id}"
        return {
            f"{prefix}_root_valid": True,
            f"{prefix}_root_x": root[0],
            f"{prefix}_root_y": root[1],
            f"{prefix}_root_z": root[2],
        }

    def set_rel(self, row, person_id, keypoint_idx, rel):
        prefix = f"p{person_id}_kpt{keypoint_idx}_rel"
        row[f"{prefix}_x"] = rel[0]
        row[f"{prefix}_y"] = rel[1]
        row[f"{prefix}_z"] = rel[2]

    def test_reconstructs_world_as_root_plus_relative_keypoint(self):
        row = self.row_with_root(1, [1.0, 2.0, 3.0])
        self.set_rel(row, 1, 0, [0.25, -0.5, 1.0])

        np.testing.assert_allclose(rv.reconstruct_keypoint(row, 1, 0), [1.25, 1.5, 4.0])

    def test_people_preserve_global_root_separation_without_recentering(self):
        row = self.row_with_root(1, [10.0, 0.0, 0.0])
        row.update(self.row_with_root(2, [11.0, 0.0, 0.0]))
        self.set_rel(row, 1, 0, [0.0, 0.0, 0.0])
        self.set_rel(row, 2, 0, [0.0, 0.0, 0.0])

        p1 = rv.reconstruct_keypoint(row, 1, 0)
        p2 = rv.reconstruct_keypoint(row, 2, 0)
        np.testing.assert_allclose(p2 - p1, [1.0, 0.0, 0.0])
        np.testing.assert_allclose(p1, [10.0, 0.0, 0.0])
        np.testing.assert_allclose(p2, [11.0, 0.0, 0.0])

    def test_touching_keypoints_reconstruct_to_same_world_coordinate(self):
        row = self.row_with_root(1, [0.0, 0.0, 0.0])
        row.update(self.row_with_root(2, [2.0, 0.0, 0.0]))
        self.set_rel(row, 1, 9, [1.0, 0.0, 0.0])
        self.set_rel(row, 2, 10, [-1.0, 0.0, 0.0])

        np.testing.assert_allclose(
            rv.reconstruct_keypoint(row, 1, 9),
            rv.reconstruct_keypoint(row, 2, 10),
        )

    def assert_point_close(self, point, expected):
        np.testing.assert_allclose([point.x, point.y, point.z], expected)

    def test_rviz_display_rotation_is_proper_90_degree_x_rotation(self):
        np.testing.assert_allclose(rv.rotate_for_rviz([1.0, 2.0, 3.0]), [1.0, 3.0, -2.0])
        self.assertAlmostEqual(float(np.linalg.det(rv.RVIZ_DISPLAY_ROTATION)), 1.0)

    def test_generated_joint_marker_points_are_rotated_for_rviz(self):
        node = rv.VisualizerNode.__new__(rv.VisualizerNode)
        node.fps = 30.0
        keypoints = [None] * rv.NUM_KEYPOINTS
        keypoints[0] = np.array([1.0, 2.0, 3.0])

        markers = node.create_person_markers(1, np.array([0.0, 0.0, 0.0]), keypoints, None)
        joint = next(marker for marker in markers if marker.ns == "person1_joints")

        self.assert_point_close(joint.points[0], [1.0, 3.0, -2.0])

    def test_bone_marker_data_z_separation_becomes_rviz_y_separation(self):
        node = rv.VisualizerNode.__new__(rv.VisualizerNode)
        node.fps = 30.0
        keypoints = [None] * rv.NUM_KEYPOINTS
        keypoints[0] = np.array([0.0, 0.0, 0.0])
        keypoints[1] = np.array([0.0, 0.0, 2.0])

        markers = node.create_person_markers(1, np.array([0.0, 0.0, 0.0]), keypoints, None)
        bones = next(marker for marker in markers if marker.ns == "person1_bones")

        self.assert_point_close(bones.points[0], [0.0, 0.0, 0.0])
        self.assert_point_close(bones.points[1], [0.0, 2.0, 0.0])

    def test_root_marker_position_is_rotated_for_rviz(self):
        node = rv.VisualizerNode.__new__(rv.VisualizerNode)
        node.fps = 30.0
        keypoints = [None] * rv.NUM_KEYPOINTS

        markers = node.create_person_markers(1, np.array([1.0, 2.0, 3.0]), keypoints, None)
        root = next(marker for marker in markers if marker.ns == "person1_root")

        self.assert_point_close(root.pose.position, [1.0, 3.0, -2.0])

    def test_gaze_marker_origin_and_end_are_rotated_for_rviz(self):
        node = rv.VisualizerNode.__new__(rv.VisualizerNode)
        node.fps = 30.0
        node.gaze_length = 2.0
        row = self.row_with_root(1, [1.0, 2.0, 3.0])
        row.update({
            "p1_gaze_dir_x": 0.0,
            "p1_gaze_dir_y": 0.0,
            "p1_gaze_dir_z": 4.0,
            "p1_gaze_observed": True,
        })
        keypoints = [None] * rv.NUM_KEYPOINTS

        marker = node.create_gaze_marker(row, 1, keypoints, None)

        self.assert_point_close(marker.points[0], [1.0, 3.0, -2.0])
        self.assert_point_close(marker.points[1], [1.0, 5.0, -2.0])

    def test_invalid_root_deletes_stale_person_markers(self):
        row = self.row_with_root(1, [1.0, 2.0, 3.0])
        row["p1_root_valid"] = False
        self.set_rel(row, 1, 0, [1.0, 0.0, 0.0])

        self.assertIsNone(rv.get_root(row, 1))
        self.assertIsNone(rv.reconstruct_keypoint(row, 1, 0))
        markers = rv.create_delete_markers(1)
        self.assertEqual({marker.ns for marker in markers}, {
            "person1_joints", "person1_bones", "person1_root", "person1_gaze"
        })
        self.assertTrue(all(marker.action == rv.Marker.DELETE for marker in markers))

    def test_gaze_origin_prefers_eye_midpoint_then_nose_then_root(self):
        row = self.row_with_root(1, [10.0, 0.0, 0.0])
        self.set_rel(row, 1, 0, [5.0, 0.0, 0.0])
        self.set_rel(row, 1, 1, [1.0, 0.0, 0.0])
        self.set_rel(row, 1, 2, [3.0, 0.0, 0.0])
        keypoints = rv.reconstruct_keypoints(row, 1, num_keypoints=3)
        np.testing.assert_allclose(rv.gaze_origin(row, 1, keypoints), [12.0, 0.0, 0.0])

        self.set_rel(row, 1, 1, [np.nan, np.nan, np.nan])
        keypoints = rv.reconstruct_keypoints(row, 1, num_keypoints=3)
        np.testing.assert_allclose(rv.gaze_origin(row, 1, keypoints), [15.0, 0.0, 0.0])

        self.set_rel(row, 1, 0, [np.nan, np.nan, np.nan])
        keypoints = rv.reconstruct_keypoints(row, 1, num_keypoints=3)
        np.testing.assert_allclose(rv.gaze_origin(row, 1, keypoints), [10.0, 0.0, 0.0])

    def test_gaze_provenance_columns_are_optional_but_respected(self):
        row = self.row_with_root(1, [0.0, 0.0, 0.0])
        row.update({"p1_gaze_dir_x": 10.0, "p1_gaze_dir_y": 0.0, "p1_gaze_dir_z": 0.0})
        np.testing.assert_allclose(rv.normalized_gaze_vector(row, 1), [1.0, 0.0, 0.0])

        row["p1_gaze_observed"] = False
        row["p1_gaze_interpolated"] = False
        self.assertIsNone(rv.normalized_gaze_vector(row, 1))

        row["p1_gaze_interpolated"] = True
        np.testing.assert_allclose(rv.normalized_gaze_vector(row, 1), [1.0, 0.0, 0.0])

    def test_parquet_roundtrip_loads_revised_schema_and_preserves_coordinates(self):
        if rv.pd is None:
            self.skipTest("pandas is not installed")

        rows = [{
            "timestamp_ms": 0.0,
            "p1_root_valid": True,
            "p1_root_x": 3.0,
            "p1_root_y": 0.0,
            "p1_root_z": 0.0,
            "p1_kpt0_rel_x": 0.25,
            "p1_kpt0_rel_y": 0.0,
            "p1_kpt0_rel_z": 0.0,
            "p1_kpt1_rel_x": 0.0,
            "p1_kpt1_rel_y": 0.1,
            "p1_kpt1_rel_z": 0.0,
            "p1_kpt2_rel_x": 0.0,
            "p1_kpt2_rel_y": -0.1,
            "p1_kpt2_rel_z": 0.0,
            "p1_gaze_dir_x": 2.0,
            "p1_gaze_dir_y": 0.0,
            "p1_gaze_dir_z": 0.0,
            "p1_gaze_observed": True,
            "p1_gaze_interpolated": False,
            "p2_root_valid": True,
            "p2_root_x": 4.0,
            "p2_root_y": 0.0,
            "p2_root_z": 0.0,
            "p2_kpt0_rel_x": -0.25,
            "p2_kpt0_rel_y": 0.0,
            "p2_kpt0_rel_z": 0.0,
        }]

        with tempfile.TemporaryDirectory() as tmp:
            session_dir = Path(tmp)
            parquet_path = session_dir / "session_ml_dataset.parquet"
            rv.pd.DataFrame(rows).to_parquet(parquet_path, engine="pyarrow", index=False)

            loaded = rv.pd.read_parquet(rv.resolve_parquet_path(session_dir=str(session_dir)))
            rv.validate_revised_schema(loaded)
            row = loaded.iloc[0].to_dict()
            np.testing.assert_allclose(rv.reconstruct_keypoint(row, 1, 0), [3.25, 0.0, 0.0])
            np.testing.assert_allclose(rv.reconstruct_keypoint(row, 2, 0), [3.75, 0.0, 0.0])
            np.testing.assert_allclose(
                rv.reconstruct_keypoint(row, 2, 0) - rv.reconstruct_keypoint(row, 1, 0),
                [0.5, 0.0, 0.0],
            )
            np.testing.assert_allclose(rv.gaze_origin(row, 1), [3.0, 0.0, 0.0])
            np.testing.assert_allclose(rv.normalized_gaze_vector(row, 1), [1.0, 0.0, 0.0])

    def test_visualizer_node_reads_parquet_when_ros_is_available(self):
        if rv.pd is None:
            self.skipTest("pandas is not installed")
        if rv.rclpy is None:
            self.skipTest("rclpy is not installed")

        with tempfile.TemporaryDirectory() as tmp:
            parquet_path = Path(tmp) / "session_ml_dataset.parquet"
            rv.pd.DataFrame([{
                "timestamp_ms": 0.0,
                "p1_root_valid": True,
                "p1_root_x": 1.0,
                "p1_root_y": 2.0,
                "p1_root_z": 3.0,
                "p1_kpt0_rel_x": 0.5,
                "p1_kpt0_rel_y": 0.0,
                "p1_kpt0_rel_z": 0.0,
                "p2_root_valid": False,
                "p2_root_x": np.nan,
                "p2_root_y": np.nan,
                "p2_root_z": np.nan,
            }]).to_parquet(parquet_path, engine="pyarrow", index=False)

            rv.rclpy.init(args=None)
            node = None
            try:
                node = rv.VisualizerNode(str(parquet_path), fps=30.0)
                self.assertEqual(len(node.rows), 1)
                np.testing.assert_allclose(rv.reconstruct_keypoint(node.rows[0], 1, 0), [1.5, 2.0, 3.0])
                node.timer_callback()
                self.assertEqual(node.frame_idx, 1)
            finally:
                if node is not None:
                    node.destroy_node()
                rv.rclpy.shutdown()

    def test_schema_rejects_old_camera_columns(self):
        with self.assertRaises(ValueError):
            rv.validate_revised_schema(SimpleNamespace(columns=["cam1_kpt0_x"]))


if __name__ == "__main__":
    unittest.main()
