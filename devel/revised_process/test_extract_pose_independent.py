import unittest

try:
    import numpy as np
except ImportError as exc:
    raise unittest.SkipTest(f"runtime dependency missing: {exc.name}")

import extract_pose_independent as epi


class ExtractPoseIndependentTests(unittest.TestCase):
    def test_depth_to_color_alignment_identity_keeps_depth_pixels(self):
        intr = {
            "fx": 1.0,
            "fy": 1.0,
            "ppx": 0.0,
            "ppy": 0.0,
            "width": 2,
            "height": 2,
        }
        depth = np.array([[1000, 2000], [0, 1500]], dtype=np.uint16)
        x_norm, y_norm = epi.build_depth_ray_grid(intr, depth.shape)

        aligned = epi.align_depth_to_color(
            depth,
            x_norm,
            y_norm,
            np.eye(3, dtype=np.float32),
            np.zeros(3, dtype=np.float32),
            intr,
            0.001,
        )

        np.testing.assert_array_equal(aligned, depth)


    def test_cam1_default_bbox_excludes_rightmost_fifth(self):
        self.assertEqual(epi.default_pose_bbox(1, 1280, 720), [0.0, 0.0, 1023.0, 719.0])
        self.assertEqual(epi.default_pose_bbox(2, 1280, 720), [0.0, 0.0, 1279.0, 719.0])


    def test_default_pose_model_paths_use_valid_model_dirs(self):
        cfg, ckpt = epi.default_model_paths('/repo', epi.POSE_MODEL_RTMPOSE2D)
        self.assertIn('/repo/models/pose/rtmw2d/', cfg)
        self.assertIn('/repo/models/pose/rtmw2d/', ckpt)
        self.assertIn('rtmpose-l_', cfg)

        cfg, ckpt = epi.default_model_paths('/repo', epi.POSE_MODEL_RTMW3D)
        self.assertIn('/repo/models/pose/rtmw3d/', cfg)
        self.assertIn('/repo/models/pose/rtmw3d/', ckpt)
        self.assertIn('rtmw3d-l_', cfg)

    def test_cam1_foreground_rejection_uses_depth_and_area(self):
        close_stats = {
            "body_valid_keypoints": 8,
            "body_median_depth_m": 0.5,
            "body_bbox_area_px": 10.0,
        }
        self.assertEqual(
            epi.rejection_reason(1, close_stats, 1280, 720, 0.7, 0.32),
            "cam1_foreground_depth",
        )

        large_stats = {
            "body_valid_keypoints": 8,
            "body_median_depth_m": 1.2,
            "body_bbox_area_px": 400000.0,
        }
        self.assertEqual(
            epi.rejection_reason(1, large_stats, 1280, 720, 0.7, 0.32),
            "cam1_foreground_bbox",
        )
        self.assertEqual(epi.rejection_reason(2, large_stats, 1280, 720, 0.7, 0.32), "")

    def test_depth_projector_uses_color_intrinsics_for_aligned_depth(self):
        cam_meta = {
            "intrinsics": {
                "fx": 10.0,
                "fy": 11.0,
                "ppx": 1.0,
                "ppy": 2.0,
                "width": 4,
                "height": 3,
            },
            "depth_storage": {
                "aligned_to": "color",
                "depth_scale_meters_per_unit": 0.001,
            },
        }

        projector = epi.DepthProjector(cam_meta)

        self.assertFalse(projector.needs_alignment)
        np.testing.assert_allclose(projector.K, [[10.0, 0.0, 1.0], [0.0, 11.0, 2.0], [0.0, 0.0, 1.0]])


if __name__ == "__main__":
    unittest.main()
