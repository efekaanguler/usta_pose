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
