import unittest

from devel.record.pcl_alignment_checker import (
    DirectionMetrics,
    PCLAlignmentChecker,
)


def direction(median_mm, p75_mm, inlier_ratio, compared_points=1000):
    return DirectionMetrics(
        source_points=2000,
        projected_points=1500,
        compared_points=compared_points,
        overlap_ratio=0.50,
        median_mm=median_mm,
        p75_mm=p75_mm,
        p90_mm=p75_mm,
        inlier_ratio=inlier_ratio,
    )


class PairGradingTest(unittest.TestCase):
    def setUp(self):
        self.checker = PCLAlignmentChecker.__new__(PCLAlignmentChecker)
        self.checker.min_compared_points = 120
        self.checker.min_overlap_ratio = 0.01
        self.checker.pass_median_mm = 35.0
        self.checker.pass_p75_mm = 65.0
        self.checker.pass_inlier_ratio = 0.35
        self.checker.warn_median_mm = 50.0
        self.checker.warn_p75_mm = 100.0
        self.checker.warn_inlier_ratio = 0.20

    def test_bidirectional_agreement_passes(self):
        status, _reason = self.checker._grade_pair(
            direction(8.0, 15.0, 0.95),
            direction(9.0, 16.0, 0.94),
        )
        self.assertEqual(status, "PASS")

    def test_visibility_asymmetry_warns_instead_of_failing(self):
        status, reason = self.checker._grade_pair(
            direction(57.0, 67.0, 0.15),
            direction(16.0, 20.0, 0.98),
        )
        self.assertEqual(status, "WARN")
        self.assertIn("visibility asymmetry", reason)

    def test_bidirectional_outlier_remains_blocking(self):
        status, _reason = self.checker._grade_pair(
            direction(70.0, 120.0, 0.10),
            direction(65.0, 110.0, 0.12),
        )
        self.assertEqual(status, "OUTLIER")

    def test_one_supported_direction_is_nonblocking(self):
        status, _reason = self.checker._grade_pair(
            direction(8.0, 15.0, 0.95, compared_points=20),
            direction(9.0, 16.0, 0.94),
        )
        self.assertEqual(status, "WARN")

    def test_any_bidirectional_outlier_blocks_overall_check(self):
        status = self.checker._grade_overall(
            failed_pairs=((1, 4),),
            warning_pairs=(),
            skipped_pairs=(),
            disconnected_cameras=(),
        )
        self.assertEqual(status, "FAIL")

    def test_redundant_low_support_pair_warns(self):
        status = self.checker._grade_overall(
            failed_pairs=(),
            warning_pairs=(),
            skipped_pairs=((1, 4),),
            disconnected_cameras=(),
        )
        self.assertEqual(status, "WARN")


if __name__ == "__main__":
    unittest.main()
