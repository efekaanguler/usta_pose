#!/usr/bin/env python3
"""Run the preserved pairwise ChArUco extrinsic capture workflow."""

import runpy
import sys
from pathlib import Path


def main():
    calibration_dir = Path(__file__).resolve().parents[1]
    target = calibration_dir / "record_extrinsic.py"
    sys.path.insert(0, str(calibration_dir))
    sys.argv = [str(target), *sys.argv[1:], "--method", "charuco"]
    runpy.run_path(str(target), run_name="__main__")


if __name__ == "__main__":
    main()
