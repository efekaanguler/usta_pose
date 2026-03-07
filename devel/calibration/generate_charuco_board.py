#!/usr/bin/env python3
"""
Generate ChArUco calibration board and save as image/PDF

This script generates a ChArUco board pattern that can be printed for camera calibration.

Usage:
    # Generate default 6x8 board
    python generate_charuco_board.py --output charuco_6x8.png

    # Generate A3-sized 8x11 board with custom measurements
    python generate_charuco_board.py --squares-x 8 --squares-y 11 \\
        --square-length 40 --marker-length 30 --paper-size A3 --output board.pdf

Tips for printing:
    - Print at 100% scale (do NOT scale to fit page)
    - Use matte paper to reduce glare
    - Mount on a rigid flat surface (foam board, plywood, etc.)
    - Measure actual printed square size to verify accuracy
"""

import argparse
import cv2
import numpy as np
from pathlib import Path


def mm_to_pixels(mm, dpi=300):
    """Convert millimeters to pixels at given DPI"""
    inches = mm / 25.4
    return int(inches * dpi)


def generate_charuco_board(squares_x, squares_y, square_length_mm, marker_length_mm,
                           aruco_dict_name, dpi=300):
    """
    Generate ChArUco board image

    Args:
        squares_x: Number of squares in X direction
        squares_y: Number of squares in Y direction
        square_length_mm: Square side length in millimeters
        marker_length_mm: Marker side length in millimeters
        aruco_dict_name: ArUco dictionary name
        dpi: Dots per inch for rendering

    Returns:
        board_image: Generated board image
    """
    # ArUco dictionary mapping
    aruco_dict_map = {
        '4X4_50': cv2.aruco.DICT_4X4_50,
        '4X4_100': cv2.aruco.DICT_4X4_100,
        '4X4_250': cv2.aruco.DICT_4X4_250,
        '4X4_1000': cv2.aruco.DICT_4X4_1000,
        '5X5_50': cv2.aruco.DICT_5X5_50,
        '5X5_100': cv2.aruco.DICT_5X5_100,
        '5X5_250': cv2.aruco.DICT_5X5_250,
        '5X5_1000': cv2.aruco.DICT_5X5_1000,
        '6X6_50': cv2.aruco.DICT_6X6_50,
        '6X6_100': cv2.aruco.DICT_6X6_100,
        '6X6_250': cv2.aruco.DICT_6X6_250,
        '6X6_1000': cv2.aruco.DICT_6X6_1000,
    }

    aruco_dict = cv2.aruco.getPredefinedDictionary(aruco_dict_map[aruco_dict_name])

    # Create ChArUco board
    board = cv2.aruco.CharucoBoard(
        (squares_x, squares_y),
        square_length_mm / 1000.0,  # Convert to meters for board definition
        marker_length_mm / 1000.0,
        aruco_dict
    )

    # Calculate image size in pixels
    square_size_pixels = mm_to_pixels(square_length_mm, dpi)
    img_width = squares_x * square_size_pixels
    img_height = squares_y * square_size_pixels

    # Generate board image
    board_image = board.generateImage((img_width, img_height), marginSize=0, borderBits=1)

    return board_image


def add_measurement_guides(image, squares_x, squares_y, square_length_mm, dpi=300):
    """Add measurement guides and information to the board image"""
    # Add white border for annotations
    border = mm_to_pixels(20, dpi)  # 20mm border
    h, w = image.shape
    bordered = np.ones((h + 2*border, w + 2*border), dtype=np.uint8) * 255
    bordered[border:border+h, border:border+w] = image

    # Convert to BGR for colored annotations
    bordered_bgr = cv2.cvtColor(bordered, cv2.COLOR_GRAY2BGR)

    # Add measurement markers
    square_pixels = mm_to_pixels(square_length_mm, dpi)

    # Horizontal measurement at top
    y_pos = border // 2
    x_start = border
    x_end = border + square_pixels
    cv2.line(bordered_bgr, (x_start, y_pos), (x_end, y_pos), (0, 0, 255), 2)
    cv2.line(bordered_bgr, (x_start, y_pos-10), (x_start, y_pos+10), (0, 0, 255), 2)
    cv2.line(bordered_bgr, (x_end, y_pos-10), (x_end, y_pos+10), (0, 0, 255), 2)
    text = f"{square_length_mm}mm"
    cv2.putText(bordered_bgr, text, (x_start + square_pixels//2 - 30, y_pos - 15),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)

    # Vertical measurement at left
    x_pos = border // 2
    y_start = border
    y_end = border + square_pixels
    cv2.line(bordered_bgr, (x_pos, y_start), (x_pos, y_end), (0, 0, 255), 2)
    cv2.line(bordered_bgr, (x_pos-10, y_start), (x_pos+10, y_start), (0, 0, 255), 2)
    cv2.line(bordered_bgr, (x_pos-10, y_end), (x_pos+10, y_end), (0, 0, 255), 2)

    # Add text information at bottom
    info_y = h + border + 30
    font_scale = 0.7
    cv2.putText(bordered_bgr, f"ChArUco Board: {squares_x} x {squares_y} squares",
                (border, info_y), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    cv2.putText(bordered_bgr, f"Square size: {square_length_mm}mm",
                (border, info_y + 30), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (0, 0, 0), 2)
    cv2.putText(bordered_bgr, f"PRINT AT 100% - DO NOT SCALE",
                (border, info_y + 60), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 0, 0), 2)

    return bordered_bgr


def get_paper_size_mm(paper_size):
    """Get paper dimensions in millimeters"""
    paper_sizes = {
        'A4': (210, 297),
        'A3': (297, 420),
        'A2': (420, 594),
        'A1': (594, 841),
        'Letter': (216, 279),
        'Tabloid': (279, 432),
    }
    return paper_sizes.get(paper_size, (210, 297))


def calculate_optimal_square_size(squares_x, squares_y, paper_size, margin_mm=20):
    """Calculate optimal square size to fit on paper with margin"""
    paper_width, paper_height = get_paper_size_mm(paper_size)

    # Account for margins and measurement guides
    usable_width = paper_width - 2 * margin_mm
    usable_height = paper_height - 2 * margin_mm

    # Calculate maximum square size that fits
    max_square_from_width = usable_width / squares_x
    max_square_from_height = usable_height / squares_y

    optimal_square = min(max_square_from_width, max_square_from_height)

    return int(optimal_square)


def main():
    parser = argparse.ArgumentParser(
        description="Generate ChArUco calibration board",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    # Board parameters
    parser.add_argument('--squares-x', type=int, default=6,
                       help='Number of squares in X direction (columns)')
    parser.add_argument('--squares-y', type=int, default=8,
                       help='Number of squares in Y direction (rows)')
    parser.add_argument('--square-length', type=float, default=None,
                       help='Square side length in millimeters (auto-calculated if not specified)')
    parser.add_argument('--marker-length', type=float, default=None,
                       help='Marker side length in millimeters (auto: 75% of square size)')
    parser.add_argument('--aruco-dict', type=str, default='4X4_50',
                       choices=['4X4_50', '4X4_100', '4X4_250', '4X4_1000',
                               '5X5_50', '5X5_100', '5X5_250', '5X5_1000',
                               '6X6_50', '6X6_100', '6X6_250', '6X6_1000'],
                       help='ArUco dictionary to use')

    # Paper and output settings
    parser.add_argument('--paper-size', type=str, default='A4',
                       choices=['A4', 'A3', 'A2', 'A1', 'Letter', 'Tabloid'],
                       help='Target paper size for printing')
    parser.add_argument('--dpi', type=int, default=300,
                       help='Output resolution in DPI (300 recommended for printing)')
    parser.add_argument('--output', type=str, required=True,
                       help='Output file path (PNG or PDF)')
    parser.add_argument('--no-guides', action='store_true',
                       help='Disable measurement guides and annotations')

    args = parser.parse_args()

    # Calculate optimal square size if not specified
    if args.square_length is None:
        args.square_length = calculate_optimal_square_size(
            args.squares_x, args.squares_y, args.paper_size
        )
        print(f"Auto-calculated square size: {args.square_length}mm for {args.paper_size} paper")

    # Calculate marker size if not specified (75% of square size)
    if args.marker_length is None:
        args.marker_length = args.square_length * 0.75
        print(f"Auto-calculated marker size: {args.marker_length:.1f}mm")

    # Validate marker size
    if args.marker_length >= args.square_length:
        raise ValueError("Marker length must be smaller than square length")

    # Generate board
    print(f"\nGenerating ChArUco board:")
    print(f"  Grid: {args.squares_x} × {args.squares_y}")
    print(f"  Square size: {args.square_length}mm")
    print(f"  Marker size: {args.marker_length}mm")
    print(f"  Dictionary: {args.aruco_dict}")
    print(f"  Board dimensions: {args.squares_x * args.square_length:.1f} × "
          f"{args.squares_y * args.square_length:.1f} mm")
    print(f"  ChArUco corners: {(args.squares_x-1) * (args.squares_y-1)}")

    board_image = generate_charuco_board(
        args.squares_x, args.squares_y,
        args.square_length, args.marker_length,
        args.aruco_dict, args.dpi
    )

    # Add measurement guides unless disabled
    if not args.no_guides:
        output_image = add_measurement_guides(
            board_image, args.squares_x, args.squares_y,
            args.square_length, args.dpi
        )
    else:
        output_image = cv2.cvtColor(board_image, cv2.COLOR_GRAY2BGR)

    # Save output
    output_path = Path(args.output)
    cv2.imwrite(str(output_path), output_image)

    print(f"\n✓ Board saved to: {output_path}")
    print(f"\nPrinting instructions:")
    print(f"  1. Open {output_path} in a PDF viewer or image editor")
    print(f"  2. Print at 100% scale (DISABLE 'Fit to page' or 'Scale to fit')")
    print(f"  3. Verify printed square size with a ruler (should be {args.square_length}mm)")
    print(f"  4. Mount on rigid flat surface (foam board, wood, etc.)")
    print(f"  5. Use matte finish to reduce glare during calibration")

    # Create parameter file for easy reference
    param_file = output_path.with_suffix('.txt')
    with open(param_file, 'w') as f:
        f.write("ChArUco Board Parameters\n")
        f.write("========================\n\n")
        f.write(f"Use these parameters for calibration scripts:\n\n")
        f.write(f"--squares-x {args.squares_x}\n")
        f.write(f"--squares-y {args.squares_y}\n")
        f.write(f"--square-length {args.square_length / 1000.0:.6f}  # meters\n")
        f.write(f"--marker-length {args.marker_length / 1000.0:.6f}  # meters\n")
        f.write(f"--aruco-dict {args.aruco_dict}\n")

    print(f"\n✓ Parameters saved to: {param_file}")


if __name__ == '__main__':
    main()
