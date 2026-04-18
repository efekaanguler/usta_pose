#!/usr/bin/env python3
"""
Offline depth-to-color alignment for recorded sessions.

This script aligns raw depth.h5 files (recorded without live rs.align) using the
intrinsics/extrinsics saved in metadata.json.

Input:
    recordings/session_YYYYMMDD_HHMMSS/
        metadata.json
        camX/depth.h5

Output (per camera):
    camX/depth_aligned_to_color.h5
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Optional

import h5py
import numpy as np


def _load_metadata(session_dir: Path):
    metadata_path = session_dir / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"metadata.json not found: {metadata_path}")
    with open(metadata_path, "r", encoding="utf-8") as f:
        return json.load(f), metadata_path


def _get_intrinsics(cam_meta, key):
    calib = cam_meta.get("calibration", {})
    intr = calib.get(key)
    if intr is None and key == "color_intrinsics":
        # Backward compatibility with legacy metadata.
        intr = cam_meta.get("intrinsics")
    if intr is None:
        raise KeyError(f"Missing {key} in camera metadata")
    return {
        "fx": float(intr["fx"]),
        "fy": float(intr["fy"]),
        "ppx": float(intr["ppx"]),
        "ppy": float(intr["ppy"]),
        "width": int(intr["width"]),
        "height": int(intr["height"]),
    }


def _get_depth_to_color_extrinsics(cam_meta):
    calib = cam_meta.get("calibration", {})
    extr = calib.get("depth_to_color_extrinsics")
    if extr is None:
        raise KeyError("Missing depth_to_color_extrinsics in camera metadata")

    rot = np.array(extr["rotation"], dtype=np.float32).reshape(3, 3)
    trans = np.array(extr["translation"], dtype=np.float32).reshape(3)
    return rot, trans


def _h5_compression_options(compression: str, gzip_level: int):
    if compression == "none":
        return {}
    if compression == "lzf":
        return {"compression": "lzf", "shuffle": True}
    if compression == "gzip":
        return {"compression": "gzip", "compression_opts": gzip_level, "shuffle": True}
    raise ValueError(f"Unsupported compression: {compression}")


def _attr_to_str(value):
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _build_depth_ray_grid(depth_intr, depth_w, depth_h):
    # Use metadata dimensions when they match the dataset. Otherwise trust the
    # dataset shape while still using the same fx/fy/ppx/ppy values.
    if depth_intr["width"] != depth_w or depth_intr["height"] != depth_h:
        print(
            "  [warn] depth intrinsics size does not match dataset shape "
            f"({depth_intr['width']}x{depth_intr['height']} vs {depth_w}x{depth_h}). "
            "Proceeding with dataset shape."
        )

    u = np.arange(depth_w, dtype=np.float32)
    v = np.arange(depth_h, dtype=np.float32)
    uu, vv = np.meshgrid(u, v)

    x_norm = (uu - depth_intr["ppx"]) / depth_intr["fx"]
    y_norm = (vv - depth_intr["ppy"]) / depth_intr["fy"]
    return x_norm.astype(np.float32), y_norm.astype(np.float32)


def _align_single_frame(
    depth_z16,
    x_norm,
    y_norm,
    rot,
    trans,
    color_intr,
    depth_scale,
):
    color_h = color_intr["height"]
    color_w = color_intr["width"]
    out = np.zeros((color_h, color_w), dtype=np.uint16)

    z_m = depth_z16.astype(np.float32) * depth_scale
    valid = z_m > 0.0
    if not np.any(valid):
        return out, 0

    z = z_m[valid]
    x_d = x_norm[valid] * z
    y_d = y_norm[valid] * z

    # depth camera -> color camera transform
    x_c = rot[0, 0] * x_d + rot[0, 1] * y_d + rot[0, 2] * z + trans[0]
    y_c = rot[1, 0] * x_d + rot[1, 1] * y_d + rot[1, 2] * z + trans[1]
    z_c = rot[2, 0] * x_d + rot[2, 1] * y_d + rot[2, 2] * z + trans[2]

    valid_z = z_c > 1e-6
    if not np.any(valid_z):
        return out, 0

    x_c = x_c[valid_z]
    y_c = y_c[valid_z]
    z_c = z_c[valid_z]

    u = np.rint((x_c / z_c) * color_intr["fx"] + color_intr["ppx"]).astype(np.int32)
    v = np.rint((y_c / z_c) * color_intr["fy"] + color_intr["ppy"]).astype(np.int32)

    in_bounds = (u >= 0) & (u < color_w) & (v >= 0) & (v < color_h)
    if not np.any(in_bounds):
        return out, 0

    u = u[in_bounds]
    v = v[in_bounds]
    z_c = z_c[in_bounds]

    z16_aligned = np.rint(z_c / depth_scale).astype(np.int32)
    valid_range = (z16_aligned > 0) & (z16_aligned <= 65535)
    if not np.any(valid_range):
        return out, 0

    u = u[valid_range]
    v = v[valid_range]
    z16_aligned = z16_aligned[valid_range].astype(np.uint16)

    # Resolve collisions by keeping nearest depth (smallest z).
    flat_idx = v * color_w + u
    flat = np.full(color_h * color_w, np.iinfo(np.uint16).max, dtype=np.uint16)
    np.minimum.at(flat, flat_idx, z16_aligned)

    out = flat.reshape(color_h, color_w)
    out[out == np.iinfo(np.uint16).max] = 0
    return out, int(z16_aligned.size)


def _align_camera(
    session_dir: Path,
    cam_id: int,
    cam_meta: dict,
    output_name: str,
    compression: str,
    gzip_level: int,
    overwrite: bool,
    max_frames: int,
    depth_scale_override: Optional[float],
):
    cam_dir = session_dir / f"cam{cam_id}"
    src_path = cam_dir / "depth.h5"
    dst_path = cam_dir / output_name

    if not src_path.exists():
        print(f"[cam{cam_id}] skip: {src_path} not found")
        return {"status": "skipped", "reason": "missing_depth_h5"}

    if dst_path.exists() and not overwrite:
        print(f"[cam{cam_id}] skip: {dst_path.name} already exists (use --overwrite)")
        return {"status": "skipped", "reason": "already_exists", "output": str(dst_path)}

    with h5py.File(src_path, "r") as src_h5:
        if "depth" not in src_h5:
            raise KeyError(f"[cam{cam_id}] dataset 'depth' missing in {src_path}")
        src_ds = src_h5["depth"]

        if src_ds.ndim != 3:
            raise ValueError(f"[cam{cam_id}] expected depth dataset shape (N,H,W), got {src_ds.shape}")

        total_frames = int(src_ds.shape[0])
        depth_h = int(src_ds.shape[1])
        depth_w = int(src_ds.shape[2])
        frame_count = total_frames if max_frames is None else min(total_frames, max_frames)
        source_aligned_to = _attr_to_str(src_h5.attrs.get("aligned_to", "unknown")).lower()
        source_alignment_mode = _attr_to_str(src_h5.attrs.get("alignment_mode", "unknown"))

        src_scale = float(src_h5.attrs.get("depth_scale_meters_per_unit", 0.0))
        depth_scale = depth_scale_override if depth_scale_override is not None else src_scale
        if depth_scale <= 0:
            raise ValueError(
                f"[cam{cam_id}] invalid depth scale ({depth_scale}). "
                "Pass --depth-scale-override."
            )

        comp_opts = _h5_compression_options(compression, gzip_level)
        if dst_path.exists():
            dst_path.unlink()

        # Keep backward compatibility with legacy live-rs.align recordings:
        # if source depth is already aligned to color, do not reproject again.
        if source_aligned_to == "color":
            with h5py.File(dst_path, "w") as dst_h5:
                dst_ds = dst_h5.create_dataset(
                    "depth",
                    shape=(frame_count, depth_h, depth_w),
                    dtype=np.uint16,
                    chunks=(1, depth_h, depth_w),
                    **comp_opts,
                )
                dst_ds[:] = src_ds[:frame_count]

                dst_h5.attrs["unit"] = "z16_raw"
                dst_h5.attrs["dtype"] = "uint16"
                dst_h5.attrs["source_stream_format"] = "z16"
                dst_h5.attrs["depth_scale_meters_per_unit"] = depth_scale
                dst_h5.attrs["cam_idx"] = cam_id
                dst_h5.attrs["aligned_to"] = "color"
                dst_h5.attrs["alignment_mode"] = "copied_from_already_aligned_source"
                dst_h5.attrs["source_depth_file"] = "depth.h5"
                dst_h5.attrs["source_alignment_mode"] = source_alignment_mode
                dst_h5.attrs["compression"] = compression
                if compression == "gzip":
                    dst_h5.attrs["compression_opts"] = gzip_level
                dst_h5.attrs["generated_at"] = datetime.now().isoformat()
                dst_h5.attrs["valid_projected_points_total"] = -1

            print(
                f"[cam{cam_id}] source already aligned_to=color "
                f"(mode={source_alignment_mode}); copied {frame_count} frames."
            )
            return {
                "status": "copied",
                "output": str(dst_path),
                "frames": frame_count,
                "source_frames": total_frames,
                "depth_scale": depth_scale,
                "compression": compression,
                "source_alignment_mode": source_alignment_mode,
            }

        color_intr = _get_intrinsics(cam_meta, "color_intrinsics")
        depth_intr = _get_intrinsics(cam_meta, "depth_intrinsics")
        rot, trans = _get_depth_to_color_extrinsics(cam_meta)
        x_norm, y_norm = _build_depth_ray_grid(depth_intr, depth_w, depth_h)

        with h5py.File(dst_path, "w") as dst_h5:
            dst_ds = dst_h5.create_dataset(
                "depth",
                shape=(frame_count, color_intr["height"], color_intr["width"]),
                dtype=np.uint16,
                chunks=(1, color_intr["height"], color_intr["width"]),
                **comp_opts,
            )

            valid_projection_count = 0
            t0 = time.time()
            for i in range(frame_count):
                depth_frame = src_ds[i]
                aligned, valid_points = _align_single_frame(
                    depth_frame,
                    x_norm,
                    y_norm,
                    rot,
                    trans,
                    color_intr,
                    depth_scale,
                )
                dst_ds[i] = aligned
                valid_projection_count += valid_points

                if (i + 1) % 100 == 0 or i + 1 == frame_count:
                    elapsed = time.time() - t0
                    fps = (i + 1) / elapsed if elapsed > 0 else 0.0
                    print(
                        f"[cam{cam_id}] aligned {i + 1}/{frame_count} frames "
                        f"({fps:.1f} fps offline)"
                    )

            dst_h5.attrs["unit"] = "z16_raw"
            dst_h5.attrs["dtype"] = "uint16"
            dst_h5.attrs["source_stream_format"] = "z16"
            dst_h5.attrs["depth_scale_meters_per_unit"] = depth_scale
            dst_h5.attrs["cam_idx"] = cam_id
            dst_h5.attrs["aligned_to"] = "color"
            dst_h5.attrs["alignment_mode"] = "postprocess_cpu_numpy"
            dst_h5.attrs["source_depth_file"] = "depth.h5"
            dst_h5.attrs["source_aligned_to"] = source_aligned_to
            dst_h5.attrs["source_alignment_mode"] = source_alignment_mode
            dst_h5.attrs["compression"] = compression
            if compression == "gzip":
                dst_h5.attrs["compression_opts"] = gzip_level
            dst_h5.attrs["generated_at"] = datetime.now().isoformat()
            dst_h5.attrs["valid_projected_points_total"] = valid_projection_count

    return {
        "status": "ok",
        "output": str(dst_path),
        "frames": frame_count,
        "source_frames": total_frames,
        "depth_scale": depth_scale,
        "compression": compression,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Align raw recorded depth to color in post-process",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("session_dir", type=str, help="Session directory containing metadata.json")
    parser.add_argument(
        "--cams",
        type=int,
        nargs="+",
        default=[1, 2, 3, 4],
        help="Camera ids to process (1-based)",
    )
    parser.add_argument(
        "--output-name",
        type=str,
        default="depth_aligned_to_color.h5",
        help="Output file name under each cam directory",
    )
    parser.add_argument(
        "--compression",
        type=str,
        choices=["none", "lzf", "gzip"],
        default="lzf",
        help="Compression mode for aligned depth output",
    )
    parser.add_argument(
        "--gzip-level",
        type=int,
        default=1,
        help="Gzip level (0-9) if --compression=gzip",
    )
    parser.add_argument(
        "--max-frames",
        type=int,
        default=None,
        help="Limit frames per camera for quick tests",
    )
    parser.add_argument(
        "--depth-scale-override",
        type=float,
        default=None,
        help="Override depth_scale_meters_per_unit when input metadata is invalid",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing output files",
    )

    args = parser.parse_args()

    if not (0 <= args.gzip_level <= 9):
        raise ValueError("--gzip-level must be in [0, 9]")
    if args.max_frames is not None and args.max_frames <= 0:
        raise ValueError("--max-frames must be positive")

    session_dir = Path(args.session_dir).expanduser().resolve()
    metadata, metadata_path = _load_metadata(session_dir)
    cameras_meta = metadata.get("cameras", {})

    print(f"Session: {session_dir}")
    print(f"Metadata: {metadata_path}")
    print(f"Cameras requested: {args.cams}")
    print(f"Output: {args.output_name}")
    print(f"Compression: {args.compression}")

    summary = {
        "session_dir": str(session_dir),
        "generated_at": datetime.now().isoformat(),
        "output_name": args.output_name,
        "compression": args.compression,
        "gzip_level": args.gzip_level if args.compression == "gzip" else None,
        "cameras": {},
    }

    for cam_id in args.cams:
        cam_key = str(cam_id)
        if cam_key not in cameras_meta:
            print(f"[cam{cam_id}] skip: no metadata")
            summary["cameras"][cam_key] = {"status": "skipped", "reason": "missing_metadata"}
            continue

        try:
            result = _align_camera(
                session_dir=session_dir,
                cam_id=cam_id,
                cam_meta=cameras_meta[cam_key],
                output_name=args.output_name,
                compression=args.compression,
                gzip_level=args.gzip_level,
                overwrite=args.overwrite,
                max_frames=args.max_frames,
                depth_scale_override=args.depth_scale_override,
            )
            summary["cameras"][cam_key] = result
        except Exception as exc:
            print(f"[cam{cam_id}] error: {exc}")
            summary["cameras"][cam_key] = {"status": "error", "error": str(exc)}

    summary_path = session_dir / "depth_alignment_postprocess.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Summary written to: {summary_path}")


if __name__ == "__main__":
    main()
