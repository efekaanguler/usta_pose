#!/usr/bin/env python3
"""
Collect per-session parquet files into dataset-level output folders.

Given a dataset root containing many recording/group folders, this script finds
session_YYYYMMDD_HHMMSS directories, assigns order within each direct parent by
timestamp, copies the default session_ml_dataset.parquet files, and creates the
derived RViz-aligned interaction parquets.
"""

import argparse
import os
import re
import shutil
import subprocess
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from create_interaction_parquet import (
    OUTPUT_BASENAME as FINAL_PARQUET_NAME,
    write_interaction_dataset,
)
from revised_visualize_rviz import resolve_calib_path


SESSION_RE = re.compile(r"^session_(\d{8}_\d{6})$")
DEFAULT_PARQUET_NAME = "session_ml_dataset.parquet"
DEFAULT_OUTPUT_DIRNAME = "default_parquets"
FINAL_OUTPUT_DIRNAME = "final_dataset_parquets"
DEFAULT_PIPELINE_SCRIPT = Path(__file__).resolve().parents[1] / "revised_process" / "run_revised_pipeline.sh"


@dataclass(frozen=True)
class SessionInfo:
    session_dir: Path
    group_dir: Path
    session_stamp: str
    order: int


def parse_session_stamp(path):
    match = SESSION_RE.match(path.name)
    return match.group(1) if match else None


def discover_sessions(dataset_root):
    dataset_root = Path(dataset_root).expanduser().resolve()
    grouped = defaultdict(list)

    for path in dataset_root.rglob("session_*"):
        if not path.is_dir():
            continue
        stamp = parse_session_stamp(path)
        if stamp is None:
            continue
        grouped[path.parent.resolve()].append((stamp, path.resolve()))

    sessions = []
    for group_dir in sorted(grouped):
        ordered = sorted(grouped[group_dir], key=lambda item: item[0])
        for order, (stamp, session_dir) in enumerate(ordered, start=1):
            sessions.append(SessionInfo(session_dir, group_dir, stamp, order))

    return sessions


def output_filename(session_info, include_group_name=False):
    filename = f"{session_info.session_stamp}_order{session_info.order}.parquet"
    if not include_group_name:
        return filename
    safe_group = re.sub(r"[^0-9A-Za-z_.-]+", "_", session_info.group_dir.name).strip("_")
    return f"{safe_group}_{filename}" if safe_group else filename


def ensure_can_write(path, overwrite):
    if path.exists() and not overwrite:
        raise FileExistsError(f"Output already exists, use --overwrite to replace it: {path}")


def check_duplicate_outputs(sessions, include_group_name):
    seen = {}
    for session_info in sessions:
        name = output_filename(session_info, include_group_name=include_group_name)
        if name in seen:
            first = seen[name]
            raise ValueError(
                "Output filename collision. Use --include-group-name or rename sessions: "
                f"{first.session_dir} and {session_info.session_dir} both map to {name}"
            )
        seen[name] = session_info


def check_group_counts(sessions, require_four):
    if not require_four:
        return

    counts = defaultdict(int)
    for session_info in sessions:
        counts[session_info.group_dir] += 1

    invalid = {group: count for group, count in counts.items() if count != 4}
    if invalid:
        details = ", ".join(f"{group} has {count}" for group, count in sorted(invalid.items()))
        raise ValueError(f"Expected exactly 4 sessions in each group, but {details}")


def parse_env_assignments(assignments):
    env = {}
    for assignment in assignments or []:
        if "=" not in assignment:
            raise ValueError(f"Invalid --pipeline-env value, expected KEY=VALUE: {assignment}")
        key, value = assignment.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid --pipeline-env value, empty key: {assignment}")
        env[key] = value
    return env


def run_processing_stage(
    session_info,
    *,
    process=True,
    force_reprocess=False,
    dry_run=False,
    pipeline_script=None,
    pipeline_env=None,
    pose_model="rtmpose2d",
):
    default_source = session_info.session_dir / DEFAULT_PARQUET_NAME
    if not process:
        return "disabled"
    if default_source.exists() and not force_reprocess:
        return "skipped_existing"

    pipeline_script = Path(pipeline_script or DEFAULT_PIPELINE_SCRIPT).expanduser().resolve()
    if not pipeline_script.exists():
        raise FileNotFoundError(f"Pipeline script not found: {pipeline_script}")

    if dry_run:
        return "would_run"

    env = os.environ.copy()
    env["POSE_MODEL"] = pose_model
    env.update(pipeline_env or {})

    subprocess.run(
        ["bash", str(pipeline_script), str(session_info.session_dir)],
        check=True,
        env=env,
    )

    if not default_source.exists():
        raise FileNotFoundError(
            f"Pipeline completed but default parquet was not created: {default_source}"
        )
    return "ran"


def collect_session(
    session_info,
    default_output_dir,
    final_output_dir,
    *,
    overwrite=False,
    include_group_name=False,
    dry_run=False,
    allow_missing_default=False,
    process_status="disabled",
    pose_model="rtmpose2d",
    gaze_model="puregaze",
):
    default_source = session_info.session_dir / DEFAULT_PARQUET_NAME
    if not default_source.exists() and not allow_missing_default:
        raise FileNotFoundError(f"Missing default parquet: {default_source}")

    output_name = output_filename(session_info, include_group_name=include_group_name)
    default_dest = default_output_dir / output_name
    final_dest = final_output_dir / output_name

    ensure_can_write(default_dest, overwrite)
    ensure_can_write(final_dest, overwrite)

    calib_path = resolve_calib_path(
        session_dir=str(session_info.session_dir),
        parquet_path=str(default_source),
    )
    if calib_path is None:
        raise FileNotFoundError(f"Calibration NPZ not found for session: {session_info.session_dir}")

    if dry_run:
        return {
            "session": session_info,
            "default_source": default_source,
            "default_dest": default_dest,
            "final_dest": final_dest,
            "calib_path": Path(calib_path),
            "process_status": process_status,
            "rows": None,
            "final_columns": None,
        }

    default_output_dir.mkdir(parents=True, exist_ok=True)
    final_output_dir.mkdir(parents=True, exist_ok=True)

    shutil.copy2(default_source, default_dest)
    _, final_df = write_interaction_dataset(
        str(default_source),
        str(final_dest),
        calib_path,
        session_dir=str(session_info.session_dir),
        pose_model=pose_model,
        gaze_model=gaze_model,
    )

    return {
        "session": session_info,
        "default_source": default_source,
        "default_dest": default_dest,
        "final_dest": final_dest,
        "calib_path": Path(calib_path),
        "process_status": process_status,
        "rows": len(final_df),
        "final_columns": len(final_df.columns),
    }


def check_output_paths(sessions, default_output_dir, final_output_dir, overwrite, include_group_name):
    for session_info in sessions:
        output_name = output_filename(session_info, include_group_name=include_group_name)
        ensure_can_write(default_output_dir / output_name, overwrite)
        ensure_can_write(final_output_dir / output_name, overwrite)


def collect_dataset_parquets(
    dataset_root,
    *,
    default_output_dir=None,
    final_output_dir=None,
    overwrite=False,
    include_group_name=False,
    require_four=False,
    dry_run=False,
    process=True,
    force_reprocess=False,
    pipeline_script=None,
    pipeline_env=None,
    pose_model="rtmpose2d",
    gaze_model="puregaze",
):
    dataset_root = Path(dataset_root).expanduser().resolve()
    if not dataset_root.exists():
        raise FileNotFoundError(f"Dataset root not found: {dataset_root}")
    if not dataset_root.is_dir():
        raise NotADirectoryError(f"Dataset root is not a directory: {dataset_root}")

    default_output_dir = Path(default_output_dir).expanduser().resolve() if default_output_dir else dataset_root / DEFAULT_OUTPUT_DIRNAME
    final_output_dir = Path(final_output_dir).expanduser().resolve() if final_output_dir else dataset_root / FINAL_OUTPUT_DIRNAME

    sessions = discover_sessions(dataset_root)
    if not sessions:
        raise FileNotFoundError(f"No session_YYYYMMDD_HHMMSS directories found under: {dataset_root}")

    check_group_counts(sessions, require_four)
    check_duplicate_outputs(sessions, include_group_name)
    check_output_paths(sessions, default_output_dir, final_output_dir, overwrite, include_group_name)

    results = []
    for session_info in sessions:
        process_status = run_processing_stage(
            session_info,
            process=process,
            force_reprocess=force_reprocess,
            dry_run=dry_run,
            pipeline_script=pipeline_script,
            pipeline_env=pipeline_env,
            pose_model=pose_model,
        )
        results.append(
            collect_session(
                session_info,
                default_output_dir,
                final_output_dir,
                overwrite=overwrite,
                include_group_name=include_group_name,
                dry_run=dry_run,
                allow_missing_default=dry_run and process_status in {"would_run"},
                process_status=process_status,
                pose_model=pose_model,
                gaze_model=gaze_model,
            )
        )
    return results


def print_results(results, dry_run=False):
    action = "Would collect" if dry_run else "Collected"
    print(f"{action} {len(results)} session(s).")
    for result in results:
        session_info = result["session"]
        rel_group = session_info.group_dir.name
        rows = result["rows"]
        shape_text = "" if rows is None else f" rows={rows} cols={result['final_columns']}"
        process_text = f" process={result.get('process_status', 'unknown')}"
        print(
            f"order{session_info.order} {session_info.session_stamp} "
            f"group={rel_group} -> {result['default_dest'].name}{process_text}{shape_text}"
        )


def main():
    parser = argparse.ArgumentParser(
        description="Collect default and interaction parquets from all session_YYYYMMDD_HHMMSS folders under a dataset root."
    )
    parser.add_argument("dataset_root", type=str)
    parser.add_argument("--default-output-dir", type=str, default=None)
    parser.add_argument("--final-output-dir", type=str, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--include-group-name",
        action="store_true",
        help="Prefix output names with the parent group folder to avoid collisions across groups.",
    )
    parser.add_argument(
        "--require-four",
        action="store_true",
        help="Fail if any direct parent group does not contain exactly four sessions.",
    )
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument(
        "--skip-processing",
        action="store_true",
        help="Do not run revised_process; require existing session_ml_dataset.parquet files.",
    )
    parser.add_argument(
        "--force-reprocess",
        action="store_true",
        help="Run revised_process for every session even when session_ml_dataset.parquet already exists.",
    )
    parser.add_argument(
        "--pipeline-script",
        type=str,
        default=None,
        help="Path to run_revised_pipeline.sh. Defaults to ../revised_process/run_revised_pipeline.sh.",
    )
    parser.add_argument(
        "--pipeline-env",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Extra environment assignment passed to the revised pipeline. May be repeated.",
    )
    parser.add_argument("--pose-model", type=str, default="rtmpose2d")
    parser.add_argument("--gaze-model", type=str, default="puregaze")
    args = parser.parse_args()

    try:
        results = collect_dataset_parquets(
            args.dataset_root,
            default_output_dir=args.default_output_dir,
            final_output_dir=args.final_output_dir,
            overwrite=args.overwrite,
            include_group_name=args.include_group_name,
            require_four=args.require_four,
            dry_run=args.dry_run,
            process=not args.skip_processing,
            force_reprocess=args.force_reprocess,
            pipeline_script=args.pipeline_script,
            pipeline_env=parse_env_assignments(args.pipeline_env),
            pose_model=args.pose_model,
            gaze_model=args.gaze_model,
        )
    except Exception as exc:
        parser.error(str(exc))
        return 2

    print_results(results, dry_run=args.dry_run)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
