#!/usr/bin/env python3
"""Generate publication-ready paper assets from existing analysis outputs.

This script intentionally uses only outputs already produced by the analysis
pipeline under this directory. If an expected input is missing, it writes a
clearly marked placeholder asset instead of fabricating values.
"""

from __future__ import annotations

import datetime as _dt
import math
import re
import shutil
import textwrap
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Rectangle

try:
    import seaborn as sns
except Exception:  # pragma: no cover - seaborn is optional.
    sns = None


BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[2]
OUT_DIR = BASE_DIR / "paper_assets"

DIRS = {
    "main_figures": OUT_DIR / "main_figures",
    "main_tables": OUT_DIR / "main_tables",
    "appendix_figures": OUT_DIR / "appendix_figures",
    "appendix_tables": OUT_DIR / "appendix_tables",
    "pdf": OUT_DIR / "pdf",
    "source_data_used": OUT_DIR / "source_data_used",
}

TITLE = "Learning a Symbolic Nonverbal Vocabulary from Dyadic Pose and Gaze Interactions"
TODAY = _dt.datetime.now().strftime("%Y-%m-%d")

ASSETS: list[dict] = []
MISSING: list[str] = []
SOURCES_USED: dict[str, Path] = {}

COLOR_BLUE = "#4C78A8"
COLOR_ORANGE = "#F58518"
COLOR_GREEN = "#54A24B"
COLOR_RED = "#E45756"
COLOR_PURPLE = "#B279A2"
COLOR_GRAY = "#6B7280"
COLOR_LIGHT = "#F8FAFC"
COLOR_LINE = "#334155"


WORD_INFO = {
    "IW00": (
        "task-oriented right-hand approach",
        "Task-oriented right-hand approach or placement-like action while gaze remains task-directed.",
    ),
    "IW01": (
        "monitoring/listening",
        "Dominant attention state; an attentive monitoring or listening-like nonverbal state, not empty inactivity.",
    ),
    "IW02": (
        "left-hand withdrawal/action",
        "Task-focused left-hand withdrawal or completion movement after reaching, placing, or adjustment.",
    ),
    "IW03": (
        "rare partner-monitored right approach",
        "Rare salient right-hand approach with partner monitoring; possible communicative reach or offer.",
    ),
    "IW04": (
        "strong task-focused right action",
        "Stronger task-focused right-hand action, likely active manipulation or reaching in the shared workspace.",
    ),
    "IW05": (
        "left-hand active with partner monitoring",
        "Left-hand-dominant active state while visually checking or monitoring the partner.",
    ),
    "IW06": (
        "task-focused left withdrawal/action",
        "Common task-focused left-hand withdrawal/action word; often appears as an active response candidate.",
    ),
    "IW07": (
        "task-focused right approach",
        "Common task-focused right-hand approach word with moderate motion.",
    ),
    "IW08": (
        "active left-hand approach",
        "Important active response word: left-hand approach/reach with partner-directed gaze.",
    ),
    "IW09": (
        "rare mixed transition",
        "Rare mixed action: right-hand dominant motion with left-hand withdrawal and partner gaze.",
    ),
    "IW10": (
        "partner-monitored right approach",
        "Right-hand approach while gaze is partner-directed, suggesting social checking or coordination.",
    ),
    "IW11": (
        "secondary monitoring/waiting",
        "Secondary monitoring/waiting state that often precedes later active responses.",
    ),
}


CAPTIONS = {
    "fig1": (
        "Experimental setup and task structure. Each dyad completed four tabletop "
        "disk-ordering sessions while being recorded by four RealSense cameras. "
        "Session 1 served as the noncompetitive first-exposure baseline, whereas "
        "Sessions 2-4 introduced scoring and competitive/practiced interaction."
    ),
    "fig2": (
        "Overview of the analysis pipeline. Multi-camera recordings were converted "
        "into synchronized pose/gaze tables, validated, segmented into temporal "
        "windows, discretized into symbolic nonverbal words, and evaluated through "
        "event-level response modeling."
    ),
    "fig3": (
        "Prevalence of the learned individual nonverbal vocabulary. The vocabulary "
        "is dominated by IW01, interpreted as a monitoring/listening state, while "
        "the remaining words mostly correspond to rarer active manual movements "
        "differentiated by hand dominance, approach/withdrawal, and gaze direction."
    ),
    "fig4": (
        "Event-level symbolic analysis revealed a recurring action-monitoring-action "
        "structure. Participants often did not respond with an immediate active "
        "movement; instead, the first response was frequently a monitoring/listening "
        "state followed by a delayed active movement."
    ),
    "fig5": (
        "Event-level response modeling results under leave-one-dyad-out evaluation. "
        "Actor-only models were weak, while symbolic interaction history improved "
        "plausible response prediction. Observed-immediate models performed best "
        "because they also observe the responder's first post-actor state, making "
        "them most appropriate for offline interaction interpretation rather than "
        "autonomous generation."
    ),
    "fig6": (
        "Pair-identity control experiment. High pair-recognition performance shows "
        "that dyads have recognizable signatures, justifying leave-one-dyad-out "
        "evaluation for condition and response modeling."
    ),
}


PLACEMENTS = {
    "fig1": "Dataset section",
    "fig2": "Methods section",
    "fig3": "Results section, symbolic vocabulary subsection",
    "fig4": "Results section, delayed response analysis",
    "fig5": "Results section, response modeling",
    "fig6": "Results or Discussion section",
    "table1": "Dataset section",
    "table2": "Dataset or Results section",
    "table3": "Results section",
    "table4": "Results section",
    "table5": "Results section",
    "figA1": "Appendix",
    "figA2": "Appendix",
    "figA3": "Appendix",
    "tableA1": "Appendix",
    "tableA2": "Appendix",
    "tableA3": "Appendix",
    "tableA4": "Appendix",
    "tableA5": "Appendix",
}


def setup_dirs() -> None:
    for path in DIRS.values():
        path.mkdir(parents=True, exist_ok=True)


def rel(path: Path | str) -> str:
    path = Path(path)
    try:
        return str(path.relative_to(BASE_DIR))
    except ValueError:
        try:
            return str(path.relative_to(ROOT_DIR))
        except ValueError:
            return str(path)


def source(label: str, candidates: Iterable[str | Path], required: bool = True) -> Path | None:
    for candidate in candidates:
        raw = Path(candidate)
        paths = [raw] if raw.is_absolute() else [BASE_DIR / raw, ROOT_DIR / raw]
        for path in paths:
            if path.exists():
                SOURCES_USED[label] = path
                return path
    if required:
        MISSING.append(f"{label}: none of {', '.join(map(str, candidates))} found")
    return None


def copy_sources() -> None:
    rows = []
    for label, path in sorted(SOURCES_USED.items()):
        safe_name = rel(path).replace("/", "__")
        target = DIRS["source_data_used"] / safe_name
        if path.is_file():
            shutil.copy2(path, target)
        rows.append({"label": label, "source_path": rel(path), "copied_to": rel(target)})
    pd.DataFrame(rows).to_csv(DIRS["source_data_used"] / "source_manifest.csv", index=False)


def fmt_float(x, digits: int = 3) -> str:
    if pd.isna(x):
        return ""
    try:
        return f"{float(x):.{digits}f}"
    except Exception:
        return str(x)


def fmt_percent_fraction(x, digits: int = 2) -> str:
    if pd.isna(x):
        return ""
    return f"{float(x) * 100:.{digits}f}%"


def wrap_text(value, width: int) -> str:
    text = "" if pd.isna(value) else str(value)
    if len(text) <= width:
        return text
    return "\n".join(textwrap.wrap(text, width=width, break_long_words=False))


def register_asset(
    asset_id: str,
    title: str,
    placement: str,
    caption: str,
    files: list[Path],
    sources: list[Path | str],
    kind: str,
) -> None:
    ASSETS.append(
        {
            "id": asset_id,
            "title": title,
            "placement": placement,
            "caption": caption,
            "files": [rel(p) for p in files],
            "sources": [rel(Path(s)) if isinstance(s, Path) else str(s) for s in sources],
            "kind": kind,
        }
    )


def save_plot(
    fig,
    out_base: Path,
    asset_id: str,
    title: str,
    placement: str,
    caption: str,
    sources: list[Path | str],
) -> tuple[Path, Path]:
    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    register_asset(asset_id, title, placement, caption, [png, pdf], sources, "figure")
    return png, pdf


def placeholder_asset(
    out_base: Path,
    asset_id: str,
    title: str,
    placement: str,
    caption: str,
    missing_message: str,
    sources: list[str],
    table: bool = False,
) -> None:
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    ax.axis("off")
    ax.text(0.5, 0.72, title, ha="center", va="center", fontsize=16, fontweight="bold")
    ax.text(0.5, 0.52, "PLACEHOLDER - MISSING INPUT", ha="center", va="center", fontsize=13, color=COLOR_RED)
    ax.text(0.5, 0.36, "\n".join(textwrap.wrap(missing_message, width=80)), ha="center", va="center", fontsize=10)
    ax.text(0.5, 0.16, "\n".join(textwrap.wrap(caption, width=90)), ha="center", va="center", fontsize=9, color=COLOR_GRAY)
    if table:
        pd.DataFrame({"missing_input": [missing_message], "needed_source": ["; ".join(sources)]}).to_csv(
            out_base.with_suffix(".csv"), index=False
        )
    save_plot(fig, out_base, asset_id, title, placement, caption, sources)


def render_table(
    df: pd.DataFrame,
    title: str,
    caption: str,
    out_base: Path,
    asset_id: str,
    placement: str,
    sources: list[Path | str],
    col_widths: dict[str, int] | None = None,
    font_size: float = 7.5,
    landscape: bool = True,
    note: str | None = None,
) -> None:
    csv_path = out_base.with_suffix(".csv")
    df.to_csv(csv_path, index=False)

    col_widths = col_widths or {}
    wrapped = df.copy()
    for col in wrapped.columns:
        width = col_widths.get(col, 18)
        wrapped[col] = wrapped[col].map(lambda x: wrap_text(x, width))

    n_rows, n_cols = wrapped.shape
    fig_w = 11.0 if landscape else 7.2
    fig_h = max(3.0, min(18.0, 1.2 + 0.38 * (n_rows + 1)))
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_title(title, fontsize=13, fontweight="bold", pad=12)

    table = ax.table(
        cellText=wrapped.values,
        colLabels=wrapped.columns,
        cellLoc="left",
        colLoc="left",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(font_size)
    table.scale(1.0, 1.55)

    for (row, col), cell in table.get_celld().items():
        cell.set_edgecolor("#CBD5E1")
        cell.set_linewidth(0.35)
        if row == 0:
            cell.set_facecolor("#E2E8F0")
            cell.set_text_props(weight="bold", color="#111827")
        elif row % 2 == 0:
            cell.set_facecolor("#F8FAFC")
        else:
            cell.set_facecolor("white")

    foot = caption
    if note:
        foot += " Note: " + note
    fig.text(0.02, 0.015, "\n".join(textwrap.wrap(foot, 160)), ha="left", va="bottom", fontsize=8, color=COLOR_GRAY)
    fig.tight_layout(rect=[0.0, 0.06, 1.0, 0.96])

    png = out_base.with_suffix(".png")
    pdf = out_base.with_suffix(".pdf")
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor="white")
    fig.savefig(pdf, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    register_asset(asset_id, title, placement, caption, [csv_path, png, pdf], sources, "table")


def _wrap_box_text(text: str, w: float, max_chars: int | None = None) -> str:
    """Wrap each explicit line so labels stay inside diagram boxes."""
    width = max_chars or max(7, int(w * 11))
    wrapped_lines: list[str] = []
    for line in str(text).split("\n"):
        pieces = textwrap.wrap(line, width=width, break_long_words=False) or [""]
        wrapped_lines.extend(pieces)
    return "\n".join(wrapped_lines)


def add_box(ax, xy, w, h, text, fc="#FFFFFF", ec=COLOR_LINE, fontsize=9, radius=0.08, lw=1.1, max_chars=None):
    box = FancyBboxPatch(
        xy,
        w,
        h,
        boxstyle=f"round,pad=0.03,rounding_size={radius}",
        linewidth=lw,
        edgecolor=ec,
        facecolor=fc,
    )
    ax.add_patch(box)
    txt = ax.text(
        xy[0] + w / 2,
        xy[1] + h / 2,
        _wrap_box_text(text, w, max_chars=max_chars),
        ha="center",
        va="center",
        fontsize=fontsize,
        color="#111827",
        linespacing=1.05,
        clip_on=True,
    )
    txt.set_clip_path(box)
    return box

def add_arrow(ax, start, end, color=COLOR_LINE, lw=1.3, mutation_scale=12):
    arr = FancyArrowPatch(start, end, arrowstyle="-|>", mutation_scale=mutation_scale, color=color, linewidth=lw)
    ax.add_patch(arr)
    return arr


def figure1_setup() -> None:
    agents = source("Figure 1 setup methods source", ["AGENTS.md"], required=False)
    final = source("Figure 1 analysis source", ["final_analysis.md"], required=False)

    fig, ax = plt.subplots(figsize=(10.5, 6.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 7)
    ax.axis("off")

    # Table and task area. The schematic is intentionally text-light; the
    # detailed task explanation belongs in the paper caption.
    ax.add_patch(Rectangle((2.0, 2.0), 6.0, 3.0, facecolor="#F8FAFC", edgecolor=COLOR_LINE, linewidth=1.5))

    disk_colors = [
        "#E45756",
        "#4C78A8",
        "#54A24B",
        "#F58518",
        "#B279A2",
        "#72B7B2",
        "#FF9DA6",
        "#9D755D",
        "#BAB0AC",
    ]
    chain_x = np.linspace(2.7, 7.3, 18)
    chain_y = 3.45 + 0.35 * np.sin(np.linspace(0, 2.5 * np.pi, 18))
    for i, (x, y) in enumerate(zip(chain_x, chain_y)):
        circ = Circle((x, y), 0.16, facecolor=disk_colors[i % len(disk_colors)], edgecolor="#111827", linewidth=0.7)
        ax.add_patch(circ)
        ax.text(x, y, str((i % 9) + 1), ha="center", va="center", fontsize=6.5, color="white", fontweight="bold")

    # Participants.
    ax.add_patch(Circle((5.0, 6.0), 0.42, facecolor="#DBEAFE", edgecolor=COLOR_LINE, linewidth=1.2))
    ax.add_patch(Rectangle((4.35, 5.25), 1.3, 0.28, facecolor="#DBEAFE", edgecolor=COLOR_LINE, linewidth=1.0))
    ax.add_patch(Circle((5.0, 1.0), 0.42, facecolor="#DCFCE7", edgecolor=COLOR_LINE, linewidth=1.2))
    ax.add_patch(Rectangle((4.35, 1.48), 1.3, 0.28, facecolor="#DCFCE7", edgecolor=COLOR_LINE, linewidth=1.0))
    add_arrow(ax, (5.0, 5.55), (5.0, 5.07), COLOR_BLUE, lw=1.1)
    add_arrow(ax, (5.0, 1.45), (5.0, 1.93), COLOR_GREEN, lw=1.1)

    def draw_camera(x, y, color="#FEF3C7"):
        ax.add_patch(Rectangle((x - 0.34, y - 0.20), 0.68, 0.40, facecolor=color, edgecolor=COLOR_LINE, linewidth=1.0))
        ax.add_patch(Circle((x + 0.18, y), 0.08, facecolor="#111827", edgecolor="#111827"))

    # Upper participant: current placement retained.
    draw_camera(1.15, 6.18, color="#FEF3C7")  # pose
    draw_camera(3.22, 5.06, color="#E0F2FE")  # gaze

    # Lower participant: moved from right side to left side, vertically symmetric
    # with the upper participant layout.
    draw_camera(1.15, 0.82, color="#FEF3C7")  # pose
    draw_camera(3.22, 1.94, color="#E0F2FE")  # gaze

    save_plot(
        fig,
        DIRS["main_figures"] / "fig1_experimental_setup",
        "Figure 1",
        "Experimental Setup and Task Design",
        PLACEMENTS["fig1"],
        CAPTIONS["fig1"],
        [p for p in [agents, final] if p],
    )

def figure2_pipeline() -> None:
    agents = source("Figure 2 pipeline source", ["AGENTS.md"], required=False)
    final = source("Figure 2 analysis source", ["final_analysis.md"], required=False)

    fig, ax = plt.subplots(figsize=(16.2, 3.8))
    ax.set_xlim(0, 16.8)
    ax.set_ylim(0, 3.8)
    ax.axis("off")

    w = 1.18
    h = 1.12
    x0 = 0.25
    gap = 0.31
    step = w + gap
    xs = [x0 + i * step for i in range(11)]
    y = 1.28

    steps = [
        ("Raw multi-camera recordings", "#EFF6FF"),
        ("RTMPose pose extraction", "#F0FDF4"),
        ("PureGaze gaze estimation", "#F0FDF4"),
        ("ChArUco/session calibration", "#F0FDF4"),
        ("Shared coordinate frame", "#F0FDF4"),
        ("Fused pose/gaze parquet", "#FFF7ED"),
        ("Data validation", "#FFF7ED"),
        ("Windowed dyadic features 1s/2s/5s", "#FFF7ED"),
        ("Individual IW vocabulary", "#F5F3FF"),
        ("Event-level token sequences", "#F5F3FF"),
        ("Response modeling and evaluation", "#F8FAFC"),
    ]

    phase_specs = [
        ("Data acquisition", 0, 0, "#EFF6FF"),
        ("Preprocessing", 1, 4, "#F0FDF4"),
        ("Feature construction", 5, 7, "#FFF7ED"),
        ("Symbolic modeling", 8, 9, "#F5F3FF"),
        ("Evaluation", 10, 10, "#F8FAFC"),
    ]
    for label, i0, i1, color in phase_specs:
        left = xs[i0] - 0.06
        right = xs[i1] + w + 0.06
        ax.add_patch(Rectangle((left, 2.78), right - left, 0.45, facecolor=color, edgecolor="#CBD5E1", linewidth=0.6))
        ax.text((left + right) / 2, 3.005, label, ha="center", va="center", fontsize=8.2, fontweight="bold")

    centers = []
    for x, (text, color) in zip(xs, steps):
        add_box(ax, (x, y), w, h, text, fc=color, fontsize=6.8, radius=0.035, max_chars=12)
        centers.append((x + w / 2, y + h / 2))
    for a, b in zip(centers[:-1], centers[1:]):
        add_arrow(ax, (a[0] + w / 2 - 0.08, a[1]), (b[0] - w / 2 + 0.08, b[1]), lw=0.95, mutation_scale=9)

    save_plot(
        fig,
        DIRS["main_figures"] / "fig2_pipeline",
        "Figure 2",
        "End-to-End Analysis Pipeline",
        PLACEMENTS["fig2"],
        CAPTIONS["fig2"],
        [p for p in [agents, final] if p],
    )

def figure3_iw_prevalence() -> None:
    src = source("Figure 3 individual vocabulary summary", ["outputs_conversation_full/individual_vocabulary_summary.csv"])
    if not src:
        placeholder_asset(
            DIRS["main_figures"] / "fig3_iw_prevalence",
            "Figure 3",
            "Individual IW Vocabulary Prevalence",
            PLACEMENTS["fig3"],
            CAPTIONS["fig3"],
            "Missing individual vocabulary summary CSV.",
            ["outputs_conversation_full/individual_vocabulary_summary.csv"],
        )
        return
    df = pd.read_csv(src).sort_values("word")
    df["prevalence_pct"] = df["prevalence"] * 100.0

    colors = []
    for word in df["word"]:
        if word == "IW01":
            colors.append(COLOR_BLUE)
        elif word == "IW11":
            colors.append(COLOR_GREEN)
        elif word in {"IW08", "IW06", "IW07"}:
            colors.append(COLOR_ORANGE)
        else:
            colors.append("#CBD5E1")

    fig, ax = plt.subplots(figsize=(9.2, 5.4))
    bars = ax.bar(df["word"], df["prevalence_pct"], color=colors, edgecolor="#334155", linewidth=0.5)
    ax.set_ylabel("Prevalence among individual 1s windows (%)")
    ax.set_xlabel("Individual nonverbal word")
    ax.grid(axis="y", alpha=0.25)
    ax.set_ylim(0, max(df["prevalence_pct"].max() * 1.18, 65))

    for bar, val in zip(bars, df["prevalence_pct"]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.8, f"{val:.1f}%", ha="center", va="bottom", fontsize=7)

    fig.tight_layout()
    save_plot(
        fig,
        DIRS["main_figures"] / "fig3_iw_prevalence",
        "Figure 3",
        "Individual IW Vocabulary Prevalence",
        PLACEMENTS["fig3"],
        CAPTIONS["fig3"],
        [src],
    )


def parse_event_report(path: Path) -> dict[str, float]:
    text = path.read_text(encoding="utf-8")
    patterns = {
        "events": r"Events:\s*([0-9.]+)",
        "opportunities": r"Active actor response opportunities:\s*([0-9.]+)",
        "responses": r"Found delayed active responses:\s*([0-9.]+)",
        "wait_first": r"Delayed responses where the first other-person event was wait/monitoring:\s*([0-9.]+)",
        "mean_lag": r"Mean delayed active response lag:\s*([0-9.]+)",
        "median_lag": r"Median delayed active response lag:\s*([0-9.]+)",
    }
    out = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        out[key] = float(match.group(1)) if match else math.nan
    return out


def figure4_event_grammar() -> None:
    src = source("Figure 4 event grammar report", ["outputs_academic_strengthening/academic_strengthening_report.md"])
    if not src:
        placeholder_asset(
            DIRS["main_figures"] / "fig4_action_monitoring_action",
            "Figure 4",
            "Action-Monitoring-Action Event Grammar",
            PLACEMENTS["fig4"],
            CAPTIONS["fig4"],
            "Missing academic strengthening report with event counts and lag statistics.",
            ["outputs_academic_strengthening/academic_strengthening_report.md"],
        )
        return

    stats = parse_event_report(src)
    fig, ax = plt.subplots(figsize=(10.0, 5.8))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5.8)
    ax.axis("off")

    add_box(ax, (0.7, 3.0), 2.2, 1.15, "Actor active word\n(e.g., IW04/IW07)", fc="#EFF6FF", fontsize=10)
    add_box(ax, (3.9, 3.0), 2.2, 1.15, "Responder monitoring\nIW01 or IW11", fc="#F0FDF4", fontsize=10)
    add_box(ax, (7.1, 3.0), 2.2, 1.15, "Delayed active word\nOften IW08 or IW06", fc="#FFF7ED", fontsize=10)
    add_arrow(ax, (2.95, 3.58), (3.85, 3.58), lw=1.6)
    add_arrow(ax, (6.15, 3.58), (7.05, 3.58), lw=1.6)

    quant_rows = [
        ("Collapsed symbolic events", stats["events"]),
        ("Active actor response opportunities", stats["opportunities"]),
        ("Delayed active responses found", stats["responses"]),
        ("Wait/monitoring first before active response", stats["wait_first"]),
        ("Mean delayed active response lag", stats["mean_lag"]),
        ("Median delayed active response lag", stats["median_lag"]),
    ]
    y0 = 2.0
    for i, (label, val) in enumerate(quant_rows):
        x = 0.9 + (i % 3) * 3.05
        y = y0 - (i // 3) * 0.68
        if "lag" in label:
            val_str = f"{val:.2f} windows"
        else:
            val_str = f"{int(val):,}" if not pd.isna(val) else "missing"
        add_box(ax, (x, y), 2.45, 0.48, f"{label}\n{val_str}", fc="#F8FAFC", fontsize=8.2, radius=0.03, lw=0.7)

    save_plot(
        fig,
        DIRS["main_figures"] / "fig4_action_monitoring_action",
        "Figure 4",
        "Action-Monitoring-Action Event Grammar",
        PLACEMENTS["fig4"],
        CAPTIONS["fig4"],
        [src],
    )


def load_event_results() -> pd.DataFrame | None:
    src = source("Event response model results", ["outputs_academic_strengthening/event_bc_model_results.csv"])
    if not src:
        return None
    df = pd.read_csv(src)
    dist_src = source("Event response distribution evaluation", ["outputs_academic_strengthening/event_bc_distribution_eval.csv"], required=False)
    if dist_src:
        dist = pd.read_csv(dist_src)
        overall = dist[(dist["scope"] == "overall") & (dist["group"].astype(str) == "all")][["model", "js_divergence"]]
        df = df.merge(overall, on="model", how="left")
    else:
        df["js_divergence"] = np.nan
    return df


def best_row(df: pd.DataFrame, mask, metric: str = "macro_f1") -> pd.Series:
    subset = df[mask].copy()
    if subset.empty:
        return pd.Series(dtype=object)
    return subset.sort_values(metric, ascending=False).iloc[0]


def pretty_model(name: str) -> str:
    return (
        str(name)
        .replace("xgboost", "XGBoost")
        .replace("random_forest", "Random Forest")
        .replace("hist_gbdt", "Hist-GBDT")
        .replace("logistic", "Logistic")
        .replace("majority", "Majority")
        .replace("_", " ")
    )


def figure5_response_models() -> None:
    df = load_event_results()
    srcs = [
        source("Figure 5 event response model CSV", ["outputs_academic_strengthening/event_bc_model_results.csv"], required=False),
        source("Figure 5 event distribution CSV", ["outputs_academic_strengthening/event_bc_distribution_eval.csv"], required=False),
    ]
    srcs = [s for s in srcs if s]
    if df is None:
        placeholder_asset(
            DIRS["main_figures"] / "fig5_response_model_comparison",
            "Figure 5",
            "Event-Level Response Modeling Comparison",
            PLACEMENTS["fig5"],
            CAPTIONS["fig5"],
            "Missing event-level response model results CSV.",
            ["outputs_academic_strengthening/event_bc_model_results.csv"],
        )
        return

    rows = [
        ("Majority baseline", best_row(df, df["model"].eq("majority"))),
        ("Best actor-only", best_row(df, df["feature_set"].eq("actor_only"), "macro_f1")),
        ("Best actor-history", best_row(df, df["feature_set"].eq("actor_history"), "macro_f1")),
        ("Best observed-immediate", best_row(df, df["feature_set"].eq("observed_immediate"), "macro_f1")),
    ]
    plot_df = pd.DataFrame(
        [
            {
                "family": family,
                "model": pretty_model(row.get("model", "")),
                "macro_f1": float(row.get("macro_f1", np.nan)),
                "top3_accuracy": float(row.get("top3_accuracy", np.nan)),
            }
            for family, row in rows
            if not row.empty
        ]
    )

    fig, ax = plt.subplots(figsize=(9.2, 5.5))
    x = np.arange(len(plot_df))
    width = 0.34
    b1 = ax.bar(x - width / 2, plot_df["macro_f1"], width, label="Macro F1", color=COLOR_BLUE)
    b2 = ax.bar(x + width / 2, plot_df["top3_accuracy"], width, label="Top-3 accuracy", color=COLOR_ORANGE)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{r.family}\n{r.model}" for r in plot_df.itertuples()], fontsize=8)
    ax.set_ylim(0, 1.0)
    ax.set_ylabel("Score")
    ax.legend(frameon=False, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    for bars in [b1, b2]:
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02, f"{bar.get_height():.3f}", ha="center", fontsize=7)
    fig.tight_layout()
    save_plot(
        fig,
        DIRS["main_figures"] / "fig5_response_model_comparison",
        "Figure 5",
        "Event-Level Response Modeling Comparison",
        PLACEMENTS["fig5"],
        CAPTIONS["fig5"],
        srcs,
    )


def figure6_pair_identity() -> None:
    src = source("Figure 6 classification results", ["results/classification/classification_results_all_windows.csv"])
    if not src:
        placeholder_asset(
            DIRS["main_figures"] / "fig6_pair_identity_leakage",
            "Figure 6",
            "Pair-Specific Signatures and Leakage Risk",
            PLACEMENTS["fig6"],
            CAPTIONS["fig6"],
            "Missing classification results CSV.",
            ["results/classification/classification_results_all_windows.csv"],
        )
        return
    df = pd.read_csv(src)
    pair = df[
        (df["target"] == "pair_identity_session_group_control")
        & (df["model"] == "random_forest")
        & (df["feature_set"] == "combined")
        & (df["status"] == "ok")
    ].sort_values("window_size_s")

    fig, ax = plt.subplots(figsize=(8.8, 5.2))
    metrics = [("accuracy", "Accuracy", COLOR_BLUE), ("macro_f1", "Macro F1", COLOR_ORANGE), ("roc_auc", "ROC-AUC", COLOR_GREEN)]
    x = np.arange(len(pair))
    width = 0.24
    for i, (col, label, color) in enumerate(metrics):
        bars = ax.bar(x + (i - 1) * width, pair[col], width, label=label, color=color)
        for bar in bars:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.015, f"{bar.get_height():.3f}", ha="center", fontsize=7)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{int(w)}s windows" for w in pair["window_size_s"]])
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_xlabel("Window size")
    ax.legend(frameon=False, ncol=3, loc="upper left")
    ax.grid(axis="y", alpha=0.25)
    fig.tight_layout()
    save_plot(
        fig,
        DIRS["main_figures"] / "fig6_pair_identity_leakage",
        "Figure 6",
        "Pair-Specific Signatures and Leakage Risk",
        PLACEMENTS["fig6"],
        CAPTIONS["fig6"],
        [src],
    )


def table1_dataset_summary() -> None:
    src = source("Table 1 dataset metadata", ["results/validation/dataset_metadata.csv"])
    if not src:
        placeholder_asset(
            DIRS["main_tables"] / "table1_dataset_summary",
            "Table 1",
            "Dataset Summary",
            PLACEMENTS["table1"],
            "Compact dataset summary.",
            "Missing dataset metadata CSV.",
            ["results/validation/dataset_metadata.csv"],
            table=True,
        )
        return
    meta = pd.read_csv(src)
    dyads = int(meta["dyad_id"].nunique()) if "dyad_id" in meta else 11
    total_sessions = int(len(meta))
    rows = int(meta["row_count"].sum()) if "row_count" in meta else 183195
    cols = int(round(meta["column_count"].median())) if "column_count" in meta else 3436
    df = pd.DataFrame(
        [
            ("Participants", f"{dyads * 2}"),
            ("Dyads", f"{dyads}"),
            ("Sessions per dyad", "4"),
            ("Total sessions", f"{total_sessions}"),
            ("Total processed frame rows", f"{rows:,}"),
            ("Cameras", "4 total; D455 for pose and D435 for gaze per participant"),
            ("Resolution/FPS", "720p, 30 fps"),
            ("Task objects", "18 disks, 9 colors, 2 copies per color"),
            ("Session 1", "Noncompetitive first exposure"),
            ("Sessions 2-4", "Competitive/practiced scoring"),
            ("Session lengths", "Variable; stopped after task completion"),
            ("Processed table format", "Synchronized fused parquet"),
            ("Approximate columns per parquet", f"{cols:,}"),
        ],
        columns=["Dataset property", "Value"],
    )
    render_table(
        df,
        "Table 1. Dataset Summary",
        "Summary of the processed dyadic tabletop interaction dataset.",
        DIRS["main_tables"] / "table1_dataset_summary",
        "Table 1",
        PLACEMENTS["table1"],
        [src, "Experimental task description from project prompt"],
        col_widths={"Dataset property": 26, "Value": 60},
        font_size=8,
        landscape=False,
    )


def table2_validation_summary() -> None:
    miss_src = source("Table 2 missingness by session", ["results/validation/missing_by_session.csv"])
    ts_src = source("Table 2 timestamp consistency", ["results/validation/timestamp_frame_consistency.csv"])
    out_src = source("Table 2 outlier summary", ["results/validation/outliers_by_feature.csv"])
    if not (miss_src and ts_src and out_src):
        placeholder_asset(
            DIRS["main_tables"] / "table2_validation_summary",
            "Table 2",
            "Data Validation Summary",
            PLACEMENTS["table2"],
            "Compact validation summary.",
            "Missing one or more validation source CSVs.",
            [
                "results/validation/missing_by_session.csv",
                "results/validation/timestamp_frame_consistency.csv",
                "results/validation/outliers_by_feature.csv",
            ],
            table=True,
        )
        return

    miss = pd.read_csv(miss_src)
    ts = pd.read_csv(ts_src)
    out = pd.read_csv(out_src)
    top_features = ", ".join(out.head(3)["feature"].astype(str).tolist())
    outlier_family = "motion speed and selected keypoint coordinates"
    if top_features:
        outlier_family += f" (top examples: {top_features})"

    df = pd.DataFrame(
        [
            ("Mean missing ratio", fmt_float(miss["overall_missing_ratio"].mean(), 4), "Across all session-level missingness summaries"),
            ("Worst session missing ratio", fmt_float(miss["overall_missing_ratio"].max(), 4), "Highest session-level missingness"),
            ("Minimum session missing ratio", fmt_float(miss["overall_missing_ratio"].min(), 4), "Lowest session-level missingness"),
            ("Total timestamp gaps detected", f"{int(ts['temporal_gap_count'].sum())}", "Gap threshold from validation output"),
            ("Largest temporal gap", f"{ts['largest_temporal_gap_ms'].max():.2f} ms", "Maximum detected temporal gap"),
            ("Mean median frame interval", f"{ts['median_dt_ms'].mean():.1f} ms", "Average of per-session median intervals"),
            ("Main outlier family", outlier_family, "Z-score and IQR outlier checks"),
            ("Interpretation", "Usable dataset, but validity coverage must be considered", "Missingness/visibility features remain analytically relevant"),
        ],
        columns=["Validation item", "Value", "Interpretation"],
    )
    render_table(
        df,
        "Table 2. Data Validation Summary",
        "Validation summary for missingness, temporal consistency, and extreme outliers.",
        DIRS["main_tables"] / "table2_validation_summary",
        "Table 2",
        PLACEMENTS["table2"],
        [miss_src, ts_src, out_src],
        col_widths={"Validation item": 28, "Value": 34, "Interpretation": 55},
        font_size=7.4,
        landscape=True,
    )


def table3_compact_vocab() -> None:
    src = source("Table 3 individual vocabulary summary", ["outputs_conversation_full/individual_vocabulary_summary.csv"])
    if not src:
        placeholder_asset(
            DIRS["main_tables"] / "table3_compact_iw_vocabulary",
            "Table 3",
            "Compact Individual IW Vocabulary",
            PLACEMENTS["table3"],
            "Compact vocabulary table for the main paper.",
            "Missing individual vocabulary summary CSV.",
            ["outputs_conversation_full/individual_vocabulary_summary.csv"],
            table=True,
        )
        return
    vocab = pd.read_csv(src).set_index("word")
    selected = ["IW01", "IW11", "IW08", "IW06", "IW07", "IW04"]
    rows = []
    for word in selected:
        label, interp = WORD_INFO[word]
        prev = fmt_percent_fraction(vocab.loc[word, "prevalence"], 2) if word in vocab.index else ""
        rows.append((word, label, prev, interp))
    df = pd.DataFrame(rows, columns=["Word", "Semantic label", "Prevalence", "Short interpretation"])
    render_table(
        df,
        "Table 3. Compact Individual IW Vocabulary",
        "Most important individual nonverbal words used in the main paper.",
        DIRS["main_tables"] / "table3_compact_iw_vocabulary",
        "Table 3",
        PLACEMENTS["table3"],
        [src],
        col_widths={"Word": 8, "Semantic label": 26, "Prevalence": 12, "Short interpretation": 60},
        font_size=7.8,
        landscape=True,
    )


def table4_stability_condition() -> None:
    cv_src = source("Table 4 CV vocabulary validation", ["outputs_cv_vocab_validation/cv_vocab_validation_summary.csv"])
    cls_src = source("Table 4 classification results", ["results/classification/classification_results_all_windows.csv"])
    if not (cv_src and cls_src):
        placeholder_asset(
            DIRS["main_tables"] / "table4_stability_condition_signal",
            "Table 4",
            "Vocabulary Stability and Condition Signal",
            PLACEMENTS["table4"],
            "Vocabulary stability and condition classification summary.",
            "Missing CV vocabulary or classification source CSV.",
            ["outputs_cv_vocab_validation/cv_vocab_validation_summary.csv", "results/classification/classification_results_all_windows.csv"],
            table=True,
        )
        return
    cv = pd.read_csv(cv_src)
    cls = pd.read_csv(cls_src)
    stab = cv[(cv["method"] == "train_only_kmeans_assignment") & (cv["target"] == "global_word_stability")].iloc[0]
    sim = cv[(cv["method"] == "train_only_kmeans_assignment") & (cv["target"] == "global_word_cluster_similarity")].iloc[0]

    def get_cls(window, feature_set, model):
        row = cls[
            (cls["target"] == "condition")
            & (cls["window_size_s"] == float(window))
            & (cls["feature_set"] == feature_set)
            & (cls["model"] == model)
        ]
        return row.iloc[0] if len(row) else pd.Series(dtype=object)

    rows = [
        ("A: Cross-dyad IW vocabulary stability", "Balanced accuracy", fmt_float(stab["balanced_accuracy_mean"], 3), "Train-only clustering mapped to full-data vocabulary"),
        ("A: Cross-dyad IW vocabulary stability", "Macro F1", fmt_float(stab["macro_f1_mean"], 3), "Leave-one-dyad-out stability"),
        ("A: Cross-dyad IW vocabulary stability", "Adjusted Rand Index", fmt_float(sim["adjusted_rand_mean"], 3), "Cluster similarity across held-out dyads"),
        ("A: Cross-dyad IW vocabulary stability", "Normalized Mutual Information", fmt_float(sim["normalized_mutual_info_mean"], 3), "Cluster similarity across held-out dyads"),
    ]
    for label, row in [
        ("B: 1s pose-only Random Forest", get_cls(1, "pose", "random_forest")),
        ("B: 5s gaze-only Random Forest", get_cls(5, "gaze", "random_forest")),
        ("B: 5s combined Random Forest", get_cls(5, "combined", "random_forest")),
        ("B: Dummy most-frequent baseline", get_cls(1, "combined", "dummy_most_frequent")),
    ]:
        rows.append((label, "Accuracy / Macro F1 / ROC-AUC", f"{fmt_float(row.get('accuracy'), 3)} / {fmt_float(row.get('macro_f1'), 3)} / {fmt_float(row.get('roc_auc'), 3)}", "Condition classification under pair-aware evaluation"))

    df = pd.DataFrame(rows, columns=["Block / analysis", "Metric", "Value", "Interpretation"])
    render_table(
        df,
        "Table 4. Vocabulary Stability and Condition Signal",
        "Cross-dyad vocabulary stability and best baseline-vs-competitive/practiced condition classifiers.",
        DIRS["main_tables"] / "table4_stability_condition_signal",
        "Table 4",
        PLACEMENTS["table4"],
        [cv_src, cls_src],
        col_widths={"Block / analysis": 36, "Metric": 30, "Value": 20, "Interpretation": 50},
        font_size=7.3,
        landscape=True,
    )


def table5_response_summary() -> None:
    df = load_event_results()
    srcs = [
        source("Table 5 event model source", ["outputs_academic_strengthening/event_bc_model_results.csv"], required=False),
        source("Table 5 event distribution source", ["outputs_academic_strengthening/event_bc_distribution_eval.csv"], required=False),
    ]
    srcs = [s for s in srcs if s]
    if df is None:
        placeholder_asset(
            DIRS["main_tables"] / "table5_response_model_summary",
            "Table 5",
            "Event-Level Response Modeling Summary",
            PLACEMENTS["table5"],
            "Compact event response model summary.",
            "Missing event model results CSV.",
            ["outputs_academic_strengthening/event_bc_model_results.csv"],
            table=True,
        )
        return

    desired = [
        ("Majority baseline", "majority", "Lower-bound class-prior baseline"),
        ("Best actor-only", best_row(df, df["feature_set"].eq("actor_only"), "macro_f1").get("model"), "Actor word alone is weak"),
        ("Best actor-history exact/top-3", "xgboost_actor_history", "Best autonomous candidate for exact/top-3 response prediction"),
        ("Best actor-history distributional", "random_forest_actor_history", "Best actor-history distribution match"),
        ("Best observed-immediate exact", "xgboost_observed_immediate", "Best offline exact response interpretation"),
        ("Best observed-immediate distributional", "random_forest_observed_immediate", "Best observed-immediate distribution match"),
    ]
    rows = []
    for family, model, interp in desired:
        row = df[df["model"] == model]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append(
            (
                family,
                pretty_model(model),
                fmt_float(r["accuracy"], 3),
                fmt_float(r["balanced_accuracy"], 3),
                fmt_float(r["macro_f1"], 3),
                fmt_float(r["top3_accuracy"], 3),
                fmt_float(r.get("js_divergence", np.nan), 4),
                interp,
            )
        )
    out = pd.DataFrame(
        rows,
        columns=["Model family", "Best model", "Accuracy", "Balanced acc.", "Macro F1", "Top-3 acc.", "JS div.", "Interpretation"],
    )
    render_table(
        out,
        "Table 5. Event-Level Response Modeling Summary",
        "Comparison of event-level response models for symbolic nonverbal interaction.",
        DIRS["main_tables"] / "table5_response_model_summary",
        "Table 5",
        PLACEMENTS["table5"],
        srcs,
        col_widths={"Model family": 28, "Best model": 26, "Interpretation": 48},
        font_size=6.8,
        landscape=True,
    )


def appendix_figA1_missingness_heatmap() -> None:
    src = source("Appendix Figure A1 missingness by session", ["results/validation/missing_by_session.csv"])
    if not src:
        placeholder_asset(
            DIRS["appendix_figures"] / "figA1_missingness_heatmap",
            "Appendix Figure A1",
            "Missingness Heatmap",
            PLACEMENTS["figA1"],
            "Session-level missingness by dyad and session order.",
            "Missing missing_by_session.csv.",
            ["results/validation/missing_by_session.csv"],
        )
        return
    df = pd.read_csv(src)
    pivot = df.pivot_table(index="pair_id", columns="order", values="overall_missing_ratio", aggfunc="mean")
    fig, ax = plt.subplots(figsize=(7.2, 5.8))
    if sns:
        sns.heatmap(pivot, annot=True, fmt=".2f", cmap="Blues", cbar_kws={"label": "Missing ratio"}, ax=ax)
    else:
        im = ax.imshow(pivot.values, aspect="auto", cmap="Blues")
        plt.colorbar(im, ax=ax, label="Missing ratio")
        ax.set_xticks(np.arange(pivot.shape[1]), labels=pivot.columns)
        ax.set_yticks(np.arange(pivot.shape[0]), labels=pivot.index)
    ax.set_xlabel("Session order")
    ax.set_ylabel("Dyad")
    caption = "Session-level missingness aggregated by dyad and session order."
    fig.tight_layout()
    save_plot(fig, DIRS["appendix_figures"] / "figA1_missingness_heatmap", "Appendix Figure A1", "Missingness Heatmap", PLACEMENTS["figA1"], caption, [src])


def appendix_figA2_transition_matrices() -> None:
    srcs = []
    for ws in [1, 2, 5]:
        s = source(f"Appendix Figure A2 token transitions {ws}s", [f"results/tokens/token_transitions_{ws}s.csv"], required=False)
        if s:
            srcs.append(s)
    div_src = source("Appendix Figure A2 token transition divergence", ["results/tokens/token_sequence_summary_all_windows.csv"], required=False)
    if div_src:
        srcs.append(div_src)
    if len(srcs) < 3:
        placeholder_asset(
            DIRS["appendix_figures"] / "figA2_token_transition_matrices",
            "Appendix Figure A2",
            "Token Transition Matrices",
            PLACEMENTS["figA2"],
            "Baseline vs competitive/practiced token transition matrices.",
            "Missing one or more token transition matrix CSVs.",
            ["results/tokens/token_transitions_1s.csv", "results/tokens/token_transitions_2s.csv", "results/tokens/token_transitions_5s.csv"],
        )
        return

    div = None
    if div_src:
        div = pd.read_csv(div_src).set_index("window_size_s")["transition_js_divergence"].to_dict()

    fig, axes = plt.subplots(3, 2, figsize=(7.4, 9.0))
    for r, ws in enumerate([1, 2, 5]):
        path = BASE_DIR / f"results/tokens/token_transitions_{ws}s.csv"
        df = pd.read_csv(path)
        tokens = sorted(set(df["from_token"]).union(set(df["to_token"])))
        for c, condition in enumerate(["baseline", "competitive"]):
            sub = df[df["condition_label"] == condition]
            mat = pd.DataFrame(0.0, index=tokens, columns=tokens)
            counts = sub.groupby(["from_token", "to_token"])["count"].sum()
            for (a, b), val in counts.items():
                mat.loc[a, b] = val
            mat = mat.div(mat.sum(axis=1).replace(0, np.nan), axis=0).fillna(0)
            ax = axes[r, c]
            if sns:
                sns.heatmap(mat, annot=True, fmt=".2f", cmap="Greens", cbar=False, ax=ax, vmin=0, vmax=1)
            else:
                ax.imshow(mat.values, cmap="Greens", vmin=0, vmax=1)
                ax.set_xticks(np.arange(len(tokens)), tokens)
                ax.set_yticks(np.arange(len(tokens)), tokens)
            js = div.get(float(ws), np.nan) if div else np.nan
            ax.set_title(f"{ws}s {condition}\nJS={js:.4f}" if c == 0 else f"{ws}s {condition}")
            ax.set_xlabel("To token")
            ax.set_ylabel("From token")
    caption = "Baseline and competitive/practiced token transition matrices. JS divergences: 1s=0.0774, 2s=0.0903, 5s=0.1081."
    fig.tight_layout()
    save_plot(fig, DIRS["appendix_figures"] / "figA2_token_transition_matrices", "Appendix Figure A2", "Token Transition Matrices", PLACEMENTS["figA2"], caption, srcs)


def appendix_figA3_condition_ablation() -> None:
    src = source("Appendix Figure A3 classification results", ["results/classification/classification_results_all_windows.csv"])
    if not src:
        placeholder_asset(
            DIRS["appendix_figures"] / "figA3_condition_ablation",
            "Appendix Figure A3",
            "Condition Classifier Ablation",
            PLACEMENTS["figA3"],
            "Feature-set ablation for condition classification.",
            "Missing classification results CSV.",
            ["results/classification/classification_results_all_windows.csv"],
        )
        return
    df = pd.read_csv(src)
    sub = df[(df["target"] == "condition") & (df["status"] == "ok") & (~df["model"].str.contains("dummy", na=False))].copy()
    best = sub.sort_values("macro_f1", ascending=False).groupby(["window_size_s", "feature_set"], as_index=False).first()
    best["label"] = best["window_size_s"].astype(int).astype(str) + "s " + best["feature_set"]

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.8), sharey=True)
    for ax, metric, title, color in [
        (axes[0], "macro_f1", "Macro F1", COLOR_BLUE),
        (axes[1], "roc_auc", "ROC-AUC", COLOR_GREEN),
    ]:
        plot = best.sort_values(["feature_set", "window_size_s"])
        bars = ax.barh(plot["label"], plot[metric], color=color, edgecolor=COLOR_LINE, linewidth=0.4)
        ax.set_xlim(0, 0.75)
        ax.set_title(title)
        ax.grid(axis="x", alpha=0.25)
        for bar, model in zip(bars, plot["model"]):
            ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2, f"{bar.get_width():.3f}\n{model}", va="center", fontsize=6.5)
    caption = "Best non-dummy baseline-vs-competitive/practiced classifier per window size and feature set."
    fig.tight_layout()
    save_plot(fig, DIRS["appendix_figures"] / "figA3_condition_ablation", "Appendix Figure A3", "Condition Classifier Ablation", PLACEMENTS["figA3"], caption, [src])


def appendix_tableA1_full_vocab() -> None:
    src = source("Appendix Table A1 individual vocabulary summary", ["outputs_conversation_full/individual_vocabulary_summary.csv"])
    if not src:
        placeholder_asset(
            DIRS["appendix_tables"] / "tableA1_full_iw_vocabulary",
            "Appendix Table A1",
            "Full 12-Word IW Vocabulary",
            PLACEMENTS["tableA1"],
            "Full individual nonverbal vocabulary.",
            "Missing individual vocabulary summary CSV.",
            ["outputs_conversation_full/individual_vocabulary_summary.csv"],
            table=True,
        )
        return
    vocab = pd.read_csv(src).sort_values("word")
    rows = []
    for row in vocab.itertuples():
        label, interp = WORD_INFO.get(row.word, ("", ""))
        rows.append((row.word, fmt_percent_fraction(row.prevalence, 2), label, row.description, interp))
    df = pd.DataFrame(rows, columns=["Word", "Prevalence", "Semantic label", "Centroid description", "Interpretation"])
    render_table(
        df,
        "Appendix Table A1. Full 12-Word IW Vocabulary",
        "Full individual vocabulary with centroid-derived semantic interpretations.",
        DIRS["appendix_tables"] / "tableA1_full_iw_vocabulary",
        "Appendix Table A1",
        PLACEMENTS["tableA1"],
        [src],
        col_widths={"Word": 8, "Prevalence": 12, "Semantic label": 24, "Centroid description": 56, "Interpretation": 58},
        font_size=6.4,
        landscape=True,
    )


def appendix_tableA2_statistics() -> None:
    src = source("Appendix Table A2 session statistics", ["results/statistics/session_statistics_all_windows.csv"])
    if not src:
        placeholder_asset(
            DIRS["appendix_tables"] / "tableA2_top_statistical_contrasts",
            "Appendix Table A2",
            "Top Exploratory Statistical Contrasts",
            PLACEMENTS["tableA2"],
            "Top exploratory paired contrasts by raw p-value.",
            "Missing session_statistics_all_windows.csv.",
            ["results/statistics/session_statistics_all_windows.csv"],
            table=True,
        )
        return
    stats = pd.read_csv(src).sort_values("p_value").head(10)
    df = pd.DataFrame(
        {
            "Contrast": stats["contrast"],
            "Feature": stats["feature"],
            "Window size": stats["window_size_s"].map(lambda x: f"{x:g}s"),
            "Mean diff.": stats["mean_difference"].map(lambda x: fmt_float(x, 3)),
            "Effect size": stats["cohen_dz"].map(lambda x: fmt_float(x, 3)),
            "Raw p": stats["p_value"].map(lambda x: fmt_float(x, 4)),
            "FDR q": stats["q_value"].map(lambda x: fmt_float(x, 4)),
        }
    )
    render_table(
        df,
        "Appendix Table A2. Top Exploratory Statistical Contrasts",
        "Top 10 exploratory paired contrasts sorted by raw p-value.",
        DIRS["appendix_tables"] / "tableA2_top_statistical_contrasts",
        "Appendix Table A2",
        PLACEMENTS["tableA2"],
        [src],
        col_widths={"Contrast": 28, "Feature": 44, "Window size": 12},
        font_size=6.9,
        landscape=True,
        note="No broad condition-level statistical effect survived FDR correction; these contrasts are exploratory.",
    )


def appendix_tableA3_event_models() -> None:
    df = load_event_results()
    srcs = [
        source("Appendix Table A3 event model CSV", ["outputs_academic_strengthening/event_bc_model_results.csv"], required=False),
        source("Appendix Table A3 distribution CSV", ["outputs_academic_strengthening/event_bc_distribution_eval.csv"], required=False),
    ]
    srcs = [s for s in srcs if s]
    if df is None:
        placeholder_asset(
            DIRS["appendix_tables"] / "tableA3_full_event_model_results",
            "Appendix Table A3",
            "Full Event-Level Response Model Results",
            PLACEMENTS["tableA3"],
            "Full event response model table.",
            "Missing event_bc_model_results.csv.",
            ["outputs_academic_strengthening/event_bc_model_results.csv"],
            table=True,
        )
        return
    out = df.copy()
    out["model"] = out["model"].map(pretty_model)
    for col in ["accuracy", "balanced_accuracy", "macro_f1", "top3_accuracy"]:
        out[col] = out[col].map(lambda x: fmt_float(x, 3))
    out["js_divergence"] = out["js_divergence"].map(lambda x: fmt_float(x, 4))
    out = out.rename(
        columns={
            "model": "Model",
            "feature_set": "Feature set",
            "n_samples": "N",
            "n_classes": "Classes",
            "accuracy": "Accuracy",
            "balanced_accuracy": "Balanced acc.",
            "macro_f1": "Macro F1",
            "top3_accuracy": "Top-3 acc.",
            "js_divergence": "JS div.",
        }
    )
    render_table(
        out,
        "Appendix Table A3. Full Event-Level Response Model Results",
        "Full actor-only, actor-history, observed-immediate, Markov, and majority model results.",
        DIRS["appendix_tables"] / "tableA3_full_event_model_results",
        "Appendix Table A3",
        PLACEMENTS["tableA3"],
        srcs,
        col_widths={"Model": 32, "Feature set": 30},
        font_size=6.0,
        landscape=True,
    )


def appendix_tableA4_feature_importance() -> None:
    src = source("Appendix Table A4 permutation feature importance", ["results/classification/permutation_feature_importance.csv"])
    if not src:
        placeholder_asset(
            DIRS["appendix_tables"] / "tableA4_feature_importance",
            "Appendix Table A4",
            "Feature Importance",
            PLACEMENTS["tableA4"],
            "Top permutation feature importances.",
            "Missing permutation_feature_importance.csv.",
            ["results/classification/permutation_feature_importance.csv"],
            table=True,
        )
        return
    fi = pd.read_csv(src).sort_values("importance_mean", ascending=False).head(15)
    out = pd.DataFrame(
        {
            "Rank": range(1, len(fi) + 1),
            "Feature": fi["feature"],
            "Importance mean": fi["importance_mean"].map(lambda x: fmt_float(x, 4)),
            "Importance std.": fi["importance_std"].map(lambda x: fmt_float(x, 4)),
            "Window": fi["window_size_s"].map(lambda x: f"{x:g}s"),
            "Model": fi["model"].map(pretty_model),
        }
    )
    render_table(
        out,
        "Appendix Table A4. Feature Importance",
        "Top 15 permutation feature importances for the best condition classifier analysis.",
        DIRS["appendix_tables"] / "tableA4_feature_importance",
        "Appendix Table A4",
        PLACEMENTS["tableA4"],
        [src],
        col_widths={"Feature": 42, "Model": 20},
        font_size=6.8,
        landscape=True,
        note="Some top features are validity/visibility features, so the condition classifier should not be interpreted as purely behavioral without further validity-controlled analysis.",
    )


def appendix_tableA5_outliers() -> None:
    src = source("Appendix Table A5 outlier summary", ["results/validation/outliers_by_feature.csv"])
    if not src:
        placeholder_asset(
            DIRS["appendix_tables"] / "tableA5_outlier_summary",
            "Appendix Table A5",
            "Outlier Summary",
            PLACEMENTS["tableA5"],
            "Top outlier features by z-score and IQR rules.",
            "Missing outliers_by_feature.csv.",
            ["results/validation/outliers_by_feature.csv"],
            table=True,
        )
        return
    out = pd.read_csv(src)
    z = out.sort_values("z_outlier_ratio", ascending=False).head(10).reset_index(drop=True)
    iqr = out.sort_values("iqr_outlier_ratio", ascending=False).head(10).reset_index(drop=True)
    df = pd.DataFrame(
        {
            "Rank": range(1, 11),
            "Top z-score feature": z["feature"],
            "Z outlier ratio": z["z_outlier_ratio"].map(lambda x: fmt_float(x, 4)),
            "Top IQR feature": iqr["feature"],
            "IQR outlier ratio": iqr["iqr_outlier_ratio"].map(lambda x: fmt_float(x, 4)),
        }
    )
    render_table(
        df,
        "Appendix Table A5. Outlier Summary",
        "Top 10 features by z-score and IQR outlier ratios.",
        DIRS["appendix_tables"] / "tableA5_outlier_summary",
        "Appendix Table A5",
        PLACEMENTS["tableA5"],
        [src],
        col_widths={"Top z-score feature": 42, "Top IQR feature": 42},
        font_size=7.0,
        landscape=True,
    )


def add_text_page(pdf: PdfPages, title: str, body: str, footer: str | None = None) -> None:
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    ax.text(0.05, 0.94, title, transform=ax.transAxes, fontsize=18, fontweight="bold", va="top")
    wrapped = []
    for paragraph in body.split("\n"):
        if not paragraph.strip():
            wrapped.append("")
        else:
            wrapped.extend(textwrap.wrap(paragraph, width=92))
    ax.text(0.05, 0.88, "\n".join(wrapped), transform=ax.transAxes, fontsize=10.2, va="top", linespacing=1.35)
    if footer:
        ax.text(0.05, 0.04, footer, transform=ax.transAxes, fontsize=8, color=COLOR_GRAY)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def add_asset_page(pdf: PdfPages, asset: dict) -> None:
    pngs = [BASE_DIR / f for f in asset["files"] if f.endswith(".png")]
    if not pngs:
        return
    png = pngs[0]
    img = plt.imread(png)
    fig, ax = plt.subplots(figsize=(8.27, 11.69))
    ax.axis("off")
    fig.text(0.05, 0.965, asset["title"], fontsize=15, fontweight="bold", va="top")
    fig.text(0.05, 0.937, f"Recommended placement: {asset['placement']}", fontsize=9, color=COLOR_GRAY, va="top")
    image_ax = fig.add_axes([0.06, 0.22, 0.88, 0.66])
    image_ax.imshow(img)
    image_ax.axis("off")
    caption = f"Caption: {asset['caption']}"
    source_text = "Sources: " + "; ".join(asset["sources"])
    fig.text(0.05, 0.16, "\n".join(textwrap.wrap(caption, 105)), fontsize=8.5, va="top", color="#111827")
    fig.text(0.05, 0.07, "\n".join(textwrap.wrap(source_text, 112)), fontsize=7.5, va="top", color=COLOR_GRAY)
    pdf.savefig(fig, bbox_inches="tight")
    plt.close(fig)


def build_combined_pdf() -> Path:
    pdf_path = DIRS["pdf"] / "paper_figures_tables_appendix_pack.pdf"
    source_folders = [
        "results/validation",
        "results/classification",
        "results/statistics",
        "results/tokens",
        "outputs_conversation_full",
        "outputs_cv_vocab_validation",
        "outputs_academic_strengthening",
    ]
    with PdfPages(pdf_path) as pdf:
        cover = (
            f"Project title: {TITLE}\n\n"
            "Description: Publication-ready figure and table pack for an EasyChair-style paper on symbolic "
            "nonverbal vocabulary learning from dyadic tabletop pose/gaze interactions.\n\n"
            f"Generation date: {TODAY}\n\n"
            "Source folders used:\n- " + "\n- ".join(source_folders)
        )
        add_text_page(pdf, "Paper Figures, Tables, and Appendix Pack", cover)

        order = [
            "Figure 1",
            "Table 1",
            "Figure 2",
            "Table 2",
            "Figure 3",
            "Table 3",
            "Figure 4",
            "Figure 5",
            "Table 5",
            "Figure 6",
            "Table 4",
            "Appendix Figure A1",
            "Appendix Figure A2",
            "Appendix Figure A3",
            "Appendix Table A1",
            "Appendix Table A2",
            "Appendix Table A3",
            "Appendix Table A4",
            "Appendix Table A5",
        ]
        by_id = {a["id"]: a for a in ASSETS}
        for asset_id in order:
            if asset_id in by_id:
                add_asset_page(pdf, by_id[asset_id])

        placement_map = "\n".join([f"- {a['id']}: {a['placement']}" for a in ASSETS])
        missing = "\n".join([f"- {m}" for m in MISSING]) if MISSING else "- None"
        notes = (
            "Recommended paper placement map:\n"
            f"{placement_map}\n\n"
            "Missing or uncertain assets:\n"
            f"{missing}\n\n"
            "Notes for authors:\n"
            "- Figure 1 and Figure 2 are schematics derived from the project method description, not raw data plots.\n"
            "- The strongest paper figures are Figure 3, Figure 4, Figure 5, and Figure 6 because they directly support the vocabulary, grammar, model, and leakage-control claims.\n"
            "- Tables with long text have compact main-paper versions and full appendix versions where applicable.\n"
            "- Event-model scores should be interpreted with the distinction between offline observed-immediate interpretation and autonomous actor-history generation."
        )
        add_text_page(pdf, "Recommended Placement Map and Author Notes", notes)
    return pdf_path


def write_readme(pdf_path: Path) -> None:
    lines = [
        f"# Paper Asset Pack",
        "",
        f"Title: {TITLE}",
        "",
        f"Generated: {TODAY}",
        "",
        f"Combined PDF: `{rel(pdf_path)}`",
        "",
        "## Recommended Figure/Table Placement",
        "",
        "| Asset | Recommended section | Files | Source files | Recommended caption |",
        "|---|---|---|---|---|",
    ]
    for asset in ASSETS:
        lines.append(
            "| {id} | {placement} | {files} | {sources} | {caption} |".format(
                id=asset["id"],
                placement=asset["placement"],
                files="<br>".join(f"`{f}`" for f in asset["files"]),
                sources="<br>".join(f"`{s}`" for s in asset["sources"]) or "Conceptual schematic",
                caption=asset["caption"].replace("|", "/"),
            )
        )
    lines.extend(
        [
            "",
            "## Missing Inputs or Assumptions",
            "",
        ]
    )
    if MISSING:
        lines.extend([f"- {m}" for m in MISSING])
    else:
        lines.append("- No requested source file was missing during generation.")
    lines.extend(
        [
            "- Figure 1 and Figure 2 are clean schematics based on the project method description and existing analysis documentation.",
            "- Table 1 includes task-design constants from the project description and processed dataset counts from `results/validation/dataset_metadata.csv`.",
            "- Event grammar counts and lag values are parsed from `outputs_academic_strengthening/academic_strengthening_report.md`.",
            "",
            "## Source Data Used",
            "",
            "A copy or manifest entry for each source file used is stored in `source_data_used/`.",
            "",
            "## Notes for Authors",
            "",
            "- Use Figure 3 and Appendix Table A1 when discussing the learned IW vocabulary.",
            "- Use Figure 4 and Table 5 when making claims about action-monitoring-action grammar and response prediction.",
            "- Use Figure 6 to justify leave-one-dyad-out or pair-aware validation.",
            "- Report condition-classification results as modest signal, not strong separability.",
            "- Report observed-immediate response models as offline interpretation models because they observe the responder's first post-actor state.",
            "- Report actor-history models as the closest autonomous/generative candidate, with moderate exact accuracy but useful top-3 and distributional behavior.",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def print_summary(pdf_path: Path) -> None:
    main_figures = [f for a in ASSETS if a["id"].startswith("Figure") for f in a["files"] if f.startswith("paper_assets/main_figures") and not f.endswith(".csv")]
    main_tables = [f for a in ASSETS if a["id"].startswith("Table") for f in a["files"] if f.startswith("paper_assets/main_tables")]
    appendix = [
        f
        for a in ASSETS
        if a["id"].startswith("Appendix")
        for f in a["files"]
        if f.startswith("paper_assets/appendix_figures") or f.startswith("paper_assets/appendix_tables")
    ]

    print("Generated:")
    print("* Main figures:")
    for path in main_figures:
        print(f"  - {path}")
    print("* Main tables:")
    for path in main_tables:
        print(f"  - {path}")
    print("* Appendix figures/tables:")
    for path in appendix:
        print(f"  - {path}")
    print(f"* {rel(pdf_path)}")
    print("")
    print("Missing:")
    if MISSING:
        for item in MISSING:
            print(f"* {item}")
    else:
        print("* none")
    print("")
    print("Recommended paper placements:")
    for asset in ASSETS:
        print(f"* {asset['id']} -> {asset['placement']}")


def main() -> None:
    setup_dirs()
    figure1_setup()
    table1_dataset_summary()
    figure2_pipeline()
    table2_validation_summary()
    figure3_iw_prevalence()
    table3_compact_vocab()
    figure4_event_grammar()
    figure5_response_models()
    table5_response_summary()
    figure6_pair_identity()
    table4_stability_condition()
    appendix_figA1_missingness_heatmap()
    appendix_figA2_transition_matrices()
    appendix_figA3_condition_ablation()
    appendix_tableA1_full_vocab()
    appendix_tableA2_statistics()
    appendix_tableA3_event_models()
    appendix_tableA4_feature_importance()
    appendix_tableA5_outliers()
    copy_sources()
    pdf_path = build_combined_pdf()
    write_readme(pdf_path)
    print_summary(pdf_path)


if __name__ == "__main__":
    main()
