from __future__ import annotations

import json
import os
import platform
from pathlib import Path

SHOW_PLOTS = os.environ.get(
    "PAPER1_MOCK2_REDRAW_SHOW_PLOTS",
    "1",
) != "0"

import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. Input and output
# ============================================================

DEFAULT_ROOT_DIR = Path(
    'mock_data'
)

ROOT_DIR = Path(
    os.environ.get(
        "PAPER1_ROOT_DIR",
        str(DEFAULT_ROOT_DIR),
    )
)

INPUT_DIR = Path(
    os.environ.get(
        "PAPER1_MOCK2_FINAL_SENSITIVITY_DIR",
        str(
            ROOT_DIR
            / "mock2_final_geometry_local_sensitivity_v1"
        ),
    )
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_MOCK2_FINAL_FIGURE_DIR",
        str(INPUT_DIR),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

INPUT_PREFIX = "mock2_final_geometry_local_sensitivity"
OUTPUT_PREFIX = "mock2_final_geometry_local_sensitivity_redraw"

GEOMETRY_AGGREGATE_FILE = (
    INPUT_DIR
    / f"{INPUT_PREFIX}_geometry_aggregate.csv"
)
GEOMETRY_PAIRED_SUMMARY_FILE = (
    INPUT_DIR
    / f"{INPUT_PREFIX}_geometry_paired_summary.csv"
)
LOCAL_AGGREGATE_FILE = (
    INPUT_DIR
    / f"{INPUT_PREFIX}_local_aggregate.csv"
)
LOCAL_PAIRED_SUMMARY_FILE = (
    INPUT_DIR
    / f"{INPUT_PREFIX}_local_paired_summary.csv"
)
LOCAL_RUNS_FILE = (
    INPUT_DIR
    / f"{INPUT_PREFIX}_local_runs.csv"
)
GEOMETRY_RUNS_FILE = (
    INPUT_DIR
    / f"{INPUT_PREFIX}_geometry_runs.csv"
)

SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 300


# ============================================================
# 2. Figure conventions
# ============================================================

GEOMETRY_ROLE_ORDER = [
    "primary_unweighted",
    "primary_loss_weighted",
    "legacy_geometry_only",
    "legacy_geometry_plus_loss",
    "full_inverse_geometry_only",
    "full_inverse_joint",
]

GEOMETRY_ROLE_LABELS = {
    "primary_unweighted": (
        "Primary geometry,\nunweighted loss"
    ),
    "primary_loss_weighted": (
        "Primary geometry,\nselection-weighted loss"
    ),
    "legacy_geometry_only": (
        "Legacy geometry,\nunweighted loss"
    ),
    "legacy_geometry_plus_loss": (
        "Legacy geometry,\nselection-weighted loss"
    ),
    "full_inverse_geometry_only": (
        "Full-inverse geometry,\nunweighted loss"
    ),
    "full_inverse_joint": (
        "Full-inverse geometry,\nselection-weighted loss"
    ),
}

GEOMETRY_MARKERS = {
    "primary_unweighted": "o",
    "primary_loss_weighted": "s",
    "legacy_geometry_only": "D",
    "legacy_geometry_plus_loss": "^",
    "full_inverse_geometry_only": "v",
    "full_inverse_joint": "P",
}

GEOMETRY_CONTRAST_ORDER = [
    "primary_loss_minus_primary_unweighted",
    "legacy_geometry_only_minus_primary_unweighted",
    "legacy_geometry_plus_loss_minus_primary_loss",
    "full_inverse_geometry_only_minus_primary_unweighted",
    "full_inverse_joint_minus_primary_loss",
]

GEOMETRY_CONTRAST_LABELS = {
    "primary_loss_minus_primary_unweighted": (
        "Selection-weighted loss\nminus unweighted loss"
    ),
    "legacy_geometry_only_minus_primary_unweighted": (
        "Legacy geometry\nminus primary geometry"
    ),
    "legacy_geometry_plus_loss_minus_primary_loss": (
        "Legacy geometry + weighted loss\nminus primary weighted loss"
    ),
    "full_inverse_geometry_only_minus_primary_unweighted": (
        "Full-inverse geometry\nminus primary geometry"
    ),
    "full_inverse_joint_minus_primary_loss": (
        "Full-inverse joint\nminus primary weighted loss"
    ),
}

LOCAL_ROLE_ORDER = [
    "primary_unweighted",
    "primary_loss_weighted",
    "local_unweighted",
    "local_full_inverse",
    "local_selection_optimized",
]

LOCAL_ROLE_LABELS = {
    "primary_unweighted": (
        "Primary diffusion,\nunweighted loss"
    ),
    "primary_loss_weighted": (
        "Primary diffusion,\nselection-weighted loss"
    ),
    "local_unweighted": (
        "Local kernel,\nunweighted"
    ),
    "local_full_inverse": (
        "Local kernel,\nfull inverse"
    ),
    "local_selection_optimized": (
        "Local kernel,\nselection-optimized"
    ),
}

LOCAL_MARKERS = {
    "primary_unweighted": "o",
    "primary_loss_weighted": "s",
    "local_unweighted": "D",
    "local_full_inverse": "^",
    "local_selection_optimized": "v",
}

LOCAL_CONTRAST_ORDER = [
    "local_unweighted_minus_primary_unweighted",
    "local_full_inverse_minus_primary_unweighted",
    "local_selection_optimized_minus_primary_unweighted",
    "primary_loss_minus_local_selection_optimized",
]

LOCAL_CONTRAST_LABELS = {
    "local_unweighted_minus_primary_unweighted": (
        "Local unweighted\nminus primary diffusion"
    ),
    "local_full_inverse_minus_primary_unweighted": (
        "Local full inverse\nminus primary diffusion"
    ),
    "local_selection_optimized_minus_primary_unweighted": (
        "Local optimized\nminus primary diffusion"
    ),
    "primary_loss_minus_local_selection_optimized": (
        "Primary weighted diffusion\nminus local optimized"
    ),
}

RISK_PANEL_SPECS = {
    "parent": (
        "parent_normalized_rmse_mean",
        "parent_normalized_rmse_std",
        "Observed parent-weighted risk",
    ),
    "observed": (
        "unweighted_normalized_rmse_mean",
        "unweighted_normalized_rmse_std",
        "Observed-unweighted risk",
    ),
    "query": (
        "query_primary_normalized_rmse_mean",
        "query_primary_normalized_rmse_std",
        "Primary complete-query risk",
    ),
}


# ============================================================
# 3. Utilities
# ============================================================


def configure_font() -> str:
    candidates = [
        "DejaVu Sans",
        "Liberation Sans",
        "Arial",
        "Helvetica",
    ]
    installed = {
        font.name
        for font in fm.fontManager.ttflist
    }
    selected = next(
        (
            name
            for name in candidates
            if name in installed
        ),
        "DejaVu Sans",
    )

    matplotlib.rcParams.update(
        {
            "font.family": selected,
            "font.size": 11.2,
            "axes.titlesize": 13.5,
            "axes.labelsize": 12.0,
            "xtick.labelsize": 10.8,
            "ytick.labelsize": 10.8,
            "legend.fontsize": 9.3,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
        }
    )
    return selected



def require_file(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            "Required input file was not found:\n"
            f"{path}"
        )



def require_columns(
    frame: pd.DataFrame,
    columns: list[str],
    source: Path,
) -> None:
    missing = [
        column
        for column in columns
        if column not in frame.columns
    ]
    if missing:
        raise KeyError(
            f"Missing columns in {source}: "
            + ", ".join(missing)
        )



def finite_array(values, name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if not np.all(np.isfinite(array)):
        raise FloatingPointError(
            f"Non-finite values were found in {name}."
        )
    return array



def add_panel_label(
    ax: plt.Axes,
    label: str,
) -> None:
    ax.text(
        -0.11,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.5,
        fontweight="bold",
        clip_on=False,
    )



def padded_limits(
    low_values: np.ndarray,
    high_values: np.ndarray,
    include_zero: bool = False,
    fraction: float = 0.10,
) -> tuple[float, float]:
    low_values = finite_array(low_values, "lower limits")
    high_values = finite_array(high_values, "upper limits")

    lower = float(np.min(low_values))
    upper = float(np.max(high_values))

    if include_zero:
        lower = min(lower, 0.0)
        upper = max(upper, 0.0)

    span = upper - lower
    if span <= 0.0:
        span = max(abs(lower), abs(upper), 1.0)

    padding = fraction * span
    return lower - padding, upper + padding



def save_figure(
    fig: plt.Figure,
    stem: str,
    show: bool = True,
) -> tuple[Path | None, Path | None]:
    pdf_path = (
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.pdf"
        if SAVE_PDF
        else None
    )
    png_path = (
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.png"
        if SAVE_PNG
        else None
    )

    if pdf_path is not None:
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.06,
        )
    if png_path is not None:
        fig.savefig(
            png_path,
            dpi=PNG_DPI,
            bbox_inches="tight",
            pad_inches=0.06,
        )

    if SHOW_PLOTS and show:
        plt.show()
    else:
        plt.close(fig)

    return pdf_path, png_path


# ============================================================
# 4. Data loading
# ============================================================

for required_path in [
    GEOMETRY_AGGREGATE_FILE,
    GEOMETRY_PAIRED_SUMMARY_FILE,
    LOCAL_AGGREGATE_FILE,
    LOCAL_PAIRED_SUMMARY_FILE,
]:
    require_file(required_path)

geometry_aggregate = pd.read_csv(
    GEOMETRY_AGGREGATE_FILE
)
geometry_paired_summary = pd.read_csv(
    GEOMETRY_PAIRED_SUMMARY_FILE
)
local_aggregate = pd.read_csv(
    LOCAL_AGGREGATE_FILE
)
local_paired_summary = pd.read_csv(
    LOCAL_PAIRED_SUMMARY_FILE
)

geometry_runs = (
    pd.read_csv(GEOMETRY_RUNS_FILE)
    if GEOMETRY_RUNS_FILE.is_file()
    else None
)
local_runs = (
    pd.read_csv(LOCAL_RUNS_FILE)
    if LOCAL_RUNS_FILE.is_file()
    else None
)

aggregate_columns = [
    "role",
    "parent_normalized_rmse_mean",
    "parent_normalized_rmse_std",
    "unweighted_normalized_rmse_mean",
    "unweighted_normalized_rmse_std",
    "query_primary_normalized_rmse_mean",
    "query_primary_normalized_rmse_std",
]
paired_columns = [
    "contrast",
    "metric",
    "mean_difference",
    "ci_low",
    "ci_high",
]

require_columns(
    geometry_aggregate,
    aggregate_columns,
    GEOMETRY_AGGREGATE_FILE,
)
require_columns(
    local_aggregate,
    aggregate_columns,
    LOCAL_AGGREGATE_FILE,
)
require_columns(
    geometry_paired_summary,
    paired_columns,
    GEOMETRY_PAIRED_SUMMARY_FILE,
)
require_columns(
    local_paired_summary,
    paired_columns,
    LOCAL_PAIRED_SUMMARY_FILE,
)


# ============================================================
# 5. Drawing helpers
# ============================================================


def ordered_subset(
    frame: pd.DataFrame,
    role_order: list[str],
) -> pd.DataFrame:
    available = set(frame["role"])
    retained = [
        role
        for role in role_order
        if role in available
    ]
    if not retained:
        raise ValueError(
            "None of the requested roles were found."
        )

    indexed = frame.set_index("role")
    return indexed.loc[retained].reset_index()



def draw_risk_summary(
    ax: plt.Axes,
    aggregate: pd.DataFrame,
    role_order: list[str],
    role_labels: dict[str, str],
    markers: dict[str, str],
    metric_key: str,
    panel_label: str,
) -> None:
    mean_column, std_column, title = (
        RISK_PANEL_SPECS[metric_key]
    )
    selected = ordered_subset(
        aggregate,
        role_order,
    )

    means = finite_array(
        selected[mean_column],
        mean_column,
    )
    stds = finite_array(
        selected[std_column],
        std_column,
    )
    positions = np.arange(selected.shape[0])

    for position, role, mean, std in zip(
        positions,
        selected["role"],
        means,
        stds,
        strict=True,
    ):
        ax.errorbar(
            mean,
            position,
            xerr=std,
            marker=markers.get(role, "o"),
            linestyle="none",
            markersize=7.2,
            capsize=4,
            elinewidth=1.4,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [
            role_labels.get(role, role)
            for role in selected["role"]
        ]
    )
    ax.invert_yaxis()
    ax.set_xlim(
        padded_limits(
            means - stds,
            means + stds,
            include_zero=False,
            fraction=0.10,
        )
    )
    ax.set_xlabel("NRMSE$_{3D}$")
    ax.set_title(title, pad=10)
    add_panel_label(ax, panel_label)



def draw_forest_plot(
    ax: plt.Axes,
    summary: pd.DataFrame,
    metric: str,
    contrast_order: list[str],
    contrast_labels: dict[str, str],
    panel_label: str,
    title: str,
    x_label: str,
) -> None:
    selected = summary.loc[
        summary["metric"] == metric
    ].copy()

    available = set(selected["contrast"])
    retained = [
        contrast
        for contrast in contrast_order
        if contrast in available
    ]
    if not retained:
        raise ValueError(
            f"No contrasts were found for metric={metric}."
        )

    indexed = selected.set_index("contrast")
    selected = indexed.loc[retained].reset_index()

    means = finite_array(
        selected["mean_difference"],
        "mean_difference",
    )
    low = finite_array(
        selected["ci_low"],
        "ci_low",
    )
    high = finite_array(
        selected["ci_high"],
        "ci_high",
    )
    positions = np.arange(selected.shape[0])

    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
        zorder=0,
    )

    marker_cycle = [
        "o",
        "s",
        "D",
        "^",
        "v",
        "P",
    ]
    for index, (
        position,
        mean,
        lower,
        upper,
    ) in enumerate(
        zip(
            positions,
            means,
            low,
            high,
            strict=True,
        )
    ):
        ax.errorbar(
            mean,
            position,
            xerr=np.array(
                [
                    [mean - lower],
                    [upper - mean],
                ]
            ),
            marker=marker_cycle[
                index % len(marker_cycle)
            ],
            linestyle="none",
            markersize=7.2,
            capsize=4,
            elinewidth=1.4,
        )

    ax.set_yticks(positions)
    ax.set_yticklabels(
        [
            contrast_labels.get(
                contrast,
                contrast.replace("_", " "),
            )
            for contrast in selected["contrast"]
        ]
    )
    ax.invert_yaxis()
    ax.set_xlim(
        padded_limits(
            low,
            high,
            include_zero=True,
            fraction=0.10,
        )
    )
    ax.set_xlabel(x_label)
    ax.set_title(title, pad=10)
    add_panel_label(ax, panel_label)


# ============================================================
# 6. Revised geometry-sensitivity figure
# ============================================================


def draw_geometry_composite() -> plt.Figure:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(16.5, 10.5),
        constrained_layout=True,
    )

    draw_risk_summary(
        axes[0, 0],
        geometry_aggregate,
        GEOMETRY_ROLE_ORDER,
        GEOMETRY_ROLE_LABELS,
        GEOMETRY_MARKERS,
        "parent",
        "(a)",
    )
    draw_risk_summary(
        axes[0, 1],
        geometry_aggregate,
        GEOMETRY_ROLE_ORDER,
        GEOMETRY_ROLE_LABELS,
        GEOMETRY_MARKERS,
        "observed",
        "(b)",
    )
    draw_risk_summary(
        axes[1, 0],
        geometry_aggregate,
        GEOMETRY_ROLE_ORDER,
        GEOMETRY_ROLE_LABELS,
        GEOMETRY_MARKERS,
        "query",
        "(c)",
    )
    draw_forest_plot(
        axes[1, 1],
        geometry_paired_summary,
        "parent",
        GEOMETRY_CONTRAST_ORDER,
        GEOMETRY_CONTRAST_LABELS,
        "(d)",
        "Restricted geometry contrasts",
        "Mean paired parent-risk difference",
    )

    return fig


# ============================================================
# 7. Revised local-kernel comparison figure
# ============================================================


def draw_local_composite() -> plt.Figure:
    combined_aggregate = pd.concat(
        [
            geometry_aggregate.loc[
                geometry_aggregate["role"].isin(
                    [
                        "primary_unweighted",
                        "primary_loss_weighted",
                    ]
                )
            ],
            local_aggregate,
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(15.8, 9.8),
        constrained_layout=True,
    )

    draw_risk_summary(
        axes[0, 0],
        combined_aggregate,
        LOCAL_ROLE_ORDER,
        LOCAL_ROLE_LABELS,
        LOCAL_MARKERS,
        "parent",
        "(a)",
    )
    draw_risk_summary(
        axes[0, 1],
        combined_aggregate,
        LOCAL_ROLE_ORDER,
        LOCAL_ROLE_LABELS,
        LOCAL_MARKERS,
        "query",
        "(b)",
    )
    draw_forest_plot(
        axes[1, 0],
        local_paired_summary,
        "parent",
        LOCAL_CONTRAST_ORDER,
        LOCAL_CONTRAST_LABELS,
        "(c)",
        "Paired parent-risk contrasts",
        "Mean paired difference",
    )
    draw_forest_plot(
        axes[1, 1],
        local_paired_summary,
        "query_primary",
        LOCAL_CONTRAST_ORDER,
        LOCAL_CONTRAST_LABELS,
        "(d)",
        "Paired complete-query contrasts",
        "Mean paired difference",
    )

    return fig


# ============================================================
# 8. Optional split-wise diagnostic
# ============================================================


def draw_local_split_diagnostic() -> plt.Figure | None:
    if geometry_runs is None or local_runs is None:
        return None

    require_columns(
        geometry_runs,
        [
            "role",
            "replicate",
            "parent_normalized_rmse",
            "query_primary_normalized_rmse",
        ],
        GEOMETRY_RUNS_FILE,
    )
    require_columns(
        local_runs,
        [
            "role",
            "replicate",
            "parent_normalized_rmse",
            "query_primary_normalized_rmse",
        ],
        LOCAL_RUNS_FILE,
    )

    combined_runs = pd.concat(
        [
            geometry_runs.loc[
                geometry_runs["role"].isin(
                    [
                        "primary_unweighted",
                        "primary_loss_weighted",
                    ]
                )
            ],
            local_runs,
        ],
        ignore_index=True,
    )

    fig, axes = plt.subplots(
        1,
        2,
        figsize=(13.8, 5.2),
        constrained_layout=True,
    )

    for panel_index, (
        ax,
        metric,
        title,
    ) in enumerate(
        [
            (
                axes[0],
                "parent_normalized_rmse",
                "Observed parent-weighted risk by split",
            ),
            (
                axes[1],
                "query_primary_normalized_rmse",
                "Primary complete-query risk by split",
            ),
        ]
    ):
        for role in LOCAL_ROLE_ORDER:
            selected = combined_runs.loc[
                combined_runs["role"] == role
            ].sort_values("replicate")
            if selected.empty:
                continue
            ax.plot(
                selected["replicate"],
                selected[metric],
                marker=LOCAL_MARKERS.get(role, "o"),
                label=LOCAL_ROLE_LABELS.get(
                    role,
                    role,
                ).replace("\n", " "),
            )

        ax.set_xticks(
            sorted(
                int(value)
                for value in combined_runs[
                    "replicate"
                ].unique()
            )
        )
        ax.set_xlabel("Label-split replicate")
        ax.set_ylabel("NRMSE$_{3D}$")
        ax.set_title(title, pad=10)
        ax.legend(
            loc="best",
            fontsize=8.4,
        )
        add_panel_label(
            ax,
            f"({chr(ord('a') + panel_index)})",
        )

    return fig


# ============================================================
# 9. Individual panels
# ============================================================


def save_individual_panels() -> dict[str, str]:
    outputs: dict[str, str] = {}

    geometry_specs = [
        ("parent", "geometry_parent_risk", "(a)"),
        ("observed", "geometry_observed_risk", "(b)"),
        ("query", "geometry_query_risk", "(c)"),
    ]
    for metric_key, stem, panel_label in geometry_specs:
        fig, ax = plt.subplots(
            figsize=(8.6, 5.8),
            constrained_layout=True,
        )
        draw_risk_summary(
            ax,
            geometry_aggregate,
            GEOMETRY_ROLE_ORDER,
            GEOMETRY_ROLE_LABELS,
            GEOMETRY_MARKERS,
            metric_key,
            panel_label,
        )
        pdf_path, png_path = save_figure(fig, stem, show=False)
        outputs[f"{stem}_pdf"] = str(pdf_path or "")
        outputs[f"{stem}_png"] = str(png_path or "")

    fig, ax = plt.subplots(
        figsize=(9.4, 5.8),
        constrained_layout=True,
    )
    draw_forest_plot(
        ax,
        geometry_paired_summary,
        "parent",
        GEOMETRY_CONTRAST_ORDER,
        GEOMETRY_CONTRAST_LABELS,
        "(d)",
        "Restricted geometry contrasts",
        "Mean paired parent-risk difference",
    )
    pdf_path, png_path = save_figure(
        fig,
        "geometry_parent_contrasts",
        show=False,
    )
    outputs["geometry_parent_contrasts_pdf"] = str(
        pdf_path or ""
    )
    outputs["geometry_parent_contrasts_png"] = str(
        png_path or ""
    )

    combined_aggregate = pd.concat(
        [
            geometry_aggregate.loc[
                geometry_aggregate["role"].isin(
                    [
                        "primary_unweighted",
                        "primary_loss_weighted",
                    ]
                )
            ],
            local_aggregate,
        ],
        ignore_index=True,
    )

    local_specs = [
        ("parent", "local_parent_risk", "(a)"),
        ("query", "local_query_risk", "(b)"),
    ]
    for metric_key, stem, panel_label in local_specs:
        fig, ax = plt.subplots(
            figsize=(8.6, 5.5),
            constrained_layout=True,
        )
        draw_risk_summary(
            ax,
            combined_aggregate,
            LOCAL_ROLE_ORDER,
            LOCAL_ROLE_LABELS,
            LOCAL_MARKERS,
            metric_key,
            panel_label,
        )
        pdf_path, png_path = save_figure(fig, stem, show=False)
        outputs[f"{stem}_pdf"] = str(pdf_path or "")
        outputs[f"{stem}_png"] = str(png_path or "")

    for metric, stem, panel_label, title in [
        (
            "parent",
            "local_parent_contrasts",
            "(c)",
            "Paired parent-risk contrasts",
        ),
        (
            "query_primary",
            "local_query_contrasts",
            "(d)",
            "Paired complete-query contrasts",
        ),
    ]:
        fig, ax = plt.subplots(
            figsize=(9.4, 5.3),
            constrained_layout=True,
        )
        draw_forest_plot(
            ax,
            local_paired_summary,
            metric,
            LOCAL_CONTRAST_ORDER,
            LOCAL_CONTRAST_LABELS,
            panel_label,
            title,
            "Mean paired difference",
        )
        pdf_path, png_path = save_figure(fig, stem, show=False)
        outputs[f"{stem}_pdf"] = str(pdf_path or "")
        outputs[f"{stem}_png"] = str(png_path or "")

    return outputs


# ============================================================
# 10. Main
# ============================================================

font_name = configure_font()

print("=" * 112)
print("Paper I: Mock-2 final sensitivity figure redraw")
print("=" * 112)
print(f"Python                   : {platform.python_version()}")
print(f"Matplotlib font          : {font_name}")
print(f"Input                    : {INPUT_DIR}")
print(f"Output                   : {OUTPUT_DIR}")
print("No reconstruction or eigensystem calculation is performed.")

geometry_fig = draw_geometry_composite()
geometry_pdf, geometry_png = save_figure(
    geometry_fig,
    "geometry_sensitivity_final_composite",
)

local_fig = draw_local_composite()
local_pdf, local_png = save_figure(
    local_fig,
    "local_comparison_final_composite",
)

split_fig = draw_local_split_diagnostic()
if split_fig is not None:
    split_pdf, split_png = save_figure(
        split_fig,
        "local_split_diagnostic",
    )
else:
    split_pdf = None
    split_png = None

individual_outputs = save_individual_panels()

summary_payload = {
    "inputs": {
        "geometry_aggregate": str(
            GEOMETRY_AGGREGATE_FILE
        ),
        "geometry_paired_summary": str(
            GEOMETRY_PAIRED_SUMMARY_FILE
        ),
        "local_aggregate": str(
            LOCAL_AGGREGATE_FILE
        ),
        "local_paired_summary": str(
            LOCAL_PAIRED_SUMMARY_FILE
        ),
        "geometry_runs": (
            str(GEOMETRY_RUNS_FILE)
            if GEOMETRY_RUNS_FILE.is_file()
            else ""
        ),
        "local_runs": (
            str(LOCAL_RUNS_FILE)
            if LOCAL_RUNS_FILE.is_file()
            else ""
        ),
    },
    "outputs": {
        "geometry_composite_pdf": str(
            geometry_pdf or ""
        ),
        "geometry_composite_png": str(
            geometry_png or ""
        ),
        "local_composite_pdf": str(
            local_pdf or ""
        ),
        "local_composite_png": str(
            local_png or ""
        ),
        "local_split_pdf": str(split_pdf or ""),
        "local_split_png": str(split_png or ""),
        **individual_outputs,
    },
}

summary_path = (
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_summary.json"
)
summary_path.write_text(
    json.dumps(
        summary_payload,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("-" * 112)
print(f"Geometry composite PDF   : {geometry_pdf}")
print(f"Geometry composite PNG   : {geometry_png}")
print(f"Local composite PDF      : {local_pdf}")
print(f"Local composite PNG      : {local_png}")
print(f"Split diagnostic PDF     : {split_pdf}")
print(f"Split diagnostic PNG     : {split_png}")
print(f"Summary JSON             : {summary_path}")
print("=" * 112)
