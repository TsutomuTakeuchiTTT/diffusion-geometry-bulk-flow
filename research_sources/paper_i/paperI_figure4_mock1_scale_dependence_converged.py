from __future__ import annotations

import json
import os
import platform
from pathlib import Path

SHOW_PLOTS = os.environ.get(
    "PAPER1_FIGURE4_SHOW_PLOTS",
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
# 1. Input and output paths
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

MODE_CONVERGENCE_DIR = (
    ROOT_DIR
    / "mock1_mode_number_convergence_v2"
)

MODE_RUNS_FILE = (
    MODE_CONVERGENCE_DIR
    / "mock1_mode_number_convergence_runs_to_2048.csv"
)

PRACTICAL_SELECTION_FILE = (
    MODE_CONVERGENCE_DIR
    / "mock1_mode_number_convergence_practical_selection.csv"
)

LOCAL_RUNS_FILE = (
    ROOT_DIR
    / "mock1_bulk_seed_scale_robustness_runs.csv"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_FIGURE4_OUTPUT",
        str(
            ROOT_DIR
            / "paperI_figures"
            / "figure4_mock1_scale_dependence_converged"
        ),
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PREFIX = (
    "paperI_fig4_mock1_scale_dependence_converged"
)

SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 300


# ============================================================
# 2. Fixed labels and plotting setup
# ============================================================

SCALE_ORDER = [
    "fine",
    "fiducial",
    "coarse",
]

SCALE_LABELS = {
    "fine": "Fine",
    "fiducial": "Fiducial",
    "coarse": "Coarse",
}

LOCAL_MODEL_NAME = (
    "adaptive local-kernel regression"
)

PANEL_LABELS = [
    "(a)",
    "(b)",
    "(c)",
    "(d)",
]

PANEL_FIGSIZE = (
    6.3,
    4.9,
)

COMPOSITE_FIGSIZE = (
    13.0,
    10.0,
)


# ============================================================
# 3. Utility functions
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
            "font.size": 11.5,
            "axes.unicode_minus": False,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "mathtext.fontset": "dejavusans",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "savefig.facecolor": "white",
            "savefig.edgecolor": "white",
        }
    )

    return selected



def add_panel_label(
    ax: plt.Axes,
    label: str,
) -> None:
    ax.text(
        -0.10,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.0,
        fontweight="bold",
        clip_on=False,
    )



def save_figure_pair(
    fig: plt.Figure,
    stem: str,
) -> tuple[Path | None, Path | None]:
    pdf_path = (
        OUTPUT_DIR / f"{stem}.pdf"
        if SAVE_PDF
        else None
    )
    png_path = (
        OUTPUT_DIR / f"{stem}.png"
        if SAVE_PNG
        else None
    )

    if pdf_path is not None:
        fig.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.05,
        )

    if png_path is not None:
        fig.savefig(
            png_path,
            dpi=PNG_DPI,
            bbox_inches="tight",
            pad_inches=0.05,
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return (
        pdf_path,
        png_path,
    )



def require_columns(
    frame: pd.DataFrame,
    required: list[str],
    source_name: str,
) -> None:
    missing = [
        column
        for column in required
        if column not in frame.columns
    ]

    if missing:
        raise KeyError(
            f"{source_name} is missing required columns: "
            + ", ".join(missing)
        )



def finite_array(
    values: pd.Series | np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )

    if not np.all(np.isfinite(array)):
        raise FloatingPointError(
            f"{name} contains non-finite values."
        )

    return array


# ============================================================
# 4. Load and assemble final practical-mode results
# ============================================================


def load_inputs() -> dict[str, pd.DataFrame]:
    required_files = [
        MODE_RUNS_FILE,
        PRACTICAL_SELECTION_FILE,
        LOCAL_RUNS_FILE,
    ]

    missing_files = [
        path
        for path in required_files
        if not path.is_file()
    ]

    if missing_files:
        raise FileNotFoundError(
            "Required Figure 4 inputs were not found:\n"
            + "\n".join(
                str(path)
                for path in missing_files
            )
        )

    mode_runs = pd.read_csv(
        MODE_RUNS_FILE
    )
    practical = pd.read_csv(
        PRACTICAL_SELECTION_FILE
    )
    local_runs = pd.read_csv(
        LOCAL_RUNS_FILE
    )

    require_columns(
        mode_runs,
        [
            "replicate",
            "scale",
            "n_basis",
            "median_reference_bandwidth",
            "selected_normalized_rmse",
            "selected_median_direction_error_deg",
            "selected_median_abs_relative_speed_error",
            "selected_calibration_slope_x",
            "selected_calibration_slope_y",
            "selected_calibration_slope_z",
            "selected_vector_gain_through_origin",
        ],
        str(MODE_RUNS_FILE),
    )

    require_columns(
        practical,
        [
            "scale",
            "one_standard_error_basis",
            "practical_plateau_basis",
        ],
        str(PRACTICAL_SELECTION_FILE),
    )

    require_columns(
        local_runs,
        [
            "replicate",
            "scale",
            "model",
            "normalized_rmse",
            "median_direction_error_deg",
        ],
        str(LOCAL_RUNS_FILE),
    )

    practical = practical.loc[
        practical["scale"].isin(SCALE_ORDER)
    ].copy()

    if practical.shape[0] != len(SCALE_ORDER):
        raise ValueError(
            "The practical-selection table must contain one row "
            "for each of fine, fiducial, and coarse."
        )

    practical["one_standard_error_basis"] = practical[
        "one_standard_error_basis"
    ].astype(np.int64)
    practical["practical_plateau_basis"] = practical[
        "practical_plateau_basis"
    ].astype(np.int64)

    if not np.array_equal(
        practical["one_standard_error_basis"].to_numpy(),
        practical["practical_plateau_basis"].to_numpy(),
    ):
        raise ValueError(
            "The one-standard-error and practical-plateau "
            "basis selections do not agree."
        )

    practical = practical.rename(
        columns={
            "one_standard_error_basis": "selected_basis",
        }
    )[
        [
            "scale",
            "selected_basis",
        ]
    ]

    diffusion = mode_runs.merge(
        practical,
        on="scale",
        how="inner",
    )
    diffusion = diffusion.loc[
        diffusion["n_basis"]
        == diffusion["selected_basis"]
    ].copy()

    local = local_runs.loc[
        local_runs["model"] == LOCAL_MODEL_NAME,
        [
            "replicate",
            "scale",
            "normalized_rmse",
            "median_direction_error_deg",
        ],
    ].rename(
        columns={
            "normalized_rmse": "local_normalized_rmse",
            "median_direction_error_deg": (
                "local_median_direction_error_deg"
            ),
        }
    )

    paired = diffusion.merge(
        local,
        on=[
            "replicate",
            "scale",
        ],
        how="inner",
        validate="one_to_one",
    )

    paired["diffusion_minus_local_nrmse"] = (
        paired["selected_normalized_rmse"]
        - paired["local_normalized_rmse"]
    )

    expected_replicates = set(
        range(1, 6)
    )

    for scale in SCALE_ORDER:
        selected = paired.loc[
            paired["scale"] == scale
        ]

        actual_replicates = set(
            int(value)
            for value in selected["replicate"]
        )

        if actual_replicates != expected_replicates:
            raise ValueError(
                f"Scale {scale!r} does not contain replicates 1--5."
            )

        basis_values = selected[
            "selected_basis"
        ].unique()

        if basis_values.size != 1:
            raise ValueError(
                f"Scale {scale!r} has a non-unique practical basis."
            )

    paired["scale"] = pd.Categorical(
        paired["scale"],
        categories=SCALE_ORDER,
        ordered=True,
    )
    paired = paired.sort_values(
        [
            "scale",
            "replicate",
        ]
    ).reset_index(
        drop=True
    )

    summary_records: list[dict[str, float | int | str]] = []

    for scale in SCALE_ORDER:
        selected = paired.loc[
            paired["scale"] == scale
        ]

        record: dict[str, float | int | str] = {
            "scale": scale,
            "scale_label": SCALE_LABELS[scale],
            "selected_basis": int(
                selected["selected_basis"].iloc[0]
            ),
            "n_replicates": int(
                selected.shape[0]
            ),
            "mean_reference_bandwidth": float(
                selected[
                    "median_reference_bandwidth"
                ].mean()
            ),
            "std_reference_bandwidth": float(
                selected[
                    "median_reference_bandwidth"
                ].std(ddof=1)
            ),
        }

        metric_columns = [
            "selected_normalized_rmse",
            "local_normalized_rmse",
            "selected_median_direction_error_deg",
            "local_median_direction_error_deg",
            "selected_median_abs_relative_speed_error",
            "selected_calibration_slope_x",
            "selected_calibration_slope_y",
            "selected_calibration_slope_z",
            "selected_vector_gain_through_origin",
            "diffusion_minus_local_nrmse",
        ]

        for column in metric_columns:
            values = finite_array(
                selected[column],
                f"{scale}:{column}",
            )
            record[f"{column}_mean"] = float(
                np.mean(values)
            )
            record[f"{column}_std"] = float(
                np.std(
                    values,
                    ddof=1,
                )
            )

        record["diffusion_win_count"] = int(
            np.sum(
                selected[
                    "diffusion_minus_local_nrmse"
                ].to_numpy(
                    dtype=np.float64
                )
                < 0.0
            )
        )

        summary_records.append(
            record
        )

    summary = pd.DataFrame(
        summary_records
    )
    summary["scale"] = pd.Categorical(
        summary["scale"],
        categories=SCALE_ORDER,
        ordered=True,
    )
    summary = summary.sort_values(
        "scale"
    ).reset_index(
        drop=True
    )

    return {
        "mode_runs": mode_runs,
        "practical": practical,
        "local_runs": local_runs,
        "paired": paired,
        "summary": summary,
    }


# ============================================================
# 5. Panel drawing functions
# ============================================================


def draw_nrmse_panel(
    ax: plt.Axes,
    data: dict[str, pd.DataFrame],
    panel_label: str,
) -> None:
    summary = data["summary"]
    x_values = finite_array(
        summary["mean_reference_bandwidth"],
        "mean_reference_bandwidth",
    )

    ax.errorbar(
        x_values,
        summary[
            "selected_normalized_rmse_mean"
        ],
        yerr=summary[
            "selected_normalized_rmse_std"
        ],
        marker="o",
        capsize=4,
        label="Diffusion spectral",
    )
    ax.errorbar(
        x_values,
        summary[
            "local_normalized_rmse_mean"
        ],
        yerr=summary[
            "local_normalized_rmse_std"
        ],
        marker="o",
        capsize=4,
        label="Adaptive local kernel",
    )

    ax.set_xlabel(
        r"Median reference smoothing scale "
        r"$[h^{-1}\,\mathrm{Mpc}]$"
    )
    ax.set_ylabel(
        r"Test $\mathrm{NRMSE}_{3\mathrm{D}}$"
    )
    ax.set_title(
        "Reconstruction error at practical truncation"
    )
    ax.legend(
        loc="upper right",
        frameon=True,
    )
    ax.margins(x=0.08)
    add_panel_label(
        ax,
        panel_label,
    )



def draw_direction_panel(
    ax: plt.Axes,
    data: dict[str, pd.DataFrame],
    panel_label: str,
) -> None:
    summary = data["summary"]
    x_values = finite_array(
        summary["mean_reference_bandwidth"],
        "mean_reference_bandwidth",
    )

    ax.errorbar(
        x_values,
        summary[
            "selected_median_direction_error_deg_mean"
        ],
        yerr=summary[
            "selected_median_direction_error_deg_std"
        ],
        marker="o",
        capsize=4,
        label="Diffusion spectral",
    )
    ax.errorbar(
        x_values,
        summary[
            "local_median_direction_error_deg_mean"
        ],
        yerr=summary[
            "local_median_direction_error_deg_std"
        ],
        marker="o",
        capsize=4,
        label="Adaptive local kernel",
    )

    ax.set_xlabel(
        r"Median reference smoothing scale "
        r"$[h^{-1}\,\mathrm{Mpc}]$"
    )
    ax.set_ylabel(
        "Median direction error [deg]"
    )
    ax.set_title(
        "Directional accuracy at practical truncation"
    )
    ax.legend(
        loc="upper right",
        frameon=True,
    )
    ax.margins(x=0.08)
    add_panel_label(
        ax,
        panel_label,
    )



def draw_calibration_panel(
    ax: plt.Axes,
    data: dict[str, pd.DataFrame],
    panel_label: str,
) -> None:
    summary = data["summary"]
    x_values = finite_array(
        summary["mean_reference_bandwidth"],
        "mean_reference_bandwidth",
    )

    for component in [
        "x",
        "y",
        "z",
    ]:
        ax.errorbar(
            x_values,
            summary[
                f"selected_calibration_slope_{component}_mean"
            ],
            yerr=summary[
                f"selected_calibration_slope_{component}_std"
            ],
            marker="o",
            capsize=4,
            label=rf"$v_{component}$",
        )

    ax.axhline(
        1.0,
        linestyle="--",
        linewidth=1.0,
        label="unity",
    )
    ax.set_xlabel(
        r"Median reference smoothing scale "
        r"$[h^{-1}\,\mathrm{Mpc}]$"
    )
    ax.set_ylabel(
        "Calibration slope"
    )
    ax.set_title(
        "Cartesian-component calibration"
    )
    ax.legend(
        loc="lower right",
        ncol=2,
        frameon=True,
    )
    ax.margins(x=0.08)
    add_panel_label(
        ax,
        panel_label,
    )



def draw_paired_panel(
    ax: plt.Axes,
    data: dict[str, pd.DataFrame],
    panel_label: str,
) -> None:
    paired = data["paired"]
    summary = data["summary"]

    positions = np.arange(
        len(SCALE_ORDER),
        dtype=np.float64,
    )
    jitter = np.linspace(
        -0.12,
        0.12,
        5,
    )

    means = summary[
        "diffusion_minus_local_nrmse_mean"
    ].to_numpy(
        dtype=np.float64
    )
    standard_deviations = summary[
        "diffusion_minus_local_nrmse_std"
    ].to_numpy(
        dtype=np.float64
    )

    mean_artist = ax.errorbar(
        positions,
        means,
        yerr=standard_deviations,
        marker="D",
        capsize=5,
        linewidth=1.5,
        label=r"Mean $\pm$ 1 s.d.",
    )

    point_color = mean_artist.lines[0].get_color()

    for scale_index, scale in enumerate(SCALE_ORDER):
        selected = paired.loc[
            paired["scale"] == scale
        ].sort_values(
            "replicate"
        )
        values = selected[
            "diffusion_minus_local_nrmse"
        ].to_numpy(
            dtype=np.float64
        )

        ax.scatter(
            positions[scale_index]
            + jitter,
            values,
            s=34,
            color=point_color,
            alpha=0.75,
        )

    ax.axhline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )

    tick_labels = []
    for row in summary.itertuples(
        index=False
    ):
        tick_labels.append(
            f"{row.scale_label}\n"
            f"{row.mean_reference_bandwidth:.2f}\n"
            rf"$n_{{\mathrm{{b}}}}={int(row.selected_basis)}$"
        )

    ax.set_xticks(
        positions
    )
    ax.set_xticklabels(
        tick_labels
    )
    ax.set_xlabel(
        r"Reference smoothing scale "
        r"$[h^{-1}\,\mathrm{Mpc}]$"
    )
    ax.set_ylabel(
        r"Paired test $\mathrm{NRMSE}_{3\mathrm{D}}$ difference "
        r"(diffusion $-$ local)"
    )
    ax.set_title(
        "Paired method comparison"
    )
    ax.legend(
        loc="upper right",
        frameon=True,
    )
    ax.text(
        0.03,
        0.04,
        "Negative values favor diffusion spectral",
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=9.2,
    )
    add_panel_label(
        ax,
        panel_label,
    )


# ============================================================
# 6. Individual and composite figures
# ============================================================


def make_individual_figures(
    data: dict[str, pd.DataFrame],
) -> dict[str, str]:
    outputs: dict[str, str] = {}

    panel_specs = [
        (
            draw_nrmse_panel,
            PANEL_LABELS[0],
            "nrmse",
        ),
        (
            draw_direction_panel,
            PANEL_LABELS[1],
            "direction",
        ),
        (
            draw_calibration_panel,
            PANEL_LABELS[2],
            "calibration",
        ),
        (
            draw_paired_panel,
            PANEL_LABELS[3],
            "paired_difference",
        ),
    ]

    for draw_function, panel_label, suffix in panel_specs:
        fig, ax = plt.subplots(
            figsize=PANEL_FIGSIZE,
            constrained_layout=True,
        )
        draw_function(
            ax,
            data,
            panel_label,
        )
        pdf_path, png_path = save_figure_pair(
            fig,
            f"{OUTPUT_PREFIX}_{suffix}",
        )
        outputs[f"{suffix}_pdf"] = (
            str(pdf_path)
            if pdf_path is not None
            else ""
        )
        outputs[f"{suffix}_png"] = (
            str(png_path)
            if png_path is not None
            else ""
        )

    return outputs



def make_composite_figure(
    data: dict[str, pd.DataFrame],
) -> tuple[Path | None, Path | None]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=COMPOSITE_FIGSIZE,
        constrained_layout=True,
    )

    draw_nrmse_panel(
        axes[0, 0],
        data,
        PANEL_LABELS[0],
    )
    draw_direction_panel(
        axes[0, 1],
        data,
        PANEL_LABELS[1],
    )
    draw_calibration_panel(
        axes[1, 0],
        data,
        PANEL_LABELS[2],
    )
    draw_paired_panel(
        axes[1, 1],
        data,
        PANEL_LABELS[3],
    )

    return save_figure_pair(
        fig,
        f"{OUTPUT_PREFIX}_composite",
    )



def write_latex_template() -> Path:
    path = (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_figure_template.tex"
    )

    text = (
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        "\\begin{minipage}[t]{0.49\\textwidth}\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{OUTPUT_PREFIX}_nrmse}}\n"
        "\\end{minipage}\n"
        "\\hfill\n"
        "\\begin{minipage}[t]{0.49\\textwidth}\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{OUTPUT_PREFIX}_direction}}\n"
        "\\end{minipage}\n"
        "\\par\n"
        "\\vspace{0.5em}\n"
        "\\begin{minipage}[t]{0.49\\textwidth}\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{OUTPUT_PREFIX}_calibration}}\n"
        "\\end{minipage}\n"
        "\\hfill\n"
        "\\begin{minipage}[t]{0.49\\textwidth}\n"
        "\\centering\n"
        f"\\includegraphics[width=\\linewidth]{{{OUTPUT_PREFIX}_paired_difference}}\n"
        "\\end{minipage}\n"
        "\\caption{CAPTION TO BE ADDED AFTER VISUAL INSPECTION.}\n"
        "\\label{fig:mock1_scale_dependence}\n"
        "\\end{figure*}\n"
    )

    path.write_text(
        text,
        encoding="utf-8",
    )

    return path


# ============================================================
# 7. Main
# ============================================================

font_name = configure_font()
data = load_inputs()
summary = data["summary"]
paired = data["paired"]

print("=" * 108)
print("Paper I Figure 4: Mock-1 scale dependence at practical mode truncation")
print("=" * 108)
print(f"Python                   : {platform.python_version()}")
print(f"Matplotlib font          : {font_name}")
print(f"Mode-convergence runs    : {MODE_RUNS_FILE}")
print(f"Practical selection      : {PRACTICAL_SELECTION_FILE}")
print(f"Local-baseline runs      : {LOCAL_RUNS_FILE}")
print(f"Output                   : {OUTPUT_DIR}")

individual_outputs = make_individual_figures(
    data
)
composite_pdf, composite_png = make_composite_figure(
    data
)
latex_template = write_latex_template()

summary_csv = (
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_summary.csv"
)
paired_csv = (
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_paired.csv"
)
summary_json = (
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_summary.json"
)

summary.to_csv(
    summary_csv,
    index=False,
)
paired.to_csv(
    paired_csv,
    index=False,
)

payload = {
    "inputs": {
        "mode_runs_csv": str(MODE_RUNS_FILE),
        "practical_selection_csv": str(PRACTICAL_SELECTION_FILE),
        "local_runs_csv": str(LOCAL_RUNS_FILE),
    },
    "summary": summary.to_dict(
        orient="records"
    ),
    "outputs": {
        **individual_outputs,
        "composite_pdf": (
            str(composite_pdf)
            if composite_pdf is not None
            else ""
        ),
        "composite_png": (
            str(composite_png)
            if composite_png is not None
            else ""
        ),
        "summary_csv": str(summary_csv),
        "paired_csv": str(paired_csv),
        "latex_template": str(latex_template),
    },
}

summary_json.write_text(
    json.dumps(
        payload,
        indent=2,
        ensure_ascii=False,
        default=str,
    ),
    encoding="utf-8",
)

print("-" * 108)
for row in summary.itertuples(index=False):
    print(
        f"{row.scale_label:9s}: "
        f"n_b={int(row.selected_basis):4d}, "
        f"h_ref={row.mean_reference_bandwidth:.3f}, "
        f"diffusion={row.selected_normalized_rmse_mean:.5f}"
        f" +/- {row.selected_normalized_rmse_std:.5f}, "
        f"local={row.local_normalized_rmse_mean:.5f}"
        f" +/- {row.local_normalized_rmse_std:.5f}, "
        f"paired={row.diffusion_minus_local_nrmse_mean:+.5f}"
        f" +/- {row.diffusion_minus_local_nrmse_std:.5f}, "
        f"wins={int(row.diffusion_win_count)}/5"
    )

print(f"Composite PDF            : {composite_pdf}")
print(f"Composite PNG            : {composite_png}")
print(f"Summary CSV              : {summary_csv}")
print(f"Paired CSV               : {paired_csv}")
print(f"LaTeX template           : {latex_template}")
print(f"Summary JSON             : {summary_json}")
print("=" * 108)
