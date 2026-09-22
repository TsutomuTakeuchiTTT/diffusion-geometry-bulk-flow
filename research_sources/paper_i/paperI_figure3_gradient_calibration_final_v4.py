from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# ============================================================
# 1. 入出力
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

DIAGNOSTIC_CANDIDATES = [
    (
        ROOT_DIR / "mock1_potential_flow_constrained_v4"
        / "mock1_potential_flow_constrained_gradient_diagnostics.csv"
    ),
    (
        ROOT_DIR / "mock1_potential_flow_full_robustness_v5"
        / "mock1_potential_flow_full_robustness_gradient_diagnostics.csv"
    ),
]

GRADIENT_CACHE_FILE = (
    ROOT_DIR
    / "mock1_potential_flow_gradient_basis_cache_v4.npz"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_FIGURE3_OUTPUT",
        str(
            ROOT_DIR
            / "paperI_figures"
            / "figure3_gradient_calibration"
        ),
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PREFIX = "paperI_fig3_gradient_calibration_final_v4"

SAVE_PDF = True
SAVE_PNG = True
SHOW_PLOTS = os.environ.get("PAPER1_FIGURE3_SHOW_PLOTS", "1") != "0"
PNG_DPI = 300


# ============================================================
# 2. 描画設定
# ============================================================

COMPOSITE_FIGSIZE = (12.0, 5.2)
SINGLE_PANEL_FIGSIZE = (6.2, 4.9)
N_CONDITION_BINS = 50

PANEL_LABELS = ["(a)", "(b)"]


# ============================================================
# 3. 基本関数
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
    mpl.rcParams.update(
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



def first_existing_file(
    candidates: list[Path],
) -> Path:
    for path in candidates:
        if path.is_file():
            return path
    raise FileNotFoundError(
        "Gradient-diagnostics CSVが見つかりません。\n"
        + "\n".join(
            str(path)
            for path in candidates
        )
    )



def finite_positive(
    values: np.ndarray,
    name: str,
) -> np.ndarray:
    array = np.asarray(
        values,
        dtype=np.float64,
    )
    selected = array[
        np.isfinite(array)
        & (array > 0.0)
    ]
    if selected.size == 0:
        raise ValueError(
            f"{name}に有限かつ正の値がありません。"
        )
    return selected



def add_panel_label(
    ax: plt.Axes,
    label: str,
) -> None:
    ax.text(
        -0.06,
        1.02,
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
            pad_inches=0.04,
        )

    if png_path is not None:
        fig.savefig(
            png_path,
            dpi=PNG_DPI,
            bbox_inches="tight",
            pad_inches=0.04,
        )

    return (
        pdf_path,
        png_path,
    )


# ============================================================
# 4. 入力読込み
# ============================================================


def load_diagnostics() -> dict[str, object]:
    diagnostics_file = first_existing_file(
        DIAGNOSTIC_CANDIDATES
    )

    if not GRADIENT_CACHE_FILE.is_file():
        raise FileNotFoundError(
            "Gradient cacheが見つかりません。\n"
            f"{GRADIENT_CACHE_FILE}"
        )

    diagnostics_df = pd.read_csv(
        diagnostics_file
    )

    if "diagnostic" not in diagnostics_df.columns:
        raise KeyError(
            "Gradient-diagnostics CSVに"
            "'diagnostic'列がありません。"
        )

    coordinate_df = diagnostics_df.loc[
        diagnostics_df[
            "diagnostic"
        ]
        .astype(str)
        .str.startswith(
            "coordinate_"
        )
    ].copy()

    if coordinate_df.shape[0] != 3:
        raise ValueError(
            "coordinate_x, coordinate_y, coordinate_zの"
            "3行を取得できません。"
        )

    required_coordinate_columns = [
        "coordinate",
        "naive_rms_error",
        "calibrated_rms_error",
    ]
    missing_coordinate_columns = [
        column
        for column in required_coordinate_columns
        if column not in coordinate_df.columns
    ]
    if missing_coordinate_columns:
        raise KeyError(
            "Coordinate diagnosticsに必要な列がありません: "
            + ", ".join(
                missing_coordinate_columns
            )
        )

    coordinate_order = {
        "x": 0,
        "y": 1,
        "z": 2,
    }
    coordinate_df[
        "_order"
    ] = coordinate_df[
        "coordinate"
    ].map(
        coordinate_order
    )
    coordinate_df = (
        coordinate_df
        .sort_values(
            "_order"
        )
        .reset_index(
            drop=True
        )
    )

    affine_rows = diagnostics_df.loc[
        diagnostics_df[
            "diagnostic"
        ]
        == "affine_and_quadratic"
    ]
    if affine_rows.shape[0] != 1:
        raise ValueError(
            "affine_and_quadratic行を一意に取得できません。"
        )
    affine_row = affine_rows.iloc[0]

    metric_rows = diagnostics_df.loc[
        diagnostics_df[
            "diagnostic"
        ]
        == "metric_condition"
    ]
    if metric_rows.shape[0] != 1:
        raise ValueError(
            "metric_condition行を一意に取得できません。"
        )
    metric_row = metric_rows.iloc[0]

    required_affine_columns = [
        "affine_rms_error",
        "quadratic_gradient_nrmse",
    ]
    for column in required_affine_columns:
        if column not in diagnostics_df.columns:
            raise KeyError(
                f"Diagnostics CSVに{column}列がありません。"
            )

    required_metric_columns = [
        "metric_condition_median",
        "metric_condition_p90",
        "metric_condition_p99",
        "metric_condition_max",
        "metric_clipped_point_fraction",
    ]
    for column in required_metric_columns:
        if column not in diagnostics_df.columns:
            raise KeyError(
                f"Diagnostics CSVに{column}列がありません。"
            )

    with np.load(
        GRADIENT_CACHE_FILE,
        allow_pickle=False,
    ) as loaded:
        if "metric_condition" not in loaded.files:
            raise KeyError(
                "Gradient cacheにmetric_conditionがありません。"
            )
        metric_condition = np.asarray(
            loaded[
                "metric_condition"
            ],
            dtype=np.float64,
        ).reshape(-1)

    metric_condition = finite_positive(
        metric_condition,
        "metric_condition",
    )

    return {
        "diagnostics_file": diagnostics_file,
        "coordinate_df": coordinate_df,
        "affine_rms_error": float(
            affine_row[
                "affine_rms_error"
            ]
        ),
        "quadratic_gradient_nrmse": float(
            affine_row[
                "quadratic_gradient_nrmse"
            ]
        ),
        "constant_mode_gradient_rms": (
            float(
                affine_row[
                    "constant_mode_gradient_rms"
                ]
            )
            if (
                "constant_mode_gradient_rms"
                in diagnostics_df.columns
                and np.isfinite(
                    affine_row[
                        "constant_mode_gradient_rms"
                    ]
                )
            )
            else None
        ),
        "condition_median": float(
            metric_row[
                "metric_condition_median"
            ]
        ),
        "condition_p90": float(
            metric_row[
                "metric_condition_p90"
            ]
        ),
        "condition_p99": float(
            metric_row[
                "metric_condition_p99"
            ]
        ),
        "condition_max": float(
            metric_row[
                "metric_condition_max"
            ]
        ),
        "clipped_point_fraction": float(
            metric_row[
                "metric_clipped_point_fraction"
            ]
        ),
        "metric_condition": metric_condition,
    }


# ============================================================
# 5. Panel描画
# ============================================================


def draw_coordinate_panel(
    ax: plt.Axes,
    diagnostics: dict[str, object],
    panel_label: str,
) -> None:
    coordinate_df = diagnostics[
        "coordinate_df"
    ]
    if not isinstance(
        coordinate_df,
        pd.DataFrame,
    ):
        raise TypeError(
            "coordinate_dfがDataFrameではありません。"
        )

    x = np.arange(
        coordinate_df.shape[0]
    )
    width = 0.36

    raw_values = np.asarray(
        coordinate_df[
            "naive_rms_error"
        ],
        dtype=np.float64,
    )
    calibrated_values = np.asarray(
        coordinate_df[
            "calibrated_rms_error"
        ],
        dtype=np.float64,
    )

    if not np.all(
        np.isfinite(
            raw_values
        )
    ):
        raise FloatingPointError(
            "Raw coordinate-covector errorsに"
            "非有限値があります。"
        )
    if not np.all(
        np.isfinite(
            calibrated_values
        )
    ):
        raise FloatingPointError(
            "Calibrated-gradient errorsに"
            "非有限値があります。"
        )

    positive_values = finite_positive(
        np.concatenate(
            [
                raw_values,
                calibrated_values,
            ]
        ),
        "coordinate errors",
    )
    display_floor = (
        np.min(
            positive_values
        )
        / 3.0
    )

    raw_display = np.maximum(
        raw_values,
        display_floor,
    )
    calibrated_display = np.maximum(
        calibrated_values,
        display_floor,
    )

    raw_bars = ax.bar(
        x - width / 2.0,
        raw_display,
        width,
        label="raw coordinate covector",
    )
    calibrated_bars = ax.bar(
        x + width / 2.0,
        calibrated_display,
        width,
        label="metric-calibrated gradient",
    )

    ax.set_yscale(
        "log"
    )
    ax.set_xticks(
        x
    )
    ax.set_xticklabels(
        [
            r"$\nabla x$",
            r"$\nabla y$",
            r"$\nabla z$",
        ]
    )
    ax.set_ylabel(
        "RMS vector error"
    )
    ax.set_title(
        "Coordinate-gradient calibration"
    )

    # The two legend entries are stacked vertically in an intentionally
    # reserved blank region at the middle right, so that no quantitative
    # bar is obscured.
    ax.set_xlim(
        -0.55,
        4.35,
    )
    ax.legend(
        handles=[
            raw_bars,
            calibrated_bars,
        ],
        labels=[
            "raw coordinate covector",
            "metric-calibrated gradient",
        ],
        loc="center right",
        bbox_to_anchor=(0.98, 0.53),
        ncol=1,
        frameon=True,
        fontsize=9.3,
        handlelength=2.0,
        handletextpad=0.55,
        borderaxespad=0.25,
        labelspacing=0.55,
    )

    add_panel_label(
        ax,
        panel_label,
    )



def draw_condition_panel(
    ax: plt.Axes,
    diagnostics: dict[str, object],
    panel_label: str,
) -> None:
    condition = np.asarray(
        diagnostics[
            "metric_condition"
        ],
        dtype=np.float64,
    )

    condition_min = float(
        np.min(
            condition
        )
    )
    condition_max_array = float(
        np.max(
            condition
        )
    )

    bins = np.geomspace(
        condition_min,
        condition_max_array,
        N_CONDITION_BINS + 1,
    )

    ax.hist(
        condition,
        bins=bins,
    )
    ax.set_xscale(
        "log"
    )
    ax.set_xlabel(
        r"Local Gram-matrix condition number $\kappa_i$"
    )
    ax.set_ylabel(
        "Number of tracers"
    )
    ax.set_title(
        "Local Gram-matrix conditioning"
    )

    median = float(
        diagnostics[
            "condition_median"
        ]
    )
    p90 = float(
        diagnostics[
            "condition_p90"
        ]
    )
    p99 = float(
        diagnostics[
            "condition_p99"
        ]
    )

    median_line = ax.axvline(
        median,
        linestyle="-",
        linewidth=1.2,
        label=rf"median $={median:.3g}$",
    )
    p90_line = ax.axvline(
        p90,
        linestyle="--",
        linewidth=1.2,
        label=rf"p90 $={p90:.3g}$",
    )
    p99_line = ax.axvline(
        p99,
        linestyle=":",
        linewidth=1.5,
        label=rf"p99 $={p99:.3g}$",
    )
    ax.legend(
        handles=[
            median_line,
            p99_line,
            p90_line,
        ],
        labels=[
            rf"median $={median:.3g}$",
            rf"p99 $={p99:.3g}$",
            rf"p90 $={p90:.3g}$",
        ],
        loc="center right",
        bbox_to_anchor=(0.98, 0.53),
        ncol=2,
        frameon=True,
        fontsize=9.1,
        columnspacing=0.9,
        handlelength=2.0,
        handletextpad=0.45,
        borderaxespad=0.25,
    )

    clipped_fraction = float(
        diagnostics[
            "clipped_point_fraction"
        ]
    )
    diagnostic_max = float(
        diagnostics[
            "condition_max"
        ]
    )

    ax.text(
        0.97,
        0.95,
        (
            rf"max $\kappa_i={diagnostic_max:.3g}$"
            "\n"
            rf"clipped-point fraction $={clipped_fraction:.3e}$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.3,
        bbox={
            "boxstyle": "round",
            "facecolor": "white",
            "alpha": 0.88,
        },
    )

    add_panel_label(
        ax,
        panel_label,
    )


# ============================================================
# 6. 出力
# ============================================================


def make_composite(
    diagnostics: dict[str, object],
) -> tuple[Path | None, Path | None]:
    fig, axes = plt.subplots(
        1,
        2,
        figsize=COMPOSITE_FIGSIZE,
        constrained_layout=True,
    )

    draw_coordinate_panel(
        axes[0],
        diagnostics,
        PANEL_LABELS[0],
    )
    draw_condition_panel(
        axes[1],
        diagnostics,
        PANEL_LABELS[1],
    )

    paths = save_figure_pair(
        fig,
        f"{OUTPUT_PREFIX}_composite",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(
            fig
        )

    return paths



def make_individual_panels(
    diagnostics: dict[str, object],
) -> dict[str, str]:
    outputs: dict[str, str] = {}

    fig_a, ax_a = plt.subplots(
        figsize=SINGLE_PANEL_FIGSIZE,
        constrained_layout=True,
    )
    draw_coordinate_panel(
        ax_a,
        diagnostics,
        PANEL_LABELS[0],
    )
    pdf_a, png_a = save_figure_pair(
        fig_a,
        f"{OUTPUT_PREFIX}_coordinate_calibration",
    )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(
            fig_a
        )
    outputs[
        "coordinate_pdf"
    ] = (
        str(pdf_a)
        if pdf_a is not None
        else ""
    )
    outputs[
        "coordinate_png"
    ] = (
        str(png_a)
        if png_a is not None
        else ""
    )

    fig_b, ax_b = plt.subplots(
        figsize=SINGLE_PANEL_FIGSIZE,
        constrained_layout=True,
    )
    draw_condition_panel(
        ax_b,
        diagnostics,
        PANEL_LABELS[1],
    )
    pdf_b, png_b = save_figure_pair(
        fig_b,
        f"{OUTPUT_PREFIX}_metric_condition",
    )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(
            fig_b
        )
    outputs[
        "condition_pdf"
    ] = (
        str(pdf_b)
        if pdf_b is not None
        else ""
    )
    outputs[
        "condition_png"
    ] = (
        str(png_b)
        if png_b is not None
        else ""
    )

    return outputs


# ============================================================
# 7. Main
# ============================================================

font_name = configure_font()
diagnostics = load_diagnostics()

print("=" * 100)
print("Paper I Figure 3: metric-calibrated gradient diagnostics")
print("=" * 100)
print(f"Python                   : {platform.python_version()}")
print(f"Matplotlib font          : {font_name}")
print(f"Diagnostics CSV          : {diagnostics['diagnostics_file']}")
print(f"Gradient cache           : {GRADIENT_CACHE_FILE}")
print(f"Metric-condition points  : {len(diagnostics['metric_condition']):,}")
print(f"Output                   : {OUTPUT_DIR}")
print(
    "Coordinate RMS errors    : "
    + ", ".join(
        f"{row.coordinate} raw={row.naive_rms_error:.3e}, calibrated={row.calibrated_rms_error:.3e}"
        for row in diagnostics["coordinate_df"].itertuples(index=False)
    )
)
print(f"Affine-gradient RMS      : {diagnostics['affine_rms_error']:.3e}")
print(f"Quadratic-gradient NRMSE : {diagnostics['quadratic_gradient_nrmse']:.4f}")
if diagnostics["constant_mode_gradient_rms"] is not None:
    print(f"Constant-mode grad RMS   : {diagnostics['constant_mode_gradient_rms']:.3e}")
print(f"Condition median         : {diagnostics['condition_median']:.3g}")
print(f"Condition p90            : {diagnostics['condition_p90']:.3g}")
print(f"Condition p99            : {diagnostics['condition_p99']:.3g}")
print(f"Condition max            : {diagnostics['condition_max']:.3g}")
print(f"Clipped-point fraction   : {diagnostics['clipped_point_fraction']:.3e}")

composite_pdf, composite_png = make_composite(
    diagnostics
)
individual_outputs = make_individual_panels(
    diagnostics
)

coordinate_df = diagnostics[
    "coordinate_df"
]
if not isinstance(
    coordinate_df,
    pd.DataFrame,
):
    raise TypeError(
        "coordinate_dfがDataFrameではありません。"
    )

summary = {
    "diagnostics_file": str(
        diagnostics[
            "diagnostics_file"
        ]
    ),
    "gradient_cache_file": str(
        GRADIENT_CACHE_FILE
    ),
    "coordinate_records": (
        coordinate_df[
            [
                "coordinate",
                "naive_rms_error",
                "calibrated_rms_error",
            ]
        ]
        .to_dict(
            orient="records"
        )
    ),
    "affine_rms_error": float(
        diagnostics[
            "affine_rms_error"
        ]
    ),
    "quadratic_gradient_nrmse": float(
        diagnostics[
            "quadratic_gradient_nrmse"
        ]
    ),
    "constant_mode_gradient_rms": (
        diagnostics[
            "constant_mode_gradient_rms"
        ]
    ),
    "metric_condition": {
        "n": int(
            len(
                diagnostics[
                    "metric_condition"
                ]
            )
        ),
        "median": float(
            diagnostics[
                "condition_median"
            ]
        ),
        "p90": float(
            diagnostics[
                "condition_p90"
            ]
        ),
        "p99": float(
            diagnostics[
                "condition_p99"
            ]
        ),
        "max": float(
            diagnostics[
                "condition_max"
            ]
        ),
        "clipped_point_fraction": float(
            diagnostics[
                "clipped_point_fraction"
            ]
        ),
    },
    "outputs": {
        "composite_pdf": (
            str(
                composite_pdf
            )
            if composite_pdf is not None
            else ""
        ),
        "composite_png": (
            str(
                composite_png
            )
            if composite_png is not None
            else ""
        ),
        **individual_outputs,
    },
}

summary_file = (
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_summary.json"
)
summary_file.write_text(
    json.dumps(
        summary,
        indent=2,
        ensure_ascii=False,
    ),
    encoding="utf-8",
)

print("-" * 100)
print(f"Composite PDF            : {composite_pdf}")
print(f"Composite PNG            : {composite_png}")
print(f"Summary                  : {summary_file}")
print("=" * 100)
