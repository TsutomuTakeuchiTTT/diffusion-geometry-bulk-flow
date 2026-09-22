from __future__ import annotations

import json
import os
import platform
from pathlib import Path

import matplotlib as mpl
import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from mpl_toolkits.axes_grid1 import make_axes_locatable


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

DATA_FILE = ROOT_DIR / "mock1_complete_sphere" / "mock1.npz"
GEOMETRY_CACHE_FILE = ROOT_DIR / "mock1_diffusion_geometry_bulk_cache_m2048.npz"

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_FIGURE2_OUTPUT",
        str(ROOT_DIR / "paperI_figures" / "figure2_diffusion_modes"),
    )
)

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

OUTPUT_PREFIX = "paperI_fig2_diffusion_modes_m2048_final"

SAVE_PDF = True
SAVE_PNG = True
SHOW_PLOTS = True


# ============================================================
# 2. 可視化設定
# ============================================================

SPHERE_CENTER = np.array(
    [250.0, 250.0, 250.0],
    dtype=np.float64,
)

# 0, 1, 2 はそれぞれ x, y, z に対応する。
SLAB_AXIS = 2
SLAB_HALF_THICKNESS = 12.5
MIN_SLAB_POINTS = 700
MAX_SLAB_HALF_THICKNESS = 30.0
SLAB_EXPANSION_STEP = 2.5

# 定数 mode を除いた低周波 mode の順位。
NONTRIVIAL_MODE_RANKS = [1, 8, 32]

ROBUST_SCALE_QUANTILE = 0.99
POINT_SIZE = 8.0
POINT_ALPHA = 0.92
PANEL_FIGSIZE = (5.7, 5.1)
SPECTRUM_FIGSIZE = (5.7, 5.1)
COMPOSITE_FIGSIZE = (11.6, 10.0)
PNG_DPI = 300
ORTHOGONALITY_AUDIT_MODES = 128

POSITION_LABELS = [
    r"$x\ [h^{-1}\,\mathrm{Mpc}]$",
    r"$y\ [h^{-1}\,\mathrm{Mpc}]$",
    r"$z\ [h^{-1}\,\mathrm{Mpc}]$",
]

PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


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
            "font.size": 12.0,
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


def first_existing_key(
    data: np.lib.npyio.NpzFile,
    candidates: list[str],
) -> str:
    for key in candidates:
        if key in data.files:
            return key
    raise KeyError(
        "次の候補 key が入力に見つかりません: "
        + ", ".join(candidates)
    )


def load_inputs() -> tuple[
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray | None,
    np.ndarray | None,
]:
    missing = [
        str(path)
        for path in [
            DATA_FILE,
            GEOMETRY_CACHE_FILE,
        ]
        if not path.is_file()
    ]
    if missing:
        raise FileNotFoundError(
            "必要な入力ファイルが見つかりません。\n"
            + "\n".join(missing)
        )

    with np.load(
        DATA_FILE,
        allow_pickle=False,
    ) as data:
        position_key = first_existing_key(
            data,
            [
                "pos",
                "positions",
                "position",
                "x",
            ],
        )
        positions_absolute = np.asarray(
            data[position_key],
            dtype=np.float64,
        )

    if (
        positions_absolute.ndim != 2
        or positions_absolute.shape[1] != 3
    ):
        raise ValueError(
            "銀河位置は shape (N, 3) でなければなりません。"
        )

    with np.load(
        GEOMETRY_CACHE_FILE,
        allow_pickle=False,
    ) as data:
        eigenfunction_key = first_existing_key(
            data,
            [
                "eigenfunctions",
                "phi",
                "diffusion_eigenfunctions",
            ],
        )
        eigenfunctions = np.asarray(
            data[eigenfunction_key],
            dtype=np.float64,
        )

        if "generator_eigenvalues" in data.files:
            generator_eigenvalues = np.asarray(
                data["generator_eigenvalues"],
                dtype=np.float64,
            )
        elif "eigenvalues_markov" in data.files:
            markov_eigenvalues = np.asarray(
                data["eigenvalues_markov"],
                dtype=np.float64,
            )
            generator_eigenvalues = np.maximum(
                1.0 - markov_eigenvalues,
                0.0,
            )
        elif "eigenvalues" in data.files:
            markov_eigenvalues = np.asarray(
                data["eigenvalues"],
                dtype=np.float64,
            )
            generator_eigenvalues = np.maximum(
                1.0 - markov_eigenvalues,
                0.0,
            )
        else:
            raise KeyError(
                "generator_eigenvalues または "
                "Markov eigenvalues が cache にありません。"
            )

        stationary_measure = (
            np.asarray(
                data["stationary_measure"],
                dtype=np.float64,
            )
            if "stationary_measure" in data.files
            else None
        )

        markov_eigenvalues = None
        for key in [
            "eigenvalues_markov",
            "markov_eigenvalues",
            "eigenvalues",
        ]:
            if key in data.files:
                markov_eigenvalues = np.asarray(
                    data[key],
                    dtype=np.float64,
                )
                break

    n_objects = positions_absolute.shape[0]

    if eigenfunctions.ndim != 2:
        raise ValueError(
            "eigenfunctions は2次元配列でなければなりません。"
        )

    if eigenfunctions.shape[0] != n_objects:
        if eigenfunctions.shape[1] == n_objects:
            eigenfunctions = eigenfunctions.T
        else:
            raise ValueError(
                "eigenfunctions の点数が mock catalog と一致しません。"
            )

    n_modes = eigenfunctions.shape[1]

    generator_eigenvalues = np.ravel(
        generator_eigenvalues
    )
    if generator_eigenvalues.size < n_modes:
        raise ValueError(
            "generator_eigenvalues の個数が "
            "eigenfunctions の mode 数より少ないです。"
        )
    generator_eigenvalues = generator_eigenvalues[
        :n_modes
    ]

    if markov_eigenvalues is not None:
        markov_eigenvalues = np.ravel(
            markov_eigenvalues
        )[:n_modes]

    if stationary_measure is not None:
        stationary_measure = np.ravel(
            stationary_measure
        )
        if stationary_measure.size != n_objects:
            stationary_measure = None

    if not np.all(
        np.isfinite(
            positions_absolute
        )
    ):
        raise FloatingPointError(
            "positions に非有限値があります。"
        )
    if not np.all(
        np.isfinite(
            eigenfunctions
        )
    ):
        raise FloatingPointError(
            "eigenfunctions に非有限値があります。"
        )
    if not np.all(
        np.isfinite(
            generator_eigenvalues
        )
    ):
        raise FloatingPointError(
            "generator_eigenvalues に非有限値があります。"
        )

    centered_positions = (
        positions_absolute
        - SPHERE_CENTER[None, :]
    )

    return (
        centered_positions,
        eigenfunctions,
        generator_eigenvalues,
        markov_eigenvalues,
        stationary_measure,
    )


def determine_constant_mode(
    generator_eigenvalues: np.ndarray,
) -> int:
    return int(
        np.argmin(
            generator_eigenvalues
        )
    )


def ordered_nonconstant_modes(
    generator_eigenvalues: np.ndarray,
    constant_mode_index: int,
) -> np.ndarray:
    ordering = np.argsort(
        generator_eigenvalues,
        kind="stable",
    )
    nonconstant = ordering[
        ordering != constant_mode_index
    ]
    return nonconstant.astype(
        np.int64,
        copy=False,
    )


def choose_selected_modes(
    generator_eigenvalues: np.ndarray,
) -> tuple[int, np.ndarray]:
    constant_mode_index = determine_constant_mode(
        generator_eigenvalues
    )
    nonconstant = ordered_nonconstant_modes(
        generator_eigenvalues,
        constant_mode_index,
    )

    maximum_rank = max(
        NONTRIVIAL_MODE_RANKS
    )
    if nonconstant.size < maximum_rank:
        raise ValueError(
            "指定した非自明 mode rank を選べるだけの "
            "mode 数がありません。"
        )

    selected = np.array(
        [
            nonconstant[rank - 1]
            for rank in NONTRIVIAL_MODE_RANKS
        ],
        dtype=np.int64,
    )

    return (
        constant_mode_index,
        selected,
    )


def orient_mode(
    values: np.ndarray,
) -> np.ndarray:
    result = np.asarray(
        values,
        dtype=np.float64,
    ).copy()

    pivot = int(
        np.argmax(
            np.abs(
                result
            )
        )
    )
    if result[pivot] < 0.0:
        result *= -1.0

    return result


def choose_slab(
    centered_positions: np.ndarray,
) -> tuple[np.ndarray, float]:
    half_thickness = float(
        SLAB_HALF_THICKNESS
    )

    while True:
        mask = (
            np.abs(
                centered_positions[
                    :,
                    SLAB_AXIS,
                ]
            )
            <= half_thickness
        )

        if (
            np.sum(mask)
            >= MIN_SLAB_POINTS
            or half_thickness
            >= MAX_SLAB_HALF_THICKNESS
        ):
            return (
                mask,
                half_thickness,
            )

        half_thickness += (
            SLAB_EXPANSION_STEP
        )


def plane_axes(
    slab_axis: int,
) -> tuple[int, int]:
    axes = [
        axis
        for axis in range(3)
        if axis != slab_axis
    ]
    return (
        axes[0],
        axes[1],
    )


def robust_scaled_values(
    values: np.ndarray,
    mask: np.ndarray,
) -> tuple[np.ndarray, float]:
    slab_values = values[mask]
    scale = float(
        np.quantile(
            np.abs(
                slab_values
            ),
            ROBUST_SCALE_QUANTILE,
        )
    )

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        scale = float(
            np.max(
                np.abs(
                    slab_values
                )
            )
        )

    if (
        not np.isfinite(scale)
        or scale <= 0.0
    ):
        raise FloatingPointError(
            "固有関数の表示 scale を決定できません。"
        )

    scaled = np.clip(
        values / scale,
        -1.0,
        1.0,
    )
    return (
        scaled,
        scale,
    )


def common_spatial_limit(
    centered_positions: np.ndarray,
    axis_x: int,
    axis_y: int,
) -> float:
    maximum = float(
        np.max(
            np.abs(
                centered_positions[
                    :,
                    [axis_x, axis_y],
                ]
            )
        )
    )
    return float(
        np.ceil(
            maximum / 10.0
        )
        * 10.0
    )


def add_panel_label(
    ax: plt.Axes,
    label: str,
) -> None:
    ax.text(
        0.02,
        0.98,
        label,
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14.0,
        fontweight="bold",
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
            pad_inches=0.03,
        )

    if png_path is not None:
        fig.savefig(
            png_path,
            dpi=PNG_DPI,
            bbox_inches="tight",
            pad_inches=0.03,
        )

    return (
        pdf_path,
        png_path,
    )


# ============================================================
# 4. 固有関数 panel
# ============================================================

def make_mode_panel(
    centered_positions: np.ndarray,
    eigenfunctions: np.ndarray,
    generator_eigenvalues: np.ndarray,
    slab_mask: np.ndarray,
    slab_half_thickness: float,
    mode_index: int,
    nontrivial_rank: int,
    panel_label: str,
    output_suffix: str,
    spatial_limit: float,
) -> dict[str, float | int | str]:
    axis_x, axis_y = plane_axes(
        SLAB_AXIS
    )

    oriented = orient_mode(
        eigenfunctions[
            :,
            mode_index,
        ]
    )

    scaled, robust_scale = (
        robust_scaled_values(
            oriented,
            slab_mask,
        )
    )

    x = centered_positions[
        slab_mask,
        axis_x,
    ]
    y = centered_positions[
        slab_mask,
        axis_y,
    ]
    color_values = scaled[
        slab_mask
    ]

    draw_order = np.argsort(
        np.abs(
            color_values
        ),
        kind="stable",
    )

    fig = plt.figure(
        figsize=PANEL_FIGSIZE,
        facecolor="white",
    )
    ax = fig.add_axes(
        [0.14, 0.13, 0.70, 0.78],
        facecolor="white",
    )

    scatter = ax.scatter(
        x[draw_order],
        y[draw_order],
        c=color_values[draw_order],
        s=POINT_SIZE,
        alpha=POINT_ALPHA,
        linewidths=0.0,
        norm=Normalize(
            vmin=-1.0,
            vmax=1.0,
        ),
        rasterized=True,
    )

    divider = make_axes_locatable(
        ax
    )
    colorbar_axis = divider.append_axes(
        "right",
        size="4.5%",
        pad=0.10,
    )
    colorbar_axis.set_facecolor(
        "white"
    )
    colorbar = fig.colorbar(
        scatter,
        cax=colorbar_axis,
    )
    colorbar.set_label(
        r"Display-scaled $\phi_a$",
        fontsize=11.0,
    )
    colorbar.ax.tick_params(
        labelsize=10.0,
    )

    ax.set_xlabel(
        POSITION_LABELS[
            axis_x
        ]
    )
    ax.set_ylabel(
        POSITION_LABELS[
            axis_y
        ]
    )
    ax.set_aspect(
        "equal",
        adjustable="box",
    )
    ax.set_xlim(
        -spatial_limit,
        spatial_limit,
    )
    ax.set_ylim(
        -spatial_limit,
        spatial_limit,
    )

    eta = float(
        generator_eigenvalues[
            mode_index
        ]
    )

    ax.set_title(
        (
            f"Rank {nontrivial_rank} "
            f"($a={mode_index}$, "
            f"$\\widetilde{{\\eta}}_a={eta:.4g}$)"
        ),
        fontsize=12.0,
    )

    add_panel_label(
        ax,
        panel_label,
    )

    axis_name = [
        "x",
        "y",
        "z",
    ][SLAB_AXIS]
    ax.text(
        0.98,
        0.02,
        (
            rf"$|{axis_name}|\leq"
            rf"{slab_half_thickness:g}\,"
            rf"h^{{-1}}\,\mathrm{{Mpc}}$"
        ),
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
    )

    output_stem = (
        f"{OUTPUT_PREFIX}_"
        f"{output_suffix}"
    )
    pdf_path, png_path = (
        save_figure_pair(
            fig,
            output_stem,
        )
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(
            fig
        )

    return {
        "panel": panel_label,
        "nontrivial_rank": int(
            nontrivial_rank
        ),
        "mode_index": int(
            mode_index
        ),
        "generator_weight_tilde": eta,
        "robust_scale": float(
            robust_scale
        ),
        "slab_points": int(
            np.sum(
                slab_mask
            )
        ),
        "pdf": (
            str(pdf_path)
            if pdf_path is not None
            else ""
        ),
        "png": (
            str(png_path)
            if png_path is not None
            else ""
        ),
    }


# ============================================================
# 5. Spectrum panel
# ============================================================

def make_spectrum_panel(
    generator_eigenvalues: np.ndarray,
    constant_mode_index: int,
    selected_mode_indices: np.ndarray,
) -> dict[str, str | int | float]:
    nonconstant = ordered_nonconstant_modes(
        generator_eigenvalues,
        constant_mode_index,
    )
    eta = generator_eigenvalues[
        nonconstant
    ]

    positive = (
        np.isfinite(
            eta
        )
        & (eta > 0.0)
    )
    if not np.any(
        positive
    ):
        raise ValueError(
            "正の generator eigenvalue がありません。"
        )

    ranks = np.arange(
        1,
        nonconstant.size + 1,
        dtype=np.int64,
    )

    fig = plt.figure(
        figsize=SPECTRUM_FIGSIZE,
        facecolor="white",
    )
    ax = fig.add_axes(
        [0.16, 0.14, 0.78, 0.77],
        facecolor="white",
    )

    ax.plot(
        ranks[positive],
        eta[positive],
        linewidth=1.8,
    )

    selected_ranks = []
    selected_eta = []

    for mode_index in selected_mode_indices:
        location = np.flatnonzero(
            nonconstant
            == mode_index
        )
        if location.size != 1:
            raise RuntimeError(
                "選択 mode の非自明 rank を決定できません。"
            )
        rank = int(
            location[0] + 1
        )
        selected_ranks.append(
            rank
        )
        selected_eta.append(
            float(
                generator_eigenvalues[
                    mode_index
                ]
            )
        )

    ax.scatter(
        selected_ranks,
        selected_eta,
        s=44.0,
        zorder=4,
    )

    for rank, value, mode_index in zip(
        selected_ranks,
        selected_eta,
        selected_mode_indices,
    ):
        ax.annotate(
            f"$a={int(mode_index)}$",
            xy=(
                rank,
                value,
            ),
            xytext=(
                8,
                10,
            ),
            textcoords="offset points",
            fontsize=9.5,
        )

    ax.set_yscale(
        "log"
    )
    ax.set_xlabel(
        "Nonconstant mode rank"
    )
    ax.set_ylabel(
        r"Normalized generator weight $\widetilde{\eta}_a$"
    )
    ax.set_title(
        "Normalized diffusion-generator spectrum",
        fontsize=12.0,
    )
    ax.set_xlim(
        -5.0,
        nonconstant.size + 0.5,
    )
    ax.set_xticks(
        [
            tick
            for tick in [1, 500, 1000, 1500, 2000]
            if tick <= nonconstant.size
        ]
    )
    add_panel_label(
        ax,
        PANEL_LABELS[3],
    )

    output_stem = (
        f"{OUTPUT_PREFIX}_spectrum"
    )
    pdf_path, png_path = (
        save_figure_pair(
            fig,
            output_stem,
        )
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(
            fig
        )

    return {
        "panel": PANEL_LABELS[3],
        "n_nonconstant_modes": int(
            nonconstant.size
        ),
        "minimum_positive_eta_tilde": float(
            np.min(
                eta[positive]
            )
        ),
        "maximum_eta_tilde": float(
            np.max(
                eta[positive]
            )
        ),
        "pdf": (
            str(pdf_path)
            if pdf_path is not None
            else ""
        ),
        "png": (
            str(png_path)
            if png_path is not None
            else ""
        ),
    }


# ============================================================
# 6. Composite figure
# ============================================================

def make_composite_figure(
    centered_positions: np.ndarray,
    eigenfunctions: np.ndarray,
    generator_eigenvalues: np.ndarray,
    constant_mode_index: int,
    selected_mode_indices: np.ndarray,
    slab_mask: np.ndarray,
    slab_half_thickness: float,
    spatial_limit: float,
) -> dict[str, str]:
    axis_x, axis_y = plane_axes(
        SLAB_AXIS
    )

    fig = plt.figure(
        figsize=COMPOSITE_FIGSIZE,
        facecolor="white",
    )
    gs = fig.add_gridspec(
        2,
        2,
        left=0.08,
        right=0.92,
        bottom=0.07,
        top=0.94,
        wspace=0.22,
        hspace=0.24,
    )

    map_axes = [
        fig.add_subplot(gs[0, 0], facecolor="white"),
        fig.add_subplot(gs[0, 1], facecolor="white"),
        fig.add_subplot(gs[1, 0], facecolor="white"),
    ]
    spec_ax = fig.add_subplot(gs[1, 1], facecolor="white")

    # --- (a)--(c): eigenfunction maps
    scatter_for_colorbar = None
    mode_records: list[dict[str, float | int]] = []

    for index, (rank, mode_index, label, ax) in enumerate(
        zip(
            NONTRIVIAL_MODE_RANKS,
            selected_mode_indices,
            PANEL_LABELS[:3],
            map_axes,
        )
    ):
        oriented = orient_mode(
            eigenfunctions[:, mode_index]
        )
        scaled, robust_scale = robust_scaled_values(
            oriented,
            slab_mask,
        )

        x = centered_positions[
            slab_mask,
            axis_x,
        ]
        y = centered_positions[
            slab_mask,
            axis_y,
        ]
        color_values = scaled[
            slab_mask
        ]
        draw_order = np.argsort(
            np.abs(
                color_values
            ),
            kind="stable",
        )

        scatter = ax.scatter(
            x[draw_order],
            y[draw_order],
            c=color_values[draw_order],
            s=POINT_SIZE,
            alpha=POINT_ALPHA,
            linewidths=0.0,
            norm=Normalize(
                vmin=-1.0,
                vmax=1.0,
            ),
            rasterized=True,
        )
        scatter_for_colorbar = scatter

        ax.set_aspect(
            "equal",
            adjustable="box",
        )
        ax.set_xlim(
            -spatial_limit,
            spatial_limit,
        )
        ax.set_ylim(
            -spatial_limit,
            spatial_limit,
        )
        ax.set_xlabel(
            POSITION_LABELS[axis_x]
        )
        ax.set_ylabel(
            POSITION_LABELS[axis_y]
        )

        eta = float(
            generator_eigenvalues[
                mode_index
            ]
        )
        ax.set_title(
            (
                f"Rank {rank} "
                f"($a={int(mode_index)}$, "
                f"$\\widetilde{{\\eta}}_a={eta:.4g}$)"
            ),
            fontsize=12.0,
        )
        add_panel_label(
            ax,
            label,
        )

        axis_name = [
            "x",
            "y",
            "z",
        ][SLAB_AXIS]
        ax.text(
            0.98,
            0.02,
            (
                rf"$|{axis_name}|\leq"
                rf"{slab_half_thickness:g}\,"
                rf"h^{{-1}}\,\mathrm{{Mpc}}$"
            ),
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=9.2,
        )

        mode_records.append(
            {
                "panel": label,
                "nontrivial_rank": int(rank),
                "mode_index": int(mode_index),
                "generator_weight_tilde": eta,
                "display_scale_99pct": float(robust_scale),
            }
        )

    if scatter_for_colorbar is None:
        raise RuntimeError(
            "Composite figure colorbar could not be initialized."
        )

    # shared colorbar for (a)--(c)
    colorbar = fig.colorbar(
        scatter_for_colorbar,
        ax=map_axes,
        fraction=0.025,
        pad=0.02,
        location="right",
    )
    colorbar.set_label(
        r"Display-scaled $\phi_a$",
        fontsize=11.5,
    )
    colorbar.ax.tick_params(
        labelsize=10.0,
    )

    # --- (d): spectrum
    nonconstant = ordered_nonconstant_modes(
        generator_eigenvalues,
        constant_mode_index,
    )
    eta = generator_eigenvalues[
        nonconstant
    ]
    positive = (
        np.isfinite(
            eta
        )
        & (eta > 0.0)
    )

    ranks = np.arange(
        1,
        nonconstant.size + 1,
        dtype=np.int64,
    )

    spec_ax.plot(
        ranks[positive],
        eta[positive],
        linewidth=1.8,
    )

    selected_ranks = []
    selected_eta = []
    for mode_index in selected_mode_indices:
        location = np.flatnonzero(
            nonconstant
            == mode_index
        )
        if location.size != 1:
            raise RuntimeError(
                "選択 mode の rank 決定に失敗しました。"
            )
        selected_ranks.append(
            int(location[0] + 1)
        )
        selected_eta.append(
            float(
                generator_eigenvalues[
                    mode_index
                ]
            )
        )

    spec_ax.scatter(
        selected_ranks,
        selected_eta,
        s=44.0,
        zorder=4,
    )
    for rank, value, mode_index in zip(
        selected_ranks,
        selected_eta,
        selected_mode_indices,
    ):
        spec_ax.annotate(
            f"$a={int(mode_index)}$",
            xy=(rank, value),
            xytext=(8, 10),
            textcoords="offset points",
            fontsize=9.5,
        )

    spec_ax.set_yscale(
        "log"
    )
    spec_ax.set_xlabel(
        "Nonconstant mode rank"
    )
    spec_ax.set_ylabel(
        r"Normalized generator weight $\widetilde{\eta}_a$"
    )
    spec_ax.set_title(
        "Normalized diffusion-generator spectrum",
        fontsize=12.0,
    )
    spec_ax.set_xlim(
        -5.0,
        nonconstant.size + 0.5,
    )
    spec_ax.set_xticks(
        [
            tick
            for tick in [1, 500, 1000, 1500, 2000]
            if tick <= nonconstant.size
        ]
    )
    add_panel_label(
        spec_ax,
        PANEL_LABELS[3],
    )

    output_stem = f"{OUTPUT_PREFIX}_composite"
    pdf_path, png_path = save_figure_pair(
        fig,
        output_stem,
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return {
        "pdf": str(pdf_path) if pdf_path is not None else "",
        "png": str(png_path) if png_path is not None else "",
    }


# ============================================================
# 7. Main
# ============================================================

font_name = configure_font()

(
    centered_positions,
    eigenfunctions,
    generator_eigenvalues,
    markov_eigenvalues,
    stationary_measure,
) = load_inputs()

(
    constant_mode_index,
    selected_mode_indices,
) = choose_selected_modes(
    generator_eigenvalues
)

slab_mask, slab_half_thickness = (
    choose_slab(
        centered_positions
    )
)

axis_x, axis_y = plane_axes(
    SLAB_AXIS
)
spatial_limit = common_spatial_limit(
    centered_positions,
    axis_x,
    axis_y,
)

print("=" * 100)
print("Paper I Figure 2: 2048-mode diffusion eigenmodes and normalized generator spectrum")
print("=" * 100)
print(f"Python                   : {platform.python_version()}")
print(f"Matplotlib font          : {font_name}")
print(f"Data file                : {DATA_FILE}")
print(f"Geometry cache           : {GEOMETRY_CACHE_FILE}")
print(f"Objects                  : {centered_positions.shape[0]:,}")
print(f"Stored modes             : {eigenfunctions.shape[1]:,}")
print(f"Constant mode index      : {constant_mode_index}")
print(f"Selected mode indices    : {selected_mode_indices.tolist()}")
print(f"Slab half-thickness      : {slab_half_thickness:g} h^-1 Mpc")
print(f"Slab points              : {np.sum(slab_mask):,}")
print(f"Output                   : {OUTPUT_DIR}")

if stationary_measure is not None:
    audit_modes = min(
        ORTHOGONALITY_AUDIT_MODES,
        eigenfunctions.shape[1],
    )
    audited_eigenfunctions = eigenfunctions[:, :audit_modes]
    weighted_gram = (
        audited_eigenfunctions.T
        @ (
            stationary_measure[:, None]
            * audited_eigenfunctions
        )
    )
    orthogonality_error = float(
        np.linalg.norm(
            weighted_gram
            - np.eye(audit_modes),
            ord="fro",
        )
        / np.sqrt(audit_modes)
    )
    print(
        "Low-mode weighted orthogonality RMS "
        f"(first {audit_modes} modes): "
        f"{orthogonality_error:.3e}"
    )
else:
    audit_modes = 0
    orthogonality_error = None

panel_records = []

for (
    rank,
    mode_index,
    label,
    suffix,
) in zip(
    NONTRIVIAL_MODE_RANKS,
    selected_mode_indices,
    PANEL_LABELS[:3],
    [
        "mode_rank001",
        "mode_rank008",
        "mode_rank032",
    ],
):
    panel_records.append(
        make_mode_panel(
            centered_positions=centered_positions,
            eigenfunctions=eigenfunctions,
            generator_eigenvalues=generator_eigenvalues,
            slab_mask=slab_mask,
            slab_half_thickness=slab_half_thickness,
            mode_index=int(mode_index),
            nontrivial_rank=int(rank),
            panel_label=label,
            output_suffix=suffix,
            spatial_limit=spatial_limit,
        )
    )

spectrum_record = make_spectrum_panel(
    generator_eigenvalues=generator_eigenvalues,
    constant_mode_index=constant_mode_index,
    selected_mode_indices=selected_mode_indices,
)

composite_record = make_composite_figure(
    centered_positions=centered_positions,
    eigenfunctions=eigenfunctions,
    generator_eigenvalues=generator_eigenvalues,
    constant_mode_index=constant_mode_index,
    selected_mode_indices=selected_mode_indices,
    slab_mask=slab_mask,
    slab_half_thickness=slab_half_thickness,
    spatial_limit=spatial_limit,
)

summary = {
    "data_file": str(DATA_FILE),
    "geometry_cache_file": str(GEOMETRY_CACHE_FILE),
    "n_objects": int(centered_positions.shape[0]),
    "n_modes": int(eigenfunctions.shape[1]),
    "constant_mode_index": int(constant_mode_index),
    "selected_nontrivial_ranks": [
        int(value)
        for value in NONTRIVIAL_MODE_RANKS
    ],
    "selected_mode_indices": [
        int(value)
        for value in selected_mode_indices
    ],
    "slab_axis": int(SLAB_AXIS),
    "slab_half_thickness": float(slab_half_thickness),
    "slab_points": int(np.sum(slab_mask)),
    "orthogonality_audit_modes": int(audit_modes),
    "low_mode_weighted_orthogonality_rms": orthogonality_error,
    "mode_panels": panel_records,
    "spectrum_panel": spectrum_record,
    "composite_panel": composite_record,
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
print("Saved panels")
for record in panel_records:
    print(
        record["panel"],
        record["pdf"],
    )
print(
    spectrum_record["panel"],
    spectrum_record["pdf"],
)
print("Composite", composite_record["pdf"])
print(f"Summary                  : {summary_file}")
print("=" * 100)
