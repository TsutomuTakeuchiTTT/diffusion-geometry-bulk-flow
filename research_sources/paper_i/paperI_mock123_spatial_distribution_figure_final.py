# -*- coding: utf-8 -*-
"""
Paper I: spatial distributions of Mock-1, Mock-2, and Mock-3

This script generates two publication-oriented visualizations:

1. A one-row, three-panel orthographic 3D point-cloud comparison.
   Radius from the observer is encoded by a common color scale.
2. A three-by-three matrix of orthogonal projections (xy, xz, yz).
   This version is less visually dramatic but makes anisotropic support easier
   to inspect.

The two-dimensional scatter layers are rasterized while text and axes remain
vector graphics. Matplotlib does not support per-artist rasterization of 3D
Path3DCollection objects reliably, so the 3D PDF remains vector-based and the
high-resolution PNG is the recommended publication asset for that panel.

Expected catalog keys
---------------------
Mock-1:
    pos, vel, ids, dist

Mock-2:
    pos, vel, ids, dist, m_app, M_abs, m_lim

Mock-3:
    pos, vel, ids, dist, m_app, M_abs, m_lim,
    octant, mlim_per_octant, delta_m
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import Normalize
from matplotlib.patches import Circle


# =============================================================================
# 1. Paths and plotting configuration
# =============================================================================

ROOT_DIR = Path(
    os.environ.get(
        "COSMIC_DIPOLE_MOCK_DATA_ROOT",
        'mock_data',
    )
)

MOCK1_CANDIDATES = [
    ROOT_DIR / "mock1_complete_sphere" / "mock1.npz",
    ROOT_DIR / "mock1.npz",
]
MOCK2_CANDIDATES = [
    ROOT_DIR / "mock2_schechter_selection" / "mock2.npz",
    ROOT_DIR / "mock2.npz",
]
MOCK3_CANDIDATES = [
    ROOT_DIR / "mock3_inhomogeneous_survey" / "mock3.npz",
    ROOT_DIR / "mock3.npz",
]

OUTPUT_DIR = (
    ROOT_DIR
    / "paperI_figures"
    / "mock_catalog_spatial_distributions"
)

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)
SPHERE_RADIUS = 169.35

# Fixed camera used for all three mocks.
VIEW_ELEVATION_DEG = 20.0
VIEW_AZIMUTH_DEG = -52.0

# Display all objects by default. Set, for example, 8000 to make a lighter
# exploratory preview while retaining a deterministic random subset.
MAX_DISPLAY_POINTS: int | None = None
DISPLAY_SEED = 20260825

POINT_SIZE_3D = 2.2
POINT_ALPHA_3D = 0.45
POINT_SIZE_2D = 1.5
POINT_ALPHA_2D = 0.38

PNG_DPI = 350
PDF_DPI = 350

MAKE_3D_COMPOSITE = True
MAKE_ORTHOGONAL_COMPOSITE = True
SHOW_PLOTS = True

# Mock-3 panel only: show x=0, y=0, or z=0 boundaries in the orthogonal
# projections. This makes the octant-dependent support easier to read.
SHOW_MOCK3_OCTANT_BOUNDARIES = True


# =============================================================================
# 2. Data containers and loading
# =============================================================================

@dataclass(frozen=True)
class MockCatalog:
    name: str
    short_title: str
    path: Path
    position: np.ndarray
    centered_position: np.ndarray
    radius: np.ndarray
    stored_distance: np.ndarray | None
    octant: np.ndarray | None

    @property
    def n_objects(self) -> int:
        return int(self.position.shape[0])


def _first_existing(paths: Iterable[Path], label: str) -> Path:
    candidates = [Path(path) for path in paths]
    for path in candidates:
        if path.is_file():
            return path
    candidate_text = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(
        f"Could not locate {label}. Tried:\n{candidate_text}"
    )


def _load_catalog(
    path: Path,
    name: str,
    short_title: str,
    require_octant: bool = False,
) -> MockCatalog:
    with np.load(path, allow_pickle=False) as data:
        if "pos" not in data.files:
            raise KeyError(f"{path} does not contain the required key 'pos'.")

        position = np.asarray(data["pos"], dtype=np.float64)
        if position.ndim != 2 or position.shape[1] != 3:
            raise ValueError(
                f"{path}: 'pos' must have shape (N, 3), got {position.shape}."
            )
        if not np.all(np.isfinite(position)):
            raise ValueError(f"{path}: nonfinite positions were found.")

        stored_distance = None
        if "dist" in data.files:
            stored_distance = np.asarray(data["dist"], dtype=np.float64)
            if stored_distance.shape != (position.shape[0],):
                raise ValueError(
                    f"{path}: 'dist' must have shape ({position.shape[0]},)."
                )

        octant = None
        if "octant" in data.files:
            octant = np.asarray(data["octant"], dtype=np.int64)
            if octant.shape != (position.shape[0],):
                raise ValueError(
                    f"{path}: 'octant' must have shape ({position.shape[0]},)."
                )
        elif require_octant:
            raise KeyError(f"{path} does not contain the required key 'octant'.")

    centered = position - SPHERE_CENTER[None, :]
    radius = np.linalg.norm(centered, axis=1)

    if np.any(radius > SPHERE_RADIUS + 1.0e-6):
        count = int(np.count_nonzero(radius > SPHERE_RADIUS + 1.0e-6))
        maximum = float(np.max(radius))
        raise ValueError(
            f"{path}: {count} positions lie outside the adopted sphere; "
            f"maximum radius={maximum:.6f}."
        )

    if octant is not None:
        expected_octant = (
            (centered[:, 0] >= 0.0).astype(np.int64)
            + 2 * (centered[:, 1] >= 0.0).astype(np.int64)
            + 4 * (centered[:, 2] >= 0.0).astype(np.int64)
        )
        if not np.array_equal(octant, expected_octant):
            raise ValueError(
                f"{path}: stored octant labels do not match Cartesian signs."
            )

    return MockCatalog(
        name=name,
        short_title=short_title,
        path=path,
        position=position,
        centered_position=centered,
        radius=radius,
        stored_distance=stored_distance,
        octant=octant,
    )


def load_all_catalogs() -> list[MockCatalog]:
    mock1_path = _first_existing(MOCK1_CANDIDATES, "Mock-1")
    mock2_path = _first_existing(MOCK2_CANDIDATES, "Mock-2")
    mock3_path = _first_existing(MOCK3_CANDIDATES, "Mock-3")

    return [
        _load_catalog(
            mock1_path,
            name="Mock-1",
            short_title="Complete sample",
        ),
        _load_catalog(
            mock2_path,
            name="Mock-2",
            short_title="Single flux limit",
        ),
        _load_catalog(
            mock3_path,
            name="Mock-3",
            short_title="Octant-dependent limits",
            require_octant=True,
        ),
    ]


# =============================================================================
# 3. Plotting helpers
# =============================================================================

def _display_indices(n_objects: int, mock_number: int) -> np.ndarray:
    if MAX_DISPLAY_POINTS is None or n_objects <= MAX_DISPLAY_POINTS:
        size = n_objects
    else:
        size = int(MAX_DISPLAY_POINTS)

    rng = np.random.default_rng(DISPLAY_SEED + 100 * mock_number)
    return rng.choice(n_objects, size=size, replace=False)


def _panel_title(catalog: MockCatalog) -> str:
    return (
        f"{catalog.name}: {catalog.short_title}\n"
        rf"$N={catalog.n_objects:,}$"
    )


def _configure_3d_axis(ax) -> None:
    ax.set_xlim(-SPHERE_RADIUS, SPHERE_RADIUS)
    ax.set_ylim(-SPHERE_RADIUS, SPHERE_RADIUS)
    ax.set_zlim(-SPHERE_RADIUS, SPHERE_RADIUS)
    ax.set_box_aspect((1.0, 1.0, 1.0))

    try:
        ax.set_proj_type("ortho")
    except (AttributeError, TypeError):
        # Older Matplotlib versions may not provide orthographic projection.
        pass

    ax.view_init(
        elev=VIEW_ELEVATION_DEG,
        azim=VIEW_AZIMUTH_DEG,
    )

    ticks = [-150, 0, 150]
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_zticks(ticks)

    ax.set_xlabel(r"$x\ [h^{-1}\,\mathrm{Mpc}]$", labelpad=0.0)
    ax.set_ylabel(r"$y\ [h^{-1}\,\mathrm{Mpc}]$", labelpad=0.0)
    ax.set_zlabel(r"$z\ [h^{-1}\,\mathrm{Mpc}]$", labelpad=0.0)

    ax.tick_params(axis="both", which="major", labelsize=8, pad=0.0)
    ax.grid(False)

    # Remove pane fills but retain the three-dimensional axes.
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        try:
            axis.pane.fill = False
        except AttributeError:
            pass


def _configure_2d_axis(
    ax,
    x_index: int,
    y_index: int,
    show_xlabel: bool,
    show_ylabel: bool,
) -> None:
    coordinate_labels = ("x", "y", "z")

    ax.set_xlim(-SPHERE_RADIUS, SPHERE_RADIUS)
    ax.set_ylim(-SPHERE_RADIUS, SPHERE_RADIUS)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([-150, 0, 150])
    ax.set_yticks([-150, 0, 150])
    ax.tick_params(labelsize=8)
    ax.grid(False)

    if show_xlabel:
        ax.set_xlabel(
            rf"${coordinate_labels[x_index]}\ [h^{{-1}}\,\mathrm{{Mpc}}]$"
        )
    else:
        ax.set_xticklabels([])

    if show_ylabel:
        ax.set_ylabel(
            rf"${coordinate_labels[y_index]}\ [h^{{-1}}\,\mathrm{{Mpc}}]$"
        )
    else:
        ax.set_yticklabels([])

    # The circle shows the common spherical extraction boundary in projection.
    ax.add_patch(
        Circle(
            (0.0, 0.0),
            SPHERE_RADIUS,
            fill=False,
            linestyle=":",
            linewidth=0.8,
            alpha=0.45,
        )
    )


def _save_figure(fig, stem: Path) -> dict[str, str]:
    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")

    fig.savefig(
        pdf_path,
        dpi=PDF_DPI,
        bbox_inches="tight",
        pad_inches=0.04,
    )
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        bbox_inches="tight",
        pad_inches=0.04,
    )

    return {
        "pdf": str(pdf_path),
        "png": str(png_path),
    }


def _save_3d_figure(fig, stem: Path) -> dict[str, str]:
    """Save the 3D composite without a tight bounding box.

    Matplotlib can underestimate the extent of three-dimensional axes and clip
    the rightmost z axis or its label when ``bbox_inches="tight"`` is used.
    The 3D layout therefore reserves explicit margins and is exported with the
    full figure canvas.
    """

    pdf_path = stem.with_suffix(".pdf")
    png_path = stem.with_suffix(".png")

    # Force a draw before saving so all 3D projections and text extents have
    # been finalized on the current canvas.
    fig.canvas.draw()

    fig.savefig(
        pdf_path,
        dpi=PDF_DPI,
        facecolor="white",
    )
    fig.savefig(
        png_path,
        dpi=PNG_DPI,
        facecolor="white",
    )

    return {
        "pdf": str(pdf_path),
        "png": str(png_path),
    }


# =============================================================================
# 4. Main 3D composite
# =============================================================================

def make_3d_composite(
    catalogs: list[MockCatalog],
    norm: Normalize,
) -> dict[str, str]:
    # A slightly wider canvas and a deliberately generous right margin prevent
    # clipping of the rightmost 3D axis and z label.
    fig = plt.figure(figsize=(16.4, 5.8))
    grid = fig.add_gridspec(
        1,
        3,
        left=0.025,
        right=0.945,
        bottom=0.19,
        top=0.90,
        wspace=0.10,
    )
    axes = [
        fig.add_subplot(grid[0, panel], projection="3d")
        for panel in range(3)
    ]

    panel_labels = ("(a)", "(b)", "(c)")
    last_scatter = None

    for mock_number, (catalog, ax, panel_label) in enumerate(
        zip(catalogs, axes, panel_labels),
        start=1,
    ):
        selected = _display_indices(catalog.n_objects, mock_number)
        xyz = catalog.centered_position[selected]
        radius = catalog.radius[selected]

        last_scatter = ax.scatter(
            xyz[:, 0],
            xyz[:, 1],
            xyz[:, 2],
            c=radius,
            norm=norm,
            s=POINT_SIZE_3D,
            alpha=POINT_ALPHA_3D,
            linewidths=0.0,
            depthshade=False,
        )

        # Observer position.
        observer = ax.scatter(
            [0.0],
            [0.0],
            [0.0],
            marker="*",
            s=58,
            linewidths=0.8,
            depthshade=False,
            zorder=20,
        )

        _configure_3d_axis(ax)
        ax.set_title(_panel_title(catalog), fontsize=12.5, pad=10)
        ax.text2D(
            0.00,
            0.98,
            panel_label,
            transform=ax.transAxes,
            fontsize=15,
            fontweight="bold",
            va="top",
        )

    # The axes margins are controlled by GridSpec above. Do not call
    # tight_layout or constrained_layout for this 3D figure.
    colorbar_axis = fig.add_axes([0.29, 0.075, 0.42, 0.027])
    if last_scatter is None:
        raise RuntimeError("No scatter object was created.")
    colorbar = fig.colorbar(
        last_scatter,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label(
        r"Observer-centered radius $r\ [h^{-1}\,\mathrm{Mpc}]$"
    )

    output = _save_3d_figure(
        fig,
        OUTPUT_DIR / "paperI_mock123_spatial_distribution_3d",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return output


# =============================================================================
# 5. Orthogonal-projection diagnostic
# =============================================================================

def make_orthogonal_composite(
    catalogs: list[MockCatalog],
    norm: Normalize,
) -> dict[str, str]:
    # Rows: xy, xz, yz. Columns: Mock-1, Mock-2, Mock-3.
    projection_pairs = (
        (0, 1, r"$xy$ projection"),
        (0, 2, r"$xz$ projection"),
        (1, 2, r"$yz$ projection"),
    )

    fig, axes = plt.subplots(
        3,
        3,
        figsize=(12.8, 12.0),
        sharex=False,
        sharey=False,
    )

    panel_counter = 0
    last_scatter = None

    for column, catalog in enumerate(catalogs):
        selected = _display_indices(catalog.n_objects, column + 1)
        xyz = catalog.centered_position[selected]
        radius = catalog.radius[selected]

        for row, (x_index, y_index, projection_title) in enumerate(
            projection_pairs
        ):
            ax = axes[row, column]
            last_scatter = ax.scatter(
                xyz[:, x_index],
                xyz[:, y_index],
                c=radius,
                norm=norm,
                s=POINT_SIZE_2D,
                alpha=POINT_ALPHA_2D,
                linewidths=0.0,
                rasterized=True,
            )
            last_scatter.set_rasterized(True)

            _configure_2d_axis(
                ax,
                x_index=x_index,
                y_index=y_index,
                show_xlabel=(row == 2),
                show_ylabel=(column == 0),
            )

            if row == 0:
                ax.set_title(
                    _panel_title(catalog),
                    fontsize=12.0,
                    pad=7,
                )

            if column == 0:
                ax.text(
                    -0.22,
                    0.50,
                    projection_title,
                    transform=ax.transAxes,
                    rotation=90,
                    va="center",
                    ha="center",
                    fontsize=10.5,
                )

            # In the Mock-3 column, the zero-coordinate lines are the projected
            # boundaries of the octants.
            if SHOW_MOCK3_OCTANT_BOUNDARIES and catalog.name == "Mock-3":
                ax.axhline(0.0, linestyle="--", linewidth=0.75, alpha=0.55)
                ax.axvline(0.0, linestyle="--", linewidth=0.75, alpha=0.55)

            panel_label = f"({chr(ord('a') + panel_counter)})"
            ax.text(
                0.02,
                0.98,
                panel_label,
                transform=ax.transAxes,
                fontsize=12.5,
                fontweight="bold",
                va="top",
            )
            panel_counter += 1

    fig.subplots_adjust(
        left=0.09,
        right=0.985,
        bottom=0.105,
        top=0.935,
        wspace=0.08,
        hspace=0.11,
    )

    colorbar_axis = fig.add_axes([0.29, 0.045, 0.42, 0.022])
    if last_scatter is None:
        raise RuntimeError("No scatter object was created.")
    colorbar = fig.colorbar(
        last_scatter,
        cax=colorbar_axis,
        orientation="horizontal",
    )
    colorbar.set_label(
        r"Observer-centered radius $r\ [h^{-1}\,\mathrm{Mpc}]$"
    )

    output = _save_figure(
        fig,
        OUTPUT_DIR / "paperI_mock123_spatial_distribution_orthogonal",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return output


# =============================================================================
# 6. Summary and execution
# =============================================================================

def _catalog_summary(catalog: MockCatalog) -> dict:
    quantiles = np.quantile(
        catalog.radius,
        [0.10, 0.25, 0.50, 0.75, 0.90, 0.99],
    )

    summary = {
        "name": catalog.name,
        "title": catalog.short_title,
        "path": str(catalog.path),
        "n_objects": catalog.n_objects,
        "radius_min": float(np.min(catalog.radius)),
        "radius_max": float(np.max(catalog.radius)),
        "radius_quantiles": {
            "q10": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q90": float(quantiles[4]),
            "q99": float(quantiles[5]),
        },
    }

    if catalog.stored_distance is not None:
        summary["stored_distance_max_abs_difference"] = float(
            np.max(np.abs(catalog.stored_distance - catalog.radius))
        )

    if catalog.octant is not None:
        counts = np.bincount(catalog.octant, minlength=8)
        summary["octant_counts"] = [int(value) for value in counts]

    return summary


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.linewidth": 0.9,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )

    catalogs = load_all_catalogs()
    radius_norm = Normalize(vmin=0.0, vmax=SPHERE_RADIUS)

    print("=" * 104)
    print("Paper I: three-dimensional spatial distributions of Mock-1, Mock-2, and Mock-3")
    print("=" * 104)
    print(f"Python                   : {platform.python_version()}")
    print(f"Matplotlib font          : {plt.rcParams['font.family']}")
    print(f"Root                     : {ROOT_DIR}")
    print(f"Output                   : {OUTPUT_DIR}")
    print(f"Sphere center            : {SPHERE_CENTER.tolist()} h^-1 Mpc")
    print(f"Sphere radius            : {SPHERE_RADIUS:.2f} h^-1 Mpc")
    print(f"3D view                  : elevation={VIEW_ELEVATION_DEG:.1f} deg, "
          f"azimuth={VIEW_AZIMUTH_DEG:.1f} deg")
    print(f"Display cap              : {MAX_DISPLAY_POINTS}")
    print("3D export                : fixed canvas margins; no tight bounding box")
    print("-" * 104)

    for catalog in catalogs:
        print(
            f"{catalog.name:8s} | N={catalog.n_objects:6,d} | "
            f"r_med={np.median(catalog.radius):8.3f} | "
            f"r_max={np.max(catalog.radius):8.3f} | "
            f"{catalog.path}"
        )
        if catalog.octant is not None:
            counts = np.bincount(catalog.octant, minlength=8)
            print(f"          octant counts: {counts.tolist()}")

    outputs: dict[str, dict[str, str]] = {}

    if MAKE_3D_COMPOSITE:
        outputs["three_dimensional"] = make_3d_composite(
            catalogs,
            radius_norm,
        )

    if MAKE_ORTHOGONAL_COMPOSITE:
        outputs["orthogonal"] = make_orthogonal_composite(
            catalogs,
            radius_norm,
        )

    summary = {
        "python": platform.python_version(),
        "root_dir": str(ROOT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "sphere_center_hinv_mpc": SPHERE_CENTER.tolist(),
        "sphere_radius_hinv_mpc": SPHERE_RADIUS,
        "view_elevation_deg": VIEW_ELEVATION_DEG,
        "view_azimuth_deg": VIEW_AZIMUTH_DEG,
        "max_display_points": MAX_DISPLAY_POINTS,
        "catalogs": [_catalog_summary(catalog) for catalog in catalogs],
        "outputs": outputs,
    }

    summary_path = (
        OUTPUT_DIR
        / "paperI_mock123_spatial_distribution_summary.json"
    )
    summary_path.write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("-" * 104)
    for family, files in outputs.items():
        print(f"{family:24s} PDF: {files['pdf']}")
        print(f"{'':24s} PNG: {files['png']}")
    print(f"{'summary':24s} JSON: {summary_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
