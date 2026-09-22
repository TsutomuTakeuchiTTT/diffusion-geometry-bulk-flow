# -*- coding: utf-8 -*-
"""
Publication figures for Paper II: radial-velocity reconstruction with diffusion geometry.

This script creates two figures that are intentionally distinct from the spatial-distribution
figures used in Paper I.

Figure 1
--------
A data-independent schematic of
  (a) pointwise radial projection,
  (b) the tangential null space, and
  (c) global potential-flow completion.

Figure 2
--------
A data-driven summary of the controlled radial mocks:
  (a) empirical radial cumulative distributions,
  (b) Mock-1, Mock-2, and direction-averaged Mock-3 selection probabilities,
  (c) octant-specific Mock-3 selection probabilities, and
  (d) a Mollweide map of Mock-3 angular support colored by observer-centered radius.

Expected catalog keys
---------------------
Mock-1: pos, vel, ids, dist
Mock-2: pos, vel, ids, dist, m_app, M_abs, m_lim
Mock-3: pos, vel, ids, dist, m_app, M_abs, m_lim, octant,
        mlim_per_octant, delta_m

The script searches both the original subdirectory layout and a flat directory containing
mock1.npz, mock2.npz, and mock3.npz. Set COSMIC_DIPOLE_MOCK_DATA_ROOT to override the
catalog root and PAPERII_FIGURE_OUTPUT to override the output directory.
"""

from __future__ import annotations

import json
import os
import platform
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import matplotlib

SHOW_PLOTS = os.environ.get("PAPERII_SHOW_PLOTS", "1") != "0"
if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D
from scipy.integrate import cumulative_trapezoid


# =============================================================================
# 1. Paths and fixed scientific settings
# =============================================================================

ROOT_DIR = Path(
    os.environ.get(
        "COSMIC_DIPOLE_MOCK_DATA_ROOT",
        'mock_data',
    )
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPERII_FIGURE_OUTPUT",
        str(ROOT_DIR / "paperII_figures" / "radial_observation_overview"),
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

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)
SPHERE_RADIUS = 169.35

HUBBLE_h = 0.6774
SCHECHTER_MSTAR = -22.101
SCHECHTER_ALPHA = -1.077
M_BRIGHT = -24.0
M_FAINT = -14.0

SELECTION_THRESHOLDS = (1.0e-2, 1.0e-3)

PNG_DPI = 350
PDF_DPI = 350


# =============================================================================
# 2. Data containers and validation
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
    m_lim_scalar: float | None
    mlim_per_octant: np.ndarray | None

    @property
    def n_objects(self) -> int:
        return int(self.position.shape[0])


def _first_existing(paths: Iterable[Path], label: str) -> Path:
    candidates = [Path(path) for path in paths]
    for path in candidates:
        if path.is_file():
            return path
    tried = "\n".join(f"  - {path}" for path in candidates)
    raise FileNotFoundError(f"Could not locate {label}. Tried:\n{tried}")


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
            raise ValueError(f"{path}: 'pos' must have shape (N, 3), got {position.shape}.")
        if not np.all(np.isfinite(position)):
            raise ValueError(f"{path}: nonfinite positions were found.")

        stored_distance = None
        if "dist" in data.files:
            stored_distance = np.asarray(data["dist"], dtype=np.float64)
            if stored_distance.shape != (position.shape[0],):
                raise ValueError(
                    f"{path}: 'dist' must have shape ({position.shape[0]},), "
                    f"got {stored_distance.shape}."
                )

        octant = None
        if "octant" in data.files:
            octant = np.asarray(data["octant"], dtype=np.int64)
            if octant.shape != (position.shape[0],):
                raise ValueError(
                    f"{path}: 'octant' must have shape ({position.shape[0]},), "
                    f"got {octant.shape}."
                )
        elif require_octant:
            raise KeyError(f"{path} does not contain the required key 'octant'.")

        m_lim_scalar = None
        if "m_lim" in data.files:
            m_lim_array = np.asarray(data["m_lim"], dtype=np.float64)
            if m_lim_array.ndim == 0:
                m_lim_scalar = float(m_lim_array)

        mlim_per_octant = None
        if "mlim_per_octant" in data.files:
            mlim_per_octant = np.asarray(data["mlim_per_octant"], dtype=np.float64)
            if mlim_per_octant.shape != (8,):
                raise ValueError(
                    f"{path}: 'mlim_per_octant' must have shape (8,), "
                    f"got {mlim_per_octant.shape}."
                )

    centered = position - SPHERE_CENTER[None, :]
    radius = np.linalg.norm(centered, axis=1)

    outside = radius > SPHERE_RADIUS + 1.0e-6
    if np.any(outside):
        raise ValueError(
            f"{path}: {int(np.count_nonzero(outside))} points lie outside the adopted sphere; "
            f"maximum radius={float(np.max(radius)):.6f}."
        )

    if stored_distance is not None:
        discrepancy = float(np.max(np.abs(stored_distance - radius)))
        if discrepancy > 1.0e-6:
            raise ValueError(
                f"{path}: stored and recomputed distances differ by up to {discrepancy:.6e}."
            )

    if octant is not None:
        expected_octant = (
            (centered[:, 0] >= 0.0).astype(np.int64)
            + 2 * (centered[:, 1] >= 0.0).astype(np.int64)
            + 4 * (centered[:, 2] >= 0.0).astype(np.int64)
        )
        if not np.array_equal(octant, expected_octant):
            raise ValueError(f"{path}: stored octant labels do not match Cartesian signs.")

    return MockCatalog(
        name=name,
        short_title=short_title,
        path=path,
        position=position,
        centered_position=centered,
        radius=radius,
        stored_distance=stored_distance,
        octant=octant,
        m_lim_scalar=m_lim_scalar,
        mlim_per_octant=mlim_per_octant,
    )


def load_all_catalogs() -> list[MockCatalog]:
    return [
        _load_catalog(
            _first_existing(MOCK1_CANDIDATES, "Mock-1"),
            name="Mock-1",
            short_title="Complete sample",
        ),
        _load_catalog(
            _first_existing(MOCK2_CANDIDATES, "Mock-2"),
            name="Mock-2",
            short_title="Single flux limit",
        ),
        _load_catalog(
            _first_existing(MOCK3_CANDIDATES, "Mock-3"),
            name="Mock-3",
            short_title="Octant-dependent limits",
            require_octant=True,
        ),
    ]


# =============================================================================
# 3. Common plotting and selection helpers
# =============================================================================


def _configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "axes.linewidth": 0.9,
            "axes.titlesize": 11.5,
            "axes.labelsize": 10.5,
            "legend.fontsize": 8.8,
            "xtick.labelsize": 9.0,
            "ytick.labelsize": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def _save_figure(fig: plt.Figure, stem: Path) -> dict[str, str]:
    stem.parent.mkdir(parents=True, exist_ok=True)
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
    return {"pdf": str(pdf_path), "png": str(png_path)}


def _panel_label(ax, label: str, x: float = 0.01, y: float = 0.99) -> None:
    ax.text(
        x,
        y,
        label,
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=13.5,
        fontweight="bold",
    )


def _arrow(ax, start: np.ndarray, vector: np.ndarray, **kwargs) -> None:
    end = start + vector
    defaults = {
        "arrowstyle": "-|>",
        "mutation_scale": 12,
        "linewidth": 1.6,
        "shrinkA": 0.0,
        "shrinkB": 0.0,
    }
    defaults.update(kwargs)
    ax.annotate("", xy=end, xytext=start, arrowprops=defaults)


class SchechterSelection:
    """Numerically stable bounded Schechter selection function in magnitude space."""

    def __init__(self, n_grid: int = 40001) -> None:
        self.magnitude_grid = np.linspace(M_BRIGHT, M_FAINT, n_grid)
        luminosity_ratio = 10.0 ** (
            0.4 * (SCHECHTER_MSTAR - self.magnitude_grid)
        )
        shape = (
            10.0
            ** (
                0.4
                * (SCHECHTER_ALPHA + 1.0)
                * (SCHECHTER_MSTAR - self.magnitude_grid)
            )
            * np.exp(-luminosity_ratio)
        )
        cumulative = cumulative_trapezoid(
            shape,
            self.magnitude_grid,
            initial=0.0,
        )
        if not np.isfinite(cumulative[-1]) or cumulative[-1] <= 0.0:
            raise RuntimeError("Schechter normalization failed.")
        self.cumulative = cumulative / cumulative[-1]

    def __call__(self, radius: np.ndarray, apparent_limit: float) -> np.ndarray:
        r = np.asarray(radius, dtype=np.float64)
        result = np.ones_like(r)

        positive = r > 0.0
        limiting_absolute = np.full_like(r, np.inf)
        limiting_absolute[positive] = (
            apparent_limit
            - 5.0 * np.log10(r[positive] / HUBBLE_h)
            - 25.0
        )

        too_shallow = limiting_absolute < M_BRIGHT
        intermediate = (
            (limiting_absolute >= M_BRIGHT)
            & (limiting_absolute < M_FAINT)
        )
        complete = limiting_absolute >= M_FAINT

        result[too_shallow] = 0.0
        result[intermediate] = np.interp(
            limiting_absolute[intermediate],
            self.magnitude_grid,
            self.cumulative,
        )
        result[complete] = 1.0
        return result


def _support_radius(
    radius_grid: np.ndarray,
    selection: np.ndarray,
    threshold: float,
) -> float | None:
    supported = selection >= threshold
    if not np.any(supported):
        return None
    return float(np.max(radius_grid[supported]))


# =============================================================================
# 4. Figure 1: radial observation geometry and structural completion
# =============================================================================


def _analytic_potential(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    term_a = np.exp(-((x - 0.70) ** 2 + (y + 0.30) ** 2) / 1.20)
    term_b = np.exp(-((x + 0.95) ** 2 + (y - 0.75) ** 2) / 0.85)

    phi = 0.50 * x + 0.22 * y + 0.68 * term_a - 0.52 * term_b
    grad_x = (
        0.50
        - (2.0 * 0.68 / 1.20) * (x - 0.70) * term_a
        + (2.0 * 0.52 / 0.85) * (x + 0.95) * term_b
    )
    grad_y = (
        0.22
        - (2.0 * 0.68 / 1.20) * (y + 0.30) * term_a
        + (2.0 * 0.52 / 0.85) * (y - 0.75) * term_b
    )
    return phi, grad_x, grad_y


def _draw_right_angle_marker(
    ax,
    vertex: np.ndarray,
    direction_a: np.ndarray,
    direction_b: np.ndarray,
    size: float = 0.12,
    **kwargs,
) -> None:
    """Draw a small right-angle marker at *vertex* between two unit directions."""
    p1 = vertex + size * direction_a
    p2 = p1 + size * direction_b
    p3 = vertex + size * direction_b
    ax.plot(
        [p1[0], p2[0], p3[0]],
        [p1[1], p2[1], p3[1]],
        **kwargs,
    )


def make_figure1() -> dict[str, str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    fig = plt.figure(figsize=(14.4, 4.65))
    grid = fig.add_gridspec(
        1,
        3,
        width_ratios=(1.0, 1.0, 1.22),
        left=0.025,
        right=0.985,
        bottom=0.08,
        top=0.90,
        wspace=0.16,
    )
    axes = [fig.add_subplot(grid[0, i]) for i in range(3)]

    # -------------------------------------------------------------------------
    # (a) Pointwise projection and orthogonal decomposition
    # -------------------------------------------------------------------------
    ax = axes[0]
    observer = np.array([0.20, 0.24])
    galaxy = np.array([1.70, 0.98])
    displacement = galaxy - observer
    n_hat = displacement / np.linalg.norm(displacement)
    tangent = np.array([-n_hat[1], n_hat[0]])

    velocity = 0.92 * n_hat + 0.88 * tangent
    radial = np.dot(velocity, n_hat) * n_hat
    tangential = velocity - radial
    radial_endpoint = galaxy + radial
    velocity_endpoint = galaxy + velocity

    ax.plot(
        [observer[0], galaxy[0]],
        [observer[1], galaxy[1]],
        linewidth=1.25,
        alpha=0.72,
    )
    ax.scatter(*observer, marker="*", s=105, zorder=10)
    ax.scatter(*galaxy, marker="o", s=42, zorder=10)

    _arrow(ax, galaxy, velocity, color=cycle[0], linewidth=2.35)
    _arrow(ax, galaxy, radial, color=cycle[1], linewidth=2.35)
    _arrow(
        ax,
        radial_endpoint,
        tangential,
        color=cycle[2],
        linewidth=1.85,
        linestyle="--",
    )
    _draw_right_angle_marker(
        ax,
        radial_endpoint,
        -n_hat,
        tangent,
        size=0.105,
        linewidth=0.9,
        alpha=0.75,
    )

    ax.text(observer[0] - 0.02, observer[1] - 0.24, "observer", ha="left")
    ax.text(galaxy[0] - 0.02, galaxy[1] - 0.25, "galaxy", ha="center")
    ax.text(
        observer[0] + 0.50 * displacement[0],
        observer[1] + 0.50 * displacement[1] - 0.18,
        r"$\widehat{\boldsymbol{n}}_i$",
        ha="center",
    )
    ax.text(
        *(galaxy + 0.58 * velocity + np.array([-0.10, 0.03])),
        r"$\boldsymbol{v}_i$",
    )
    ax.text(
        *(galaxy + 0.56 * radial + np.array([0.02, -0.20])),
        r"$u_i\widehat{\boldsymbol{n}}_i$",
    )
    ax.text(
        *(radial_endpoint + 0.48 * tangential + np.array([0.04, 0.02])),
        r"$\boldsymbol{v}_{t,i}$",
    )
    ax.text(
        0.50,
        0.035,
        r"$\boldsymbol{v}_i=u_i\widehat{\boldsymbol{n}}_i+\boldsymbol{v}_{t,i}$",
        transform=ax.transAxes,
        ha="center",
        fontsize=11.2,
    )

    ax.set_title("Pointwise radial observation", pad=8)
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(-0.06, 2.60)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    _panel_label(ax, "(a)", x=0.00, y=1.00)

    # -------------------------------------------------------------------------
    # (b) Tangential null-space ambiguity
    # -------------------------------------------------------------------------
    ax = axes[1]
    observer = np.array([0.20, 0.24])
    galaxy = np.array([1.56, 0.93])
    displacement = galaxy - observer
    n_hat = displacement / np.linalg.norm(displacement)
    tangent = np.array([-n_hat[1], n_hat[0]])
    radial = 0.92 * n_hat
    radial_endpoint = galaxy + radial

    ax.plot(
        [observer[0], galaxy[0]],
        [observer[1], galaxy[1]],
        linewidth=1.25,
        alpha=0.72,
    )
    ax.scatter(*observer, marker="*", s=105, zorder=10)
    ax.scatter(*galaxy, marker="o", s=42, zorder=10)
    _arrow(ax, galaxy, radial, color=cycle[1], linewidth=2.35)

    offsets = (-0.82, 0.40, 1.02)
    vector_colors = (cycle[3], cycle[4], cycle[5])
    for offset, color in zip(offsets, vector_colors):
        _arrow(
            ax,
            galaxy,
            radial + offset * tangent,
            color=color,
            linewidth=1.95,
            alpha=0.95,
        )

    line_extent = 1.25
    p0 = radial_endpoint - line_extent * tangent
    p1 = radial_endpoint + line_extent * tangent
    ax.plot(
        [p0[0], p1[0]],
        [p0[1], p1[1]],
        linestyle="--",
        linewidth=1.15,
        alpha=0.72,
    )
    _draw_right_angle_marker(
        ax,
        radial_endpoint,
        -n_hat,
        tangent,
        size=0.105,
        linewidth=0.9,
        alpha=0.75,
    )

    ax.text(observer[0] - 0.02, observer[1] - 0.24, "observer", ha="left")
    ax.text(galaxy[0] - 0.02, galaxy[1] - 0.25, "galaxy", ha="center")
    ax.text(
        *(galaxy + 0.55 * radial + np.array([0.04, -0.19])),
        r"same $u_i$",
    )
    ax.text(
        0.63,
        0.80,
        r"$\widehat{\boldsymbol{n}}_i^{\mathsf{T}}\boldsymbol{w}_i=0$",
        transform=ax.transAxes,
        ha="center",
    )
    ax.text(
        0.50,
        0.035,
        r"$\mathcal{P}_{\mathrm{r}}(\boldsymbol{v}+\boldsymbol{w})"
        r"=\mathcal{P}_{\mathrm{r}}\boldsymbol{v}$",
        transform=ax.transAxes,
        ha="center",
        fontsize=11.2,
    )

    ax.set_title("Tangential null space", pad=8)
    ax.set_xlim(-0.05, 3.05)
    ax.set_ylim(-0.06, 2.60)
    ax.set_aspect("equal", adjustable="box")
    ax.axis("off")
    _panel_label(ax, "(b)", x=0.00, y=1.00)

    # -------------------------------------------------------------------------
    # (c) Global potential-flow completion
    # -------------------------------------------------------------------------
    ax = axes[2]
    grid_values = np.linspace(-2.45, 2.45, 180)
    gx, gy = np.meshgrid(grid_values, grid_values)
    phi, grad_x, grad_y = _analytic_potential(gx, gy)
    domain_mask = gx**2 + gy**2 <= 2.38**2
    masked_phi = np.ma.masked_where(~domain_mask, phi)

    ax.contourf(gx, gy, masked_phi, levels=16, alpha=0.13)
    ax.contour(gx, gy, masked_phi, levels=10, linewidths=0.55, alpha=0.45)
    ax.add_patch(
        plt.Circle(
            (0.0, 0.0),
            2.38,
            fill=False,
            linewidth=1.0,
            linestyle=":",
            alpha=0.75,
        )
    )

    q_values = np.linspace(-2.05, 2.05, 8)
    qx, qy = np.meshgrid(q_values, q_values)
    _, qgrad_x, qgrad_y = _analytic_potential(qx, qy)
    qmask = qx**2 + qy**2 <= 2.18**2
    qnorm = np.hypot(qgrad_x, qgrad_y)
    safe_qnorm = np.where(qnorm > 0.0, qnorm, 1.0)
    ax.quiver(
        qx[qmask],
        qy[qmask],
        (qgrad_x / safe_qnorm)[qmask],
        (qgrad_y / safe_qnorm)[qmask],
        angles="xy",
        scale_units="xy",
        scale=3.05,
        width=0.006,
        alpha=0.42,
        color=cycle[0],
    )

    rng = np.random.default_rng(20260906)
    angles = rng.uniform(-np.pi, np.pi, 22)
    radii = 2.08 * np.sqrt(rng.uniform(0.10, 1.0, 22))
    px = radii * np.cos(angles)
    py = radii * np.sin(angles)
    ax.scatter(px, py, s=18, marker="o", zorder=5)
    ax.scatter(0.0, 0.0, marker="*", s=110, zorder=10)

    _, point_grad_x, point_grad_y = _analytic_potential(px, py)
    point_radius = np.hypot(px, py)
    n_x = px / point_radius
    n_y = py / point_radius
    radial_scalar = n_x * point_grad_x + n_y * point_grad_y

    selected = np.array([0, 3, 6, 9, 12, 15, 18, 20])
    radial_scale = 0.43
    ax.quiver(
        px[selected],
        py[selected],
        radial_scale * radial_scalar[selected] * n_x[selected],
        radial_scale * radial_scalar[selected] * n_y[selected],
        angles="xy",
        scale_units="xy",
        scale=1.0,
        width=0.010,
        alpha=0.98,
        color=cycle[1],
    )

    ax.text(
        0.97,
        0.955,
        r"completed field: $\boldsymbol{v}=\nabla\Phi$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10.8,
    )
    ax.text(
        0.04,
        0.905,
        r"shared scalar potential $\Phi$",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=10.2,
    )

    # Compact direct key in the blank corner outside the circular domain.
    ax.annotate(
        "",
        xy=(0.145, 0.065),
        xytext=(0.045, 0.065),
        xycoords=ax.transAxes,
        textcoords=ax.transAxes,
        arrowprops={"arrowstyle": "-|>", "color": cycle[1], "linewidth": 2.0},
    )
    ax.text(
        0.155,
        0.065,
        "observed radial labels",
        transform=ax.transAxes,
        va="center",
        fontsize=9.3,
    )

    ax.set_title("Global potential-flow completion", pad=8)
    ax.set_xlim(-2.53, 2.53)
    ax.set_ylim(-2.53, 2.53)
    ax.set_aspect("equal", adjustable="box")
    ax.set_xticks([])
    ax.set_yticks([])
    _panel_label(ax, "(c)", x=0.01, y=0.99)

    output = _save_figure(
        fig,
        OUTPUT_DIR / "paperII_radial_observation_and_completion",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    return output


# =============================================================================
# 5. Figure 2: controlled radial mocks and selection geometry
# =============================================================================


def _empirical_cdf(radius: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.asarray(radius, dtype=np.float64))
    cumulative = np.arange(1, ordered.size + 1, dtype=np.float64) / ordered.size
    return ordered, cumulative


def _mollweide_coordinates(centered_position: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    radius = np.linalg.norm(centered_position, axis=1)
    if np.any(radius <= 0.0):
        raise ValueError("Observer-coincident points cannot be displayed on the sky.")
    longitude = np.arctan2(centered_position[:, 1], centered_position[:, 0])
    latitude = np.arcsin(np.clip(centered_position[:, 2] / radius, -1.0, 1.0))
    return longitude, latitude


def make_figure2(catalogs: list[MockCatalog]) -> tuple[dict[str, str], dict]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    mock1, mock2, mock3 = catalogs

    if mock2.m_lim_scalar is None:
        raise ValueError("Mock-2 must contain a scalar 'm_lim'.")
    if mock3.mlim_per_octant is None:
        raise ValueError("Mock-3 must contain 'mlim_per_octant'.")
    if mock3.octant is None:
        raise ValueError("Mock-3 must contain octant labels.")

    selection_model = SchechterSelection()
    radius_grid = np.linspace(0.05, SPHERE_RADIUS, 2400)

    selection_mock1 = np.ones_like(radius_grid)
    selection_mock2 = selection_model(radius_grid, mock2.m_lim_scalar)
    selection_octants = np.vstack(
        [
            selection_model(radius_grid, float(apparent_limit))
            for apparent_limit in mock3.mlim_per_octant
        ]
    )
    selection_mock3_global = np.mean(selection_octants, axis=0)

    fig = plt.figure(figsize=(13.8, 9.6))
    grid = fig.add_gridspec(
        2,
        2,
        left=0.075,
        right=0.965,
        bottom=0.075,
        top=0.955,
        wspace=0.23,
        hspace=0.29,
    )

    ax_a = fig.add_subplot(grid[0, 0])
    ax_b = fig.add_subplot(grid[0, 1])
    ax_c = fig.add_subplot(grid[1, 0])
    ax_d = fig.add_subplot(grid[1, 1], projection="mollweide")

    # -------------------------------------------------------------------------
    # (a) Empirical radial CDFs
    # -------------------------------------------------------------------------
    for catalog in catalogs:
        ordered, cumulative = _empirical_cdf(catalog.radius)
        ax_a.plot(
            ordered,
            cumulative,
            linewidth=2.0,
            label=f"{catalog.name} ($N={catalog.n_objects:,}$)",
        )

    ax_a.axvline(SPHERE_RADIUS, linestyle=":", linewidth=1.0, alpha=0.75)
    ax_a.text(
        SPHERE_RADIUS - 2.0,
        0.48,
        r"$R_{\mathrm{mock}}$",
        rotation=90,
        va="center",
        ha="right",
    )
    ax_a.set_xlim(0.0, SPHERE_RADIUS + 3.0)
    ax_a.set_ylim(0.0, 1.02)
    ax_a.set_xlabel(r"Observer-centered radius $r\ [h^{-1}\,\mathrm{Mpc}]$")
    ax_a.set_ylabel(r"Empirical cumulative fraction $F_N(r)$")
    ax_a.set_title("Radial support of the three catalogs")
    ax_a.grid(alpha=0.20)
    ax_a.legend(frameon=False, loc="lower right")
    _panel_label(ax_a, "(a)")

    # -------------------------------------------------------------------------
    # (b) Primary radial selection functions
    # -------------------------------------------------------------------------
    positive_floor = 1.0e-8
    ax_b.plot(
        radius_grid,
        np.clip(selection_mock1, positive_floor, None),
        linewidth=2.0,
        label="Mock-1: complete",
    )
    ax_b.plot(
        radius_grid,
        np.clip(selection_mock2, positive_floor, None),
        linewidth=2.0,
        label="Mock-2: single limit",
    )
    ax_b.plot(
        radius_grid,
        np.clip(selection_mock3_global, positive_floor, None),
        linewidth=2.0,
        label="Mock-3: direction average",
    )

    for threshold, linestyle in zip(SELECTION_THRESHOLDS, ("--", ":")):
        ax_b.axhline(threshold, linestyle=linestyle, linewidth=1.0, alpha=0.75)
        ax_b.text(
            SPHERE_RADIUS - 2.0,
            threshold * 1.12,
            rf"$S={threshold:.0e}$",
            ha="right",
            va="bottom",
        )

    ax_b.set_yscale("log")
    ax_b.set_xlim(0.0, SPHERE_RADIUS)
    ax_b.set_ylim(1.0e-4, 1.25)
    ax_b.set_xlabel(r"Observer-centered radius $r\ [h^{-1}\,\mathrm{Mpc}]$")
    ax_b.set_ylabel(r"Selection probability $S(r)$")
    ax_b.set_title("Complete, radial, and direction-averaged selection")
    ax_b.grid(alpha=0.20, which="both")
    ax_b.legend(frameon=False, loc="upper right")
    _panel_label(ax_b, "(b)")

    # -------------------------------------------------------------------------
    # (c) Mock-3 octant-dependent selection functions
    # -------------------------------------------------------------------------
    for octant_index in range(8):
        ax_c.plot(
            radius_grid,
            np.clip(selection_octants[octant_index], positive_floor, None),
            linewidth=1.45,
            label=rf"$o={octant_index}$",
        )

    ax_c.plot(
        radius_grid,
        np.clip(selection_mock3_global, positive_floor, None),
        linewidth=2.6,
        linestyle="--",
        label="direction average",
    )

    for threshold, linestyle in zip(SELECTION_THRESHOLDS, ("--", ":")):
        ax_c.axhline(threshold, linestyle=linestyle, linewidth=0.9, alpha=0.65)

    ax_c.set_yscale("log")
    ax_c.set_xlim(0.0, SPHERE_RADIUS)
    ax_c.set_ylim(1.0e-4, 1.25)
    ax_c.set_xlabel(r"Observer-centered radius $r\ [h^{-1}\,\mathrm{Mpc}]$")
    ax_c.set_ylabel(r"Octant selection probability $S_{\mathrm{oct}}(r,o)$")
    ax_c.set_title("Direction-dependent support in Mock-3")
    ax_c.grid(alpha=0.20, which="both")
    ax_c.legend(
        frameon=False,
        ncol=3,
        loc="lower left",
        columnspacing=0.9,
        handlelength=1.8,
    )
    _panel_label(ax_c, "(c)")

    # -------------------------------------------------------------------------
    # (d) Mock-3 angular support
    # -------------------------------------------------------------------------
    longitude, latitude = _mollweide_coordinates(mock3.centered_position)
    draw_order = np.argsort(mock3.radius)
    scatter = ax_d.scatter(
        longitude[draw_order],
        latitude[draw_order],
        c=mock3.radius[draw_order],
        s=2.8,
        alpha=0.58,
        linewidths=0.0,
        rasterized=True,
    )

    # Cartesian sign changes define the eight octants. Their great-circle traces
    # are z=0 (latitude 0), y=0 (longitude 0 and the map edge), and x=0
    # (longitude +/- pi/2).
    ax_d.axhline(0.0, linestyle="--", linewidth=0.9, alpha=0.70)
    ax_d.axvline(0.0, linestyle="--", linewidth=0.9, alpha=0.70)
    ax_d.axvline(-0.5 * np.pi, linestyle="--", linewidth=0.9, alpha=0.70)
    ax_d.axvline(0.5 * np.pi, linestyle="--", linewidth=0.9, alpha=0.70)

    ax_d.grid(alpha=0.22)
    ax_d.set_title("Mock-3 sightlines and radial reach", pad=14)
    ax_d.set_xlabel("Observer-centered longitude", labelpad=14)
    ax_d.set_ylabel("Latitude", labelpad=7)
    _panel_label(ax_d, "(d)", x=-0.02, y=1.06)

    colorbar = fig.colorbar(
        scatter,
        ax=ax_d,
        orientation="horizontal",
        fraction=0.060,
        pad=0.12,
        aspect=34,
    )
    colorbar.set_label(r"Observer-centered radius $r\ [h^{-1}\,\mathrm{Mpc}]$")

    output = _save_figure(
        fig,
        OUTPUT_DIR / "paperII_radial_mock_selection_overview",
    )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)

    support_summary: dict[str, object] = {
        "mock2_m_lim": float(mock2.m_lim_scalar),
        "mock3_mlim_per_octant": [float(v) for v in mock3.mlim_per_octant],
        "selection_thresholds": list(SELECTION_THRESHOLDS),
        "mock2_support_radius_hinv_mpc": {
            f"S_ge_{threshold:.0e}": _support_radius(
                radius_grid,
                selection_mock2,
                threshold,
            )
            for threshold in SELECTION_THRESHOLDS
        },
        "mock3_global_support_radius_hinv_mpc": {
            f"S_ge_{threshold:.0e}": _support_radius(
                radius_grid,
                selection_mock3_global,
                threshold,
            )
            for threshold in SELECTION_THRESHOLDS
        },
        "mock3_octant_support_radius_hinv_mpc": {
            str(octant_index): {
                f"S_ge_{threshold:.0e}": _support_radius(
                    radius_grid,
                    selection_octants[octant_index],
                    threshold,
                )
                for threshold in SELECTION_THRESHOLDS
            }
            for octant_index in range(8)
        },
    }

    return output, support_summary


# =============================================================================
# 6. Summary and execution
# =============================================================================


def _catalog_summary(catalog: MockCatalog) -> dict[str, object]:
    quantiles = np.quantile(
        catalog.radius,
        [0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99],
    )
    summary: dict[str, object] = {
        "name": catalog.name,
        "title": catalog.short_title,
        "path": str(catalog.path),
        "n_objects": catalog.n_objects,
        "radius_min_hinv_mpc": float(np.min(catalog.radius)),
        "radius_max_hinv_mpc": float(np.max(catalog.radius)),
        "radius_quantiles_hinv_mpc": {
            "q10": float(quantiles[0]),
            "q25": float(quantiles[1]),
            "q50": float(quantiles[2]),
            "q75": float(quantiles[3]),
            "q90": float(quantiles[4]),
            "q95": float(quantiles[5]),
            "q99": float(quantiles[6]),
        },
    }
    if catalog.octant is not None:
        summary["octant_counts"] = [
            int(value) for value in np.bincount(catalog.octant, minlength=8)
        ]
    return summary


def main() -> None:
    _configure_matplotlib()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 104)
    print("Paper II: radial observation and controlled-mock overview figures")
    print("=" * 104)
    print(f"Python             : {platform.python_version()}")
    print(f"NumPy              : {np.__version__}")
    print(f"Matplotlib         : {matplotlib.__version__}")
    print(f"Catalog root       : {ROOT_DIR}")
    print(f"Output directory   : {OUTPUT_DIR}")
    print(f"Show plots         : {SHOW_PLOTS}")
    print("-" * 104)

    figure1_output = make_figure1()
    catalogs = load_all_catalogs()
    figure2_output, support_summary = make_figure2(catalogs)

    for catalog in catalogs:
        print(
            f"{catalog.name:8s} | N={catalog.n_objects:6,d} | "
            f"r_med={np.median(catalog.radius):8.3f} | "
            f"r_max={np.max(catalog.radius):8.3f} | {catalog.path}"
        )

    summary = {
        "script": Path(__file__).name,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "matplotlib": matplotlib.__version__,
        "root_dir": str(ROOT_DIR),
        "output_dir": str(OUTPUT_DIR),
        "sphere_center_hinv_mpc": SPHERE_CENTER.tolist(),
        "sphere_radius_hinv_mpc": SPHERE_RADIUS,
        "schechter_parameters": {
            "M_star": SCHECHTER_MSTAR,
            "alpha": SCHECHTER_ALPHA,
            "M_bright": M_BRIGHT,
            "M_faint": M_FAINT,
            "h": HUBBLE_h,
        },
        "catalogs": [_catalog_summary(catalog) for catalog in catalogs],
        "selection_support": support_summary,
        "outputs": {
            "figure1": figure1_output,
            "figure2": figure2_output,
        },
    }

    summary_path = OUTPUT_DIR / "paperII_radial_overview_figure_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print("-" * 104)
    print(f"Figure 1 PDF       : {figure1_output['pdf']}")
    print(f"Figure 1 PNG       : {figure1_output['png']}")
    print(f"Figure 2 PDF       : {figure2_output['pdf']}")
    print(f"Figure 2 PNG       : {figure2_output['png']}")
    print(f"Summary JSON       : {summary_path}")
    print("=" * 104)


if __name__ == "__main__":
    main()
