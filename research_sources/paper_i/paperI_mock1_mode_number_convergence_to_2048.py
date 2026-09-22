from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import time
import warnings
from pathlib import Path
from typing import Any

SHOW_PLOTS = os.environ.get(
    "PAPER1_MODE_CONVERGENCE_SHOW_PLOTS",
    "1",
) != "0"

import matplotlib

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.font_manager as fm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import linalg, sparse
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.spatial import cKDTree


# ==================================================================================================
# 1. Environment parsers
# ==================================================================================================


def parse_int_list(name: str, default: list[int]) -> list[int]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = sorted({int(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value <= 0 for value in values):
        raise ValueError(f"{name}には正の整数をコンマ区切りで指定してください。")
    return values



def parse_float_list(name: str, default: list[float]) -> list[float]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = sorted({float(item.strip()) for item in raw.split(",") if item.strip()})
    if not values or any(value < 0.0 for value in values):
        raise ValueError(f"{name}には非負の数をコンマ区切りで指定してください。")
    return values



def parse_string_list(name: str, default: list[str]) -> list[str]:
    raw = os.environ.get(name)
    if raw is None or not raw.strip():
        return list(default)
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError(f"{name}が空です。")
    return values



def parse_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None:
        return default
    return raw.strip().lower() not in {"0", "false", "no", "off"}


# ==================================================================================================
# 2. Paths and experiment design
# ==================================================================================================

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
LEGACY_GEOMETRY_CACHE_FILE = ROOT_DIR / "mock1_diffusion_geometry_bulk_cache.npz"
PREVIOUS_GEOMETRY_CACHE_FILE = (
    ROOT_DIR / "mock1_diffusion_geometry_bulk_cache_m1024.npz"
)
REFERENCE_CACHE_DIR = ROOT_DIR / "mock1_bulk_robustness_cache"
LOCAL_BASELINE_RUNS_FILE = ROOT_DIR / "mock1_bulk_seed_scale_robustness_runs.csv"

# This run only has to recompute the high-mode part.  The previous 256--1024
# results are merged automatically when available.
NEW_BASIS_SIZES = parse_int_list(
    "PAPER1_MODE_CONVERGENCE_BASIS_SIZES",
    [1024, 1280, 1536, 1792, 2048],
)
MAX_MODES = max(NEW_BASIS_SIZES)
REFERENCE_BASIS = int(
    os.environ.get(
        "PAPER1_MODE_CONVERGENCE_REFERENCE_BASIS",
        "1024",
    )
)
if REFERENCE_BASIS not in NEW_BASIS_SIZES:
    raise ValueError(
        "REFERENCE_BASISは今回再計算するBASIS_SIZESに含めてください。"
    )

EXTENDED_GEOMETRY_CACHE_FILE = (
    ROOT_DIR / f"mock1_diffusion_geometry_bulk_cache_m{MAX_MODES}.npz"
)

PREVIOUS_OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_MODE_CONVERGENCE_PREVIOUS_OUTPUT",
        str(ROOT_DIR / "mock1_mode_number_convergence_v1"),
    )
)
PREVIOUS_RUNS_FILE = PREVIOUS_OUTPUT_DIR / "mock1_mode_number_convergence_runs.csv"
PREVIOUS_VALIDATION_FILE = (
    PREVIOUS_OUTPUT_DIR / "mock1_mode_number_convergence_validation.csv"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_MODE_CONVERGENCE_OUTPUT",
        str(ROOT_DIR / "mock1_mode_number_convergence_v2"),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = "mock1_mode_number_convergence"
SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 300

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)
POSITION_UNIT_LABEL = r"$h^{-1}\,\mathrm{Mpc}$"

SEED_PAIRS_ALL = [
    (20260805, 20260804),
    (20260815, 20260814),
    (20260825, 20260824),
    (20260905, 20260904),
    (20260915, 20260914),
]
N_REPLICATES = int(
    os.environ.get(
        "PAPER1_MODE_CONVERGENCE_N_REPLICATES",
        "5",
    )
)
if not 1 <= N_REPLICATES <= len(SEED_PAIRS_ALL):
    raise ValueError("PAPER1_MODE_CONVERGENCE_N_REPLICATESは1--5で指定してください。")
SEED_PAIRS = SEED_PAIRS_ALL[:N_REPLICATES]

REFERENCE_FRACTION = 0.50
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

SMOOTHING_CONFIGS_ALL = [
    {
        "scale": "fine",
        "n_neighbors": 16,
        "bandwidth_neighbor": 6,
        "bandwidth_multiplier": 1.0,
    },
    {
        "scale": "fiducial",
        "n_neighbors": 32,
        "bandwidth_neighbor": 12,
        "bandwidth_multiplier": 1.0,
    },
    {
        "scale": "coarse",
        "n_neighbors": 64,
        "bandwidth_neighbor": 24,
        "bandwidth_multiplier": 1.0,
    },
]
SCALES_TO_RUN = parse_string_list(
    "PAPER1_MODE_CONVERGENCE_SCALES",
    ["fine", "fiducial", "coarse"],
)
valid_scale_names = {item["scale"] for item in SMOOTHING_CONFIGS_ALL}
unknown_scales = sorted(set(SCALES_TO_RUN).difference(valid_scale_names))
if unknown_scales:
    raise ValueError(f"未知のscaleです: {unknown_scales}")
SMOOTHING_CONFIGS = [
    item for item in SMOOTHING_CONFIGS_ALL if item["scale"] in SCALES_TO_RUN
]

REGULARIZATION_STRENGTHS = parse_float_list(
    "PAPER1_MODE_CONVERGENCE_REGULARIZATION",
    [1.0e-4, 1.0e-2, 1.0],
)
FIXED_REGULARIZATION = float(
    os.environ.get(
        "PAPER1_MODE_CONVERGENCE_FIXED_LAMBDA",
        "1.0e-2",
    )
)
ALL_REGULARIZATION_STRENGTHS = sorted(
    set(REGULARIZATION_STRENGTHS + [FIXED_REGULARIZATION])
)
NUMERICAL_RIDGE = 1.0e-10

# Preserve exactly the generator normalization used by the 512- and 1024-mode runs.
GENERATOR_NORMALIZATION_REFERENCE_MODES = int(
    os.environ.get(
        "PAPER1_MODE_CONVERGENCE_GENERATOR_REFERENCE_MODES",
        str(min(512, MAX_MODES)),
    )
)
if not 2 <= GENERATOR_NORMALIZATION_REFERENCE_MODES <= MAX_MODES:
    raise ValueError(
        "GENERATOR_NORMALIZATION_REFERENCE_MODESは2以上MAX_MODES以下にしてください。"
    )

GEOMETRY_NEIGHBORS = int(os.environ.get("PAPER1_GEOMETRY_NEIGHBORS", "48"))
GEOMETRY_BANDWIDTH_NEIGHBOR = int(
    os.environ.get("PAPER1_GEOMETRY_BANDWIDTH_NEIGHBOR", "16")
)
GEOMETRY_BANDWIDTH_MULTIPLIER = float(
    os.environ.get("PAPER1_GEOMETRY_BANDWIDTH_MULTIPLIER", "1.0")
)
ALPHA_NORMALIZATION = 0.0
EIGEN_SOLVER_SEED = 20260807
EIGEN_SOLVER_TOLERANCE = float(
    os.environ.get("PAPER1_MODE_CONVERGENCE_EIGEN_TOL", "1.0e-8")
)
EIGEN_NCV_RAW = int(os.environ.get("PAPER1_MODE_CONVERGENCE_EIGEN_NCV", "0"))
EIGEN_NCV = None if EIGEN_NCV_RAW <= 0 else EIGEN_NCV_RAW
EIGEN_MAXITER_FACTOR = int(
    os.environ.get("PAPER1_MODE_CONVERGENCE_EIGEN_MAXITER_FACTOR", "50")
)

USE_EXTENDED_GEOMETRY_CACHE = parse_bool(
    "PAPER1_MODE_CONVERGENCE_USE_GEOMETRY_CACHE",
    True,
)
USE_REFERENCE_FIELD_CACHE = parse_bool(
    "PAPER1_MODE_CONVERGENCE_USE_REFERENCE_CACHE",
    True,
)
USE_CHECKPOINTS = parse_bool(
    "PAPER1_MODE_CONVERGENCE_USE_CHECKPOINTS",
    True,
)
MERGE_PREVIOUS_RESULTS = parse_bool(
    "PAPER1_MODE_CONVERGENCE_MERGE_PREVIOUS",
    True,
)
SAVE_GEOMETRY_COMPRESSED = parse_bool(
    "PAPER1_MODE_CONVERGENCE_COMPRESS_GEOMETRY_CACHE",
    False,
)
FULL_ORTHOGONALITY_AUDIT = parse_bool(
    "PAPER1_MODE_CONVERGENCE_FULL_ORTHOGONALITY",
    False,
)
ORTHOGONALITY_SAMPLE_SIZE = int(
    os.environ.get("PAPER1_MODE_CONVERGENCE_ORTHOGONALITY_SAMPLE", "192")
)

PLATEAU_RELATIVE_THRESHOLD = float(
    os.environ.get("PAPER1_MODE_CONVERGENCE_PLATEAU_THRESHOLD", "0.02")
)
PLATEAU_CONSECUTIVE_STEPS = int(
    os.environ.get("PAPER1_MODE_CONVERGENCE_PLATEAU_STEPS", "2")
)
if PLATEAU_RELATIVE_THRESHOLD <= 0.0:
    raise ValueError("PLATEAU_RELATIVE_THRESHOLDは正にしてください。")
if PLATEAU_CONSECUTIVE_STEPS < 1:
    raise ValueError("PLATEAU_CONSECUTIVE_STEPSは1以上にしてください。")

MODEL_NAME = "corrected Cartesian diffusion spectral"
LOCAL_MODEL_NAME = "adaptive local-kernel regression"
SCALE_LABELS = {
    "fine": "Fine",
    "fiducial": "Fiducial",
    "coarse": "Coarse",
}
PANEL_LABELS = ["(a)", "(b)", "(c)", "(d)"]


# ==================================================================================================
# 3. Generic utilities
# ==================================================================================================


def configure_font() -> str:
    candidates = ["DejaVu Sans", "Liberation Sans", "Arial", "Helvetica"]
    installed = {font.name for font in fm.fontManager.ttflist}
    selected = next((name for name in candidates if name in installed), "DejaVu Sans")
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
        }
    )
    return selected



def file_signature(path: Path) -> dict[str, object]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }



def stable_signature(payload: dict[str, object]) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()



def make_reference_model_split(
    n: int,
    reference_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_reference = int(round(reference_fraction * n))
    return np.sort(order[:n_reference]), np.sort(order[n_reference:])



def make_label_split(
    n: int,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    n_validation = int(round(validation_fraction * n))
    return {
        "train": np.sort(order[:n_train]),
        "validation": np.sort(order[n_train:n_train + n_validation]),
        "test": np.sort(order[n_train + n_validation:]),
    }



def finite_median(values: np.ndarray) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else math.nan



def finite_quantile(values: np.ndarray, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else math.nan



def direction_errors(true: np.ndarray, predicted: np.ndarray) -> np.ndarray:
    true_speed = np.linalg.norm(true, axis=1)
    predicted_speed = np.linalg.norm(predicted, axis=1)
    output = np.full(true.shape[0], np.nan, dtype=np.float64)
    valid = (true_speed > 0.0) & (predicted_speed > 0.0)
    if np.any(valid):
        cosine = np.sum(true[valid] * predicted[valid], axis=1) / (
            true_speed[valid] * predicted_speed[valid]
        )
        output[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return output



def velocity_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    residual = predicted - true
    true_speed = np.linalg.norm(true, axis=1)
    predicted_speed = np.linalg.norm(predicted, axis=1)

    rmse_3d = float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1))))
    reference_rms = float(np.sqrt(np.mean(np.sum(np.square(true), axis=1))))
    normalized_rmse = rmse_3d / reference_rms if reference_rms > 0.0 else math.nan
    component_rmse = np.sqrt(np.mean(np.square(residual), axis=0))

    correlations: list[float] = []
    for component in range(3):
        if np.std(true[:, component]) == 0.0 or np.std(predicted[:, component]) == 0.0:
            correlations.append(math.nan)
        else:
            correlations.append(
                float(np.corrcoef(true[:, component], predicted[:, component])[0, 1])
            )

    angles = direction_errors(true, predicted)
    relative_speed_error = np.full(true.shape[0], np.nan, dtype=np.float64)
    positive = true_speed > 0.0
    relative_speed_error[positive] = (
        predicted_speed[positive] - true_speed[positive]
    ) / true_speed[positive]

    return {
        "rmse_3d": rmse_3d,
        "normalized_rmse": float(normalized_rmse),
        "rmse_x": float(component_rmse[0]),
        "rmse_y": float(component_rmse[1]),
        "rmse_z": float(component_rmse[2]),
        "corr_x": correlations[0],
        "corr_y": correlations[1],
        "corr_z": correlations[2],
        "median_direction_error_deg": finite_median(angles),
        "p90_direction_error_deg": finite_quantile(angles, 0.90),
        "median_abs_relative_speed_error": finite_median(
            np.abs(relative_speed_error)
        ),
    }



def calibration_metrics(true: np.ndarray, predicted: np.ndarray) -> dict[str, float]:
    records: dict[str, float] = {}
    slopes: list[float] = []
    for component, label in enumerate(["x", "y", "z"]):
        reference = true[:, component]
        reconstructed = predicted[:, component]
        centered = reference - np.mean(reference)
        denominator = float(np.dot(centered, centered))
        if denominator > 0.0:
            slope = float(
                np.dot(centered, reconstructed - np.mean(reconstructed))
                / denominator
            )
            intercept = float(np.mean(reconstructed) - slope * np.mean(reference))
        else:
            slope = math.nan
            intercept = math.nan
        records[f"calibration_slope_{label}"] = slope
        records[f"calibration_intercept_{label}"] = intercept
        slopes.append(slope)

    vector_denominator = float(np.sum(np.square(true)))
    records["vector_gain_through_origin"] = (
        float(np.sum(true * predicted) / vector_denominator)
        if vector_denominator > 0.0
        else math.nan
    )
    finite_slopes = np.asarray(slopes, dtype=np.float64)
    finite_slopes = finite_slopes[np.isfinite(finite_slopes)]
    records["mean_component_calibration_slope"] = (
        float(np.mean(finite_slopes)) if finite_slopes.size else math.nan
    )
    return records



def factor_regularized_system(
    base_gram: np.ndarray,
    regularization: np.ndarray,
    strength: float,
) -> tuple[str, object]:
    system = np.array(base_gram, dtype=np.float64, order="F", copy=True)
    diagonal = np.diag_indices_from(system)
    system[diagonal] += strength * regularization + NUMERICAL_RIDGE
    try:
        factor = linalg.cho_factor(
            system,
            lower=True,
            overwrite_a=True,
            check_finite=False,
        )
        return "cholesky", factor
    except linalg.LinAlgError:
        # cho_factor may overwrite its input before failing; rebuild the matrix.
        system = np.array(base_gram, dtype=np.float64, order="F", copy=True)
        system[diagonal] += strength * regularization + NUMERICAL_RIDGE
        return "direct", system



def solve_factored_dense(
    factor_kind: str,
    factor: object,
    rhs: np.ndarray,
) -> np.ndarray:
    if factor_kind == "cholesky":
        return linalg.cho_solve(factor, rhs, check_finite=False)
    return linalg.solve(
        np.asarray(factor),
        rhs,
        assume_a="sym",
        check_finite=False,
    )



def save_figure(fig: plt.Figure, stem: str) -> tuple[Path | None, Path | None]:
    pdf_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.pdf" if SAVE_PDF else None
    png_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.png" if SAVE_PNG else None
    if pdf_path is not None:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
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
    return pdf_path, png_path



def add_panel_label(ax: plt.Axes, label: str) -> None:
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


# ==================================================================================================
# 4. Extended diffusion geometry
# ==================================================================================================


def build_variable_bandwidth_kernel(
    positions: np.ndarray,
    n_neighbors: int,
    bandwidth_neighbor: int,
    bandwidth_multiplier: float,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    n = positions.shape[0]
    if n_neighbors >= n:
        raise ValueError("GEOMETRY_NEIGHBORSは天体数より小さくしてください。")
    if not 2 <= bandwidth_neighbor <= n_neighbors:
        raise ValueError(
            "GEOMETRY_BANDWIDTH_NEIGHBORは2以上GEOMETRY_NEIGHBORS以下にしてください。"
        )

    print(f"[Geometry] {n_neighbors}近傍のvariable-bandwidth kernelを構成します。")
    tree = cKDTree(positions)
    try:
        distances, indices = tree.query(
            positions,
            k=n_neighbors + 1,
            workers=-1,
        )
    except TypeError:
        distances, indices = tree.query(positions, k=n_neighbors + 1)

    distances = np.asarray(distances[:, 1:], dtype=np.float64)
    indices = np.asarray(indices[:, 1:], dtype=np.int64)
    rho = bandwidth_multiplier * distances[:, bandwidth_neighbor - 1]
    positive = rho[rho > 0.0]
    if positive.size == 0:
        raise ValueError("局所bandwidthがすべて0です。")
    rho = np.maximum(
        rho,
        max(float(np.median(positive)) * 1.0e-8, np.finfo(float).eps),
    )

    row = np.repeat(np.arange(n, dtype=np.int64), n_neighbors)
    col = indices.reshape(-1)
    squared_distance = np.square(distances.reshape(-1))
    denominator = rho[row] * rho[col]
    weights = np.exp(-squared_distance / denominator)

    kernel = sparse.coo_matrix((weights, (row, col)), shape=(n, n)).tocsr()
    kernel = kernel.maximum(kernel.T)
    kernel.setdiag(1.0)
    kernel.eliminate_zeros()

    print(
        "[Geometry] "
        f"nnz={kernel.nnz:,}, mean degree={kernel.nnz / n:.2f}, "
        f"median rho={np.median(rho):.6g}"
    )
    return kernel, rho



def build_symmetric_markov_operator(
    kernel: sparse.csr_matrix,
) -> tuple[sparse.csr_matrix, np.ndarray]:
    q = np.asarray(kernel.sum(axis=1)).ravel()
    if np.any(q <= 0.0):
        raise ValueError("kernelに孤立点があります。")

    if ALPHA_NORMALIZATION == 0.0:
        kernel_alpha = kernel
    else:
        q_inverse = np.power(q, -ALPHA_NORMALIZATION)
        kernel_alpha = (
            sparse.diags(q_inverse) @ kernel @ sparse.diags(q_inverse)
        ).tocsr()

    degree = np.asarray(kernel_alpha.sum(axis=1)).ravel()
    stationary = degree / np.sum(degree)
    inverse_sqrt_degree = 1.0 / np.sqrt(degree)
    symmetric_operator = (
        sparse.diags(inverse_sqrt_degree)
        @ kernel_alpha
        @ sparse.diags(inverse_sqrt_degree)
    )
    symmetric_operator = (
        0.5 * (symmetric_operator + symmetric_operator.T)
    ).tocsr()
    return symmetric_operator, stationary



def orthogonality_audit(
    eigenvectors: np.ndarray,
) -> tuple[float, float, str, int]:
    column_norm_error = float(
        np.max(np.abs(np.sum(np.square(eigenvectors), axis=0) - 1.0))
    )
    n_modes = eigenvectors.shape[1]

    if FULL_ORTHOGONALITY_AUDIT:
        selected = np.arange(n_modes, dtype=np.int64)
        audit_kind = "full"
    else:
        sample_size = min(max(16, ORTHOGONALITY_SAMPLE_SIZE), n_modes)
        edge = min(64, sample_size // 3)
        selected = np.unique(
            np.concatenate(
                [
                    np.arange(edge, dtype=np.int64),
                    np.linspace(0, n_modes - 1, sample_size, dtype=np.int64),
                    np.arange(max(0, n_modes - edge), n_modes, dtype=np.int64),
                ]
            )
        )
        audit_kind = "sampled"

    sample = eigenvectors[:, selected]
    gram = sample.T @ sample
    orthogonality_rms = float(
        np.linalg.norm(gram - np.eye(selected.size), ord="fro")
        / np.sqrt(selected.size)
    )
    return orthogonality_rms, column_norm_error, audit_kind, int(selected.size)



def compute_diffusion_eigenbasis(
    symmetric_operator: sparse.csr_matrix,
    stationary_measure: np.ndarray,
    max_modes: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, object]]:
    n = symmetric_operator.shape[0]
    if max_modes >= n:
        raise ValueError("MAX_MODESは天体数より小さくしてください。")
    if EIGEN_NCV is not None and not max_modes < EIGEN_NCV <= n:
        raise ValueError("EIGEN_NCVはMAX_MODESより大きく天体数以下にしてください。")

    print(f"[Geometry] 先頭{max_modes}個のdiffusion eigenfunctionsを計算します。")
    start = time.perf_counter()
    kwargs: dict[str, Any] = {
        "v0": np.random.default_rng(EIGEN_SOLVER_SEED).normal(size=n),
        "k": max_modes,
        "which": "LA",
        "tol": EIGEN_SOLVER_TOLERANCE,
        "maxiter": max(20000, EIGEN_MAXITER_FACTOR * n),
    }
    if EIGEN_NCV is not None:
        kwargs["ncv"] = EIGEN_NCV

    try:
        eigenvalues, eigenvectors = eigsh(symmetric_operator, **kwargs)
    except ArpackNoConvergence as exc:
        n_converged = 0 if exc.eigenvalues is None else len(exc.eigenvalues)
        raise RuntimeError(
            "ARPACKの固有値計算が収束しませんでした。\n"
            f"収束した固有対: {n_converged}/{max_modes}\n"
            "メモリに余裕があればPAPER1_MODE_CONVERGENCE_EIGEN_NCVを増やしてください。"
        ) from exc

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)

    for column in range(eigenvectors.shape[1]):
        pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
        if eigenvectors[pivot, column] < 0.0:
            eigenvectors[:, column] *= -1.0

    (
        orthogonality_rms,
        column_norm_error,
        orthogonality_kind,
        orthogonality_modes,
    ) = orthogonality_audit(eigenvectors)

    eigenfunctions = eigenvectors / np.sqrt(stationary_measure)[:, None]
    raw_generator = np.maximum(1.0 - eigenvalues, 0.0)
    reference_positive = raw_generator[
        1:GENERATOR_NORMALIZATION_REFERENCE_MODES
    ]
    reference_positive = reference_positive[reference_positive > 0.0]
    if reference_positive.size == 0:
        raise ValueError("generator normalizationに使える正固有値がありません。")
    generator_scale = float(np.median(reference_positive))
    generator_eigenvalues = raw_generator / generator_scale

    audit = {
        "elapsed_seconds": float(time.perf_counter() - start),
        "orthogonality_rms": orthogonality_rms,
        "orthogonality_kind": orthogonality_kind,
        "orthogonality_modes": orthogonality_modes,
        "column_norm_max_abs_error": column_norm_error,
        "generator_scale": generator_scale,
    }
    print(
        "[Geometry] eigensystem completed: "
        f"elapsed={audit['elapsed_seconds']:.1f} s, "
        f"{orthogonality_kind} orthogonality RMS={orthogonality_rms:.3e}, "
        f"generator scale={generator_scale:.6e}"
    )
    return eigenvalues, generator_eigenvalues, eigenfunctions, audit



def geometry_signature() -> str:
    return stable_signature(
        {
            "data": file_signature(DATA_FILE),
            "center": SPHERE_CENTER.tolist(),
            "n_neighbors": GEOMETRY_NEIGHBORS,
            "bandwidth_neighbor": GEOMETRY_BANDWIDTH_NEIGHBOR,
            "bandwidth_multiplier": GEOMETRY_BANDWIDTH_MULTIPLIER,
            "alpha": ALPHA_NORMALIZATION,
            "eigen_solver_seed": EIGEN_SOLVER_SEED,
            "eigen_tolerance": EIGEN_SOLVER_TOLERANCE,
            "eigen_ncv": EIGEN_NCV,
            "max_modes": MAX_MODES,
            "generator_reference_modes": GENERATOR_NORMALIZATION_REFERENCE_MODES,
        }
    )



def load_or_build_extended_geometry(
    positions: np.ndarray,
) -> dict[str, object]:
    signature = geometry_signature()
    if USE_EXTENDED_GEOMETRY_CACHE and EXTENDED_GEOMETRY_CACHE_FILE.is_file():
        try:
            with np.load(EXTENDED_GEOMETRY_CACHE_FILE, allow_pickle=False) as cached:
                if str(cached["signature"].item()) == signature:
                    print(
                        "[Cache] Extended geometryを再利用します: "
                        f"{EXTENDED_GEOMETRY_CACHE_FILE}"
                    )
                    return {
                        "signature": signature,
                        "stationary_measure": np.asarray(
                            cached["stationary_measure"], dtype=np.float64
                        ),
                        "bandwidth": np.asarray(cached["bandwidth"], dtype=np.float64),
                        "eigenvalues_markov": np.asarray(
                            cached["eigenvalues_markov"], dtype=np.float64
                        ),
                        "generator_eigenvalues": np.asarray(
                            cached["generator_eigenvalues"], dtype=np.float64
                        ),
                        "eigenfunctions": np.asarray(
                            cached["eigenfunctions"], dtype=np.float64
                        ),
                        "generator_scale": float(cached["generator_scale"].item()),
                        "orthogonality_rms": float(
                            cached["orthogonality_rms"].item()
                        ),
                        "orthogonality_kind": str(
                            cached["orthogonality_kind"].item()
                        ),
                        "orthogonality_modes": int(
                            cached["orthogonality_modes"].item()
                        ),
                        "column_norm_max_abs_error": float(
                            cached["column_norm_max_abs_error"].item()
                        ),
                        "kernel_nnz": int(cached["kernel_nnz"].item()),
                    }
                print("[Cache] Extended geometry signature mismatch; rebuilding.")
        except Exception as exc:
            warnings.warn(f"Extended geometry cacheを読めないため再計算します: {exc}")

    kernel, bandwidth = build_variable_bandwidth_kernel(
        positions,
        GEOMETRY_NEIGHBORS,
        GEOMETRY_BANDWIDTH_NEIGHBOR,
        GEOMETRY_BANDWIDTH_MULTIPLIER,
    )
    symmetric_operator, stationary_measure = build_symmetric_markov_operator(kernel)
    (
        eigenvalues_markov,
        generator_eigenvalues,
        eigenfunctions,
        eigen_audit,
    ) = compute_diffusion_eigenbasis(
        symmetric_operator,
        stationary_measure,
        MAX_MODES,
    )

    save_function = np.savez_compressed if SAVE_GEOMETRY_COMPRESSED else np.savez
    if USE_EXTENDED_GEOMETRY_CACHE:
        save_function(
            EXTENDED_GEOMETRY_CACHE_FILE,
            signature=np.array(signature),
            stationary_measure=stationary_measure,
            bandwidth=bandwidth,
            eigenvalues_markov=eigenvalues_markov,
            generator_eigenvalues=generator_eigenvalues,
            eigenfunctions=eigenfunctions,
            generator_scale=np.array(eigen_audit["generator_scale"]),
            orthogonality_rms=np.array(eigen_audit["orthogonality_rms"]),
            orthogonality_kind=np.array(eigen_audit["orthogonality_kind"]),
            orthogonality_modes=np.array(eigen_audit["orthogonality_modes"]),
            column_norm_max_abs_error=np.array(
                eigen_audit["column_norm_max_abs_error"]
            ),
            kernel_nnz=np.array(kernel.nnz, dtype=np.int64),
        )
        print(f"[Cache] Extended geometry saved: {EXTENDED_GEOMETRY_CACHE_FILE}")

    return {
        "signature": signature,
        "stationary_measure": stationary_measure,
        "bandwidth": bandwidth,
        "eigenvalues_markov": eigenvalues_markov,
        "generator_eigenvalues": generator_eigenvalues,
        "eigenfunctions": eigenfunctions,
        "generator_scale": float(eigen_audit["generator_scale"]),
        "orthogonality_rms": float(eigen_audit["orthogonality_rms"]),
        "orthogonality_kind": str(eigen_audit["orthogonality_kind"]),
        "orthogonality_modes": int(eigen_audit["orthogonality_modes"]),
        "column_norm_max_abs_error": float(
            eigen_audit["column_norm_max_abs_error"]
        ),
        "kernel_nnz": int(kernel.nnz),
    }



def audit_geometry_against_cache(
    geometry: dict[str, object],
    cache_path: Path,
    comparison_name: str,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    records: list[dict[str, object]] = []
    summary: dict[str, object] = {
        "comparison": comparison_name,
        "cache": str(cache_path),
        "cache_found": bool(cache_path.is_file()),
    }
    if not cache_path.is_file():
        return records, summary

    try:
        with np.load(cache_path, allow_pickle=False) as previous:
            previous_stationary = np.asarray(
                previous["stationary_measure"], dtype=np.float64
            )
            previous_eigenvalues = np.asarray(
                previous["eigenvalues_markov"], dtype=np.float64
            )
            previous_generator = np.asarray(
                previous["generator_eigenvalues"], dtype=np.float64
            )
            previous_eigenfunctions = np.asarray(
                previous["eigenfunctions"], dtype=np.float64
            )
    except Exception as exc:
        summary["error"] = str(exc)
        return records, summary

    current_stationary = np.asarray(geometry["stationary_measure"], dtype=np.float64)
    current_eigenvalues = np.asarray(geometry["eigenvalues_markov"], dtype=np.float64)
    current_generator = np.asarray(
        geometry["generator_eigenvalues"], dtype=np.float64
    )
    current_eigenfunctions = np.asarray(geometry["eigenfunctions"], dtype=np.float64)

    n_common = min(previous_eigenvalues.size, current_eigenvalues.size)
    n_mode_audit = min(64, previous_eigenfunctions.shape[1], current_eigenfunctions.shape[1])
    weighted_overlap = (
        previous_eigenfunctions[:, :n_mode_audit].T
        @ (current_stationary[:, None] * current_eigenfunctions[:, :n_mode_audit])
    )
    diagonal_correlations = np.abs(np.diag(weighted_overlap))
    singular_values = np.linalg.svd(weighted_overlap, compute_uv=False)

    quantities = {
        "stationary_max_abs_difference": float(
            np.max(np.abs(current_stationary - previous_stationary))
        ),
        "markov_eigenvalue_max_abs_difference": float(
            np.max(np.abs(current_eigenvalues[:n_common] - previous_eigenvalues[:n_common]))
        ),
        "generator_eigenvalue_max_abs_difference": float(
            np.max(np.abs(current_generator[:n_common] - previous_generator[:n_common]))
        ),
        "first_64_mode_correlation_median": float(
            np.median(diagonal_correlations)
        ),
        "first_64_mode_correlation_minimum": float(
            np.min(diagonal_correlations)
        ),
        "first_64_subspace_minimum_singular_value": float(
            np.min(singular_values)
        ),
    }
    for quantity, value in quantities.items():
        records.append(
            {
                "comparison": comparison_name,
                "quantity": quantity,
                "value": value,
            }
        )
    summary.update(quantities)
    summary["n_common_modes"] = int(n_common)
    return records, summary



def audit_extended_geometry(
    geometry: dict[str, object],
) -> tuple[pd.DataFrame, list[dict[str, object]]]:
    records: list[dict[str, object]] = []
    summaries: list[dict[str, object]] = []
    for path, name in [
        (PREVIOUS_GEOMETRY_CACHE_FILE, "m2048_vs_m1024"),
        (LEGACY_GEOMETRY_CACHE_FILE, "m2048_vs_legacy_m512"),
    ]:
        item_records, item_summary = audit_geometry_against_cache(
            geometry,
            path,
            name,
        )
        records.extend(item_records)
        summaries.append(item_summary)
    return pd.DataFrame(records), summaries


# ==================================================================================================
# 5. Reference bulk fields
# ==================================================================================================


def query_reference_neighbors(
    query_positions: np.ndarray,
    reference_positions: np.ndarray,
    max_neighbors: int,
) -> tuple[np.ndarray, np.ndarray]:
    tree = cKDTree(reference_positions)
    try:
        distances, indices = tree.query(
            query_positions,
            k=max_neighbors,
            workers=-1,
        )
    except TypeError:
        distances, indices = tree.query(query_positions, k=max_neighbors)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if max_neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return distances, indices



def bulk_field_from_neighbors(
    distances_max: np.ndarray,
    indices_max: np.ndarray,
    reference_velocities: np.ndarray,
    n_neighbors: int,
    bandwidth_neighbor: int,
    bandwidth_multiplier: float,
) -> dict[str, np.ndarray]:
    distances = distances_max[:, :n_neighbors]
    indices = indices_max[:, :n_neighbors]
    bandwidth = bandwidth_multiplier * distances[:, bandwidth_neighbor - 1]
    positive = bandwidth[bandwidth > 0.0]
    if positive.size == 0:
        raise ValueError("Reference-field bandwidthがすべて0です。")
    bandwidth = np.maximum(
        bandwidth,
        max(float(np.median(positive)) * 1.0e-8, np.finfo(float).eps),
    )
    weights = np.exp(-np.square(distances / bandwidth[:, None]))
    weight_sum = np.sum(weights, axis=1)
    neighbor_velocities = reference_velocities[indices]
    bulk_velocity = np.einsum(
        "ij,ijk->ik",
        weights,
        neighbor_velocities,
        optimize=True,
    ) / weight_sum[:, None]
    residual = neighbor_velocities - bulk_velocity[:, None, :]
    dispersion_squared = np.einsum(
        "ij,ijk->i",
        weights,
        np.square(residual),
        optimize=True,
    ) / weight_sum
    effective_neighbors = np.square(weight_sum) / np.sum(np.square(weights), axis=1)
    return {
        "bulk_velocity": bulk_velocity,
        "bandwidth": bandwidth,
        "dispersion": np.sqrt(np.maximum(dispersion_squared, 0.0)),
        "effective_neighbors": effective_neighbors,
    }



def reference_cache_path(reference_seed: int, scale: str) -> Path:
    return REFERENCE_CACHE_DIR / f"reference_seed_{reference_seed}_scale_{scale}.npz"



def reference_signature(
    reference_seed: int,
    reference_indices: np.ndarray,
    model_indices: np.ndarray,
    config: dict[str, object],
) -> str:
    return stable_signature(
        {
            "data": file_signature(DATA_FILE),
            "reference_seed": int(reference_seed),
            "reference_fraction": float(REFERENCE_FRACTION),
            "reference_indices_hash": hashlib.sha256(
                reference_indices.tobytes()
            ).hexdigest(),
            "model_indices_hash": hashlib.sha256(model_indices.tobytes()).hexdigest(),
            **config,
        }
    )



def load_or_build_reference_fields(
    positions: np.ndarray,
    velocities: np.ndarray,
    reference_indices: np.ndarray,
    model_indices: np.ndarray,
    reference_seed: int,
) -> dict[str, dict[str, np.ndarray]]:
    REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, np.ndarray]] = {}
    missing_configs: list[dict[str, object]] = []

    for config in SMOOTHING_CONFIGS:
        scale = str(config["scale"])
        cache_path = reference_cache_path(reference_seed, scale)
        signature = reference_signature(
            reference_seed,
            reference_indices,
            model_indices,
            config,
        )
        loaded = False
        if USE_REFERENCE_FIELD_CACHE and cache_path.is_file():
            try:
                with np.load(cache_path, allow_pickle=False) as cached:
                    if str(cached["signature"].item()) == signature:
                        results[scale] = {
                            "bulk_velocity": np.asarray(
                                cached["bulk_velocity"], dtype=np.float64
                            ),
                            "bandwidth": np.asarray(cached["bandwidth"], dtype=np.float64),
                            "dispersion": np.asarray(cached["dispersion"], dtype=np.float64),
                            "effective_neighbors": np.asarray(
                                cached["effective_neighbors"], dtype=np.float64
                            ),
                        }
                        loaded = True
            except Exception as exc:
                warnings.warn(f"Reference cache {cache_path.name}を読めません: {exc}")
        if not loaded:
            missing_configs.append(config)

    if missing_configs:
        max_neighbors = max(int(item["n_neighbors"]) for item in missing_configs)
        distances_max, indices_max = query_reference_neighbors(
            positions[model_indices],
            positions[reference_indices],
            max_neighbors,
        )
        for config in missing_configs:
            scale = str(config["scale"])
            result = bulk_field_from_neighbors(
                distances_max,
                indices_max,
                velocities[reference_indices],
                int(config["n_neighbors"]),
                int(config["bandwidth_neighbor"]),
                float(config["bandwidth_multiplier"]),
            )
            results[scale] = result
            if USE_REFERENCE_FIELD_CACHE:
                np.savez_compressed(
                    reference_cache_path(reference_seed, scale),
                    signature=np.array(
                        reference_signature(
                            reference_seed,
                            reference_indices,
                            model_indices,
                            config,
                        )
                    ),
                    **result,
                )

    return results


# ==================================================================================================
# 6. Memory-controlled high-mode regression
# ==================================================================================================


def regularization_diagonal(
    generator_eigenvalues: np.ndarray,
    n_basis: int,
) -> np.ndarray:
    return np.maximum(generator_eigenvalues[:n_basis], NUMERICAL_RIDGE)



def run_signature(geometry_signature_value: str) -> str:
    return stable_signature(
        {
            "data": file_signature(DATA_FILE),
            "geometry_signature": geometry_signature_value,
            "new_basis_sizes": NEW_BASIS_SIZES,
            "regularization_strengths": REGULARIZATION_STRENGTHS,
            "fixed_regularization": FIXED_REGULARIZATION,
            "scales": SCALES_TO_RUN,
            "reference_fraction": REFERENCE_FRACTION,
            "train_fraction": TRAIN_FRACTION,
            "validation_fraction": VALIDATION_FRACTION,
            "seed_pairs": SEED_PAIRS,
        }
    )



def checkpoint_paths(replicate: int) -> tuple[Path, Path, Path]:
    stem = f"replicate_{replicate:02d}"
    return (
        CHECKPOINT_DIR / f"{stem}_runs.csv",
        CHECKPOINT_DIR / f"{stem}_validation.csv",
        CHECKPOINT_DIR / f"{stem}_meta.json",
    )



def load_checkpoint(
    replicate: int,
    signature: str,
) -> tuple[pd.DataFrame, pd.DataFrame] | None:
    if not USE_CHECKPOINTS:
        return None
    runs_path, validation_path, meta_path = checkpoint_paths(replicate)
    if not (runs_path.is_file() and validation_path.is_file() and meta_path.is_file()):
        return None
    try:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("signature") != signature:
            return None
        runs = pd.read_csv(runs_path)
        validation = pd.read_csv(validation_path)
        expected_runs = len(SMOOTHING_CONFIGS) * len(NEW_BASIS_SIZES)
        expected_validation = expected_runs * len(REGULARIZATION_STRENGTHS)
        if runs.shape[0] != expected_runs or validation.shape[0] != expected_validation:
            return None
        print(f"[Checkpoint] replicate {replicate}を再利用します。")
        return runs, validation
    except Exception as exc:
        warnings.warn(f"Checkpoint replicate {replicate}を読めません: {exc}")
        return None



def save_checkpoint(
    replicate: int,
    signature: str,
    runs: pd.DataFrame,
    validation: pd.DataFrame,
) -> None:
    if not USE_CHECKPOINTS:
        return
    runs_path, validation_path, meta_path = checkpoint_paths(replicate)
    runs.to_csv(runs_path, index=False)
    validation.to_csv(validation_path, index=False)
    meta_path.write_text(
        json.dumps(
            {
                "signature": signature,
                "replicate": replicate,
                "n_runs": int(runs.shape[0]),
                "n_validation": int(validation.shape[0]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )



def run_single_replicate(
    replicate: int,
    reference_seed: int,
    label_seed: int,
    positions: np.ndarray,
    velocities: np.ndarray,
    eigenfunctions: np.ndarray,
    generator_eigenvalues: np.ndarray,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    print("-" * 116)
    print(
        f"Replicate {replicate}/{len(SEED_PAIRS)}: "
        f"reference_seed={reference_seed}, label_seed={label_seed}"
    )
    reference_indices, model_indices = make_reference_model_split(
        positions.shape[0],
        REFERENCE_FRACTION,
        reference_seed,
    )
    split = make_label_split(
        model_indices.size,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        label_seed,
    )
    train_global = model_indices[split["train"]]
    validation_global = model_indices[split["validation"]]
    test_global = model_indices[split["test"]]

    phi_train = np.asarray(
        eigenfunctions[train_global, :MAX_MODES],
        dtype=np.float64,
        order="C",
    )
    phi_validation = np.asarray(
        eigenfunctions[validation_global, :MAX_MODES],
        dtype=np.float64,
        order="C",
    )
    phi_test = np.asarray(
        eigenfunctions[test_global, :MAX_MODES],
        dtype=np.float64,
        order="C",
    )

    start_gram = time.perf_counter()
    train_gram_max = (phi_train.T @ phi_train) / phi_train.shape[0]
    validation_gram_max = (
        phi_validation.T @ phi_validation
    ) / phi_validation.shape[0]
    n_fit = phi_train.shape[0] + phi_validation.shape[0]
    fit_gram_max = (
        phi_train.shape[0] * train_gram_max
        + phi_validation.shape[0] * validation_gram_max
    ) / n_fit
    print(
        "[Regression] full Gram matrices completed: "
        f"elapsed={time.perf_counter() - start_gram:.1f} s"
    )

    reference_fields = load_or_build_reference_fields(
        positions,
        velocities,
        reference_indices,
        model_indices,
        reference_seed,
    )

    scale_data: dict[str, dict[str, Any]] = {}
    for config in SMOOTHING_CONFIGS:
        scale = str(config["scale"])
        reference_result = reference_fields[scale]
        bulk_model = reference_result["bulk_velocity"]
        bulk_train = bulk_model[split["train"]]
        bulk_validation = bulk_model[split["validation"]]
        bulk_test = bulk_model[split["test"]]

        rhs_train_max = (phi_train.T @ bulk_train) / phi_train.shape[0]
        rhs_validation_max = (
            phi_validation.T @ bulk_validation
        ) / phi_validation.shape[0]
        rhs_fit_max = (
            phi_train.shape[0] * rhs_train_max
            + phi_validation.shape[0] * rhs_validation_max
        ) / n_fit

        scale_data[scale] = {
            "bulk_validation": bulk_validation,
            "bulk_test": bulk_test,
            "rhs_train_max": rhs_train_max,
            "rhs_fit_max": rhs_fit_max,
            "median_reference_bandwidth": float(
                np.median(reference_result["bandwidth"])
            ),
        }
        print(
            f"  scale={scale:8s}, median h="
            f"{scale_data[scale]['median_reference_bandwidth']:.3f}"
        )

    # The training basis is no longer needed after the Gram matrices and RHS blocks are built.
    del phi_train
    gc.collect()

    scale_order = [str(config["scale"]) for config in SMOOTHING_CONFIGS]
    rhs_train_concat_max = np.hstack(
        [scale_data[scale]["rhs_train_max"] for scale in scale_order]
    )
    rhs_fit_concat_max = np.hstack(
        [scale_data[scale]["rhs_fit_max"] for scale in scale_order]
    )

    run_records: list[dict[str, object]] = []
    validation_records: list[dict[str, object]] = []

    for n_basis in NEW_BASIS_SIZES:
        basis_start = time.perf_counter()
        train_base = train_gram_max[:n_basis, :n_basis]
        fit_base = fit_gram_max[:n_basis, :n_basis]
        regularization = regularization_diagonal(generator_eigenvalues, n_basis)

        best_by_scale: dict[str, tuple[float, float, dict[str, float]]] = {}

        for strength in REGULARIZATION_STRENGTHS:
            factor_kind, factor = factor_regularized_system(
                train_base,
                regularization,
                strength,
            )
            coefficients_concat = solve_factored_dense(
                factor_kind,
                factor,
                rhs_train_concat_max[:n_basis],
            )
            validation_prediction_concat = (
                phi_validation[:, :n_basis] @ coefficients_concat
            )
            del factor, coefficients_concat

            for scale_index, scale in enumerate(scale_order):
                column_slice = slice(3 * scale_index, 3 * (scale_index + 1))
                prediction = validation_prediction_concat[:, column_slice]
                metrics = velocity_metrics(
                    scale_data[scale]["bulk_validation"],
                    prediction,
                )
                validation_records.append(
                    {
                        "replicate": replicate,
                        "reference_seed": reference_seed,
                        "label_seed": label_seed,
                        "scale": scale,
                        "n_basis": n_basis,
                        "regularization_strength": strength,
                        **metrics,
                    }
                )
                candidate = (
                    metrics["normalized_rmse"],
                    strength,
                    metrics,
                )
                current = best_by_scale.get(scale)
                if current is None or candidate[:2] < current[:2]:
                    best_by_scale[scale] = candidate

            del validation_prediction_concat
            gc.collect()

        needed_fit_strengths = sorted(
            set(
                [best_by_scale[scale][1] for scale in scale_order]
                + [FIXED_REGULARIZATION]
            )
        )
        fit_results: dict[float, dict[str, tuple[np.ndarray, np.ndarray]]] = {}

        for strength in needed_fit_strengths:
            factor_kind, factor = factor_regularized_system(
                fit_base,
                regularization,
                strength,
            )
            coefficients_concat = solve_factored_dense(
                factor_kind,
                factor,
                rhs_fit_concat_max[:n_basis],
            )
            test_prediction_concat = phi_test[:, :n_basis] @ coefficients_concat
            fit_results[strength] = {}
            for scale_index, scale in enumerate(scale_order):
                column_slice = slice(3 * scale_index, 3 * (scale_index + 1))
                fit_results[strength][scale] = (
                    np.asarray(test_prediction_concat[:, column_slice]),
                    np.asarray(coefficients_concat[:, column_slice]),
                )
            del factor, coefficients_concat, test_prediction_concat
            gc.collect()

        for scale in scale_order:
            (
                selected_validation_nrmse,
                selected_strength,
                selected_validation_metrics,
            ) = best_by_scale[scale]
            selected_prediction, selected_coefficients = fit_results[
                selected_strength
            ][scale]
            fixed_prediction, fixed_coefficients = fit_results[
                FIXED_REGULARIZATION
            ][scale]

            selected_metrics = velocity_metrics(
                scale_data[scale]["bulk_test"],
                selected_prediction,
            )
            selected_calibration = calibration_metrics(
                scale_data[scale]["bulk_test"],
                selected_prediction,
            )
            fixed_metrics = velocity_metrics(
                scale_data[scale]["bulk_test"],
                fixed_prediction,
            )
            fixed_calibration = calibration_metrics(
                scale_data[scale]["bulk_test"],
                fixed_prediction,
            )

            record: dict[str, object] = {
                "replicate": replicate,
                "reference_seed": reference_seed,
                "label_seed": label_seed,
                "scale": scale,
                "n_basis": n_basis,
                "selected_regularization": selected_strength,
                "validation_normalized_rmse": selected_validation_nrmse,
                "validation_median_direction_error_deg": selected_validation_metrics[
                    "median_direction_error_deg"
                ],
                "median_reference_bandwidth": scale_data[scale][
                    "median_reference_bandwidth"
                ],
                "n_train": int(split["train"].size),
                "n_validation": int(split["validation"].size),
                "n_test": int(split["test"].size),
                "selected_coefficient_frobenius_norm": float(
                    np.linalg.norm(selected_coefficients, ord="fro")
                ),
                "fixed_coefficient_frobenius_norm": float(
                    np.linalg.norm(fixed_coefficients, ord="fro")
                ),
            }
            record.update(
                {f"selected_{key}": value for key, value in selected_metrics.items()}
            )
            record.update(
                {
                    f"selected_{key}": value
                    for key, value in selected_calibration.items()
                }
            )
            record.update(
                {f"fixed_{key}": value for key, value in fixed_metrics.items()}
            )
            record.update(
                {f"fixed_{key}": value for key, value in fixed_calibration.items()}
            )
            run_records.append(record)

            print(
                f"    n_b={n_basis:4d}, scale={scale:8s}, "
                f"selected lambda={selected_strength:7.1e}, "
                f"validation={selected_validation_nrmse:.5f}, "
                f"test={selected_metrics['normalized_rmse']:.5f}"
            )

        print(
            f"[Regression] n_b={n_basis} completed: "
            f"elapsed={time.perf_counter() - basis_start:.1f} s"
        )
        del fit_results
        gc.collect()

    return pd.DataFrame(run_records), pd.DataFrame(validation_records)



def run_convergence(
    positions: np.ndarray,
    velocities: np.ndarray,
    eigenfunctions: np.ndarray,
    generator_eigenvalues: np.ndarray,
    geometry_signature_value: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    signature = run_signature(geometry_signature_value)
    all_runs: list[pd.DataFrame] = []
    all_validation: list[pd.DataFrame] = []

    for replicate, (reference_seed, label_seed) in enumerate(SEED_PAIRS, start=1):
        checkpoint = load_checkpoint(replicate, signature)
        if checkpoint is not None:
            runs, validation = checkpoint
        else:
            runs, validation = run_single_replicate(
                replicate,
                reference_seed,
                label_seed,
                positions,
                velocities,
                eigenfunctions,
                generator_eigenvalues,
            )
            save_checkpoint(replicate, signature, runs, validation)
        all_runs.append(runs)
        all_validation.append(validation)

    return (
        pd.concat(all_runs, ignore_index=True),
        pd.concat(all_validation, ignore_index=True),
    )


# ==================================================================================================
# 7. Merge, audits, aggregation, and practical convergence diagnostics
# ==================================================================================================


def merge_previous_results(
    new_runs: pd.DataFrame,
    new_validation: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    continuity_records: list[dict[str, object]] = []
    if not MERGE_PREVIOUS_RESULTS or not PREVIOUS_RUNS_FILE.is_file():
        return new_runs.copy(), new_validation.copy(), pd.DataFrame()

    previous_runs = pd.read_csv(PREVIOUS_RUNS_FILE)
    previous_validation = (
        pd.read_csv(PREVIOUS_VALIDATION_FILE)
        if PREVIOUS_VALIDATION_FILE.is_file()
        else pd.DataFrame()
    )

    join_basis = REFERENCE_BASIS
    old_join = previous_runs.loc[
        previous_runs["n_basis"] == join_basis
    ].copy()
    new_join = new_runs.loc[new_runs["n_basis"] == join_basis].copy()
    if not old_join.empty and not new_join.empty:
        metrics = [
            "validation_normalized_rmse",
            "selected_normalized_rmse",
            "selected_median_direction_error_deg",
            "selected_vector_gain_through_origin",
            "selected_regularization",
        ]
        merged = new_join.merge(
            old_join,
            on=["replicate", "scale", "n_basis"],
            suffixes=("_new", "_previous"),
            how="inner",
        )
        for metric in metrics:
            if f"{metric}_new" not in merged.columns or f"{metric}_previous" not in merged.columns:
                continue
            difference = (
                merged[f"{metric}_new"] - merged[f"{metric}_previous"]
            )
            continuity_records.append(
                {
                    "n_basis": join_basis,
                    "metric": metric,
                    "n_rows": int(difference.size),
                    "max_abs_difference": float(np.max(np.abs(difference))),
                    "mean_difference": float(np.mean(difference)),
                }
            )

    previous_runs_keep = previous_runs.loc[
        ~previous_runs["n_basis"].isin(NEW_BASIS_SIZES)
    ].copy()
    combined_runs = pd.concat(
        [previous_runs_keep, new_runs],
        ignore_index=True,
        sort=False,
    )

    if previous_validation.empty:
        combined_validation = new_validation.copy()
    else:
        previous_validation_keep = previous_validation.loc[
            ~previous_validation["n_basis"].isin(NEW_BASIS_SIZES)
        ].copy()
        combined_validation = pd.concat(
            [previous_validation_keep, new_validation],
            ignore_index=True,
            sort=False,
        )

    combined_runs = combined_runs.sort_values(
        ["scale", "n_basis", "replicate"]
    ).reset_index(drop=True)
    combined_validation = combined_validation.sort_values(
        ["scale", "n_basis", "replicate", "regularization_strength"]
    ).reset_index(drop=True)
    return combined_runs, combined_validation, pd.DataFrame(continuity_records)



def aggregate_results(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metric_stems = [
        "validation_normalized_rmse",
        "selected_normalized_rmse",
        "selected_median_direction_error_deg",
        "selected_vector_gain_through_origin",
        "selected_mean_component_calibration_slope",
        "fixed_normalized_rmse",
        "fixed_median_direction_error_deg",
        "fixed_vector_gain_through_origin",
    ]
    aggregate_records: list[dict[str, object]] = []
    for (scale, n_basis), group in runs.groupby(["scale", "n_basis"], sort=False):
        record: dict[str, object] = {
            "scale": scale,
            "n_basis": int(n_basis),
            "n_replicates": int(group.shape[0]),
            "mean_reference_bandwidth": float(
                group["median_reference_bandwidth"].mean()
            ),
            "selected_regularization_mean": float(
                group["selected_regularization"].mean()
            ),
        }
        for stem in metric_stems:
            mean_value = float(group[stem].mean())
            std_value = float(group[stem].std(ddof=1))
            if not np.isfinite(std_value):
                std_value = 0.0
            record[f"{stem}_mean"] = mean_value
            record[f"{stem}_std"] = std_value
            record[f"{stem}_sem"] = float(
                std_value / np.sqrt(group.shape[0])
            )
        aggregate_records.append(record)
    aggregate = pd.DataFrame(aggregate_records).sort_values(
        ["scale", "n_basis"]
    ).reset_index(drop=True)

    frequency = (
        runs.groupby(
            ["scale", "n_basis", "selected_regularization"],
            as_index=False,
        )
        .size()
        .rename(columns={"size": "count"})
        .sort_values(["scale", "n_basis", "count"], ascending=[True, True, False])
    )

    reference = runs.loc[
        runs["n_basis"] == REFERENCE_BASIS,
        [
            "replicate",
            "scale",
            "selected_normalized_rmse",
            "selected_median_direction_error_deg",
            "selected_vector_gain_through_origin",
        ],
    ].rename(
        columns={
            "selected_normalized_rmse": "reference_normalized_rmse",
            "selected_median_direction_error_deg": "reference_direction_error_deg",
            "selected_vector_gain_through_origin": "reference_vector_gain",
        }
    )
    if reference.empty:
        raise ValueError(
            f"REFERENCE_BASIS={REFERENCE_BASIS}のrunsがありません。"
        )

    paired = runs.merge(reference, on=["replicate", "scale"], how="left")
    paired["nrmse_difference_from_reference_basis"] = (
        paired["selected_normalized_rmse"] - paired["reference_normalized_rmse"]
    )
    paired["direction_difference_from_reference_basis_deg"] = (
        paired["selected_median_direction_error_deg"]
        - paired["reference_direction_error_deg"]
    )
    paired["vector_gain_difference_from_reference_basis"] = (
        paired["selected_vector_gain_through_origin"] - paired["reference_vector_gain"]
    )

    paired_summary = (
        paired.groupby(["scale", "n_basis"], as_index=False)
        .agg(
            n_replicates=("replicate", "size"),
            mean_nrmse_difference=("nrmse_difference_from_reference_basis", "mean"),
            std_nrmse_difference=("nrmse_difference_from_reference_basis", "std"),
            improvement_count=(
                "nrmse_difference_from_reference_basis",
                lambda values: int(np.sum(np.asarray(values) < 0.0)),
            ),
            mean_direction_difference_deg=(
                "direction_difference_from_reference_basis_deg",
                "mean",
            ),
            mean_vector_gain_difference=(
                "vector_gain_difference_from_reference_basis",
                "mean",
            ),
        )
    )
    return aggregate, frequency, paired, paired_summary



def practical_convergence_diagnostics(
    aggregate: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    diagnostic_records: list[dict[str, object]] = []
    recommendation_records: list[dict[str, object]] = []

    for scale in [item["scale"] for item in SMOOTHING_CONFIGS]:
        selected = aggregate.loc[aggregate["scale"] == scale].sort_values("n_basis")
        selected = selected.reset_index(drop=True)
        if selected.empty:
            continue

        validation_values = selected["validation_normalized_rmse_mean"].to_numpy(
            dtype=np.float64
        )
        test_values = selected["selected_normalized_rmse_mean"].to_numpy(
            dtype=np.float64
        )
        basis_values = selected["n_basis"].to_numpy(dtype=np.int64)

        validation_improvements = np.full(selected.shape[0], np.nan)
        test_improvements = np.full(selected.shape[0], np.nan)
        validation_improvements[1:] = (
            validation_values[:-1] - validation_values[1:]
        ) / validation_values[:-1]
        test_improvements[1:] = (
            test_values[:-1] - test_values[1:]
        ) / test_values[:-1]

        for index, row in selected.iterrows():
            diagnostic_records.append(
                {
                    "scale": scale,
                    "n_basis": int(row["n_basis"]),
                    "previous_n_basis": (
                        int(selected.loc[index - 1, "n_basis"]) if index > 0 else math.nan
                    ),
                    "validation_relative_improvement_from_previous": float(
                        validation_improvements[index]
                    ),
                    "test_relative_improvement_from_previous": float(
                        test_improvements[index]
                    ),
                    "below_validation_plateau_threshold": bool(
                        index > 0
                        and validation_improvements[index]
                        < PLATEAU_RELATIVE_THRESHOLD
                    ),
                }
            )

        minimum_index = int(np.argmin(validation_values))
        minimum_mean = float(validation_values[minimum_index])
        minimum_sem = float(
            selected.loc[minimum_index, "validation_normalized_rmse_sem"]
        )
        if not np.isfinite(minimum_sem):
            minimum_sem = 0.0
        one_se_threshold = minimum_mean + minimum_sem
        eligible = selected.loc[
            selected["validation_normalized_rmse_mean"] <= one_se_threshold
        ]
        if eligible.empty:
            one_se_basis = int(basis_values[minimum_index])
        else:
            one_se_basis = int(eligible["n_basis"].min())

        plateau_basis: int | None = None
        for start_index in range(selected.shape[0] - PLATEAU_CONSECUTIVE_STEPS):
            next_improvements = validation_improvements[
                start_index + 1:
                start_index + 1 + PLATEAU_CONSECUTIVE_STEPS
            ]
            if np.all(next_improvements < PLATEAU_RELATIVE_THRESHOLD):
                plateau_basis = int(basis_values[start_index])
                break

        recommendation_records.append(
            {
                "scale": scale,
                "minimum_validation_basis": int(basis_values[minimum_index]),
                "minimum_validation_mean": minimum_mean,
                "minimum_validation_sem": minimum_sem,
                "one_standard_error_threshold": one_se_threshold,
                "one_standard_error_basis": one_se_basis,
                "practical_plateau_basis": plateau_basis,
                "plateau_relative_threshold": PLATEAU_RELATIVE_THRESHOLD,
                "plateau_consecutive_steps": PLATEAU_CONSECUTIVE_STEPS,
                "last_validation_relative_improvement": float(
                    validation_improvements[-1]
                ),
                "last_test_relative_improvement": float(test_improvements[-1]),
            }
        )

    return pd.DataFrame(diagnostic_records), pd.DataFrame(recommendation_records)



def audit_against_previous_join_basis(
    continuity_audit: pd.DataFrame,
) -> dict[str, object]:
    if continuity_audit.empty:
        return {
            "available": False,
            "join_basis": REFERENCE_BASIS,
        }
    return {
        "available": True,
        "join_basis": REFERENCE_BASIS,
        "max_abs_difference_over_metrics": float(
            continuity_audit["max_abs_difference"].max()
        ),
        "metrics": continuity_audit.to_dict(orient="records"),
    }



def compare_with_local_baseline(
    runs: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if not LOCAL_BASELINE_RUNS_FILE.is_file():
        return pd.DataFrame(), pd.DataFrame()
    local_runs = pd.read_csv(LOCAL_BASELINE_RUNS_FILE)
    required = {"replicate", "scale", "model", "normalized_rmse"}
    if not required.issubset(local_runs.columns):
        warnings.warn("Local-baseline runs CSVのschemaが想定と異なります。")
        return pd.DataFrame(), pd.DataFrame()

    local = local_runs.loc[
        local_runs["model"] == LOCAL_MODEL_NAME,
        ["replicate", "scale", "normalized_rmse"],
    ].rename(columns={"normalized_rmse": "local_normalized_rmse"})
    diffusion = runs[
        ["replicate", "scale", "n_basis", "selected_normalized_rmse"]
    ].copy()
    comparison = diffusion.merge(local, on=["replicate", "scale"], how="inner")
    comparison["diffusion_minus_local_nrmse"] = (
        comparison["selected_normalized_rmse"] - comparison["local_normalized_rmse"]
    )
    summary = (
        comparison.groupby(["scale", "n_basis"], as_index=False)
        .agg(
            n_replicates=("replicate", "size"),
            diffusion_nrmse_mean=("selected_normalized_rmse", "mean"),
            local_nrmse_mean=("local_normalized_rmse", "mean"),
            mean_diffusion_minus_local=("diffusion_minus_local_nrmse", "mean"),
            std_diffusion_minus_local=("diffusion_minus_local_nrmse", "std"),
            diffusion_win_count=(
                "diffusion_minus_local_nrmse",
                lambda values: int(np.sum(np.asarray(values) < 0.0)),
            ),
        )
    )
    return comparison, summary


# ==================================================================================================
# 8. Figures
# ==================================================================================================


def plot_composite(
    aggregate: pd.DataFrame,
    paired_summary: pd.DataFrame,
) -> tuple[Path | None, Path | None]:
    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.0, 10.0),
        constrained_layout=True,
    )
    full_basis_sizes = sorted(int(value) for value in aggregate["n_basis"].unique())

    for scale in [item["scale"] for item in SMOOTHING_CONFIGS]:
        selected = aggregate.loc[aggregate["scale"] == scale].sort_values("n_basis")
        axes[0, 0].errorbar(
            selected["n_basis"],
            selected["selected_normalized_rmse_mean"],
            yerr=selected["selected_normalized_rmse_std"],
            marker="o",
            capsize=4,
            label=SCALE_LABELS[scale],
        )
        axes[0, 1].errorbar(
            selected["n_basis"],
            selected["selected_median_direction_error_deg_mean"],
            yerr=selected["selected_median_direction_error_deg_std"],
            marker="o",
            capsize=4,
            label=SCALE_LABELS[scale],
        )
        axes[1, 0].errorbar(
            selected["n_basis"],
            selected["selected_vector_gain_through_origin_mean"],
            yerr=selected["selected_vector_gain_through_origin_std"],
            marker="o",
            capsize=4,
            label=SCALE_LABELS[scale],
        )

        paired_selected = paired_summary.loc[
            paired_summary["scale"] == scale
        ].sort_values("n_basis")
        axes[1, 1].errorbar(
            paired_selected["n_basis"],
            paired_selected["mean_nrmse_difference"],
            yerr=paired_selected["std_nrmse_difference"],
            marker="o",
            capsize=4,
            label=SCALE_LABELS[scale],
        )

    axes[0, 0].set_title("Mode-number convergence of reconstruction error")
    axes[0, 0].set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
    axes[0, 0].set_ylabel(r"Test $\mathrm{NRMSE}_{3\mathrm{D}}$")
    axes[0, 0].legend(frameon=True)
    add_panel_label(axes[0, 0], PANEL_LABELS[0])

    axes[0, 1].set_title("Directional accuracy")
    axes[0, 1].set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
    axes[0, 1].set_ylabel("Median direction error [deg]")
    axes[0, 1].legend(frameon=True)
    add_panel_label(axes[0, 1], PANEL_LABELS[1])

    axes[1, 0].axhline(1.0, linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("Vector-amplitude calibration")
    axes[1, 0].set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
    axes[1, 0].set_ylabel("Vector gain")
    axes[1, 0].legend(frameon=True)
    add_panel_label(axes[1, 0], PANEL_LABELS[2])

    axes[1, 1].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1, 1].set_title(
        rf"Paired error change relative to $n_{{\mathrm{{b}}}}={REFERENCE_BASIS}$"
    )
    axes[1, 1].set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
    axes[1, 1].set_ylabel(r"Paired test $\Delta\mathrm{NRMSE}_{3\mathrm{D}}$")
    axes[1, 1].legend(frameon=True)
    add_panel_label(axes[1, 1], PANEL_LABELS[3])

    for ax in axes.ravel():
        ax.set_xticks(full_basis_sizes)
        ax.tick_params(axis="x", rotation=28)

    return save_figure(fig, "composite_to_2048")



def plot_selected_vs_fixed(
    aggregate: pd.DataFrame,
) -> tuple[Path | None, Path | None]:
    fig, ax = plt.subplots(figsize=(8.0, 5.2), constrained_layout=True)
    full_basis_sizes = sorted(int(value) for value in aggregate["n_basis"].unique())
    for scale in [item["scale"] for item in SMOOTHING_CONFIGS]:
        selected = aggregate.loc[aggregate["scale"] == scale].sort_values("n_basis")
        ax.plot(
            selected["n_basis"],
            selected["selected_normalized_rmse_mean"],
            marker="o",
            label=rf"{SCALE_LABELS[scale]}: validation-selected $\lambda$",
        )
        ax.plot(
            selected["n_basis"],
            selected["fixed_normalized_rmse_mean"],
            marker="s",
            linestyle="--",
            label=rf"{SCALE_LABELS[scale]}: fixed $\lambda=10^{{-2}}$",
        )
    ax.set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
    ax.set_ylabel(r"Test $\mathrm{NRMSE}_{3\mathrm{D}}$")
    ax.set_title("Regularization-selection sensitivity")
    ax.set_xticks(full_basis_sizes)
    ax.tick_params(axis="x", rotation=28)
    ax.legend(frameon=True, fontsize=9.0, ncol=2)
    return save_figure(fig, "selected_vs_fixed_lambda_to_2048")



def plot_plateau_diagnostics(
    diagnostics: pd.DataFrame,
) -> tuple[Path | None, Path | None]:
    fig, ax = plt.subplots(figsize=(7.5, 5.0), constrained_layout=True)
    for scale in [item["scale"] for item in SMOOTHING_CONFIGS]:
        selected = diagnostics.loc[
            (diagnostics["scale"] == scale)
            & diagnostics["validation_relative_improvement_from_previous"].notna()
        ].sort_values("n_basis")
        ax.plot(
            selected["n_basis"],
            100.0 * selected["validation_relative_improvement_from_previous"],
            marker="o",
            label=SCALE_LABELS[scale],
        )
    ax.axhline(
        100.0 * PLATEAU_RELATIVE_THRESHOLD,
        linestyle="--",
        linewidth=1.0,
        label=rf"{100.0 * PLATEAU_RELATIVE_THRESHOLD:.1f}\% threshold",
    )
    ax.set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
    ax.set_ylabel("Adjacent validation-error reduction [per cent]")
    ax.set_title("Practical plateau diagnostic")
    ax.legend(frameon=True)
    return save_figure(fig, "plateau_diagnostic")



def write_latex_template() -> Path:
    path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_figure_template_to_2048.tex"
    text = (
        "\\begin{figure*}[t]\n"
        "\\centering\n"
        "\\includegraphics[width=\\textwidth]"
        "{mock1_mode_number_convergence_composite_to_2048}\n"
        "\\caption{CAPTION TO BE ADDED AFTER VISUAL INSPECTION.}\n"
        "\\label{fig:mock1_mode_number_convergence}\n"
        "\\end{figure*}\n"
    )
    path.write_text(text, encoding="utf-8")
    return path


# ==================================================================================================
# 9. Main
# ==================================================================================================


def main() -> None:
    font_name = configure_font()
    if not DATA_FILE.is_file():
        raise FileNotFoundError(DATA_FILE)

    with np.load(DATA_FILE, allow_pickle=False) as loaded:
        positions_absolute = np.asarray(loaded["pos"], dtype=np.float64)
        velocities = np.asarray(loaded["vel"], dtype=np.float64)
    positions = positions_absolute - SPHERE_CENTER[None, :]

    if MAX_MODES >= positions.shape[0]:
        raise ValueError("最大mode数は天体数より小さくしてください。")
    if max(int(item["n_neighbors"]) for item in SMOOTHING_CONFIGS) >= int(
        round(REFERENCE_FRACTION * positions.shape[0])
    ):
        raise ValueError("Reference-field neighbor数がreference sample数以上です。")

    print("=" * 116)
    print("Paper I: Mock-1 mode-number convergence to 2048 modes")
    print("=" * 116)
    print(
        f"Python / NumPy / SciPy : {platform.python_version()} / "
        f"{np.__version__} / {scipy.__version__}"
    )
    print(f"Matplotlib font         : {font_name}")
    print(f"Input                   : {DATA_FILE}")
    print(f"Objects                 : {positions.shape[0]:,}")
    print(f"Seed pairs              : {len(SEED_PAIRS)}")
    print(f"Scales                  : {', '.join(SCALES_TO_RUN)}")
    print(f"New basis sizes         : {NEW_BASIS_SIZES}")
    print(f"Reference basis         : {REFERENCE_BASIS}")
    print(f"Lambda grid             : {REGULARIZATION_STRENGTHS}")
    print(f"Fixed lambda            : {FIXED_REGULARIZATION:g}")
    print(f"Extended geometry cache : {EXTENDED_GEOMETRY_CACHE_FILE}")
    print(f"Previous output         : {PREVIOUS_OUTPUT_DIR}")
    print(f"Output                  : {OUTPUT_DIR}")
    print(f"Checkpoint directory    : {CHECKPOINT_DIR}")

    geometry = load_or_build_extended_geometry(positions)
    geometry_audit_df, geometry_audit_summary = audit_extended_geometry(geometry)
    if not geometry_audit_df.empty:
        print("-" * 116)
        print("Extended-geometry audit")
        print(geometry_audit_df.to_string(index=False))

    new_runs_df, new_validation_df = run_convergence(
        positions,
        velocities,
        np.asarray(geometry["eigenfunctions"], dtype=np.float64),
        np.asarray(geometry["generator_eigenvalues"], dtype=np.float64),
        str(geometry["signature"]),
    )
    runs_df, validation_df, continuity_audit_df = merge_previous_results(
        new_runs_df,
        new_validation_df,
    )
    aggregate_df, frequency_df, paired_df, paired_summary_df = aggregate_results(
        runs_df
    )
    plateau_df, recommendation_df = practical_convergence_diagnostics(aggregate_df)
    local_comparison_df, local_summary_df = compare_with_local_baseline(runs_df)

    paths = {
        "new_runs_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_new_high_mode_runs.csv",
        "new_validation_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_new_high_mode_validation.csv",
        "runs_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_runs_to_2048.csv",
        "validation_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_validation_to_2048.csv",
        "aggregate_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_aggregate_to_2048.csv",
        "lambda_frequency_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_lambda_frequency_to_2048.csv",
        "paired_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_paired_relative_to_{REFERENCE_BASIS}.csv",
        "paired_summary_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_paired_summary_relative_to_{REFERENCE_BASIS}.csv",
        "geometry_audit_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_audit_to_2048.csv",
        "continuity_audit_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_continuity_audit_at_{REFERENCE_BASIS}.csv",
        "plateau_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_plateau_diagnostics.csv",
        "recommendation_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_practical_selection.csv",
        "local_comparison_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_diffusion_vs_local.csv",
        "local_summary_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_diffusion_vs_local_summary.csv",
        "summary_json": OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary_to_2048.json",
    }

    new_runs_df.to_csv(paths["new_runs_csv"], index=False)
    new_validation_df.to_csv(paths["new_validation_csv"], index=False)
    runs_df.to_csv(paths["runs_csv"], index=False)
    validation_df.to_csv(paths["validation_csv"], index=False)
    aggregate_df.to_csv(paths["aggregate_csv"], index=False)
    frequency_df.to_csv(paths["lambda_frequency_csv"], index=False)
    paired_df.to_csv(paths["paired_csv"], index=False)
    paired_summary_df.to_csv(paths["paired_summary_csv"], index=False)
    geometry_audit_df.to_csv(paths["geometry_audit_csv"], index=False)
    continuity_audit_df.to_csv(paths["continuity_audit_csv"], index=False)
    plateau_df.to_csv(paths["plateau_csv"], index=False)
    recommendation_df.to_csv(paths["recommendation_csv"], index=False)
    local_comparison_df.to_csv(paths["local_comparison_csv"], index=False)
    local_summary_df.to_csv(paths["local_summary_csv"], index=False)

    composite_pdf, composite_png = plot_composite(aggregate_df, paired_summary_df)
    sensitivity_pdf, sensitivity_png = plot_selected_vs_fixed(aggregate_df)
    plateau_pdf, plateau_png = plot_plateau_diagnostics(plateau_df)
    latex_template = write_latex_template()

    summary_payload = {
        "experiment": "Mock-1 mode-number convergence to 2048",
        "input_file": str(DATA_FILE),
        "extended_geometry_cache": str(EXTENDED_GEOMETRY_CACHE_FILE),
        "new_basis_sizes": NEW_BASIS_SIZES,
        "combined_basis_sizes": sorted(
            int(value) for value in aggregate_df["n_basis"].unique()
        ),
        "reference_basis": REFERENCE_BASIS,
        "regularization_strengths": REGULARIZATION_STRENGTHS,
        "fixed_regularization": FIXED_REGULARIZATION,
        "scales": SCALES_TO_RUN,
        "seed_pairs": [
            {"reference_seed": a, "label_seed": b} for a, b in SEED_PAIRS
        ],
        "generator_normalization_reference_modes": (
            GENERATOR_NORMALIZATION_REFERENCE_MODES
        ),
        "geometry": {
            "n_neighbors": GEOMETRY_NEIGHBORS,
            "bandwidth_neighbor": GEOMETRY_BANDWIDTH_NEIGHBOR,
            "bandwidth_multiplier": GEOMETRY_BANDWIDTH_MULTIPLIER,
            "alpha": ALPHA_NORMALIZATION,
            "generator_scale": float(geometry["generator_scale"]),
            "orthogonality_rms": float(geometry["orthogonality_rms"]),
            "orthogonality_kind": str(geometry["orthogonality_kind"]),
            "orthogonality_modes": int(geometry["orthogonality_modes"]),
            "column_norm_max_abs_error": float(
                geometry["column_norm_max_abs_error"]
            ),
            "kernel_nnz": int(geometry["kernel_nnz"]),
        },
        "geometry_audit": geometry_audit_summary,
        "continuity_audit": audit_against_previous_join_basis(
            continuity_audit_df
        ),
        "practical_selection": recommendation_df.to_dict(orient="records"),
        "maximum_mode_summary": paired_summary_df.loc[
            paired_summary_df["n_basis"] == MAX_MODES
        ].to_dict(orient="records"),
        "lambda_frequency": frequency_df.to_dict(orient="records"),
        "diffusion_vs_local_at_maximum_mode": local_summary_df.loc[
            local_summary_df["n_basis"] == MAX_MODES
        ].to_dict(orient="records")
        if not local_summary_df.empty
        else [],
        "outputs": {
            **{key: str(value) for key, value in paths.items()},
            "composite_pdf": str(composite_pdf),
            "composite_png": str(composite_png),
            "sensitivity_pdf": str(sensitivity_pdf),
            "sensitivity_png": str(sensitivity_png),
            "plateau_pdf": str(plateau_pdf),
            "plateau_png": str(plateau_png),
            "latex_template": str(latex_template),
        },
    }
    paths["summary_json"].write_text(
        json.dumps(summary_payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print("=" * 116)
    print("Mode-number convergence to 2048 completed")
    print("=" * 116)
    display_columns = [
        "scale",
        "n_basis",
        "validation_normalized_rmse_mean",
        "selected_normalized_rmse_mean",
        "selected_normalized_rmse_std",
        "selected_median_direction_error_deg_mean",
        "selected_vector_gain_through_origin_mean",
    ]
    try:
        from IPython.display import display

        display(aggregate_df[display_columns])
        display(frequency_df)
        display(paired_summary_df)
        display(recommendation_df)
        if not continuity_audit_df.empty:
            display(continuity_audit_df)
        if not local_summary_df.empty:
            display(
                local_summary_df.loc[
                    local_summary_df["n_basis"].isin(NEW_BASIS_SIZES)
                ]
            )
    except ImportError:
        print(aggregate_df[display_columns].to_string(index=False))
        print(recommendation_df.to_string(index=False))

    for name, path in paths.items():
        print(f"{name:36s}: {path}")
    print(f"composite_pdf                       : {composite_pdf}")
    print(f"composite_png                       : {composite_png}")
    print(f"sensitivity_pdf                     : {sensitivity_pdf}")
    print(f"plateau_pdf                         : {plateau_pdf}")
    print(f"latex_template                      : {latex_template}")
    print("=" * 116)


if __name__ == "__main__":
    main()
