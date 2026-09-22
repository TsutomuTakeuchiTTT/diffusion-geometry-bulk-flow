# ==================================================================================================
# Radial Mock-3: high-mode 10-label-seed robustness after H2 convergence
# ==================================================================================================
#
# Purpose
# -------
# This script extends the capacity-converged Radial Mock-3 H2 analysis from the representative
# octant-stratified label split to ten preregistered label seeds. It reuses the completed H2
# 4352-mode geometry and metric-calibrated gradient/design cache. No eigensystem or gradient basis
# is recomputed.
#
# Primary rules
# -------------
#   * operator-side unweighted alpha_DM=0 geometry is fixed;
#   * unweighted, global-radial, and octant-aware loss families are compared;
#   * within every seed, loss family, and basis, lambda and penalty power are selected using only
#     octant-parent-weighted radial validation risk;
#   * capacity curves are aggregated with equal weight across ten octant-stratified label seeds;
#   * one-standard-error and two-step 2% practical-plateau rules determine a shared practical basis
#     for each loss family;
#   * hidden tangential/3D truth, complete-query truth, octant dispersion, and boundary diagnostics
#     are used only after the observable selection policy has been fixed.
#
# Resume design
# -------------
# Candidate calculations are checkpointed separately for every seed, loss family, lambda, and
# penalty power. Final post-selection fits are checkpointed by seed, and writes use atomic
# replacement. A progress JSON file remains readable after a frontend disconnect.
#
# Standard Jupyter execution
# --------------------------
#   %run mock3_radial_potential_highmode_seed_robustness_v1.py
#
from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import time
import threading
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from IPython.display import display
from numpy.lib.format import open_memmap
from scipy import linalg, sparse
from scipy.sparse.linalg import ArpackNoConvergence, LinearOperator, eigsh
from scipy.spatial import cKDTree


# ==================================================================================================
# 1. Fixed paths and experiment configuration
# ==================================================================================================

ROOT_DIR = Path(
    os.environ.get(
        "COSMIC_DIPOLE_MOCK_DATA_ROOT",
        'mock_data',
    )
)

MOCK1_FILE = ROOT_DIR / "mock1_complete_sphere" / "mock1.npz"
MOCK3_FILE = ROOT_DIR / "mock3_inhomogeneous_survey" / "mock3.npz"
# Compatibility alias used by inherited generic helpers.
MOCK2_FILE = MOCK3_FILE

SOURCE_CANDIDATE_DIRS = [
    ROOT_DIR / "mock3_radial_potential_octant_selection_v2",
    ROOT_DIR / "mock3_radial_potential_octant_selection_v1",
]

LEGACY_GEOMETRY_CANDIDATES = [
    ROOT_DIR
    / "mock3_octant_aware_loss_optimization_v4"
    / "cache"
    / "unweighted_alpha0_geometry_m512.npz",
    ROOT_DIR
    / "mock3_radial_potential_octant_selection_v2"
    / "cache"
    / "unweighted_alpha0_geometry_m512.npz",
    ROOT_DIR
    / "mock3_radial_potential_octant_selection_v1"
    / "cache"
    / "unweighted_alpha0_geometry_m512.npz",
]

_extended_geometry_override = os.environ.get(
    "RADIAL_MOCK3_HIGHMODE_H2_GEOMETRY",
    "",
).strip()
_reference_geometry_override = os.environ.get(
    "RADIAL_MOCK3_HIGHMODE_H2_REFERENCE_GEOMETRY",
    "",
).strip()
REFERENCE_GEOMETRY_CANDIDATES = [
    ROOT_DIR
    / "mock3_primary_mode_convergence_to_4096_v1"
    / "cache"
    / "mock3_unweighted_alpha0_geometry_m4096.npz",
    ROOT_DIR
    / "mock3_primary_mode_convergence_to_4096"
    / "cache"
    / "mock3_unweighted_alpha0_geometry_m4096.npz",
]
if _reference_geometry_override:
    REFERENCE_GEOMETRY_CANDIDATES.insert(0, Path(_reference_geometry_override))

LEGACY_VALIDATION_CANDIDATES = [
    ROOT_DIR
    / "mock3_radial_potential_octant_seed_robustness_v3"
    / "mock3_radial_potential_octant_seed_robustness_validation_grid.csv",
    ROOT_DIR
    / "mock3_radial_potential_octant_seed_robustness_v2"
    / "mock3_radial_potential_octant_seed_robustness_validation_grid.csv",
]

_previous_h1_output_override = os.environ.get(
    "RADIAL_MOCK3_HIGHMODE_H2_PREVIOUS_H1_OUTPUT",
    "",
).strip()
PREVIOUS_H1_OUTPUT_DIR = Path(
    _previous_h1_output_override
    if _previous_h1_output_override
    else ROOT_DIR / "mock3_radial_potential_highmode_capacity_h1_v1"
)
PREVIOUS_H1_PREFIX = "mock3_radial_potential_highmode_capacity_h1"
PREVIOUS_H1_CACHE_DIR = PREVIOUS_H1_OUTPUT_DIR / "cache"
PREVIOUS_H1_BASIS_METADATA_FILE = PREVIOUS_H1_CACHE_DIR / "highmode_basis_metadata.json"
PREVIOUS_H1_OBSERVED_GRADIENT_FILE = PREVIOUS_H1_CACHE_DIR / "observed_nonconstant_gradient.npy"
PREVIOUS_H1_QUERY_PRIMARY_GRADIENT_FILE = PREVIOUS_H1_CACHE_DIR / "query_primary_nonconstant_gradient.npy"
PREVIOUS_H1_RADIAL_DESIGN_FILE = PREVIOUS_H1_CACHE_DIR / "observed_radial_design.npy"
PREVIOUS_H1_COLUMN_RMS_FILE = PREVIOUS_H1_CACHE_DIR / "nonconstant_column_rms.npy"
PREVIOUS_H1_CANDIDATE_GRID_FILE = (
    PREVIOUS_H1_OUTPUT_DIR / f"{PREVIOUS_H1_PREFIX}_candidate_grid.csv"
)
PREVIOUS_H1_GLOBAL_SELECTORS_FILE = (
    PREVIOUS_H1_OUTPUT_DIR / f"{PREVIOUS_H1_PREFIX}_global_selectors.csv"
)
PREVIOUS_H1_MAX_NONCONSTANT_MODES = 4095
PREVIOUS_H1_MAX_BASIS = 4098

SMOKE_TEST = os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_SMOKE_TEST", "0") == "1"
FORCE_REBUILD_CACHE = (
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_FORCE_REBUILD_CACHE", "0") == "1"
)
FORCE_REBUILD_GEOMETRY = (
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_FORCE_REBUILD_GEOMETRY", "0") == "1"
)
RESUME = os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_RESUME", "1") == "1"

_output_default = "mock3_radial_potential_highmode_capacity_h2_v1"
if SMOKE_TEST:
    _output_default += "_smoke"
OUTPUT_DIR = Path(
    os.environ.get(
        "RADIAL_MOCK3_HIGHMODE_H2_OUTPUT",
        str(ROOT_DIR / _output_default),
    )
)
CACHE_DIR = OUTPUT_DIR / "cache"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
for _directory in (OUTPUT_DIR, CACHE_DIR, CHECKPOINT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = "mock3_radial_potential_highmode_capacity_h2"

EIGEN_SOLVER_SEED = int(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_EIGEN_SEED", "20260820")
)
GEOMETRY_SOLVER = os.environ.get(
    "RADIAL_MOCK3_HIGHMODE_H2_GEOMETRY_SOLVER",
    "eigsh",
).strip().lower()
if GEOMETRY_SOLVER not in {"eigsh", "dense_eigh"}:
    raise ValueError(
        "RADIAL_MOCK3_HIGHMODE_H2_GEOMETRY_SOLVER must be 'eigsh' or 'dense_eigh'."
    )
GEOMETRY_HEARTBEAT_SECONDS = float(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_HEARTBEAT_SECONDS", "60")
)
CONTINUATION_OVERLAP_MODES = int(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_CONTINUATION_OVERLAP", "64")
)
_EIGSH_NCV_TEXT = os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_EIGSH_NCV", "").strip()
EIGSH_NCV = int(_EIGSH_NCV_TEXT) if _EIGSH_NCV_TEXT else None

CONTINUATION_BLOCK_SIZE = int(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_BLOCK_SIZE", "64")
)
CONTINUATION_EIGSH_TOL = float(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_BLOCK_TOL", "1e-8")
)
CONTINUATION_EIGSH_MAXITER = int(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_BLOCK_MAXITER", "120000")
)
CONTINUATION_DEFLATION_SHIFT = float(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_DEFLATION_SHIFT", "2.5")
)
_CONTINUATION_NCV_TEXT = os.environ.get(
    "RADIAL_MOCK3_HIGHMODE_H2_BLOCK_NCV", ""
).strip()
CONTINUATION_EIGSH_NCV = (
    int(_CONTINUATION_NCV_TEXT) if _CONTINUATION_NCV_TEXT else None
)
CONTINUATION_RESIDUAL_WARN = float(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_RESIDUAL_WARN", "1e-6")
)
CONTINUATION_RESIDUAL_FAIL = float(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_RESIDUAL_FAIL", "5e-4")
)
CONTINUATION_CHECKPOINT_DIR = CACHE_DIR / "continuation_blocks"
CONTINUATION_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BASIS_PROGRESS_FILE = CACHE_DIR / "highmode_basis_progress.json"

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)
LABEL_SPLIT_SEED = int(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_LABEL_SEED", "20261203")
)
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

GRAPH_NEIGHBORS = 48
GRAPH_BANDWIDTH_NEIGHBOR = 16
GRAPH_BANDWIDTH_MULTIPLIER = 1.0
ALPHA_DM = 0.0

METRIC_EIGENVALUE_RELATIVE_FLOOR = 1.0e-10
METRIC_EIGENVALUE_ABSOLUTE_FLOOR = 1.0e-14
GRADIENT_MODE_CHUNK_SIZE = int(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_GRADIENT_CHUNK", "32")
)

MIN_SELECTION_PROBABILITY = 1.0e-4
WEIGHT_CLIP_QUANTILE = 0.995
PRIMARY_METRIC_GAMMA = 1.0
PRIMARY_METRIC_CAP = 50.0
GLOBAL_METRIC_GAMMA = 1.0
GLOBAL_METRIC_CAP = 50.0

LOSS_CONFIGS = [
    {
        "name": "unweighted",
        "label": "Unweighted",
        "mode": "none",
        "gamma": 0.0,
        "cap": 1.0,
    },
    {
        "name": "global_g0p75_cap20",
        "label": r"Global radial ($\gamma=0.75$, cap 20)",
        "mode": "global",
        "gamma": 0.75,
        "cap": 20.0,
    },
    {
        "name": "octant_g1p00_cap20",
        "label": r"Octant-aware ($\gamma=1$, cap 20)",
        "mode": "region",
        "gamma": 1.0,
        "cap": 20.0,
    },
]
LOSS_ORDER = [item["name"] for item in LOSS_CONFIGS]
LOSS_LABELS = {item["name"]: item["label"] for item in LOSS_CONFIGS}

N_AFFINE_MODES = 3
GENERATOR_REFERENCE_NONCONSTANT_MODES = 511

_DEFAULT_MAX_NONCONSTANT = 4351
MAX_NONCONSTANT_MODES = int(
    os.environ.get(
        "RADIAL_MOCK3_HIGHMODE_H2_MAX_NONCONSTANT",
        str(96 if SMOKE_TEST else _DEFAULT_MAX_NONCONSTANT),
    )
)

_DEFAULT_DIFFUSION_MODE_COUNTS = [
    0, 32, 64, 128, 256, 384, 511, 640, 768, 896, 1024, 1280,
    1536, 1792, 2047, 2304, 2559, 2815, 3071, 3327, 3583, 3839,
    4095, 4351,
]
if SMOKE_TEST:
    _DEFAULT_DIFFUSION_MODE_COUNTS = [0, 8, 32, 64, 96]

_basis_override = os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_MODE_COUNTS", "").strip()
if _basis_override:
    POTENTIAL_DIFFUSION_MODE_COUNTS = [
        int(value) for value in _basis_override.split(",") if value.strip()
    ]
else:
    POTENTIAL_DIFFUSION_MODE_COUNTS = list(_DEFAULT_DIFFUSION_MODE_COUNTS)
POTENTIAL_DIFFUSION_MODE_COUNTS = sorted(
    {
        value
        for value in POTENTIAL_DIFFUSION_MODE_COUNTS
        if 0 <= value <= MAX_NONCONSTANT_MODES
    }
)
if not POTENTIAL_DIFFUSION_MODE_COUNTS:
    raise ValueError("POTENTIAL_DIFFUSION_MODE_COUNTS is empty.")
if POTENTIAL_DIFFUSION_MODE_COUNTS[0] != 0:
    POTENTIAL_DIFFUSION_MODE_COUNTS.insert(0, 0)
if POTENTIAL_DIFFUSION_MODE_COUNTS[-1] != MAX_NONCONSTANT_MODES:
    POTENTIAL_DIFFUSION_MODE_COUNTS.append(MAX_NONCONSTANT_MODES)

POTENTIAL_BASIS_SIZES = [
    N_AFFINE_MODES + count for count in POTENTIAL_DIFFUSION_MODE_COUNTS
]
MAX_BASIS = int(max(POTENTIAL_BASIS_SIZES))
TARGET_TOTAL_GEOMETRY_MODES = MAX_NONCONSTANT_MODES + 1
REFERENCE_TOTAL_GEOMETRY_MODES = PREVIOUS_H1_MAX_NONCONSTANT_MODES + 1
EXTENDED_GEOMETRY_FILE = (
    Path(_extended_geometry_override)
    if _extended_geometry_override
    else CACHE_DIR / f"mock3_unweighted_alpha0_geometry_m{TARGET_TOTAL_GEOMETRY_MODES}.npz"
)
EXTENDED_GEOMETRY_METADATA_FILE = CACHE_DIR / (
    f"mock3_unweighted_alpha0_geometry_m{TARGET_TOTAL_GEOMETRY_MODES}_metadata.json"
)

if SMOKE_TEST:
    POTENTIAL_REGULARIZATION_STRENGTHS = [1.0e-6, 1.0e-4]
else:
    POTENTIAL_REGULARIZATION_STRENGTHS = [
        1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0,
    ]
POTENTIAL_REGULARIZATION_POWERS = [1.0, 2.0]
NUMERICAL_RIDGE = 1.0e-12

PRACTICAL_PLATEAU_RELATIVE_THRESHOLD = 0.02
PRACTICAL_PLATEAU_CONSECUTIVE_STEPS = 2

SHOW_PLOTS = os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_SHOW_PLOTS", "1") == "1"
SAVE_PDF = os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_SAVE_PDF", "1") == "1"
SAVE_PNG = os.environ.get("RADIAL_MOCK3_HIGHMODE_H2_SAVE_PNG", "1") == "1"

BASIS_CACHE_METADATA_FILE = CACHE_DIR / "highmode_basis_metadata.json"
OBSERVED_GRADIENT_FILE = CACHE_DIR / "observed_nonconstant_gradient.npy"
QUERY_PRIMARY_GRADIENT_FILE = CACHE_DIR / "query_primary_nonconstant_gradient.npy"
RADIAL_DESIGN_FILE = CACHE_DIR / "observed_radial_design.npy"
COLUMN_RMS_FILE = CACHE_DIR / "nonconstant_column_rms.npy"
BASIS_PARTIAL_METADATA_FILE = CACHE_DIR / "highmode_basis_partial.json"

# Assigned after source discovery in main().
PREDICTIONS_FILE = ROOT_DIR / "__unresolved_mock3_predictions__.npz"
LEGACY_GRADIENT_CACHE_FILE = ROOT_DIR / "__unresolved_mock3_gradient_cache__.npz"
LEGACY_V2_VALIDATION_FILE = ROOT_DIR / "__unresolved_mock3_validation__.csv"


# ==================================================================================================
# 2. General utilities
# ==================================================================================================


def _json_ready(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, Path):
        return str(value)
    return value


def _file_signature(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stable_signature(payload: dict[str, Any]) -> str:
    text = json.dumps(_json_ready(payload), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _hash_array(values) -> str:
    array = np.ascontiguousarray(values)
    return hashlib.sha256(array.view(np.uint8)).hexdigest()


def _find_existing(paths: Iterable[Path], description: str) -> Path:
    for path in paths:
        if path.is_file():
            return path
    raise FileNotFoundError(
        f"Could not find {description}. Checked:\n"
        + "\n".join(f"  {path}" for path in paths)
    )


def _save_figure(fig, stem: str) -> None:
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(
            OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.png",
            dpi=190,
            bbox_inches="tight",
        )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


def _make_split(n: int, train_fraction: float, validation_fraction: float, seed: int):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    n_validation = int(round(validation_fraction * n))
    return {
        "train": np.sort(order[:n_train]),
        "validation": np.sort(order[n_train : n_train + n_validation]),
        "test": np.sort(order[n_train + n_validation :]),
    }


def _effective_sample_size(weights) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(np.square(weights)))
    return (
        float(np.square(np.sum(weights)) / denominator)
        if denominator > 0.0
        else math.nan
    )


def _tempered_inverse_selection_weights(probability, gamma, cap):
    probability = np.maximum(
        np.asarray(probability, dtype=np.float64),
        MIN_SELECTION_PROBABILITY,
    )
    if float(gamma) == 0.0:
        weights = np.ones(probability.shape, dtype=np.float64)
        return weights, {
            "gamma": 0.0,
            "cap": 1.0,
            "probability_min": float(np.min(probability)),
            "probability_median": float(np.median(probability)),
            "normalized_weight_max": 1.0,
            "effective_sample_size": float(weights.size),
            "effective_sample_fraction": 1.0,
        }
    raw = np.power(probability, -float(gamma))
    empirical_cap = float(np.quantile(raw, WEIGHT_CLIP_QUANTILE))
    actual_cap = min(float(cap), empirical_cap)
    weights = np.minimum(raw, actual_cap)
    weights /= np.mean(weights)
    ess = float(_effective_sample_size(weights))
    return weights, {
        "gamma": float(gamma),
        "cap": float(cap),
        "empirical_cap": empirical_cap,
        "actual_cap": actual_cap,
        "probability_min": float(np.min(probability)),
        "probability_median": float(np.median(probability)),
        "normalized_weight_max": float(np.max(weights)),
        "effective_sample_size": ess,
        "effective_sample_fraction": float(ess / weights.size),
    }


def _loss_config(name: str) -> dict[str, Any]:
    return next(item for item in LOSS_CONFIGS if item["name"] == name)


def _load_mock1(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"pos", "vel", "ids", "dist"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Mock-1 is missing keys: {sorted(missing)}")
        return {
            "pos": np.asarray(data["pos"], dtype=np.float64),
            "vel": np.asarray(data["vel"], dtype=np.float64),
            "ids": np.asarray(data["ids"]),
            "dist": np.asarray(data["dist"], dtype=np.float64),
        }


def _load_survey(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"pos", "vel", "ids", "dist", "m_app", "M_abs", "m_lim"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Mock-3 is missing keys: {sorted(missing)}")
        pos = np.asarray(data["pos"], dtype=np.float64)
        n = pos.shape[0]
        m_lim = np.asarray(data["m_lim"], dtype=np.float64)
        if m_lim.ndim == 0:
            m_lim = np.full(n, float(m_lim), dtype=np.float64)
        return {
            "pos": pos,
            "vel": np.asarray(data["vel"], dtype=np.float64),
            "ids": np.asarray(data["ids"]),
            "dist": np.asarray(data["dist"], dtype=np.float64),
            "m_app": np.asarray(data["m_app"], dtype=np.float64),
            "M_abs": np.asarray(data["M_abs"], dtype=np.float64),
            "m_lim": m_lim,
        }


def _load_geometry(path: Path) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as data:
        required = {
            "eigenvalues",
            "generator_eigenvalues",
            "eigenfunctions",
            "stationary",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Geometry cache {path} is missing keys: {sorted(missing)}")
        output = {
            "eigenvalues": np.asarray(data["eigenvalues"], dtype=np.float64),
            "generator_eigenvalues": np.asarray(
                data["generator_eigenvalues"], dtype=np.float64
            ),
            "eigenfunctions": np.asarray(data["eigenfunctions"], dtype=np.float64),
            "stationary": np.asarray(data["stationary"], dtype=np.float64),
            "alpha": float(data["alpha"].item()) if "alpha" in data.files else 0.0,
            "generator_scale": (
                float(data["generator_scale"].item())
                if "generator_scale" in data.files
                else math.nan
            ),
        }
    return output



# ==================================================================================================
# 2b. Minimal nested eigensystem continuation beyond the H2 ceiling
# ==================================================================================================


def _geometry_signature(reference_geometry_file: Path) -> str:
    return _stable_signature(
        {
            "mock3": _file_signature(MOCK3_FILE),
            "reference_geometry": _file_signature(reference_geometry_file),
            "graph_neighbors": GRAPH_NEIGHBORS,
            "graph_bandwidth_neighbor": GRAPH_BANDWIDTH_NEIGHBOR,
            "graph_bandwidth_multiplier": GRAPH_BANDWIDTH_MULTIPLIER,
            "alpha_dm": ALPHA_DM,
            "target_total_modes": TARGET_TOTAL_GEOMETRY_MODES,
            "reference_total_modes": REFERENCE_TOTAL_GEOMETRY_MODES,
            "eigen_solver_seed": EIGEN_SOLVER_SEED,
            "continuation_algorithm": "blockwise_deflated_eigsh",
            "continuation_block_size": CONTINUATION_BLOCK_SIZE,
            "continuation_tol": CONTINUATION_EIGSH_TOL,
            "continuation_maxiter": CONTINUATION_EIGSH_MAXITER,
            "continuation_ncv": CONTINUATION_EIGSH_NCV,
            "continuation_deflation_shift": CONTINUATION_DEFLATION_SHIFT,
            "version": 2,
        }
    )


def _start_heartbeat(label: str):
    stop_event = threading.Event()
    started = time.perf_counter()

    def worker():
        interval = max(float(GEOMETRY_HEARTBEAT_SECONDS), 5.0)
        while not stop_event.wait(interval):
            elapsed = (time.perf_counter() - started) / 60.0
            print(
                f"[{label}] still running: elapsed={elapsed:.1f} min",
                flush=True,
            )

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()
    return stop_event, thread, started


def _stop_heartbeat(stop_event, thread):
    stop_event.set()
    thread.join(timeout=2.0)


def _symmetric_diffusion_operator(base_kernel):
    kernel = base_kernel["kernel"]
    n = int(kernel.shape[0])
    quadrature_weights = np.ones(n, dtype=np.float64)
    q = np.asarray(kernel @ quadrature_weights, dtype=np.float64).ravel()
    if np.any(q <= 0.0):
        raise ValueError("Non-positive kernel density in H2 geometry construction.")

    q_factor = np.power(q, -float(ALPHA_DM))
    kernel_alpha = (
        sparse.diags(q_factor) @ kernel @ sparse.diags(q_factor)
    ).tocsr()
    d = np.asarray(kernel_alpha @ quadrature_weights, dtype=np.float64).ravel()
    if np.any(d <= 0.0):
        raise ValueError("Non-positive row degree in H2 geometry construction.")

    stationary = quadrature_weights * d
    stationary /= np.sum(stationary)
    symmetric_factor = np.sqrt(quadrature_weights / d)
    symmetric_operator = (
        sparse.diags(symmetric_factor)
        @ kernel_alpha
        @ sparse.diags(symmetric_factor)
    )
    symmetric_operator = (
        0.5 * (symmetric_operator + symmetric_operator.T)
    ).tocsr()
    return {
        "operator": symmetric_operator,
        "quadrature_weights": quadrature_weights,
        "q": q,
        "d": d,
        "stationary": stationary,
    }


def _atomic_write_text(path: Path, text: str):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_savez(path: Path, **payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **payload)
    os.replace(temporary, path)


def _project_out_locked(values, q_reference, locked_blocks):
    array = np.asarray(values, dtype=np.float64)
    vector_input = array.ndim == 1
    work = array[:, None].copy() if vector_input else array.copy()
    # Two projection passes suppress roundoff leakage into the 4096-dimensional
    # locked subspace without constructing its 4096 x 4096 Gram matrix.
    for _ in range(2):
        work -= q_reference @ (q_reference.T @ work)
        for block in locked_blocks:
            work -= block @ (block.T @ work)
    return work[:, 0] if vector_input else work


def _continuation_checkpoint_path(start_mode: int) -> Path:
    return CONTINUATION_CHECKPOINT_DIR / f"continuation_block_{start_mode:04d}.npz"


def _load_continuation_checkpoints(signature, q_reference, n_extra):
    blocks = []
    values = []
    residuals = []
    completed = 0
    while completed < n_extra:
        path = _continuation_checkpoint_path(completed)
        if not (RESUME and path.is_file()):
            break
        try:
            with np.load(path, allow_pickle=False) as data:
                cached_signature = str(data["signature"].item())
                cached_start = int(data["start_mode"].item())
                block_values = np.asarray(data["eigenvalues"], dtype=np.float64)
                block_vectors = np.asarray(data["eigenvectors"], dtype=np.float64)
                block_residuals = np.asarray(data["residual_norms"], dtype=np.float64)
            valid = bool(
                cached_signature == signature
                and cached_start == completed
                and block_vectors.shape == (q_reference.shape[0], block_values.size)
                and block_residuals.shape == block_values.shape
                and 0 < block_values.size <= n_extra - completed
                and np.all(np.isfinite(block_values))
                and np.all(np.isfinite(block_vectors))
            )
            if not valid:
                warnings.warn(f"Ignoring invalid continuation checkpoint: {path}")
                break
            block_vectors = _project_out_locked(block_vectors, q_reference, blocks)
            block_vectors, triangular = linalg.qr(
                block_vectors, mode="economic", check_finite=False
            )
            diagonal = np.abs(np.diag(triangular))
            if diagonal.size != block_values.size or np.min(diagonal) < 1.0e-10:
                warnings.warn(f"Ignoring rank-deficient continuation checkpoint: {path}")
                break
            blocks.append(np.asfortranarray(block_vectors))
            values.append(block_values)
            residuals.append(block_residuals)
            completed += block_values.size
            print(
                f"[Continuation checkpoint] reuse: {completed}/{n_extra} extra modes",
                flush=True,
            )
        except Exception as exc:
            warnings.warn(f"Could not reuse continuation checkpoint {path}: {exc}")
            break
    return blocks, values, residuals, completed


def _deflated_linear_operator(symmetric_operator, q_reference, locked_blocks):
    n = int(symmetric_operator.shape[0])
    shift = float(CONTINUATION_DEFLATION_SHIFT)

    def matvec(vector):
        vector = np.asarray(vector, dtype=np.float64)
        projected = _project_out_locked(vector, q_reference, locked_blocks)
        locked_component = vector - projected
        result = symmetric_operator @ projected
        result = _project_out_locked(result, q_reference, locked_blocks)
        return np.asarray(result - shift * locked_component, dtype=np.float64)

    def matmat(matrix):
        matrix = np.asarray(matrix, dtype=np.float64)
        projected = _project_out_locked(matrix, q_reference, locked_blocks)
        locked_component = matrix - projected
        result = symmetric_operator @ projected
        result = _project_out_locked(result, q_reference, locked_blocks)
        return np.asarray(result - shift * locked_component, dtype=np.float64)

    return LinearOperator(
        shape=(n, n),
        matvec=matvec,
        matmat=matmat,
        dtype=np.float64,
    )


def _solve_continuation_block(
    symmetric_operator,
    q_reference,
    locked_blocks,
    block_size,
    block_start,
):
    n = int(symmetric_operator.shape[0])
    rng = np.random.default_rng(EIGEN_SOLVER_SEED + 1009 * (block_start + 1))
    v0 = _project_out_locked(rng.normal(size=n), q_reference, locked_blocks)
    norm = float(np.linalg.norm(v0))
    if not np.isfinite(norm) or norm <= np.finfo(float).eps:
        raise RuntimeError("Could not construct a nonzero continuation start vector.")
    v0 /= norm

    operator = _deflated_linear_operator(
        symmetric_operator, q_reference, locked_blocks
    )
    base_ncv = (
        int(CONTINUATION_EIGSH_NCV)
        if CONTINUATION_EIGSH_NCV is not None
        else max(2 * int(block_size) + 24, int(block_size) + 48)
    )
    attempts = [
        (min(n - 1, base_ncv), CONTINUATION_EIGSH_TOL, CONTINUATION_EIGSH_MAXITER),
        (
            min(n - 1, max(base_ncv + block_size, 3 * block_size + 32)),
            max(CONTINUATION_EIGSH_TOL, 3.0e-8),
            2 * CONTINUATION_EIGSH_MAXITER,
        ),
    ]

    last_exception = None
    for attempt_index, (ncv, tolerance, maxiter) in enumerate(attempts, start=1):
        if ncv <= block_size:
            ncv = min(n - 1, block_size + 16)
        stop_event, heartbeat_thread, started = _start_heartbeat(
            f"Continuation block {block_start + 1}:"
            f"{block_start + block_size} attempt {attempt_index}"
        )
        print(
            f"[Continuation] block_start={block_start}, k={block_size}, "
            f"ncv={ncv}, tol={tolerance:.1e}, maxiter={maxiter}",
            flush=True,
        )
        try:
            try:
                eigenvalues, eigenvectors = eigsh(
                    operator,
                    k=int(block_size),
                    which="LA",
                    v0=v0,
                    ncv=int(ncv),
                    tol=float(tolerance),
                    maxiter=int(maxiter),
                )
            except ArpackNoConvergence as exc:
                last_exception = exc
                n_converged = (
                    0 if exc.eigenvalues is None else int(len(exc.eigenvalues))
                )
                print(
                    f"[Continuation] ARPACK partial convergence: "
                    f"{n_converged}/{block_size}",
                    flush=True,
                )
                # A substantial partial block is scientifically usable after the
                # explicit Rayleigh and residual audit below.  Otherwise retry.
                if (
                    exc.eigenvalues is None
                    or exc.eigenvectors is None
                    or n_converged < max(4, block_size // 2)
                ):
                    continue
                eigenvalues = np.asarray(exc.eigenvalues, dtype=np.float64)
                eigenvectors = np.asarray(exc.eigenvectors, dtype=np.float64)
        finally:
            _stop_heartbeat(stop_event, heartbeat_thread)

        eigenvectors = _project_out_locked(
            np.asarray(eigenvectors, dtype=np.float64),
            q_reference,
            locked_blocks,
        )
        eigenvectors, triangular = linalg.qr(
            eigenvectors, mode="economic", check_finite=False
        )
        diagonal = np.abs(np.diag(triangular))
        rank = int(np.sum(diagonal > 1.0e-10))
        if rank == 0:
            last_exception = RuntimeError("Continuation block lost numerical rank.")
            continue
        eigenvectors = eigenvectors[:, :rank]

        projected = symmetric_operator @ eigenvectors
        rayleigh = eigenvectors.T @ projected
        rayleigh = 0.5 * (rayleigh + rayleigh.T)
        values, rotation = linalg.eigh(rayleigh, check_finite=False)
        order = np.argsort(values)[::-1]
        values = np.asarray(values[order], dtype=np.float64)
        eigenvectors = np.asarray(eigenvectors @ rotation[:, order], dtype=np.float64)

        residual_matrix = (
            symmetric_operator @ eigenvectors
            - eigenvectors * values[None, :]
        )
        residual_norms = np.linalg.norm(residual_matrix, axis=0)
        if float(np.max(residual_norms)) > CONTINUATION_RESIDUAL_FAIL:
            last_exception = RuntimeError(
                "Continuation residual exceeded the failure tolerance: "
                f"{float(np.max(residual_norms)):.3e}."
            )
            continue
        if float(np.max(residual_norms)) > CONTINUATION_RESIDUAL_WARN:
            warnings.warn(
                "Continuation residual exceeds the warning tolerance: "
                f"max={float(np.max(residual_norms)):.3e}."
            )

        for column in range(eigenvectors.shape[1]):
            pivot = int(np.argmax(np.abs(eigenvectors[:, column])))
            if eigenvectors[pivot, column] < 0.0:
                eigenvectors[:, column] *= -1.0

        elapsed = time.perf_counter() - started
        print(
            f"[Continuation] accepted {eigenvectors.shape[1]} modes in "
            f"{elapsed / 60.0:.2f} min; residual max="
            f"{float(np.max(residual_norms)):.3e}",
            flush=True,
        )
        return values, np.asfortranarray(eigenvectors), residual_norms, elapsed

    raise RuntimeError(
        "The resume-safe deflated eigensolver could not complete the next "
        f"continuation block beginning at extra mode {block_start}."
    ) from last_exception


def _compute_checkpointed_continuation(
    symmetric_operator,
    reference_geometry,
    stationary,
    signature,
):
    target_modes = int(TARGET_TOTAL_GEOMETRY_MODES)
    reference_modes = min(
        int(REFERENCE_TOTAL_GEOMETRY_MODES),
        int(reference_geometry["eigenfunctions"].shape[1]),
    )
    n_extra = target_modes - reference_modes
    if n_extra <= 0:
        return {
            "eigenvalues": np.empty(0, dtype=np.float64),
            "symmetric_eigenvectors": np.empty(
                (stationary.size, 0), dtype=np.float64
            ),
            "diagnostics": {
                "construction": "reference_truncation",
                "reference_modes_retained": target_modes,
                "continuation_modes": 0,
            },
            "elapsed_seconds": 0.0,
        }

    stationary_reference = np.asarray(
        reference_geometry["stationary"], dtype=np.float64
    )
    stationary_difference = float(
        np.max(np.abs(np.asarray(stationary) - stationary_reference))
    )
    if stationary_difference > 1.0e-12:
        raise RuntimeError(
            "H2 and H1 stationary measures are inconsistent: "
            f"max abs difference={stationary_difference:.3e}."
        )

    sqrt_stationary = np.sqrt(stationary_reference)
    q_reference = np.asfortranarray(
        np.asarray(
            reference_geometry["eigenfunctions"][:, :reference_modes],
            dtype=np.float64,
        )
        * sqrt_stationary[:, None]
    )
    norms = np.linalg.norm(q_reference, axis=0)
    q_reference /= np.maximum(norms, np.finfo(float).eps)[None, :]

    blocks, value_blocks, residual_blocks, completed = (
        _load_continuation_checkpoints(signature, q_reference, n_extra)
    )
    total_elapsed = 0.0

    while completed < n_extra:
        requested = min(
            max(1, int(CONTINUATION_BLOCK_SIZE)),
            n_extra - completed,
        )
        values, vectors, residuals, elapsed = _solve_continuation_block(
            symmetric_operator,
            q_reference,
            blocks,
            requested,
            completed,
        )
        # A partial ARPACK block may contain fewer than requested modes.
        accepted = min(vectors.shape[1], n_extra - completed)
        values = values[:accepted]
        vectors = vectors[:, :accepted]
        residuals = residuals[:accepted]
        checkpoint_path = _continuation_checkpoint_path(completed)
        _atomic_savez(
            checkpoint_path,
            signature=np.array(signature),
            start_mode=np.array(completed),
            eigenvalues=values,
            eigenvectors=vectors,
            residual_norms=residuals,
            elapsed_seconds=np.array(elapsed),
        )
        blocks.append(np.asfortranarray(vectors))
        value_blocks.append(values)
        residual_blocks.append(residuals)
        completed += accepted
        total_elapsed += elapsed
        _atomic_write_text(
            CONTINUATION_CHECKPOINT_DIR / "progress.json",
            json.dumps(
                _json_ready(
                    {
                        "signature": signature,
                        "completed_extra_modes": completed,
                        "target_extra_modes": n_extra,
                        "last_checkpoint": str(checkpoint_path),
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
        )
        print(
            f"[Continuation checkpoint] saved: {completed}/{n_extra} extra modes",
            flush=True,
        )

    continuation_basis = np.asfortranarray(np.concatenate(blocks, axis=1))
    continuation_basis = _project_out_locked(
        continuation_basis, q_reference, []
    )
    continuation_basis, triangular = linalg.qr(
        continuation_basis, mode="economic", check_finite=False
    )
    diagonal = np.abs(np.diag(triangular))
    rank = int(np.sum(diagonal > 1.0e-10))
    if rank < n_extra:
        raise RuntimeError(
            f"Final continuation space has rank {rank}, expected {n_extra}."
        )
    continuation_basis = continuation_basis[:, :n_extra]

    projected = symmetric_operator @ continuation_basis
    rayleigh = continuation_basis.T @ projected
    rayleigh = 0.5 * (rayleigh + rayleigh.T)
    continuation_values, rotation = linalg.eigh(
        rayleigh, check_finite=False
    )
    order = np.argsort(continuation_values)[::-1]
    continuation_values = np.asarray(
        continuation_values[order], dtype=np.float64
    )
    continuation_basis = np.asarray(
        continuation_basis @ rotation[:, order], dtype=np.float64
    )

    for column in range(n_extra):
        pivot = int(np.argmax(np.abs(continuation_basis[:, column])))
        if continuation_basis[pivot, column] < 0.0:
            continuation_basis[:, column] *= -1.0

    residuals = np.linalg.norm(
        symmetric_operator @ continuation_basis
        - continuation_basis * continuation_values[None, :],
        axis=0,
    )
    cross = q_reference.T @ continuation_basis
    gram = continuation_basis.T @ continuation_basis
    boundary_gap = float(
        reference_geometry["eigenvalues"][reference_modes - 1]
        - continuation_values[0]
    )
    if boundary_gap < -1.0e-7:
        warnings.warn(
            "Largest continuation eigenvalue exceeds the retained H1 boundary "
            f"by {-boundary_gap:.3e}; inspect the continuation audit."
        )
    residual_max = float(np.max(residuals))
    if residual_max > CONTINUATION_RESIDUAL_FAIL:
        raise RuntimeError(
            "Final continuation residual exceeds the failure tolerance: "
            f"{residual_max:.3e}."
        )

    diagnostics = {
        "construction": "checkpointed_blockwise_deflated_eigsh",
        "reference_modes_retained": reference_modes,
        "continuation_modes": n_extra,
        "continuation_block_size": int(CONTINUATION_BLOCK_SIZE),
        "stationary_max_abs_difference": stationary_difference,
        "reference_continuation_max_abs_inner_product": float(
            np.max(np.abs(cross))
        ),
        "continuation_orthogonality_rms": float(
            np.sqrt(np.mean(np.square(gram - np.eye(n_extra))))
        ),
        "continuation_residual_max": residual_max,
        "continuation_residual_median": float(np.median(residuals)),
        "boundary_eigenvalue_gap": boundary_gap,
        "continuation_eigenvalue_max": float(continuation_values[0]),
        "continuation_eigenvalue_min": float(continuation_values[-1]),
        "checkpoint_blocks": int(len(blocks)),
    }
    return {
        "eigenvalues": continuation_values,
        "symmetric_eigenvectors": continuation_basis,
        "diagnostics": diagnostics,
        "elapsed_seconds": total_elapsed,
    }


def _assemble_nested_geometry(reference_geometry, continuation, stationary):
    target_modes = int(TARGET_TOTAL_GEOMETRY_MODES)
    reference_modes = min(
        int(REFERENCE_TOTAL_GEOMETRY_MODES),
        int(reference_geometry["eigenfunctions"].shape[1]),
    )
    if target_modes <= reference_modes:
        return {
            "eigenvalues": np.asarray(
                reference_geometry["eigenvalues"][:target_modes], dtype=np.float64
            ),
            "generator_eigenvalues": np.asarray(
                reference_geometry["generator_eigenvalues"][:target_modes],
                dtype=np.float64,
            ),
            "eigenfunctions": np.asarray(
                reference_geometry["eigenfunctions"][:, :target_modes],
                dtype=np.float64,
            ),
            "diagnostics": continuation["diagnostics"],
        }

    values = np.asarray(continuation["eigenvalues"], dtype=np.float64)
    symmetric_vectors = np.asarray(
        continuation["symmetric_eigenvectors"], dtype=np.float64
    )
    n_extra = target_modes - reference_modes
    if values.size != n_extra or symmetric_vectors.shape[1] != n_extra:
        raise RuntimeError("Continuation output has the wrong number of modes.")

    sqrt_stationary = np.sqrt(np.asarray(stationary, dtype=np.float64))
    generator_scale = float(reference_geometry["generator_scale"])
    eigenvalues = np.concatenate(
        [
            np.asarray(
                reference_geometry["eigenvalues"][:reference_modes],
                dtype=np.float64,
            ),
            values,
        ]
    )
    generator_eigenvalues = np.concatenate(
        [
            np.asarray(
                reference_geometry["generator_eigenvalues"][:reference_modes],
                dtype=np.float64,
            ),
            np.maximum(1.0 - values, 0.0) / generator_scale,
        ]
    )
    eigenfunctions = np.empty(
        (stationary.size, target_modes), dtype=np.float64
    )
    eigenfunctions[:, :reference_modes] = np.asarray(
        reference_geometry["eigenfunctions"][:, :reference_modes],
        dtype=np.float64,
    )
    eigenfunctions[:, reference_modes:] = (
        symmetric_vectors / sqrt_stationary[:, None]
    )
    return {
        "eigenvalues": eigenvalues,
        "generator_eigenvalues": generator_eigenvalues,
        "eigenfunctions": eigenfunctions,
        "diagnostics": continuation["diagnostics"],
    }


def _extended_geometry_cache_valid(path: Path, signature: str) -> bool:
    if FORCE_REBUILD_GEOMETRY or not path.is_file():
        return False
    try:
        with np.load(path, allow_pickle=False) as data:
            cached_signature = (
                str(data["signature"].item()) if "signature" in data.files else ""
            )
            return bool(
                cached_signature == signature
                and data["eigenfunctions"].shape[1]
                == TARGET_TOTAL_GEOMETRY_MODES
                and abs(float(data["alpha"].item()) - ALPHA_DM) <= 1.0e-12
            )
    except Exception as exc:
        warnings.warn(f"H2 extended geometry cache could not be reused: {exc}")
        return False


def _build_or_load_extended_geometry(
    centered_survey,
    reference_geometry_file: Path,
):
    signature = _geometry_signature(reference_geometry_file)

    # Explicit external geometry override, including a valid v1 geometry, may be
    # supplied by environment variable and is structurally audited before reuse.
    if _extended_geometry_override and EXTENDED_GEOMETRY_FILE.is_file():
        geometry = _load_geometry(EXTENDED_GEOMETRY_FILE)
        if geometry["eigenfunctions"].shape[0] != centered_survey.shape[0]:
            raise ValueError("External Mock-3 H2 geometry has the wrong survey size.")
        if geometry["eigenfunctions"].shape[1] < TARGET_TOTAL_GEOMETRY_MODES:
            raise ValueError("External Mock-3 H2 geometry contains too few eigenfunctions.")
        if abs(float(geometry["alpha"]) - ALPHA_DM) > 1.0e-12:
            raise ValueError("External Mock-3 H2 geometry is not the required alpha=0 geometry.")
        return EXTENDED_GEOMETRY_FILE, "external override"

    if _extended_geometry_cache_valid(EXTENDED_GEOMETRY_FILE, signature):
        print(f"[Geometry cache] reuse: {EXTENDED_GEOMETRY_FILE}", flush=True)
        return EXTENDED_GEOMETRY_FILE, "cache"

    reference_geometry = _load_geometry(reference_geometry_file)
    if reference_geometry["eigenfunctions"].shape[0] != centered_survey.shape[0]:
        raise ValueError("H1 reference geometry has the wrong Mock-3 survey size.")
    if abs(float(reference_geometry["alpha"]) - ALPHA_DM) > 1.0e-12:
        raise ValueError("H1 reference geometry is not the required unweighted alpha=0 geometry.")

    EXTENDED_GEOMETRY_FILE.parent.mkdir(parents=True, exist_ok=True)
    base_kernel = _build_base_kernel(
        centered_survey,
        GRAPH_NEIGHBORS,
        GRAPH_BANDWIDTH_NEIGHBOR,
        GRAPH_BANDWIDTH_MULTIPLIER,
    )
    diffusion = _symmetric_diffusion_operator(base_kernel)

    continuation = _compute_checkpointed_continuation(
        diffusion["operator"],
        reference_geometry,
        diffusion["stationary"],
        signature,
    )
    nested = _assemble_nested_geometry(
        reference_geometry,
        continuation,
        diffusion["stationary"],
    )

    geometry_payload = {
        "signature": np.array(signature),
        "eigenvalues": nested["eigenvalues"],
        "generator_eigenvalues": nested["generator_eigenvalues"],
        "eigenfunctions": nested["eigenfunctions"],
        "quadrature_weights": diffusion["quadrature_weights"],
        "q": diffusion["q"],
        "d": diffusion["d"],
        "stationary": diffusion["stationary"],
        "alpha": np.array(ALPHA_DM),
        "generator_scale": np.array(reference_geometry["generator_scale"]),
    }
    _atomic_savez(EXTENDED_GEOMETRY_FILE, **geometry_payload)

    metadata = {
        "signature": signature,
        "geometry_file": str(EXTENDED_GEOMETRY_FILE),
        "reference_geometry_file": str(reference_geometry_file),
        "target_total_modes": TARGET_TOTAL_GEOMETRY_MODES,
        "reference_total_modes": REFERENCE_TOTAL_GEOMETRY_MODES,
        "solver": "checkpointed_blockwise_deflated_eigsh",
        "eigensolver_elapsed_seconds": continuation["elapsed_seconds"],
        "diagnostics": nested["diagnostics"],
    }
    _atomic_write_text(
        EXTENDED_GEOMETRY_METADATA_FILE,
        json.dumps(_json_ready(metadata), ensure_ascii=False, indent=2),
    )
    pd.DataFrame(
        [
            {"quantity": key, "value": value}
            for key, value in nested["diagnostics"].items()
            if isinstance(value, (int, float, np.integer, np.floating))
        ]
    ).to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_eigensystem_continuation.csv",
        index=False,
    )
    print(f"[Geometry cache] saved: {EXTENDED_GEOMETRY_FILE}", flush=True)
    return EXTENDED_GEOMETRY_FILE, "new checkpointed continuation"


# ==================================================================================================
# 3. Metric and gradient construction
# ==================================================================================================


def _build_base_kernel(positions, n_neighbors, bandwidth_neighbor, multiplier):
    positions = np.asarray(positions, dtype=np.float64)
    n = positions.shape[0]
    n_neighbors = min(int(n_neighbors), n - 1)
    bandwidth_neighbor = min(int(bandwidth_neighbor), n_neighbors)
    tree = cKDTree(positions)
    try:
        distances, indices = tree.query(positions, k=n_neighbors + 1, workers=-1)
    except TypeError:
        distances, indices = tree.query(positions, k=n_neighbors + 1)
    distances = np.asarray(distances[:, 1:], dtype=np.float64)
    indices = np.asarray(indices[:, 1:], dtype=np.int64)
    rho = float(multiplier) * distances[:, bandwidth_neighbor - 1]
    positive = rho[rho > 0.0]
    floor = max(
        float(np.median(positive)) * 1.0e-8,
        np.finfo(np.float64).eps,
    )
    rho = np.maximum(rho, floor)
    row = np.repeat(np.arange(n, dtype=np.int64), n_neighbors)
    col = indices.reshape(-1)
    weights = np.exp(
        -np.square(distances.reshape(-1)) / (rho[row] * rho[col])
    )
    kernel = sparse.coo_matrix((weights, (row, col)), shape=(n, n)).tocsr()
    kernel = kernel.maximum(kernel.T)
    kernel.setdiag(1.0)
    kernel.eliminate_zeros()
    return {
        "kernel": kernel,
        "rho": rho,
        "tree": tree,
        "n_neighbors": n_neighbors,
        "bandwidth_neighbor": bandwidth_neighbor,
        "multiplier": float(multiplier),
    }


def _observed_markov_from_base_kernel(base_kernel):
    kernel = base_kernel["kernel"]
    degree = np.asarray(kernel.sum(axis=1)).ravel()
    if np.any(degree <= 0.0):
        raise ValueError("Observed graph contains a non-positive degree.")
    return (sparse.diags(1.0 / degree) @ kernel).tocsr()


def _centered_coordinate_metric_sparse(markov, positions, bandwidth):
    mean_position = markov @ positions
    metric = np.empty((positions.shape[0], 3, 3), dtype=np.float64)
    denominator = 2.0 * np.square(bandwidth)
    for j in range(3):
        for c in range(j, 3):
            second = markov @ (positions[:, j] * positions[:, c])
            covariance = second - mean_position[:, j] * mean_position[:, c]
            value = covariance / denominator
            metric[:, j, c] = value
            metric[:, c, j] = value
    return 0.5 * (metric + np.swapaxes(metric, 1, 2)), mean_position


def _invert_local_metric(metric):
    metric = 0.5 * (metric + np.swapaxes(metric, 1, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    trace_scale = np.maximum(np.trace(metric, axis1=1, axis2=2) / 3.0, 0.0)
    floor = np.maximum(
        METRIC_EIGENVALUE_ABSOLUTE_FLOOR,
        METRIC_EIGENVALUE_RELATIVE_FLOOR * np.maximum(trace_scale, 1.0),
    )
    clipped = np.maximum(eigenvalues, floor[:, None])
    inverse = np.einsum(
        "nij,nj,nkj->nik",
        eigenvectors,
        1.0 / clipped,
        eigenvectors,
        optimize=True,
    )
    condition = clipped[:, -1] / clipped[:, 0]
    clipped_count = np.sum(eigenvalues < floor[:, None], axis=1)
    return inverse, eigenvalues, condition, clipped_count


def _observed_function_gradient_block(
    markov,
    positions,
    functions,
    bandwidth,
    inverse_metric,
    mean_position,
):
    block = np.asarray(functions, dtype=np.float64)
    if block.ndim == 1:
        block = block[:, None]
    mean_block = markov @ block
    covariance_vector = np.empty((positions.shape[0], block.shape[1], 3), dtype=np.float64)
    denominator = 2.0 * np.square(bandwidth)
    for c in range(3):
        mean_product = markov @ (block * positions[:, c, None])
        covariance = mean_product - mean_block * mean_position[:, c, None]
        covariance_vector[:, :, c] = covariance / denominator[:, None]
    return np.einsum(
        "nij,nmj->nmi",
        inverse_metric,
        covariance_vector,
        optimize=True,
    )


def _query_transition(query_positions, base_kernel):
    tree = base_kernel["tree"]
    n_neighbors = base_kernel["n_neighbors"]
    bandwidth_neighbor = base_kernel["bandwidth_neighbor"]
    rho_observed = base_kernel["rho"]
    try:
        distances, indices = tree.query(query_positions, k=n_neighbors, workers=-1)
    except TypeError:
        distances, indices = tree.query(query_positions, k=n_neighbors)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    rho_query = base_kernel["multiplier"] * distances[:, bandwidth_neighbor - 1]
    positive = rho_query[rho_query > 0.0]
    floor = max(
        float(np.median(positive)) * 1.0e-8,
        np.finfo(np.float64).eps,
    )
    rho_query = np.maximum(rho_query, floor)
    kernel_query = np.exp(
        -np.square(distances) / (rho_query[:, None] * rho_observed[indices])
    )
    row_sum = np.sum(kernel_query, axis=1)
    transition = kernel_query / row_sum[:, None]
    return {
        "distances": distances,
        "indices": indices,
        "rho_query": rho_query,
        "transition": transition,
        "nearest_distance": distances[:, 0],
    }


def _query_metric(transition_info, observed_positions):
    transition = transition_info["transition"]
    indices = transition_info["indices"]
    rho_query = transition_info["rho_query"]
    neighbors = observed_positions[indices]
    mean_position = np.einsum("qk,qkc->qc", transition, neighbors, optimize=True)
    metric = np.empty((transition.shape[0], 3, 3), dtype=np.float64)
    denominator = 2.0 * np.square(rho_query)
    for j in range(3):
        for c in range(j, 3):
            second = np.einsum(
                "qk,qk->q",
                transition,
                neighbors[:, :, j] * neighbors[:, :, c],
                optimize=True,
            )
            covariance = second - mean_position[:, j] * mean_position[:, c]
            value = covariance / denominator
            metric[:, j, c] = value
            metric[:, c, j] = value
    return 0.5 * (metric + np.swapaxes(metric, 1, 2)), mean_position


def _query_function_gradient_block(
    transition_info,
    observed_positions,
    observed_functions,
    inverse_metric,
    mean_position,
):
    transition = transition_info["transition"]
    indices = transition_info["indices"]
    rho_query = transition_info["rho_query"]
    block = np.asarray(observed_functions, dtype=np.float64)
    if block.ndim == 1:
        block = block[:, None]
    neighbor_functions = block[indices]
    neighbor_positions = observed_positions[indices]
    mean_function = np.einsum(
        "qk,qkm->qm",
        transition,
        neighbor_functions,
        optimize=True,
    )
    covariance_vector = np.empty(
        (transition.shape[0], block.shape[1], 3),
        dtype=np.float64,
    )
    denominator = 2.0 * np.square(rho_query)
    for c in range(3):
        mean_product = np.einsum(
            "qk,qkm,qk->qm",
            transition,
            neighbor_functions,
            neighbor_positions[:, :, c],
            optimize=True,
        )
        covariance = mean_product - mean_function * mean_position[:, c, None]
        covariance_vector[:, :, c] = covariance / denominator[:, None]
    return np.einsum(
        "qij,qmj->qmi",
        inverse_metric,
        covariance_vector,
        optimize=True,
    )


def _geometry_continuity_and_signs(high_geometry, legacy_geometry):
    n_compare = min(
        high_geometry["eigenvalues"].size,
        legacy_geometry["eigenvalues"].size,
    )
    n_mode_audit = min(64, n_compare)
    stationary = legacy_geometry["stationary"]
    high_functions = high_geometry["eigenfunctions"][:, :n_compare]
    legacy_functions = legacy_geometry["eigenfunctions"][:, :n_compare]
    correlations = np.sum(
        stationary[:, None] * high_functions * legacy_functions,
        axis=0,
    )
    signs = np.where(correlations < 0.0, -1.0, 1.0)
    aligned = high_functions[:, :n_mode_audit] * signs[:n_mode_audit][None, :]
    cross = aligned.T @ (
        stationary[:, None] * legacy_functions[:, :n_mode_audit]
    )
    singular_values = linalg.svdvals(cross)
    audit = pd.DataFrame(
        [
            {"quantity": "legacy_geometry_available", "value": 1.0},
            {
                "quantity": "stationary_max_abs_difference",
                "value": float(
                    np.max(
                        np.abs(
                            high_geometry["stationary"]
                            - legacy_geometry["stationary"]
                        )
                    )
                ),
            },
            {
                "quantity": "markov_eigenvalue_max_abs_difference",
                "value": float(
                    np.max(
                        np.abs(
                            high_geometry["eigenvalues"][:n_compare]
                            - legacy_geometry["eigenvalues"][:n_compare]
                        )
                    )
                ),
            },
            {
                "quantity": "generator_eigenvalue_max_abs_difference",
                "value": float(
                    np.max(
                        np.abs(
                            high_geometry["generator_eigenvalues"][:n_compare]
                            - legacy_geometry["generator_eigenvalues"][:n_compare]
                        )
                    )
                ),
            },
            {
                "quantity": "first64_mode_correlation_median",
                "value": float(np.median(np.abs(correlations[:n_mode_audit]))),
            },
            {
                "quantity": "first64_mode_correlation_minimum",
                "value": float(np.min(np.abs(correlations[:n_mode_audit]))),
            },
            {
                "quantity": "first64_subspace_minimum_singular_value",
                "value": float(np.min(singular_values)),
            },
        ]
    )
    return audit, signs


def _basis_cache_signature(
    high_geometry_file,
    legacy_geometry_file,
    query_global_indices,
    primary_query_mask,
):
    return _stable_signature(
        {
            "mock1": _file_signature(MOCK1_FILE),
            "mock3": _file_signature(MOCK3_FILE),
            "predictions": _file_signature(PREDICTIONS_FILE),
            "high_geometry": _file_signature(high_geometry_file),
            "legacy_geometry": _file_signature(legacy_geometry_file),
            "query_indices_hash": _hash_array(query_global_indices),
            "primary_query_mask_hash": _hash_array(primary_query_mask.astype(np.uint8)),
            "graph_neighbors": GRAPH_NEIGHBORS,
            "graph_bandwidth_neighbor": GRAPH_BANDWIDTH_NEIGHBOR,
            "graph_multiplier": GRAPH_BANDWIDTH_MULTIPLIER,
            "metric_relative_floor": METRIC_EIGENVALUE_RELATIVE_FLOOR,
            "metric_absolute_floor": METRIC_EIGENVALUE_ABSOLUTE_FLOOR,
            "max_nonconstant_modes": MAX_NONCONSTANT_MODES,
            "generator_reference_nonconstant_modes": GENERATOR_REFERENCE_NONCONSTANT_MODES,
            "version": 3,
        }
    )


def _cache_files_valid(signature, n_survey, n_query_primary):
    required = [
        BASIS_CACHE_METADATA_FILE,
        OBSERVED_GRADIENT_FILE,
        QUERY_PRIMARY_GRADIENT_FILE,
        RADIAL_DESIGN_FILE,
        COLUMN_RMS_FILE,
    ]
    if FORCE_REBUILD_CACHE or not all(path.is_file() for path in required):
        return False
    try:
        metadata = json.loads(BASIS_CACHE_METADATA_FILE.read_text(encoding="utf-8"))
        if metadata.get("signature") != signature:
            return False
        observed_gradient = np.load(OBSERVED_GRADIENT_FILE, mmap_mode="r")
        query_gradient = np.load(QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r")
        radial_design = np.load(RADIAL_DESIGN_FILE, mmap_mode="r")
        column_rms = np.load(COLUMN_RMS_FILE, mmap_mode="r")
        return (
            observed_gradient.shape == (n_survey, MAX_NONCONSTANT_MODES, 3)
            and query_gradient.shape == (n_query_primary, MAX_NONCONSTANT_MODES, 3)
            and radial_design.shape == (n_survey, MAX_BASIS)
            and column_rms.shape == (MAX_NONCONSTANT_MODES,)
        )
    except Exception as exc:
        warnings.warn(f"High-mode basis cache could not be reused: {exc}")
        return False


def _build_or_load_highmode_basis(
    centered_survey,
    centered_query_primary,
    line_of_sight_survey,
    high_geometry_file,
    legacy_geometry_file,
    query_global_indices,
    primary_query_mask,
):
    signature = _basis_cache_signature(
        high_geometry_file,
        legacy_geometry_file,
        query_global_indices,
        primary_query_mask,
    )
    n_survey = centered_survey.shape[0]
    n_query_primary = centered_query_primary.shape[0]

    if _cache_files_valid(signature, n_survey, n_query_primary):
        metadata = json.loads(BASIS_CACHE_METADATA_FILE.read_text(encoding="utf-8"))
        print(f"[High-mode basis cache] reuse: {CACHE_DIR}")
        return {
            "observed_nonconstant_gradient": np.load(
                OBSERVED_GRADIENT_FILE, mmap_mode="r"
            ),
            "query_primary_nonconstant_gradient": np.load(
                QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r"
            ),
            "radial_design_observed": np.load(RADIAL_DESIGN_FILE, mmap_mode="r"),
            "column_rms": np.load(COLUMN_RMS_FILE, mmap_mode="r"),
            "metadata": metadata,
            "loaded_from_cache": True,
        }

    print("[High-mode basis] constructing metric-calibrated gradients...")
    high_geometry = _load_geometry(high_geometry_file)
    legacy_geometry = _load_geometry(legacy_geometry_file)
    if high_geometry["eigenfunctions"].shape[0] != n_survey:
        raise ValueError("High-mode geometry does not match the Mock-3 survey size.")
    if high_geometry["eigenfunctions"].shape[1] < MAX_NONCONSTANT_MODES + 1:
        raise ValueError(
            "High-mode geometry contains too few modes: "
            f"{high_geometry['eigenfunctions'].shape[1]} < {MAX_NONCONSTANT_MODES + 1}."
        )
    if abs(high_geometry["alpha"] - ALPHA_DM) > 1.0e-12:
        raise ValueError("High-mode geometry is not the required unweighted alpha=0 geometry.")

    geometry_audit_df, common_signs = _geometry_continuity_and_signs(
        high_geometry,
        legacy_geometry,
    )
    geometry_audit_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_continuity.csv",
        index=False,
    )

    constant_mode_index = int(np.argmin(high_geometry["generator_eigenvalues"]))
    all_mode_indices = np.arange(high_geometry["eigenfunctions"].shape[1])
    active_mode_indices = all_mode_indices[all_mode_indices != constant_mode_index][
        :MAX_NONCONSTANT_MODES
    ]
    if active_mode_indices.size != MAX_NONCONSTANT_MODES:
        raise RuntimeError("Could not select the requested nonconstant high-mode basis.")

    sign_by_raw_mode = np.ones(high_geometry["eigenfunctions"].shape[1], dtype=np.float64)
    n_common = min(common_signs.size, sign_by_raw_mode.size)
    sign_by_raw_mode[:n_common] = common_signs[:n_common]

    base_kernel = _build_base_kernel(
        centered_survey,
        GRAPH_NEIGHBORS,
        GRAPH_BANDWIDTH_NEIGHBOR,
        GRAPH_BANDWIDTH_MULTIPLIER,
    )
    markov = _observed_markov_from_base_kernel(base_kernel)
    observed_metric, observed_mean = _centered_coordinate_metric_sparse(
        markov,
        centered_survey,
        base_kernel["rho"],
    )
    (
        inverse_observed_metric,
        observed_metric_eigenvalues,
        observed_condition,
        observed_clipped_count,
    ) = _invert_local_metric(observed_metric)

    query_transition = _query_transition(centered_query_primary, base_kernel)
    query_metric, query_mean = _query_metric(query_transition, centered_survey)
    (
        inverse_query_metric,
        query_metric_eigenvalues,
        query_condition,
        query_clipped_count,
    ) = _invert_local_metric(query_metric)

    partial_files = [
        OBSERVED_GRADIENT_FILE,
        QUERY_PRIMARY_GRADIENT_FILE,
        RADIAL_DESIGN_FILE,
        COLUMN_RMS_FILE,
    ]
    progress_valid = False
    reuse_modes = 0
    if RESUME and BASIS_PROGRESS_FILE.is_file() and all(
        path.is_file() for path in partial_files
    ):
        try:
            progress = json.loads(BASIS_PROGRESS_FILE.read_text(encoding="utf-8"))
            completed_modes = int(progress.get("completed_nonconstant_modes", -1))
            if progress.get("signature") == signature and 0 <= completed_modes <= MAX_NONCONSTANT_MODES:
                observed_memmap = open_memmap(
                    OBSERVED_GRADIENT_FILE,
                    mode="r+",
                    dtype=np.float64,
                    shape=(n_survey, MAX_NONCONSTANT_MODES, 3),
                )
                query_memmap = open_memmap(
                    QUERY_PRIMARY_GRADIENT_FILE,
                    mode="r+",
                    dtype=np.float64,
                    shape=(n_query_primary, MAX_NONCONSTANT_MODES, 3),
                )
                radial_design_memmap = open_memmap(
                    RADIAL_DESIGN_FILE,
                    mode="r+",
                    dtype=np.float64,
                    shape=(n_survey, MAX_BASIS),
                )
                column_rms_memmap = open_memmap(
                    COLUMN_RMS_FILE,
                    mode="r+",
                    dtype=np.float64,
                    shape=(MAX_NONCONSTANT_MODES,),
                )
                reuse_modes = completed_modes
                progress_valid = True
                print(
                    f"[High-mode basis checkpoint] resume at "
                    f"{reuse_modes}/{MAX_NONCONSTANT_MODES} nonconstant modes",
                    flush=True,
                )
        except Exception as exc:
            warnings.warn(f"Partial high-mode basis checkpoint is invalid: {exc}")
            progress_valid = False
            reuse_modes = 0

    if not progress_valid:
        observed_memmap = open_memmap(
            OBSERVED_GRADIENT_FILE,
            mode="w+",
            dtype=np.float64,
            shape=(n_survey, MAX_NONCONSTANT_MODES, 3),
        )
        query_memmap = open_memmap(
            QUERY_PRIMARY_GRADIENT_FILE,
            mode="w+",
            dtype=np.float64,
            shape=(n_query_primary, MAX_NONCONSTANT_MODES, 3),
        )
        radial_design_memmap = open_memmap(
            RADIAL_DESIGN_FILE,
            mode="w+",
            dtype=np.float64,
            shape=(n_survey, MAX_BASIS),
        )
        column_rms_memmap = open_memmap(
            COLUMN_RMS_FILE,
            mode="w+",
            dtype=np.float64,
            shape=(MAX_NONCONSTANT_MODES,),
        )

    affine_gram = line_of_sight_survey.T @ line_of_sight_survey
    eigenfunctions = high_geometry["eigenfunctions"]

    # Preserve exact nesting by copying the complete H1 basis whenever its cache is available.
    if not progress_valid:
        previous_required = [
            PREVIOUS_H1_BASIS_METADATA_FILE,
            PREVIOUS_H1_OBSERVED_GRADIENT_FILE,
            PREVIOUS_H1_QUERY_PRIMARY_GRADIENT_FILE,
            PREVIOUS_H1_RADIAL_DESIGN_FILE,
            PREVIOUS_H1_COLUMN_RMS_FILE,
        ]
        if all(path.is_file() for path in previous_required):
            try:
                previous_metadata = json.loads(
                    PREVIOUS_H1_BASIS_METADATA_FILE.read_text(encoding="utf-8")
                )
                previous_observed = np.load(
                    PREVIOUS_H1_OBSERVED_GRADIENT_FILE, mmap_mode="r"
                )
                previous_query = np.load(
                    PREVIOUS_H1_QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r"
                )
                previous_radial = np.load(
                    PREVIOUS_H1_RADIAL_DESIGN_FILE, mmap_mode="r"
                )
                previous_column_rms = np.load(
                    PREVIOUS_H1_COLUMN_RMS_FILE, mmap_mode="r"
                )
                requested_reuse = min(
                    PREVIOUS_H1_MAX_NONCONSTANT_MODES,
                    MAX_NONCONSTANT_MODES,
                )
                previous_modes = np.asarray(
                    previous_metadata.get("active_mode_indices", [])[:requested_reuse],
                    dtype=np.int64,
                )
                current_modes = np.asarray(
                    active_mode_indices[:requested_reuse], dtype=np.int64
                )
                cache_shapes_valid = bool(
                    previous_observed.shape[0] == n_survey
                    and previous_observed.shape[1] >= requested_reuse
                    and previous_query.shape[0] == n_query_primary
                    and previous_query.shape[1] >= requested_reuse
                    and previous_radial.shape[0] == n_survey
                    and previous_radial.shape[1] >= N_AFFINE_MODES + requested_reuse
                    and previous_column_rms.shape[0] >= requested_reuse
                )
                mode_indices_valid = bool(
                    previous_modes.size == requested_reuse
                    and np.array_equal(previous_modes, current_modes)
                )
                if cache_shapes_valid and mode_indices_valid:
                    for copy_start in range(
                        0, requested_reuse, max(1, GRADIENT_MODE_CHUNK_SIZE)
                    ):
                        copy_stop = min(
                            copy_start + max(1, GRADIENT_MODE_CHUNK_SIZE),
                            requested_reuse,
                        )
                        observed_memmap[:, copy_start:copy_stop, :] = (
                            previous_observed[:, copy_start:copy_stop, :]
                        )
                        query_memmap[:, copy_start:copy_stop, :] = (
                            previous_query[:, copy_start:copy_stop, :]
                        )
                        radial_design_memmap[
                            :,
                            N_AFFINE_MODES + copy_start : N_AFFINE_MODES + copy_stop,
                        ] = previous_radial[
                            :,
                            N_AFFINE_MODES + copy_start : N_AFFINE_MODES + copy_stop,
                        ]
                        column_rms_memmap[copy_start:copy_stop] = (
                            previous_column_rms[copy_start:copy_stop]
                        )
                    reuse_modes = requested_reuse
                    print(
                        f"[High-mode basis] copied {reuse_modes} H1 nonconstant modes; "
                        f"computing only {MAX_NONCONSTANT_MODES - reuse_modes} new modes.",
                        flush=True,
                    )
                else:
                    warnings.warn(
                        "H1 basis cache was found but could not be reused because its "
                        "shape or active-mode ordering did not match the nested H2 geometry."
                    )
            except Exception as exc:
                warnings.warn(f"H1 basis-cache reuse failed; rebuilding all modes: {exc}")

        observed_memmap.flush()
        query_memmap.flush()
        radial_design_memmap.flush()
        column_rms_memmap.flush()
        _atomic_write_text(
            BASIS_PROGRESS_FILE,
            json.dumps(
                _json_ready(
                    {
                        "signature": signature,
                        "completed_nonconstant_modes": reuse_modes,
                        "target_nonconstant_modes": MAX_NONCONSTANT_MODES,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
        )

    for start in range(
        reuse_modes, MAX_NONCONSTANT_MODES, GRADIENT_MODE_CHUNK_SIZE
    ):
        stop = min(start + GRADIENT_MODE_CHUNK_SIZE, MAX_NONCONSTANT_MODES)
        raw_indices = active_mode_indices[start:stop]
        functions = np.asarray(eigenfunctions[:, raw_indices], dtype=np.float64)
        functions = functions * sign_by_raw_mode[raw_indices][None, :]

        observed_block = _observed_function_gradient_block(
            markov,
            centered_survey,
            functions,
            base_kernel["rho"],
            inverse_observed_metric,
            observed_mean,
        )
        query_block = _query_function_gradient_block(
            query_transition,
            centered_survey,
            functions,
            inverse_query_metric,
            query_mean,
        )

        observed_radial = np.einsum(
            "ic,imc->im",
            line_of_sight_survey,
            observed_block,
            optimize=True,
        )
        affine_rhs = line_of_sight_survey.T @ observed_radial
        affine_projection_vectors = linalg.solve(
            affine_gram,
            affine_rhs,
            assume_a="sym",
            check_finite=False,
        ).T
        observed_block -= affine_projection_vectors[None, :, :]
        query_block -= affine_projection_vectors[None, :, :]

        observed_memmap[:, start:stop, :] = observed_block
        query_memmap[:, start:stop, :] = query_block
        radial_design_memmap[:, N_AFFINE_MODES + start : N_AFFINE_MODES + stop] = (
            np.einsum(
                "ic,imc->im",
                line_of_sight_survey,
                observed_block,
                optimize=True,
            )
        )
        column_rms_memmap[start:stop] = np.sqrt(
            np.mean(np.sum(np.square(observed_block), axis=2), axis=0)
        )
        observed_memmap.flush()
        query_memmap.flush()
        radial_design_memmap.flush()
        column_rms_memmap.flush()
        _atomic_write_text(
            BASIS_PROGRESS_FILE,
            json.dumps(
                _json_ready(
                    {
                        "signature": signature,
                        "completed_nonconstant_modes": stop,
                        "target_nonconstant_modes": MAX_NONCONSTANT_MODES,
                    }
                ),
                ensure_ascii=False,
                indent=2,
            ),
        )
        print(
            f"[High-mode basis checkpoint] saved modes {start + 1}:{stop}/"
            f"{MAX_NONCONSTANT_MODES}",
            flush=True,
        )

    positive_reference = np.asarray(
        column_rms_memmap[: min(GENERATOR_REFERENCE_NONCONSTANT_MODES, MAX_NONCONSTANT_MODES)]
    )
    positive_reference = positive_reference[
        np.isfinite(positive_reference) & (positive_reference > 0.0)
    ]
    affine_mode_scale = (
        float(np.median(positive_reference)) if positive_reference.size else 1.0
    )
    radial_design_memmap[:, :N_AFFINE_MODES] = (
        affine_mode_scale * line_of_sight_survey
    )

    observed_memmap.flush()
    query_memmap.flush()
    radial_design_memmap.flush()
    column_rms_memmap.flush()

    # Metric-calibration diagnostics.
    affine_coefficient = np.array([0.37, -0.52, 0.81], dtype=np.float64)
    affine_observed = centered_survey @ affine_coefficient + 1.7
    affine_observed_gradient = _observed_function_gradient_block(
        markov,
        centered_survey,
        affine_observed,
        base_kernel["rho"],
        inverse_observed_metric,
        observed_mean,
    )[:, 0, :]
    affine_query_gradient = _query_function_gradient_block(
        query_transition,
        centered_survey,
        affine_observed,
        inverse_query_metric,
        query_mean,
    )[:, 0, :]

    quadratic_observed = 0.5 * np.sum(np.square(centered_survey), axis=1)
    quadratic_observed_gradient = _observed_function_gradient_block(
        markov,
        centered_survey,
        quadratic_observed,
        base_kernel["rho"],
        inverse_observed_metric,
        observed_mean,
    )[:, 0, :]
    quadratic_query_gradient = _query_function_gradient_block(
        query_transition,
        centered_survey,
        quadratic_observed,
        inverse_query_metric,
        query_mean,
    )[:, 0, :]

    constant_function = high_geometry["eigenfunctions"][:, constant_mode_index]
    constant_observed_gradient = _observed_function_gradient_block(
        markov,
        centered_survey,
        constant_function,
        base_kernel["rho"],
        inverse_observed_metric,
        observed_mean,
    )[:, 0, :]
    constant_query_gradient = _query_function_gradient_block(
        query_transition,
        centered_survey,
        constant_function,
        inverse_query_metric,
        query_mean,
    )[:, 0, :]

    diagnostics = {
        "observed_metric_condition_median": float(np.median(observed_condition)),
        "observed_metric_condition_p99": float(np.quantile(observed_condition, 0.99)),
        "observed_metric_condition_max": float(np.max(observed_condition)),
        "query_metric_condition_median": float(np.median(query_condition)),
        "query_metric_condition_p99": float(np.quantile(query_condition, 0.99)),
        "query_metric_condition_max": float(np.max(query_condition)),
        "observed_metric_clipped_fraction": float(np.mean(observed_clipped_count > 0)),
        "query_metric_clipped_fraction": float(np.mean(query_clipped_count > 0)),
        "observed_affine_rms_error": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            affine_observed_gradient - affine_coefficient[None, :]
                        ),
                        axis=1,
                    )
                )
            )
        ),
        "query_affine_rms_error": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            affine_query_gradient - affine_coefficient[None, :]
                        ),
                        axis=1,
                    )
                )
            )
        ),
        "observed_quadratic_gradient_nrmse": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(quadratic_observed_gradient - centered_survey),
                        axis=1,
                    )
                )
            )
            / max(
                float(np.sqrt(np.mean(np.sum(np.square(centered_survey), axis=1)))),
                np.finfo(float).eps,
            )
        ),
        "query_quadratic_gradient_nrmse": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            quadratic_query_gradient - centered_query_primary
                        ),
                        axis=1,
                    )
                )
            )
            / max(
                float(
                    np.sqrt(
                        np.mean(
                            np.sum(np.square(centered_query_primary), axis=1)
                        )
                    )
                ),
                np.finfo(float).eps,
            )
        ),
        "constant_mode_gradient_rms_observed": float(
            np.sqrt(np.mean(np.sum(np.square(constant_observed_gradient), axis=1)))
        ),
        "constant_mode_gradient_rms_query": float(
            np.sqrt(np.mean(np.sum(np.square(constant_query_gradient), axis=1)))
        ),
        "affine_mode_scale": affine_mode_scale,
    }

    metadata = {
        "signature": signature,
        "high_geometry_file": str(high_geometry_file),
        "legacy_geometry_file": str(legacy_geometry_file),
        "constant_mode_index": constant_mode_index,
        "active_mode_indices": active_mode_indices.tolist(),
        "active_generator_values": high_geometry["generator_eigenvalues"][
            active_mode_indices
        ].tolist(),
        "affine_mode_scale": affine_mode_scale,
        "n_survey": n_survey,
        "n_query_primary": n_query_primary,
        "max_nonconstant_modes": MAX_NONCONSTANT_MODES,
        "max_basis": MAX_BASIS,
        "diagnostics": diagnostics,
    }
    _atomic_write_text(
        BASIS_CACHE_METADATA_FILE,
        json.dumps(_json_ready(metadata), ensure_ascii=False, indent=2),
    )
    try:
        BASIS_PROGRESS_FILE.unlink(missing_ok=True)
    except TypeError:
        if BASIS_PROGRESS_FILE.exists():
            BASIS_PROGRESS_FILE.unlink()

    del high_geometry, legacy_geometry, eigenfunctions
    del observed_memmap, query_memmap, radial_design_memmap, column_rms_memmap
    gc.collect()

    return {
        "observed_nonconstant_gradient": np.load(
            OBSERVED_GRADIENT_FILE, mmap_mode="r"
        ),
        "query_primary_nonconstant_gradient": np.load(
            QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r"
        ),
        "radial_design_observed": np.load(RADIAL_DESIGN_FILE, mmap_mode="r"),
        "column_rms": np.load(COLUMN_RMS_FILE, mmap_mode="r"),
        "metadata": metadata,
        "loaded_from_cache": False,
    }


# ==================================================================================================
# 4. Metrics and potential solver
# ==================================================================================================


def _weighted_quantile(values, quantile, weights=None):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    values = values[finite]
    if values.size == 0:
        return math.nan
    if weights is None:
        return float(np.quantile(values, quantile))
    weights = np.maximum(np.asarray(weights, dtype=np.float64)[finite], 0.0)
    if np.sum(weights) <= 0.0:
        return math.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(float(quantile), cumulative, values))


def _direction_errors(true, predicted):
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
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


def _scalar_nrmse(true, predicted, weights=None):
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if weights is None:
        weights = np.ones(true.size, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    total = float(np.sum(weights))
    if total <= 0.0:
        return math.nan
    mse = float(np.sum(weights * np.square(predicted - true)) / total)
    energy = float(np.sum(weights * np.square(true)) / total)
    return math.sqrt(mse / energy) if energy > 0.0 else math.nan


def _radial_vector_metrics(true, predicted, line_of_sight, weights=None):
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    line_of_sight = np.asarray(line_of_sight, dtype=np.float64)
    n = true.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("Metric weights have zero total.")
    w = weights / weight_sum

    residual = predicted - true
    true_radial_scalar = np.einsum("ij,ij->i", line_of_sight, true)
    predicted_radial_scalar = np.einsum("ij,ij->i", line_of_sight, predicted)
    residual_radial_scalar = predicted_radial_scalar - true_radial_scalar
    true_radial = true_radial_scalar[:, None] * line_of_sight
    true_tangential = true - true_radial
    residual_radial = residual_radial_scalar[:, None] * line_of_sight
    residual_tangential = residual - residual_radial

    def ratio(error_vector, truth_vector):
        numerator = float(np.sum(w * np.sum(np.square(error_vector), axis=1)))
        denominator = float(np.sum(w * np.sum(np.square(truth_vector), axis=1)))
        return math.sqrt(numerator / denominator) if denominator > 0.0 else math.nan

    angles = _direction_errors(true, predicted)
    true_energy = float(np.sum(w[:, None] * np.square(true)))
    vector_gain = (
        float(np.sum(w[:, None] * true * predicted) / true_energy)
        if true_energy > 0.0
        else math.nan
    )
    return {
        "n": int(n),
        "nrmse_radial": float(
            _scalar_nrmse(
                true_radial_scalar,
                predicted_radial_scalar,
                weights=weights,
            )
        ),
        "nrmse_tangential": float(ratio(residual_tangential, true_tangential)),
        "nrmse_3d": float(ratio(residual, true)),
        "direction_error_median_deg": float(
            _weighted_quantile(angles, 0.5, weights=weights)
        ),
        "direction_error_p90_deg": float(
            _weighted_quantile(angles, 0.9, weights=weights)
        ),
        "vector_gain_through_origin": vector_gain,
    }


def _regularization_diagonal(generator_values, power):
    raw = np.asarray(generator_values, dtype=np.float64)
    output = np.zeros_like(raw)
    positive = raw > 0.0
    output[positive] = np.power(
        np.maximum(raw[positive], NUMERICAL_RIDGE),
        float(power),
    )
    return output


def _weighted_crossproducts(design, targets, weights):
    design = np.asarray(design, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    normalization = float(np.sum(weights))
    if normalization <= 0.0:
        raise ValueError("Loss weights have zero total.")
    sqrt_weights = np.sqrt(weights / normalization)
    weighted_design = design * sqrt_weights[:, None]
    weighted_target = targets * sqrt_weights
    gram = weighted_design.T @ weighted_design
    rhs = weighted_design.T @ weighted_target
    return gram, rhs


def _cholesky_with_jitter(system):
    system = np.asarray(system, dtype=np.float64)
    scale = max(float(np.median(np.diag(system))), 1.0)
    for jitter_factor in (0.0, 1.0e-14, 1.0e-12, 1.0e-10, 1.0e-8):
        candidate = system.copy()
        if jitter_factor > 0.0:
            candidate[np.diag_indices_from(candidate)] += jitter_factor * scale
        try:
            factor = linalg.cholesky(candidate, lower=True, check_finite=False)
            return factor, float(jitter_factor * scale)
        except linalg.LinAlgError:
            continue
    raise linalg.LinAlgError("Could not factor the regularized high-mode system.")


def _solve_leading_cholesky(factor, rhs, n_basis):
    local_factor = factor[:n_basis, :n_basis]
    intermediate = linalg.solve_triangular(
        local_factor,
        rhs[:n_basis],
        lower=True,
        check_finite=False,
    )
    return linalg.solve_triangular(
        local_factor.T,
        intermediate,
        lower=False,
        check_finite=False,
    )


def _candidate_grid_coefficients(
    gram,
    rhs,
    generator_values,
    basis_sizes,
    strengths,
    powers,
):
    max_basis = gram.shape[0]
    metadata = []
    coefficient_columns = []
    for power in powers:
        regularization = _regularization_diagonal(generator_values, power)
        for strength in strengths:
            system = gram.copy()
            system[np.diag_indices_from(system)] += (
                float(strength) * regularization + NUMERICAL_RIDGE
            )
            factor, extra_jitter = _cholesky_with_jitter(system)
            for n_basis in basis_sizes:
                coefficients = _solve_leading_cholesky(factor, rhs, int(n_basis))
                padded = np.zeros(max_basis, dtype=np.float64)
                padded[:n_basis] = coefficients
                coefficient_columns.append(padded)
                metadata.append(
                    {
                        "n_basis": int(n_basis),
                        "n_diffusion_modes": int(n_basis - N_AFFINE_MODES),
                        "regularization_strength": float(strength),
                        "penalty_power": float(power),
                        "extra_cholesky_jitter": float(extra_jitter),
                    }
                )
    return np.column_stack(coefficient_columns), pd.DataFrame(metadata)


def _many_velocity_predictions(nonconstant_gradient, coefficient_matrix, affine_scale):
    coefficients = np.asarray(coefficient_matrix, dtype=np.float64)
    nonconstant = np.einsum(
        "imc,mk->ick",
        np.asarray(nonconstant_gradient, dtype=np.float64),
        coefficients[N_AFFINE_MODES:, :],
        optimize=True,
    )
    return nonconstant + float(affine_scale) * coefficients[:N_AFFINE_MODES, :][
        None, :, :
    ]


def _single_velocity_prediction(nonconstant_gradient, coefficients, n_basis, affine_scale):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    n_diffusion = int(n_basis - N_AFFINE_MODES)
    output = np.broadcast_to(
        float(affine_scale) * coefficients[:N_AFFINE_MODES][None, :],
        (nonconstant_gradient.shape[0], 3),
    ).copy()
    if n_diffusion > 0:
        output += np.einsum(
            "imc,m->ic",
            np.asarray(nonconstant_gradient[:, :n_diffusion, :], dtype=np.float64),
            coefficients[N_AFFINE_MODES:n_basis],
            optimize=True,
        )
    return output


def _many_hidden_validation_metrics(
    true,
    predicted_many,
    line_of_sight,
    weights,
):
    true = np.asarray(true, dtype=np.float64)
    predictions = np.asarray(predicted_many, dtype=np.float64)
    line_of_sight = np.asarray(line_of_sight, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    w = weights / np.sum(weights)

    residual = predictions - true[:, :, None]
    true_radial_scalar = np.einsum("ij,ij->i", line_of_sight, true)
    predicted_radial_scalar = np.einsum(
        "ij,ijk->ik",
        line_of_sight,
        predictions,
        optimize=True,
    )
    residual_radial_scalar = predicted_radial_scalar - true_radial_scalar[:, None]
    true_radial = true_radial_scalar[:, None] * line_of_sight
    true_tangential = true - true_radial
    residual_radial = residual_radial_scalar[:, None, :] * line_of_sight[:, :, None]
    residual_tangential = residual - residual_radial

    radial_energy = float(np.sum(w * np.square(true_radial_scalar)))
    tangential_energy = float(
        np.sum(w * np.sum(np.square(true_tangential), axis=1))
    )
    total_energy = float(np.sum(w * np.sum(np.square(true), axis=1)))

    radial_mse = np.sum(w[:, None] * np.square(residual_radial_scalar), axis=0)
    tangential_mse = np.sum(
        w[:, None] * np.sum(np.square(residual_tangential), axis=1),
        axis=0,
    )
    total_mse = np.sum(
        w[:, None] * np.sum(np.square(residual), axis=1),
        axis=0,
    )
    return {
        "nrmse_radial": np.sqrt(radial_mse / radial_energy),
        "nrmse_tangential": np.sqrt(tangential_mse / tangential_energy),
        "nrmse_3d": np.sqrt(total_mse / total_energy),
    }


def _select_row(frame, primary_columns):
    tie_columns = [
        "n_basis",
        "regularization_strength",
        "penalty_power",
        "loss_gamma",
        "loss_cap",
    ]
    sort_columns = list(primary_columns) + [
        column for column in tie_columns if column in frame.columns
    ]
    return frame.sort_values(sort_columns, kind="mergesort").iloc[0].to_dict()


def _solve_selected_configurations(
    gram,
    rhs,
    generator_values,
    configurations,
):
    max_basis = gram.shape[0]
    output = {}
    grouped = defaultdict(list)
    for item in configurations:
        grouped[(float(item["penalty_power"]), float(item["regularization_strength"]))].append(item)
    for (power, strength), items in grouped.items():
        regularization = _regularization_diagonal(generator_values, power)
        system = gram.copy()
        system[np.diag_indices_from(system)] += (
            strength * regularization + NUMERICAL_RIDGE
        )
        factor, extra_jitter = _cholesky_with_jitter(system)
        for item in items:
            n_basis = int(item["n_basis"])
            coefficients = _solve_leading_cholesky(factor, rhs, n_basis)
            padded = np.zeros(max_basis, dtype=np.float64)
            padded[:n_basis] = coefficients
            output[n_basis] = {
                "coefficients": padded,
                "extra_cholesky_jitter": extra_jitter,
                **item,
            }
    return output


def _practical_plateau(basis_values, validation_values):
    basis_values = np.asarray(basis_values, dtype=int)
    validation_values = np.asarray(validation_values, dtype=np.float64)
    order = np.argsort(basis_values)
    basis_values = basis_values[order]
    validation_values = validation_values[order]
    improvements = np.full(basis_values.size, np.nan, dtype=np.float64)
    for index in range(1, basis_values.size):
        previous = validation_values[index - 1]
        current = validation_values[index]
        improvements[index] = (previous - current) / previous

    plateau_basis = math.nan
    plateau_found = False
    consecutive = int(PRACTICAL_PLATEAU_CONSECUTIVE_STEPS)
    for index in range(basis_values.size):
        future = improvements[index + 1 : index + 1 + consecutive]
        if future.size == consecutive and np.all(
            future < PRACTICAL_PLATEAU_RELATIVE_THRESHOLD
        ):
            plateau_basis = int(basis_values[index])
            plateau_found = True
            break
    return {
        "basis_values": basis_values,
        "validation_values": validation_values,
        "adjacent_improvement": improvements,
        "plateau_basis": plateau_basis,
        "plateau_found": plateau_found,
    }


def _matrix_relative_difference(a, b):
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    denominator = max(float(np.linalg.norm(b)), np.finfo(float).eps)
    return float(np.linalg.norm(a - b) / denominator)


# ==================================================================================================
# 5. Low-mode continuity audits
# ==================================================================================================


def _assemble_legacy_lowmode_basis(
    legacy_geometry_file,
    legacy_gradient_cache_file,
    line_of_sight_survey,
    primary_query_mask,
):
    legacy_geometry = _load_geometry(legacy_geometry_file)
    with np.load(legacy_gradient_cache_file, allow_pickle=False) as data:
        observed_gradient_all = np.asarray(
            data["observed_gradient_all"], dtype=np.float64
        )
        query_gradient_all = np.asarray(data["query_gradient_all"], dtype=np.float64)

    constant_index = int(np.argmin(legacy_geometry["generator_eigenvalues"]))
    pool = np.arange(legacy_geometry["generator_eigenvalues"].size)
    active = pool[pool != constant_index][:GENERATOR_REFERENCE_NONCONSTANT_MODES]
    observed_active = observed_gradient_all[:, active, :].copy()
    query_active = query_gradient_all[:, active, :].copy()

    affine_gram = line_of_sight_survey.T @ line_of_sight_survey
    radial = np.einsum(
        "ic,imc->im",
        line_of_sight_survey,
        observed_active,
        optimize=True,
    )
    affine_projection = linalg.solve(
        affine_gram,
        line_of_sight_survey.T @ radial,
        assume_a="sym",
        check_finite=False,
    ).T
    observed_active -= affine_projection[None, :, :]
    query_active -= affine_projection[None, :, :]
    column_rms = np.sqrt(
        np.mean(np.sum(np.square(observed_active), axis=2), axis=0)
    )
    positive = column_rms[np.isfinite(column_rms) & (column_rms > 0.0)]
    affine_scale = float(np.median(positive)) if positive.size else 1.0

    radial_design = np.empty(
        (line_of_sight_survey.shape[0], N_AFFINE_MODES + active.size),
        dtype=np.float64,
    )
    radial_design[:, :N_AFFINE_MODES] = affine_scale * line_of_sight_survey
    radial_design[:, N_AFFINE_MODES:] = np.einsum(
        "ic,imc->im",
        line_of_sight_survey,
        observed_active,
        optimize=True,
    )
    return {
        "observed_nonconstant_gradient": observed_active,
        "query_primary_nonconstant_gradient": query_active[primary_query_mask],
        "radial_design": radial_design,
        "affine_scale": affine_scale,
    }


def _basis_continuity_audit(
    high_basis,
    legacy_geometry_file,
    line_of_sight_survey,
    primary_query_mask,
):
    if not LEGACY_GRADIENT_CACHE_FILE.is_file():
        return pd.DataFrame(
            [{"quantity": "legacy_gradient_cache_available", "value": 0.0}]
        )
    legacy = _assemble_legacy_lowmode_basis(
        legacy_geometry_file,
        LEGACY_GRADIENT_CACHE_FILE,
        line_of_sight_survey,
        primary_query_mask,
    )
    n = min(
        GENERATOR_REFERENCE_NONCONSTANT_MODES,
        high_basis["observed_nonconstant_gradient"].shape[1],
    )
    high_obs = np.asarray(
        high_basis["observed_nonconstant_gradient"][:, :n, :],
        dtype=np.float64,
    )
    high_query = np.asarray(
        high_basis["query_primary_nonconstant_gradient"][:, :n, :],
        dtype=np.float64,
    )
    high_radial = np.asarray(
        high_basis["radial_design_observed"][:, : N_AFFINE_MODES + n],
        dtype=np.float64,
    )
    return pd.DataFrame(
        [
            {"quantity": "legacy_gradient_cache_available", "value": 1.0},
            {
                "quantity": "affine_scale_abs_difference",
                "value": abs(
                    float(high_basis["metadata"]["affine_mode_scale"])
                    - float(legacy["affine_scale"])
                ),
            },
            {
                "quantity": "observed_gradient_max_abs_difference",
                "value": float(np.max(np.abs(high_obs - legacy["observed_nonconstant_gradient"][:, :n, :]))),
            },
            {
                "quantity": "observed_gradient_relative_frobenius_difference",
                "value": _matrix_relative_difference(
                    high_obs,
                    legacy["observed_nonconstant_gradient"][:, :n, :],
                ),
            },
            {
                "quantity": "query_primary_gradient_max_abs_difference",
                "value": float(np.max(np.abs(high_query - legacy["query_primary_nonconstant_gradient"][:, :n, :]))),
            },
            {
                "quantity": "query_primary_gradient_relative_frobenius_difference",
                "value": _matrix_relative_difference(
                    high_query,
                    legacy["query_primary_nonconstant_gradient"][:, :n, :],
                ),
            },
            {
                "quantity": "radial_design_max_abs_difference",
                "value": float(np.max(np.abs(high_radial - legacy["radial_design"][:, : N_AFFINE_MODES + n]))),
            },
            {
                "quantity": "radial_design_relative_frobenius_difference",
                "value": _matrix_relative_difference(
                    high_radial,
                    legacy["radial_design"][:, : N_AFFINE_MODES + n],
                ),
            },
        ]
    )




def _atomic_write_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    os.replace(temporary, path)


def _partial_basis_cache_valid(signature, n_survey, n_query_primary):
    required = [
        BASIS_PARTIAL_METADATA_FILE,
        OBSERVED_GRADIENT_FILE,
        QUERY_PRIMARY_GRADIENT_FILE,
        RADIAL_DESIGN_FILE,
        COLUMN_RMS_FILE,
    ]
    if FORCE_REBUILD_CACHE or not all(path.is_file() for path in required):
        return False, 0
    try:
        metadata = json.loads(
            BASIS_PARTIAL_METADATA_FILE.read_text(encoding="utf-8")
        )
        if metadata.get("signature") != signature:
            return False, 0
        completed_modes = int(metadata.get("completed_modes", 0))
        observed_gradient = np.load(OBSERVED_GRADIENT_FILE, mmap_mode="r")
        query_gradient = np.load(QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r")
        radial_design = np.load(RADIAL_DESIGN_FILE, mmap_mode="r")
        column_rms = np.load(COLUMN_RMS_FILE, mmap_mode="r")
        shapes_valid = bool(
            observed_gradient.shape == (n_survey, MAX_NONCONSTANT_MODES, 3)
            and query_gradient.shape
            == (n_query_primary, MAX_NONCONSTANT_MODES, 3)
            and radial_design.shape == (n_survey, MAX_BASIS)
            and column_rms.shape == (MAX_NONCONSTANT_MODES,)
        )
        if not shapes_valid:
            return False, 0
        if not 0 <= completed_modes <= MAX_NONCONSTANT_MODES:
            return False, 0
        return True, completed_modes
    except Exception as exc:
        warnings.warn(f"Partial high-mode basis checkpoint could not be reused: {exc}")
        return False, 0


def _build_or_load_highmode_basis_legacy_unused(
    centered_survey,
    centered_query_primary,
    line_of_sight_survey,
    high_geometry_file,
    legacy_geometry_file,
    query_global_indices,
    primary_query_mask,
):
    signature = _basis_cache_signature(
        high_geometry_file,
        legacy_geometry_file,
        query_global_indices,
        primary_query_mask,
    )
    n_survey = int(centered_survey.shape[0])
    n_query_primary = int(centered_query_primary.shape[0])

    if _cache_files_valid(signature, n_survey, n_query_primary):
        metadata = json.loads(BASIS_CACHE_METADATA_FILE.read_text(encoding="utf-8"))
        print(f"[High-mode basis cache] reuse: {CACHE_DIR}")
        return {
            "observed_nonconstant_gradient": np.load(
                OBSERVED_GRADIENT_FILE, mmap_mode="r"
            ),
            "query_primary_nonconstant_gradient": np.load(
                QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r"
            ),
            "radial_design_observed": np.load(
                RADIAL_DESIGN_FILE, mmap_mode="r"
            ),
            "column_rms": np.load(COLUMN_RMS_FILE, mmap_mode="r"),
            "metadata": metadata,
            "loaded_from_cache": True,
        }

    print("[High-mode basis] constructing metric-calibrated gradients...")
    high_geometry = _load_geometry(high_geometry_file)
    legacy_geometry = _load_geometry(legacy_geometry_file)
    if high_geometry["eigenfunctions"].shape[0] != n_survey:
        raise ValueError("High-mode geometry does not match the Mock-3 survey size.")
    if high_geometry["eigenfunctions"].shape[1] < MAX_NONCONSTANT_MODES + 1:
        raise ValueError(
            "High-mode geometry contains too few modes: "
            f"{high_geometry['eigenfunctions'].shape[1]} < "
            f"{MAX_NONCONSTANT_MODES + 1}."
        )
    if abs(float(high_geometry["alpha"]) - ALPHA_DM) > 1.0e-12:
        raise ValueError(
            "High-mode geometry is not the required unweighted alpha=0 geometry."
        )

    geometry_audit_df, common_signs = _geometry_continuity_and_signs(
        high_geometry,
        legacy_geometry,
    )
    geometry_audit_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_continuity.csv",
        index=False,
    )

    constant_mode_index = int(np.argmin(high_geometry["generator_eigenvalues"]))
    all_mode_indices = np.arange(high_geometry["eigenfunctions"].shape[1])
    active_mode_indices = all_mode_indices[
        all_mode_indices != constant_mode_index
    ][:MAX_NONCONSTANT_MODES]
    if active_mode_indices.size != MAX_NONCONSTANT_MODES:
        raise RuntimeError("Could not select the requested nonconstant high-mode basis.")

    sign_by_raw_mode = np.ones(
        high_geometry["eigenfunctions"].shape[1], dtype=np.float64
    )
    n_common = min(common_signs.size, sign_by_raw_mode.size)
    sign_by_raw_mode[:n_common] = common_signs[:n_common]

    base_kernel = _build_base_kernel(
        centered_survey,
        GRAPH_NEIGHBORS,
        GRAPH_BANDWIDTH_NEIGHBOR,
        GRAPH_BANDWIDTH_MULTIPLIER,
    )
    markov = _observed_markov_from_base_kernel(base_kernel)
    observed_metric, observed_mean = _centered_coordinate_metric_sparse(
        markov,
        centered_survey,
        base_kernel["rho"],
    )
    (
        inverse_observed_metric,
        observed_metric_eigenvalues,
        observed_condition,
        observed_clipped_count,
    ) = _invert_local_metric(observed_metric)

    query_transition = _query_transition(centered_query_primary, base_kernel)
    query_metric, query_mean = _query_metric(
        query_transition,
        centered_survey,
    )
    (
        inverse_query_metric,
        query_metric_eigenvalues,
        query_condition,
        query_clipped_count,
    ) = _invert_local_metric(query_metric)

    partial_valid, completed_modes = _partial_basis_cache_valid(
        signature,
        n_survey,
        n_query_primary,
    )
    if partial_valid:
        observed_memmap = open_memmap(
            OBSERVED_GRADIENT_FILE,
            mode="r+",
            dtype=np.float64,
            shape=(n_survey, MAX_NONCONSTANT_MODES, 3),
        )
        query_memmap = open_memmap(
            QUERY_PRIMARY_GRADIENT_FILE,
            mode="r+",
            dtype=np.float64,
            shape=(n_query_primary, MAX_NONCONSTANT_MODES, 3),
        )
        radial_design_memmap = open_memmap(
            RADIAL_DESIGN_FILE,
            mode="r+",
            dtype=np.float64,
            shape=(n_survey, MAX_BASIS),
        )
        column_rms_memmap = open_memmap(
            COLUMN_RMS_FILE,
            mode="r+",
            dtype=np.float64,
            shape=(MAX_NONCONSTANT_MODES,),
        )
        print(
            f"[High-mode basis checkpoint] reuse: "
            f"{completed_modes}/{MAX_NONCONSTANT_MODES} nonconstant modes",
            flush=True,
        )
    else:
        completed_modes = 0
        observed_memmap = open_memmap(
            OBSERVED_GRADIENT_FILE,
            mode="w+",
            dtype=np.float64,
            shape=(n_survey, MAX_NONCONSTANT_MODES, 3),
        )
        query_memmap = open_memmap(
            QUERY_PRIMARY_GRADIENT_FILE,
            mode="w+",
            dtype=np.float64,
            shape=(n_query_primary, MAX_NONCONSTANT_MODES, 3),
        )
        radial_design_memmap = open_memmap(
            RADIAL_DESIGN_FILE,
            mode="w+",
            dtype=np.float64,
            shape=(n_survey, MAX_BASIS),
        )
        column_rms_memmap = open_memmap(
            COLUMN_RMS_FILE,
            mode="w+",
            dtype=np.float64,
            shape=(MAX_NONCONSTANT_MODES,),
        )
        _atomic_write_json(
            BASIS_PARTIAL_METADATA_FILE,
            {
                "signature": signature,
                "completed_modes": 0,
                "max_nonconstant_modes": MAX_NONCONSTANT_MODES,
                "status": "building",
            },
        )

    affine_gram = line_of_sight_survey.T @ line_of_sight_survey
    eigenfunctions = high_geometry["eigenfunctions"]

    for start in range(
        completed_modes,
        MAX_NONCONSTANT_MODES,
        max(1, GRADIENT_MODE_CHUNK_SIZE),
    ):
        stop = min(start + max(1, GRADIENT_MODE_CHUNK_SIZE), MAX_NONCONSTANT_MODES)
        raw_indices = active_mode_indices[start:stop]
        functions = np.asarray(
            eigenfunctions[:, raw_indices],
            dtype=np.float64,
        )
        functions = functions * sign_by_raw_mode[raw_indices][None, :]

        observed_block = _observed_function_gradient_block(
            markov,
            centered_survey,
            functions,
            base_kernel["rho"],
            inverse_observed_metric,
            observed_mean,
        )
        query_block = _query_function_gradient_block(
            query_transition,
            centered_survey,
            functions,
            inverse_query_metric,
            query_mean,
        )

        observed_radial = np.einsum(
            "ic,imc->im",
            line_of_sight_survey,
            observed_block,
            optimize=True,
        )
        affine_projection_vectors = linalg.solve(
            affine_gram,
            line_of_sight_survey.T @ observed_radial,
            assume_a="sym",
            check_finite=False,
        ).T
        observed_block -= affine_projection_vectors[None, :, :]
        query_block -= affine_projection_vectors[None, :, :]

        observed_memmap[:, start:stop, :] = observed_block
        query_memmap[:, start:stop, :] = query_block
        radial_design_memmap[
            :,
            N_AFFINE_MODES + start : N_AFFINE_MODES + stop,
        ] = np.einsum(
            "ic,imc->im",
            line_of_sight_survey,
            observed_block,
            optimize=True,
        )
        column_rms_memmap[start:stop] = np.sqrt(
            np.mean(np.sum(np.square(observed_block), axis=2), axis=0)
        )

        observed_memmap.flush()
        query_memmap.flush()
        radial_design_memmap.flush()
        column_rms_memmap.flush()
        completed_modes = stop
        _atomic_write_json(
            BASIS_PARTIAL_METADATA_FILE,
            {
                "signature": signature,
                "completed_modes": completed_modes,
                "max_nonconstant_modes": MAX_NONCONSTANT_MODES,
                "status": "building",
            },
        )
        print(
            f"[High-mode basis] nonconstant modes {start + 1}:{stop}/"
            f"{MAX_NONCONSTANT_MODES}; checkpoint saved",
            flush=True,
        )

    positive_reference = np.asarray(
        column_rms_memmap[
            : min(
                GENERATOR_REFERENCE_NONCONSTANT_MODES,
                MAX_NONCONSTANT_MODES,
            )
        ]
    )
    positive_reference = positive_reference[
        np.isfinite(positive_reference) & (positive_reference > 0.0)
    ]
    affine_mode_scale = (
        float(np.median(positive_reference)) if positive_reference.size else 1.0
    )
    radial_design_memmap[:, :N_AFFINE_MODES] = (
        affine_mode_scale * line_of_sight_survey
    )
    radial_design_memmap.flush()

    affine_coefficient = np.array([0.37, -0.52, 0.81], dtype=np.float64)
    affine_observed = centered_survey @ affine_coefficient + 1.7
    affine_observed_gradient = _observed_function_gradient_block(
        markov,
        centered_survey,
        affine_observed,
        base_kernel["rho"],
        inverse_observed_metric,
        observed_mean,
    )[:, 0, :]
    affine_query_gradient = _query_function_gradient_block(
        query_transition,
        centered_survey,
        affine_observed,
        inverse_query_metric,
        query_mean,
    )[:, 0, :]

    quadratic_observed = 0.5 * np.sum(np.square(centered_survey), axis=1)
    quadratic_observed_gradient = _observed_function_gradient_block(
        markov,
        centered_survey,
        quadratic_observed,
        base_kernel["rho"],
        inverse_observed_metric,
        observed_mean,
    )[:, 0, :]
    quadratic_query_gradient = _query_function_gradient_block(
        query_transition,
        centered_survey,
        quadratic_observed,
        inverse_query_metric,
        query_mean,
    )[:, 0, :]

    constant_function = high_geometry["eigenfunctions"][:, constant_mode_index]
    constant_observed_gradient = _observed_function_gradient_block(
        markov,
        centered_survey,
        constant_function,
        base_kernel["rho"],
        inverse_observed_metric,
        observed_mean,
    )[:, 0, :]
    constant_query_gradient = _query_function_gradient_block(
        query_transition,
        centered_survey,
        constant_function,
        inverse_query_metric,
        query_mean,
    )[:, 0, :]

    diagnostics = {
        "observed_metric_condition_median": float(np.median(observed_condition)),
        "observed_metric_condition_p99": float(
            np.quantile(observed_condition, 0.99)
        ),
        "observed_metric_condition_max": float(np.max(observed_condition)),
        "query_metric_condition_median": float(np.median(query_condition)),
        "query_metric_condition_p99": float(np.quantile(query_condition, 0.99)),
        "query_metric_condition_max": float(np.max(query_condition)),
        "observed_metric_clipped_fraction": float(
            np.mean(observed_clipped_count > 0)
        ),
        "query_metric_clipped_fraction": float(
            np.mean(query_clipped_count > 0)
        ),
        "observed_affine_rms_error": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            affine_observed_gradient
                            - affine_coefficient[None, :]
                        ),
                        axis=1,
                    )
                )
            )
        ),
        "query_affine_rms_error": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            affine_query_gradient
                            - affine_coefficient[None, :]
                        ),
                        axis=1,
                    )
                )
            )
        ),
        "observed_quadratic_gradient_nrmse": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            quadratic_observed_gradient - centered_survey
                        ),
                        axis=1,
                    )
                )
            )
            / max(
                float(
                    np.sqrt(
                        np.mean(
                            np.sum(np.square(centered_survey), axis=1)
                        )
                    )
                ),
                np.finfo(float).eps,
            )
        ),
        "query_quadratic_gradient_nrmse": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(
                            quadratic_query_gradient
                            - centered_query_primary
                        ),
                        axis=1,
                    )
                )
            )
            / max(
                float(
                    np.sqrt(
                        np.mean(
                            np.sum(
                                np.square(centered_query_primary),
                                axis=1,
                            )
                        )
                    )
                ),
                np.finfo(float).eps,
            )
        ),
        "constant_mode_gradient_rms_observed": float(
            np.sqrt(
                np.mean(
                    np.sum(np.square(constant_observed_gradient), axis=1)
                )
            )
        ),
        "constant_mode_gradient_rms_query": float(
            np.sqrt(
                np.mean(np.sum(np.square(constant_query_gradient), axis=1))
            )
        ),
        "affine_mode_scale": affine_mode_scale,
    }

    metadata = {
        "signature": signature,
        "high_geometry_file": str(high_geometry_file),
        "legacy_geometry_file": str(legacy_geometry_file),
        "constant_mode_index": constant_mode_index,
        "active_mode_indices": active_mode_indices.tolist(),
        "active_generator_values": high_geometry["generator_eigenvalues"][
            active_mode_indices
        ].tolist(),
        "affine_mode_scale": affine_mode_scale,
        "n_survey": n_survey,
        "n_query_primary": n_query_primary,
        "max_nonconstant_modes": MAX_NONCONSTANT_MODES,
        "max_basis": MAX_BASIS,
        "diagnostics": diagnostics,
    }
    _atomic_write_json(BASIS_CACHE_METADATA_FILE, metadata)
    if BASIS_PARTIAL_METADATA_FILE.exists():
        BASIS_PARTIAL_METADATA_FILE.unlink()

    del high_geometry, legacy_geometry, eigenfunctions
    del observed_memmap, query_memmap, radial_design_memmap, column_rms_memmap
    gc.collect()

    return {
        "observed_nonconstant_gradient": np.load(
            OBSERVED_GRADIENT_FILE, mmap_mode="r"
        ),
        "query_primary_nonconstant_gradient": np.load(
            QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r"
        ),
        "radial_design_observed": np.load(
            RADIAL_DESIGN_FILE, mmap_mode="r"
        ),
        "column_rms": np.load(COLUMN_RMS_FILE, mmap_mode="r"),
        "metadata": metadata,
        "loaded_from_cache": False,
    }


def _legacy_candidate_grid_audit(candidate_df):
    if not LEGACY_V2_VALIDATION_FILE.is_file():
        return pd.DataFrame(
            [{"quantity": "legacy_v2_validation_available", "value": 0.0}]
        )
    legacy = pd.read_csv(LEGACY_V2_VALIDATION_FILE)
    legacy = legacy[legacy["label_seed"] == LABEL_SPLIT_SEED].copy()
    current = candidate_df[candidate_df["n_basis"] <= 514].copy()
    keys = [
        "loss_name",
        "n_basis",
        "regularization_strength",
        "penalty_power",
    ]
    common = current.merge(
        legacy,
        on=keys,
        suffixes=("_current", "_legacy"),
    )
    records = [
        {
            "quantity": "legacy_v2_validation_available",
            "value": 1.0,
        },
        {
            "quantity": "n_common_candidate_rows",
            "value": float(common.shape[0]),
        },
    ]
    for column in [
        "weighted_validation_nrmse_radial",
        "unweighted_validation_nrmse_radial",
        "weighted_validation_nrmse_tangential",
        "weighted_validation_nrmse_3d",
    ]:
        left = f"{column}_current"
        right = f"{column}_legacy"
        if left in common and right in common:
            records.append(
                {
                    "quantity": f"{column}_max_abs_difference",
                    "value": float(np.max(np.abs(common[left] - common[right]))),
                }
            )
    return pd.DataFrame(records)


def _chunked_mode_difference(current, reference, n_modes, chunk_size=32):
    """Compare large gradient arrays without materializing the full difference."""
    max_abs = 0.0
    diff_square_sum = 0.0
    reference_square_sum = 0.0
    for start in range(0, int(n_modes), int(chunk_size)):
        stop = min(start + int(chunk_size), int(n_modes))
        current_block = np.asarray(current[:, start:stop, ...], dtype=np.float64)
        reference_block = np.asarray(reference[:, start:stop, ...], dtype=np.float64)
        difference = current_block - reference_block
        if difference.size:
            max_abs = max(max_abs, float(np.max(np.abs(difference))))
            diff_square_sum += float(np.sum(np.square(difference)))
            reference_square_sum += float(np.sum(np.square(reference_block)))
    relative = math.sqrt(diff_square_sum) / max(
        math.sqrt(reference_square_sum),
        np.finfo(float).eps,
    )
    return max_abs, float(relative)


def _chunked_column_difference(current, reference, n_columns, chunk_size=64):
    """Compare large two-dimensional design arrays column by column."""
    max_abs = 0.0
    diff_square_sum = 0.0
    reference_square_sum = 0.0
    for start in range(0, int(n_columns), int(chunk_size)):
        stop = min(start + int(chunk_size), int(n_columns))
        current_block = np.asarray(current[:, start:stop], dtype=np.float64)
        reference_block = np.asarray(reference[:, start:stop], dtype=np.float64)
        difference = current_block - reference_block
        if difference.size:
            max_abs = max(max_abs, float(np.max(np.abs(difference))))
            diff_square_sum += float(np.sum(np.square(difference)))
            reference_square_sum += float(np.sum(np.square(reference_block)))
    relative = math.sqrt(diff_square_sum) / max(
        math.sqrt(reference_square_sum),
        np.finfo(float).eps,
    )
    return max_abs, float(relative)


def _previous_h1_basis_continuity_audit(high_basis):
    required = [
        PREVIOUS_H1_BASIS_METADATA_FILE,
        PREVIOUS_H1_OBSERVED_GRADIENT_FILE,
        PREVIOUS_H1_QUERY_PRIMARY_GRADIENT_FILE,
        PREVIOUS_H1_RADIAL_DESIGN_FILE,
        PREVIOUS_H1_COLUMN_RMS_FILE,
    ]
    if not all(path.is_file() for path in required):
        return pd.DataFrame(
            [{"quantity": "previous_h1_basis_cache_available", "value": 0.0}]
        )
    try:
        metadata = json.loads(PREVIOUS_H1_BASIS_METADATA_FILE.read_text(encoding="utf-8"))
        previous_h1_observed = np.load(PREVIOUS_H1_OBSERVED_GRADIENT_FILE, mmap_mode="r")
        previous_h1_query = np.load(PREVIOUS_H1_QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r")
        previous_h1_radial = np.load(PREVIOUS_H1_RADIAL_DESIGN_FILE, mmap_mode="r")
        previous_h1_column_rms = np.load(PREVIOUS_H1_COLUMN_RMS_FILE, mmap_mode="r")

        n_modes = min(
            PREVIOUS_H1_MAX_NONCONSTANT_MODES,
            int(previous_h1_observed.shape[1]),
            int(high_basis["observed_nonconstant_gradient"].shape[1]),
        )
        if previous_h1_observed.shape[0] != high_basis["observed_nonconstant_gradient"].shape[0]:
            raise ValueError("H1 observed gradient cache has a different survey size.")
        if previous_h1_query.shape[0] != high_basis["query_primary_nonconstant_gradient"].shape[0]:
            raise ValueError("H1 query gradient cache has a different primary-query size.")

        observed_max, observed_relative = _chunked_mode_difference(
            high_basis["observed_nonconstant_gradient"], previous_h1_observed, n_modes
        )
        query_max, query_relative = _chunked_mode_difference(
            high_basis["query_primary_nonconstant_gradient"], previous_h1_query, n_modes
        )
        radial_max, radial_relative = _chunked_column_difference(
            high_basis["radial_design_observed"],
            previous_h1_radial,
            N_AFFINE_MODES + n_modes,
        )
        current_rms = np.asarray(high_basis["column_rms"][:n_modes], dtype=np.float64)
        reference_rms = np.asarray(previous_h1_column_rms[:n_modes], dtype=np.float64)
        current_modes = np.asarray(
            high_basis["metadata"].get("active_mode_indices", [])[:n_modes],
            dtype=np.int64,
        )
        previous_h1_modes = np.asarray(
            metadata.get("active_mode_indices", [])[:n_modes],
            dtype=np.int64,
        )
        active_equal = bool(
            current_modes.size == n_modes
            and previous_h1_modes.size == n_modes
            and np.array_equal(current_modes, previous_h1_modes)
        )
        return pd.DataFrame(
            [
                {"quantity": "previous_h1_basis_cache_available", "value": 1.0},
                {"quantity": "previous_h1_audited_nonconstant_modes", "value": float(n_modes)},
                {"quantity": "previous_h1_active_mode_indices_equal", "value": float(active_equal)},
                {
                    "quantity": "previous_h1_affine_scale_abs_difference",
                    "value": abs(
                        float(high_basis["metadata"]["affine_mode_scale"])
                        - float(metadata.get("affine_mode_scale", math.nan))
                    ),
                },
                {"quantity": "previous_h1_observed_gradient_max_abs_difference", "value": observed_max},
                {
                    "quantity": "previous_h1_observed_gradient_relative_frobenius_difference",
                    "value": observed_relative,
                },
                {"quantity": "previous_h1_query_gradient_max_abs_difference", "value": query_max},
                {
                    "quantity": "previous_h1_query_gradient_relative_frobenius_difference",
                    "value": query_relative,
                },
                {"quantity": "previous_h1_radial_design_max_abs_difference", "value": radial_max},
                {
                    "quantity": "previous_h1_radial_design_relative_frobenius_difference",
                    "value": radial_relative,
                },
                {
                    "quantity": "previous_h1_column_rms_max_abs_difference",
                    "value": float(np.max(np.abs(current_rms - reference_rms))),
                },
                {
                    "quantity": "previous_h1_column_rms_relative_frobenius_difference",
                    "value": _matrix_relative_difference(current_rms, reference_rms),
                },
            ]
        )
    except Exception as exc:
        warnings.warn(f"H1 basis continuity audit failed: {exc}")
        return pd.DataFrame(
            [
                {"quantity": "previous_h1_basis_cache_available", "value": 1.0},
                {"quantity": "previous_h1_basis_audit_completed", "value": 0.0},
            ]
        )


def _previous_h1_candidate_grid_audit(candidate_df):
    if not PREVIOUS_H1_CANDIDATE_GRID_FILE.is_file():
        return pd.DataFrame(
            [{"quantity": "previous_h1_candidate_grid_available", "value": 0.0}]
        )
    previous_h1 = pd.read_csv(PREVIOUS_H1_CANDIDATE_GRID_FILE)
    current = candidate_df[candidate_df["n_basis"] <= PREVIOUS_H1_MAX_BASIS].copy()
    previous_h1 = previous_h1[previous_h1["n_basis"] <= PREVIOUS_H1_MAX_BASIS].copy()
    keys = [
        "loss_name",
        "n_basis",
        "regularization_strength",
        "penalty_power",
    ]
    common = current.merge(
        previous_h1, on=keys, suffixes=("_h2", "_previous_h1")
    )
    expected_rows = int(
        len([basis for basis in POTENTIAL_BASIS_SIZES if basis <= PREVIOUS_H1_MAX_BASIS])
        * len(POTENTIAL_REGULARIZATION_STRENGTHS)
        * len(POTENTIAL_REGULARIZATION_POWERS)
        * len(LOSS_CONFIGS)
    )
    records = [
        {"quantity": "previous_h1_candidate_grid_available", "value": 1.0},
        {"quantity": "expected_common_candidate_rows", "value": float(expected_rows)},
        {"quantity": "n_common_candidate_rows", "value": float(common.shape[0])},
        {
            "quantity": "previous_h1_candidate_grid_complete",
            "value": float(common.shape[0] == expected_rows),
        },
    ]
    for column in [
        "octant_weighted_validation_nrmse_radial",
        "unweighted_validation_nrmse_radial",
        "octant_weighted_validation_nrmse_tangential",
        "octant_weighted_validation_nrmse_3d",
        "global_weighted_validation_nrmse_radial",
        "global_weighted_validation_nrmse_3d",
    ]:
        left = f"{column}_h2"
        right = f"{column}_previous_h1"
        if left in common.columns and right in common.columns and not common.empty:
            difference = np.abs(
                common[left].to_numpy(dtype=np.float64)
                - common[right].to_numpy(dtype=np.float64)
            )
            records.extend(
                [
                    {
                        "quantity": f"previous_h1_{column}_max_abs_difference",
                        "value": float(np.max(difference)),
                    },
                    {
                        "quantity": f"previous_h1_{column}_rms_difference",
                        "value": float(np.sqrt(np.mean(np.square(difference)))),
                    },
                ]
            )
    return pd.DataFrame(records)


def _previous_h1_selector_continuity_audit(candidate_df):
    if not PREVIOUS_H1_GLOBAL_SELECTORS_FILE.is_file():
        return pd.DataFrame(
            [{"quantity": "previous_h1_global_selectors_available", "value": 0.0}]
        )
    restricted = candidate_df[candidate_df["n_basis"] <= PREVIOUS_H1_MAX_BASIS].copy()
    current_rows = {
        "raw_radial_validation": _select_row(
            restricted,
            ["octant_weighted_validation_nrmse_radial", "unweighted_validation_nrmse_radial"],
        ),
        "hidden_3d_validation_diagnostic": _select_row(
            restricted,
            ["octant_weighted_validation_nrmse_3d", "octant_weighted_validation_nrmse_radial"],
        ),
    }
    previous_frame = pd.read_csv(PREVIOUS_H1_GLOBAL_SELECTORS_FILE)
    records = [{"quantity": "previous_h1_global_selectors_available", "value": 1.0}]
    for selector, current in current_rows.items():
        previous_rows = previous_frame[previous_frame["selector"] == selector]
        if previous_rows.empty:
            records.append({"quantity": f"previous_h1_{selector}_available", "value": 0.0})
            continue
        previous = previous_rows.iloc[0]
        same_configuration = bool(
            str(current["loss_name"]) == str(previous["loss_name"])
            and int(current["n_basis"]) == int(previous["n_basis"])
            and math.isclose(
                float(current["regularization_strength"]),
                float(previous["regularization_strength"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
            and math.isclose(
                float(current["penalty_power"]),
                float(previous["penalty_power"]),
                rel_tol=0.0,
                abs_tol=1.0e-15,
            )
        )
        records.extend(
            [
                {"quantity": f"previous_h1_{selector}_available", "value": 1.0},
                {
                    "quantity": f"previous_h1_{selector}_configuration_equal",
                    "value": float(same_configuration),
                },
                {
                    "quantity": f"previous_h1_{selector}_radial_nrmse_abs_difference",
                    "value": abs(
                        float(current["octant_weighted_validation_nrmse_radial"])
                        - float(previous["octant_weighted_validation_nrmse_radial"])
                    ),
                },
                {
                    "quantity": f"previous_h1_{selector}_hidden_3d_nrmse_abs_difference",
                    "value": abs(
                        float(current["octant_weighted_validation_nrmse_3d"])
                        - float(previous["octant_weighted_validation_nrmse_3d"])
                    ),
                },
            ]
        )
    return pd.DataFrame(records)


# ==================================================================================================
# 6. Main analysis
# ==================================================================================================






# ==================================================================================================
# 6. Mock-3-specific overrides and diagnostics
# ==================================================================================================


def _make_octant_stratified_split(
    region_id,
    train_fraction: float,
    validation_fraction: float,
    seed: int,
):
    region_id = np.asarray(region_id, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    train_parts = []
    validation_parts = []
    test_parts = []
    for region in range(8):
        local = rng.permutation(np.flatnonzero(region_id == region))
        if local.size < 3:
            raise ValueError(f"Octant {region} contains fewer than three objects.")
        n_train = int(round(float(train_fraction) * local.size))
        n_validation = int(round(float(validation_fraction) * local.size))
        if n_train + n_validation >= local.size:
            n_validation = max(1, local.size - n_train - 1)
        train_parts.append(local[:n_train])
        validation_parts.append(local[n_train : n_train + n_validation])
        test_parts.append(local[n_train + n_validation :])
    return {
        "train": np.sort(np.concatenate(train_parts)),
        "validation": np.sort(np.concatenate(validation_parts)),
        "test": np.sort(np.concatenate(test_parts)),
    }


def _load_mock3_metadata(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        required = {"pos", "octant", "mlim_per_octant"}
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Mock-3 is missing keys: {sorted(missing)}")
        return {
            "pos": np.asarray(data["pos"], dtype=np.float64),
            "region_id": np.asarray(data["octant"], dtype=np.int64),
            "mlim_per_region": np.asarray(
                data["mlim_per_octant"], dtype=np.float64
            ),
        }


def _load_mock1_positions(path: Path) -> np.ndarray:
    with np.load(path, allow_pickle=False) as data:
        if "pos" not in data.files:
            raise KeyError("Mock-1 is missing the position array 'pos'.")
        return np.asarray(data["pos"], dtype=np.float64)


def _loss_probability(config, probability_global, probability_region):
    if config["mode"] == "none":
        return np.ones_like(probability_region, dtype=np.float64)
    if config["mode"] == "global":
        return np.asarray(probability_global, dtype=np.float64)
    if config["mode"] == "region":
        return np.asarray(probability_region, dtype=np.float64)
    raise ValueError(f"Unknown Mock-3 loss mode: {config['mode']}")


def _octant_nrmse_summary(
    true,
    predicted,
    line_of_sight,
    region_id,
    weights=None,
):
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    line_of_sight = np.asarray(line_of_sight, dtype=np.float64)
    region_id = np.asarray(region_id, dtype=np.int64)
    if weights is not None:
        weights = np.asarray(weights, dtype=np.float64)
    values = []
    records = []
    for region in range(8):
        mask = region_id == region
        if np.sum(mask) == 0:
            continue
        local_weights = None if weights is None else weights[mask]
        metrics = _radial_vector_metrics(
            true[mask],
            predicted[mask],
            line_of_sight[mask],
            weights=local_weights,
        )
        value = float(metrics["nrmse_3d"])
        values.append(value)
        records.append(
            {
                "region_id": int(region),
                "n": int(np.sum(mask)),
                "nrmse_3d": value,
                "nrmse_radial": float(metrics["nrmse_radial"]),
                "nrmse_tangential": float(metrics["nrmse_tangential"]),
            }
        )
    array = np.asarray(values, dtype=np.float64)
    return {
        "n_octants": int(array.size),
        "octant_nrmse_3d_mean": float(np.mean(array)) if array.size else math.nan,
        "octant_nrmse_3d_std": float(np.std(array, ddof=0)) if array.size else math.nan,
        "worst_octant_nrmse_3d": float(np.max(array)) if array.size else math.nan,
        "records": records,
    }


def _basis_cache_signature(
    high_geometry_file,
    legacy_geometry_file,
    query_global_indices,
    primary_query_mask,
):
    return _stable_signature(
        {
            "mock1": _file_signature(MOCK1_FILE),
            "mock3": _file_signature(MOCK3_FILE),
            "predictions": _file_signature(PREDICTIONS_FILE),
            "high_geometry": _file_signature(high_geometry_file),
            "legacy_geometry": _file_signature(legacy_geometry_file),
            "query_indices_hash": _hash_array(query_global_indices),
            "primary_query_mask_hash": _hash_array(
                primary_query_mask.astype(np.uint8)
            ),
            "graph_neighbors": GRAPH_NEIGHBORS,
            "graph_bandwidth_neighbor": GRAPH_BANDWIDTH_NEIGHBOR,
            "graph_multiplier": GRAPH_BANDWIDTH_MULTIPLIER,
            "metric_relative_floor": METRIC_EIGENVALUE_RELATIVE_FLOOR,
            "metric_absolute_floor": METRIC_EIGENVALUE_ABSOLUTE_FLOOR,
            "max_nonconstant_modes": MAX_NONCONSTANT_MODES,
            "generator_reference_nonconstant_modes": (
                GENERATOR_REFERENCE_NONCONSTANT_MODES
            ),
            "version": 1,
        }
    )


def _legacy_candidate_grid_audit(candidate_df):
    if not LEGACY_V2_VALIDATION_FILE.is_file():
        return pd.DataFrame(
            [{"quantity": "legacy_validation_available", "value": 0.0}]
        )
    legacy = pd.read_csv(LEGACY_V2_VALIDATION_FILE)
    if "label_seed" in legacy.columns:
        legacy = legacy[legacy["label_seed"] == LABEL_SPLIT_SEED].copy()
    current = candidate_df[candidate_df["n_basis"] <= 514].copy()
    keys = [
        "loss_name",
        "n_basis",
        "regularization_strength",
        "penalty_power",
    ]
    common = current.merge(
        legacy,
        on=keys,
        suffixes=("_current", "_legacy"),
    )
    records = [
        {"quantity": "legacy_validation_available", "value": 1.0},
        {"quantity": "n_common_candidate_rows", "value": float(common.shape[0])},
    ]
    for column in [
        "octant_weighted_validation_nrmse_radial",
        "unweighted_validation_nrmse_radial",
        "octant_weighted_validation_nrmse_tangential",
        "octant_weighted_validation_nrmse_3d",
        "global_weighted_validation_nrmse_radial",
        "global_weighted_validation_nrmse_3d",
    ]:
        left = f"{column}_current"
        right = f"{column}_legacy"
        if left in common.columns and right in common.columns:
            records.append(
                {
                    "quantity": f"{column}_max_abs_difference",
                    "value": float(
                        np.max(np.abs(common[left].to_numpy() - common[right].to_numpy()))
                    ),
                }
            )
    return pd.DataFrame(records)


def _metric_record(
    *,
    protocol,
    loss_name,
    subset,
    n_basis,
    regularization_strength,
    penalty_power,
    true,
    predicted,
    line_of_sight,
    weights=None,
):
    metrics = _radial_vector_metrics(
        true,
        predicted,
        line_of_sight,
        weights=weights,
    )
    return {
        "protocol": protocol,
        "loss_name": loss_name,
        "loss_label": LOSS_LABELS[loss_name],
        "subset": subset,
        "n_basis": int(n_basis),
        "n_diffusion_modes": int(n_basis - N_AFFINE_MODES),
        "regularization_strength": float(regularization_strength),
        "penalty_power": float(penalty_power),
        **metrics,
    }





# ==================================================================================================
# 7. Ten-seed high-mode robustness after H2 convergence
# ==================================================================================================

from scipy import stats
import shutil

SEED_ROBUSTNESS_SMOKE_TEST = (
    os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_SMOKE_TEST", "0") == "1"
)
RESUME = os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_RESUME", "1") == "1"
FORCE_RECOMPUTE = (
    os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_FORCE_RECOMPUTE", "0") == "1"
)
SHOW_PLOTS = os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_SHOW_PLOTS", "1") == "1"
SAVE_PDF = os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_SAVE_PDF", "1") == "1"
SAVE_PNG = os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_SAVE_PNG", "1") == "1"
SAVE_PER_SEED_PREDICTIONS = (
    os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_SAVE_PREDICTIONS", "0") == "1"
)
HEARTBEAT_SECONDS = float(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_HEARTBEAT_SECONDS", "45")
)
GEOMETRY_HEARTBEAT_SECONDS = HEARTBEAT_SECONDS

OUTPUT_DIR = Path(
    os.environ.get(
        "RADIAL_MOCK3_HIGHMODE_10SEED_OUTPUT",
        str(
            ROOT_DIR
            / (
                "mock3_radial_potential_highmode_seed_robustness_v1_smoke"
                if SEED_ROBUSTNESS_SMOKE_TEST
                else "mock3_radial_potential_highmode_seed_robustness_v1"
            )
        ),
    )
)
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
for _directory in (OUTPUT_DIR, CHECKPOINT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = "mock3_radial_potential_highmode_seed_robustness"

H2_OUTPUT_DIR = Path(
    os.environ.get(
        "RADIAL_MOCK3_HIGHMODE_10SEED_H2_OUTPUT",
        str(ROOT_DIR / "mock3_radial_potential_highmode_capacity_h2_v1"),
    )
)
H2_PREFIX = "mock3_radial_potential_highmode_capacity_h2"
H2_CACHE_DIR = H2_OUTPUT_DIR / "cache"
H2_GEOMETRY_FILE = H2_CACHE_DIR / "mock3_unweighted_alpha0_geometry_m4352.npz"
H2_BASIS_METADATA_FILE = H2_CACHE_DIR / "highmode_basis_metadata.json"
H2_OBSERVED_GRADIENT_FILE = H2_CACHE_DIR / "observed_nonconstant_gradient.npy"
H2_QUERY_PRIMARY_GRADIENT_FILE = H2_CACHE_DIR / "query_primary_nonconstant_gradient.npy"
H2_RADIAL_DESIGN_FILE = H2_CACHE_DIR / "observed_radial_design.npy"
H2_COLUMN_RMS_FILE = H2_CACHE_DIR / "nonconstant_column_rms.npy"
H2_CANDIDATE_GRID_FILE = H2_OUTPUT_DIR / f"{H2_PREFIX}_candidate_grid.csv"
H2_GLOBAL_SELECTORS_FILE = H2_OUTPUT_DIR / f"{H2_PREFIX}_global_selectors.csv"
H2_CAPACITY_SELECTION_FILE = H2_OUTPUT_DIR / f"{H2_PREFIX}_capacity_selection.csv"
H2_DECISION_FILE = H2_OUTPUT_DIR / f"{H2_PREFIX}_decision.csv"

LEGACY_10SEED_VALIDATION_CANDIDATES = [
    ROOT_DIR
    / "mock3_radial_potential_octant_seed_robustness_v3"
    / "mock3_radial_potential_octant_seed_robustness_validation_grid.csv",
    ROOT_DIR
    / "mock3_radial_potential_octant_seed_robustness_v2"
    / "mock3_radial_potential_octant_seed_robustness_validation_grid.csv",
]
LEGACY_10SEED_VALIDATION_FILE = ROOT_DIR / "__unresolved_mock3_legacy_10seed__.csv"

LABEL_SPLIT_SEEDS = [
    20261203,
    20261213,
    20261223,
    20270103,
    20270113,
    20270123,
    20270203,
    20270213,
    20270223,
    20270303,
]

ANALYSIS_DIFFUSION_MODE_COUNTS = [
    0,
    32,
    64,
    128,
    256,
    384,
    511,
    640,
    768,
    896,
    1024,
    1280,
    1536,
    1792,
    2047,
    2304,
    2559,
    2815,
    3071,
    3327,
    3583,
    3839,
    4095,
    4351,
]

_mode_override = os.environ.get(
    "RADIAL_MOCK3_HIGHMODE_10SEED_MODE_COUNTS", ""
).strip()
if _mode_override:
    ANALYSIS_DIFFUSION_MODE_COUNTS = sorted(
        {int(value) for value in _mode_override.split(",") if value.strip()}
    )

POTENTIAL_REGULARIZATION_STRENGTHS = [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0]
POTENTIAL_REGULARIZATION_POWERS = [1.0, 2.0]

if SEED_ROBUSTNESS_SMOKE_TEST:
    LABEL_SPLIT_SEEDS = LABEL_SPLIT_SEEDS[:2]
    if not _mode_override:
        ANALYSIS_DIFFUSION_MODE_COUNTS = [0, 8, 16, 32]
    POTENTIAL_REGULARIZATION_STRENGTHS = [1.0e-6, 1.0e-4]

if not ANALYSIS_DIFFUSION_MODE_COUNTS:
    raise ValueError("ANALYSIS_DIFFUSION_MODE_COUNTS is empty.")
if ANALYSIS_DIFFUSION_MODE_COUNTS[0] != 0:
    ANALYSIS_DIFFUSION_MODE_COUNTS.insert(0, 0)
if min(ANALYSIS_DIFFUSION_MODE_COUNTS) < 0:
    raise ValueError("Diffusion-mode counts must be nonnegative.")

ANALYSIS_BASIS_SIZES = [
    N_AFFINE_MODES + int(value) for value in ANALYSIS_DIFFUSION_MODE_COUNTS
]
ANALYSIS_MAX_NONCONSTANT = int(max(ANALYSIS_DIFFUSION_MODE_COUNTS))
ANALYSIS_MAX_BASIS = int(max(ANALYSIS_BASIS_SIZES))

PRACTICAL_PLATEAU_RELATIVE_THRESHOLD = float(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_PLATEAU_THRESHOLD", "0.02")
)
PRACTICAL_PLATEAU_CONSECUTIVE_STEPS = int(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_PLATEAU_STEPS", "2")
)
SEED_BOOTSTRAP_REPLICATES = int(
    os.environ.get(
        "RADIAL_MOCK3_HIGHMODE_10SEED_BOOTSTRAP_REPLICATES",
        "2000" if SEED_ROBUSTNESS_SMOKE_TEST else "50000",
    )
)
SEED_BOOTSTRAP_SEED = int(
    os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_BOOTSTRAP_SEED", "20270831")
)


def _json_ready(value: Any):
    if isinstance(value, dict):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, Path):
        return str(value)
    return value


def _stable_int_from_text(text: str) -> int:
    digest = hashlib.sha256(text.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little", signed=False) % (2**32 - 1)


def _atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    temporary.write_text(text, encoding="utf-8")
    os.replace(temporary, path)


def _atomic_to_csv(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}")
    frame.to_csv(temporary, index=False)
    os.replace(temporary, path)


def _atomic_save_npz(path: Path, **payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp-{os.getpid()}-{time.time_ns()}.npz")
    np.savez_compressed(temporary, **payload)
    os.replace(temporary, path)


def _update_progress(stage: str, **details) -> None:
    expected_blocks = (
        len(LABEL_SPLIT_SEEDS)
        * len(LOSS_CONFIGS)
        * len(POTENTIAL_REGULARIZATION_POWERS)
        * len(POTENTIAL_REGULARIZATION_STRENGTHS)
    )
    completed_blocks = len(list(CHECKPOINT_DIR.glob("seed_*/*/candidate_p*.csv")))
    completed_final_seeds = len(
        list(CHECKPOINT_DIR.glob("seed_*/final/evaluation_*.csv"))
    )
    payload = {
        "stage": stage,
        "updated_local_time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "completed_candidate_blocks": completed_blocks,
        "expected_candidate_blocks": expected_blocks,
        "completed_final_seeds": completed_final_seeds,
        "expected_final_seeds": len(LABEL_SPLIT_SEEDS),
        **details,
    }
    _atomic_write_text(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_progress.json",
        json.dumps(_json_ready(payload), ensure_ascii=False, indent=2),
    )


def _candidate_block_tag(power: float, strength: float) -> str:
    strength_text = f"{float(strength):.0e}".replace("+", "").replace("-", "m")
    power_text = f"{float(power):g}".replace(".", "p")
    return f"p{power_text}_lambda{strength_text}"


def _same_configuration(first: dict[str, Any], second: dict[str, Any]) -> bool:
    return bool(
        first["loss_name"] == second["loss_name"]
        and int(first["n_basis"]) == int(second["n_basis"])
        and float(first["regularization_strength"])
        == float(second["regularization_strength"])
        and float(first["penalty_power"]) == float(second["penalty_power"])
    )


def _bootstrap_mean_summary(values, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {
            "mean": math.nan,
            "std": math.nan,
            "median": math.nan,
            "sem": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "probability_mean_below_zero": math.nan,
            "wins": 0,
            "ties": 0,
            "losses": 0,
            "sign_test_pvalue_two_sided": math.nan,
        }
    rng = np.random.default_rng(seed)
    sampled_means = np.empty(SEED_BOOTSTRAP_REPLICATES, dtype=np.float64)
    chunk = 5000
    for start in range(0, SEED_BOOTSTRAP_REPLICATES, chunk):
        stop = min(start + chunk, SEED_BOOTSTRAP_REPLICATES)
        indices = rng.integers(0, values.size, size=(stop - start, values.size))
        sampled_means[start:stop] = np.mean(values[indices], axis=1)
    wins = int(np.sum(values < 0.0))
    ties = int(np.sum(values == 0.0))
    losses = int(np.sum(values > 0.0))
    n_non_tied = wins + losses
    sign_p = (
        float(stats.binomtest(wins, n_non_tied, 0.5, alternative="two-sided").pvalue)
        if n_non_tied > 0
        else math.nan
    )
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    return {
        "mean": float(np.mean(values)),
        "std": std,
        "median": float(np.median(values)),
        "sem": float(std / np.sqrt(values.size)) if values.size > 0 else math.nan,
        "ci_low": float(np.quantile(sampled_means, 0.025)),
        "ci_high": float(np.quantile(sampled_means, 0.975)),
        "probability_mean_below_zero": float(np.mean(sampled_means < 0.0)),
        "wins": wins,
        "ties": ties,
        "losses": losses,
        "sign_test_pvalue_two_sided": sign_p,
    }


def _discover_mock3_source() -> Path:
    global PREDICTIONS_FILE
    explicit = os.environ.get("RADIAL_MOCK3_HIGHMODE_10SEED_PREDICTIONS", "").strip()
    if explicit:
        path = Path(explicit)
        if not path.is_file():
            raise FileNotFoundError(path)
        PREDICTIONS_FILE = path
        return path.parent
    for candidate in SOURCE_CANDIDATE_DIRS:
        path = candidate / "mock3_radial_potential_octant_selection_predictions.npz"
        if path.is_file():
            PREDICTIONS_FILE = path
            return candidate
    raise FileNotFoundError(
        "No Radial Mock-3 predictions file was found. Checked:\n"
        + "\n".join(
            f"  {path / 'mock3_radial_potential_octant_selection_predictions.npz'}"
            for path in SOURCE_CANDIDATE_DIRS
        )
    )


def _load_h2_basis_cache(n_survey: int, n_query_primary: int) -> dict[str, Any]:
    if H2_CAPACITY_SELECTION_FILE.is_file():
        h2_capacity = pd.read_csv(H2_CAPACITY_SELECTION_FILE)
        if "practical_converged" in h2_capacity.columns and not bool(
            h2_capacity["practical_converged"].astype(bool).all()
        ):
            raise RuntimeError(
                "The Mock-3 H2 source does not report practical convergence for all loss families."
            )
    else:
        warnings.warn(
            f"H2 capacity-selection file was not found: {H2_CAPACITY_SELECTION_FILE}. "
            "The basis cache will still be validated directly."
        )
    if H2_DECISION_FILE.is_file():
        h2_decision = pd.read_csv(H2_DECISION_FILE)
        if (
            "recommended_next_step" in h2_decision.columns
            and str(h2_decision.iloc[0]["recommended_next_step"])
            != "ten_seed_highmode_robustness"
        ):
            raise RuntimeError(
                "The Mock-3 H2 decision file does not authorize ten-seed high-mode robustness."
            )

    required = [
        H2_BASIS_METADATA_FILE,
        H2_OBSERVED_GRADIENT_FILE,
        H2_QUERY_PRIMARY_GRADIENT_FILE,
        H2_RADIAL_DESIGN_FILE,
        H2_COLUMN_RMS_FILE,
    ]
    missing = [path for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The completed Mock-3 H2 basis cache is required. Missing:\n"
            + "\n".join(f"  {path}" for path in missing)
            + "\nRun mock3_radial_potential_highmode_capacity_h2_v1.py to completion first."
        )

    metadata = json.loads(H2_BASIS_METADATA_FILE.read_text(encoding="utf-8"))
    observed_gradient = np.load(H2_OBSERVED_GRADIENT_FILE, mmap_mode="r")
    query_gradient = np.load(H2_QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r")
    radial_design = np.load(H2_RADIAL_DESIGN_FILE, mmap_mode="r")
    column_rms = np.load(H2_COLUMN_RMS_FILE, mmap_mode="r")
    if observed_gradient.shape[0] != n_survey:
        raise ValueError(
            f"H2 observed-gradient rows {observed_gradient.shape[0]} != {n_survey}."
        )
    if query_gradient.shape[0] != n_query_primary:
        raise ValueError(
            f"H2 query-gradient rows {query_gradient.shape[0]} != {n_query_primary}."
        )
    if observed_gradient.shape[1] < ANALYSIS_MAX_NONCONSTANT:
        raise ValueError("H2 observed-gradient cache contains too few modes.")
    if query_gradient.shape[1] < ANALYSIS_MAX_NONCONSTANT:
        raise ValueError("H2 query-gradient cache contains too few modes.")
    if radial_design.shape[0] != n_survey or radial_design.shape[1] < ANALYSIS_MAX_BASIS:
        raise ValueError("H2 radial-design cache has an incompatible shape.")
    active_generator = np.asarray(
        metadata.get("active_generator_values", []), dtype=np.float64
    )
    if active_generator.size < ANALYSIS_MAX_NONCONSTANT:
        raise ValueError("H2 metadata contains too few active generator values.")
    return {
        "metadata": metadata,
        "observed_gradient": observed_gradient,
        "query_gradient": query_gradient,
        "radial_design": radial_design,
        "column_rms": column_rms,
        "potential_generator_values": np.concatenate(
            [
                np.zeros(N_AFFINE_MODES, dtype=np.float64),
                active_generator[:ANALYSIS_MAX_NONCONSTANT],
            ]
        ),
        "affine_mode_scale": float(metadata["affine_mode_scale"]),
    }


def _candidate_block_valid(path: Path, expected_basis: list[int]) -> bool:
    if not path.is_file():
        return False
    try:
        frame = pd.read_csv(path)
        return bool(
            frame.shape[0] == len(expected_basis)
            and frame["n_basis"].astype(int).tolist() == list(expected_basis)
        )
    except Exception:
        return False


def _scan_seed_loss(
    *,
    label_seed: int,
    replicate: int,
    config: dict[str, Any],
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    radial_truth_survey: np.ndarray,
    truth_survey: np.ndarray,
    line_of_sight_survey: np.ndarray,
    radial_design_observed,
    observed_nonconstant_gradient,
    potential_generator_values: np.ndarray,
    affine_mode_scale: float,
    loss_weights: np.ndarray,
    primary_metric_weights: np.ndarray,
    global_metric_weights: np.ndarray,
    basis_cache_signature: str,
) -> pd.DataFrame:
    loss_name = config["name"]
    signature = _stable_signature(
        {
            "basis_cache_signature": basis_cache_signature,
            "label_seed": int(label_seed),
            "train_hash": _hash_array(train_indices),
            "validation_hash": _hash_array(validation_indices),
            "basis_sizes": ANALYSIS_BASIS_SIZES,
            "strengths": POTENTIAL_REGULARIZATION_STRENGTHS,
            "powers": POTENTIAL_REGULARIZATION_POWERS,
            "loss": config,
            "primary_metric_gamma": PRIMARY_METRIC_GAMMA,
            "primary_metric_cap": PRIMARY_METRIC_CAP,
            "global_metric_gamma": GLOBAL_METRIC_GAMMA,
            "global_metric_cap": GLOBAL_METRIC_CAP,
            "version": 1,
        }
    )
    seed_loss_dir = (
        CHECKPOINT_DIR
        / f"seed_{int(label_seed)}"
        / f"{loss_name}_{signature[:12]}"
    )
    if FORCE_RECOMPUTE and seed_loss_dir.is_dir():
        shutil.rmtree(seed_loss_dir)
    seed_loss_dir.mkdir(parents=True, exist_ok=True)
    metadata_file = seed_loss_dir / "metadata.json"
    if not metadata_file.is_file():
        _atomic_write_text(
            metadata_file,
            json.dumps(
                {
                    "signature": signature,
                    "label_seed": int(label_seed),
                    "loss_name": loss_name,
                },
                ensure_ascii=False,
                indent=2,
            ),
        )

    block_paths = {
        (float(power), float(strength)): seed_loss_dir
        / f"candidate_{_candidate_block_tag(power, strength)}.csv"
        for power in POTENTIAL_REGULARIZATION_POWERS
        for strength in POTENTIAL_REGULARIZATION_STRENGTHS
    }
    missing_blocks = [
        key
        for key, path in block_paths.items()
        if not (RESUME and _candidate_block_valid(path, ANALYSIS_BASIS_SIZES))
    ]
    if not missing_blocks:
        print(
            f"  [Candidate checkpoint] reuse all blocks: seed={label_seed}, loss={loss_name}",
            flush=True,
        )

    if missing_blocks:
        print(
            f"  [{label_seed} | {loss_name}] computing "
            f"{len(missing_blocks)}/{len(block_paths)} candidate blocks",
            flush=True,
        )
        stop_event, heartbeat, _ = _start_heartbeat(
            f"Crossproducts seed={label_seed}, loss={loss_name}"
        )
        try:
            A_train = np.asarray(
                radial_design_observed[train_indices, :ANALYSIS_MAX_BASIS],
                dtype=np.float64,
            )
            gram, rhs = _weighted_crossproducts(
                A_train,
                radial_truth_survey[train_indices],
                loss_weights[train_indices],
            )
        finally:
            _stop_heartbeat(stop_event, heartbeat)
        del A_train
        gc.collect()

        A_validation = np.asarray(
            radial_design_observed[validation_indices, :ANALYSIS_MAX_BASIS],
            dtype=np.float64,
        )
        gradient_validation = np.asarray(
            observed_nonconstant_gradient[
                validation_indices, :ANALYSIS_MAX_NONCONSTANT, :
            ],
            dtype=np.float64,
        )
        truth_validation = truth_survey[validation_indices]
        los_validation = line_of_sight_survey[validation_indices]
        primary_validation_weights = primary_metric_weights[validation_indices]
        global_validation_weights = global_metric_weights[validation_indices]
        radial_validation_truth = radial_truth_survey[validation_indices]
        regularization_by_power = {
            float(power): _regularization_diagonal(
                potential_generator_values, float(power)
            )
            for power in POTENTIAL_REGULARIZATION_POWERS
        }

        for block_number, (power, strength) in enumerate(missing_blocks, start=1):
            path = block_paths[(power, strength)]
            print(
                f"    block {block_number}/{len(missing_blocks)}: "
                f"p={power:g}, lambda={strength:.0e}",
                flush=True,
            )
            stop_event, heartbeat, _ = _start_heartbeat(
                f"Cholesky seed={label_seed}, loss={loss_name}, "
                f"p={power:g}, lambda={strength:.0e}"
            )
            try:
                system = gram.copy()
                system[np.diag_indices_from(system)] += (
                    float(strength) * regularization_by_power[power]
                    + NUMERICAL_RIDGE
                )
                factor, extra_jitter = _cholesky_with_jitter(system)
                columns = []
                for n_basis in ANALYSIS_BASIS_SIZES:
                    coefficients = _solve_leading_cholesky(factor, rhs, int(n_basis))
                    padded = np.zeros(ANALYSIS_MAX_BASIS, dtype=np.float64)
                    padded[: int(n_basis)] = coefficients
                    columns.append(padded)
                coefficient_matrix = np.column_stack(columns)
            finally:
                _stop_heartbeat(stop_event, heartbeat)
            del system, factor

            radial_prediction_many = A_validation @ coefficient_matrix
            hidden_prediction_many = _many_velocity_predictions(
                gradient_validation,
                coefficient_matrix,
                affine_mode_scale,
            )
            primary_hidden = _many_hidden_validation_metrics(
                truth_validation,
                hidden_prediction_many,
                los_validation,
                primary_validation_weights,
            )
            global_hidden = _many_hidden_validation_metrics(
                truth_validation,
                hidden_prediction_many,
                los_validation,
                global_validation_weights,
            )
            records = []
            loss_ess = _effective_sample_size(loss_weights[train_indices])
            for column, n_basis in enumerate(ANALYSIS_BASIS_SIZES):
                octant_radial = _scalar_nrmse(
                    radial_validation_truth,
                    radial_prediction_many[:, column],
                    weights=primary_validation_weights,
                )
                global_radial = _scalar_nrmse(
                    radial_validation_truth,
                    radial_prediction_many[:, column],
                    weights=global_validation_weights,
                )
                unweighted_radial = _scalar_nrmse(
                    radial_validation_truth,
                    radial_prediction_many[:, column],
                )
                records.append(
                    {
                        "replicate": int(replicate),
                        "label_seed": int(label_seed),
                        "loss_name": loss_name,
                        "loss_label": config["label"],
                        "loss_mode": config["mode"],
                        "loss_gamma": float(config["gamma"]),
                        "loss_cap": float(config["cap"]),
                        "loss_ess_train": float(loss_ess),
                        "n_basis": int(n_basis),
                        "n_diffusion_modes": int(n_basis - N_AFFINE_MODES),
                        "regularization_strength": float(strength),
                        "penalty_power": float(power),
                        "extra_cholesky_jitter": float(extra_jitter),
                        "octant_weighted_validation_nrmse_radial": float(octant_radial),
                        "global_weighted_validation_nrmse_radial": float(global_radial),
                        "unweighted_validation_nrmse_radial": float(unweighted_radial),
                        "octant_weighted_validation_nrmse_tangential": float(
                            primary_hidden["nrmse_tangential"][column]
                        ),
                        "octant_weighted_validation_nrmse_3d": float(
                            primary_hidden["nrmse_3d"][column]
                        ),
                        "global_weighted_validation_nrmse_3d": float(
                            global_hidden["nrmse_3d"][column]
                        ),
                        # Generic aliases used by shared capacity/selector helpers.
                        "weighted_validation_nrmse_radial": float(octant_radial),
                        "weighted_validation_nrmse_tangential": float(
                            primary_hidden["nrmse_tangential"][column]
                        ),
                        "weighted_validation_nrmse_3d": float(
                            primary_hidden["nrmse_3d"][column]
                        ),
                    }
                )
            _atomic_to_csv(pd.DataFrame(records), path)
            _update_progress(
                "candidate_scan",
                label_seed=int(label_seed),
                loss_name=loss_name,
                penalty_power=float(power),
                regularization_strength=float(strength),
                block_status="completed",
            )
            del coefficient_matrix, radial_prediction_many, hidden_prediction_many
            gc.collect()

        del gram, rhs, A_validation, gradient_validation
        gc.collect()

    frames = []
    for power in POTENTIAL_REGULARIZATION_POWERS:
        for strength in POTENTIAL_REGULARIZATION_STRENGTHS:
            path = block_paths[(float(power), float(strength))]
            if not _candidate_block_valid(path, ANALYSIS_BASIS_SIZES):
                raise RuntimeError(f"Incomplete candidate checkpoint: {path}")
            frames.append(pd.read_csv(path))
    frame = pd.concat(frames, ignore_index=True)
    frame["candidate_id"] = np.arange(frame.shape[0], dtype=int)
    return frame


def _aggregate_capacity(within_basis_df: pd.DataFrame):
    metric_columns = [
        "weighted_validation_nrmse_radial",
        "unweighted_validation_nrmse_radial",
        "weighted_validation_nrmse_tangential",
        "weighted_validation_nrmse_3d",
    ]
    aggregate_records = []
    for (loss_name, n_basis), group in within_basis_df.groupby(
        ["loss_name", "n_basis"], sort=False
    ):
        record = {
            "loss_name": loss_name,
            "loss_label": LOSS_LABELS[loss_name],
            "n_basis": int(n_basis),
            "n_diffusion_modes": int(n_basis - N_AFFINE_MODES),
            "n_seeds": int(group["label_seed"].nunique()),
        }
        for metric in metric_columns:
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_mean"] = float(np.mean(values))
            record[f"{metric}_std"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
            record[f"{metric}_sem"] = (
                float(np.std(values, ddof=1) / np.sqrt(values.size))
                if values.size > 1
                else 0.0
            )
        aggregate_records.append(record)
    aggregate_df = pd.DataFrame(aggregate_records).sort_values(
        ["loss_name", "n_basis"]
    )

    raw_per_seed_records = []
    for (label_seed, loss_name), group in within_basis_df.groupby(
        ["label_seed", "loss_name"], sort=False
    ):
        selected = _select_row(
            group,
            [
                "weighted_validation_nrmse_radial",
                "unweighted_validation_nrmse_radial",
            ],
        )
        raw_per_seed_records.append(selected)
    raw_per_seed_df = pd.DataFrame(raw_per_seed_records)

    selection_records = []
    diagnostic_records = []
    practical_basis_by_loss: dict[str, int] = {}
    for loss_name in LOSS_ORDER:
        frame = aggregate_df[aggregate_df["loss_name"] == loss_name].sort_values(
            "n_basis"
        )
        minimum_row = frame.sort_values(
            ["weighted_validation_nrmse_radial_mean", "n_basis"],
            kind="mergesort",
        ).iloc[0]
        threshold = float(
            minimum_row["weighted_validation_nrmse_radial_mean"]
            + minimum_row["weighted_validation_nrmse_radial_sem"]
        )
        one_se_eligible = frame[
            frame["weighted_validation_nrmse_radial_mean"] <= threshold
        ]
        one_se_basis = int(one_se_eligible["n_basis"].min())
        plateau = _practical_plateau(
            frame["n_basis"].to_numpy(dtype=int),
            frame["weighted_validation_nrmse_radial_mean"].to_numpy(dtype=float),
        )
        plateau_found = bool(plateau["plateau_found"])
        plateau_basis = (
            int(plateau["plateau_basis"]) if plateau_found else math.nan
        )
        combined_basis = int(
            max(one_se_basis, int(plateau_basis) if plateau_found else one_se_basis)
        )
        practical_basis_by_loss[loss_name] = combined_basis
        last_improvement = float(plateau["adjacent_improvement"][-1])
        strict_boundary = bool(int(minimum_row["n_basis"]) == ANALYSIS_MAX_BASIS)
        one_se_boundary = bool(one_se_basis == ANALYSIS_MAX_BASIS)
        maximum_fraction = float(
            np.mean(
                raw_per_seed_df[
                    raw_per_seed_df["loss_name"] == loss_name
                ]["n_basis"].to_numpy(dtype=int)
                == ANALYSIS_MAX_BASIS
            )
        )
        final_increment_small = bool(
            np.isfinite(last_improvement)
            and last_improvement < PRACTICAL_PLATEAU_RELATIVE_THRESHOLD
        )
        practical_converged = bool(
            plateau_found and not one_se_boundary and final_increment_small
        )
        needs_larger = bool(not practical_converged)
        selection_records.append(
            {
                "loss_name": loss_name,
                "loss_label": LOSS_LABELS[loss_name],
                "n_seed_splits": len(LABEL_SPLIT_SEEDS),
                "validation_minimum_basis": int(minimum_row["n_basis"]),
                "validation_minimum_mean": float(
                    minimum_row["weighted_validation_nrmse_radial_mean"]
                ),
                "validation_minimum_sem": float(
                    minimum_row["weighted_validation_nrmse_radial_sem"]
                ),
                "one_standard_error_threshold": threshold,
                "one_standard_error_basis": one_se_basis,
                "practical_plateau_basis": plateau_basis,
                "plateau_found": plateau_found,
                "combined_practical_basis": combined_basis,
                "strict_argmin_boundary": strict_boundary,
                "one_se_boundary": one_se_boundary,
                "maximum_basis_selected_fraction": maximum_fraction,
                "last_adjacent_validation_improvement": last_improvement,
                "practical_converged": practical_converged,
                "needs_larger_eigensystem": needs_larger,
                "convergence_status": (
                    "converged" if practical_converged else "requires_review"
                ),
            }
        )
        for basis, value, improvement in zip(
            plateau["basis_values"],
            plateau["validation_values"],
            plateau["adjacent_improvement"],
        ):
            diagnostic_records.append(
                {
                    "loss_name": loss_name,
                    "n_basis": int(basis),
                    "validation_mean": float(value),
                    "adjacent_validation_improvement": float(improvement),
                }
            )
    return (
        aggregate_df,
        pd.DataFrame(selection_records),
        pd.DataFrame(diagnostic_records),
        raw_per_seed_df,
        practical_basis_by_loss,
    )

def _build_selector_tables(
    candidate_df: pd.DataFrame,
    within_basis_df: pd.DataFrame,
    practical_basis_by_loss: dict[str, int],
):
    selector_records = []
    family_practical_records = []
    unweighted_selector_records = []

    for replicate, label_seed in enumerate(LABEL_SPLIT_SEEDS, start=1):
        seed_candidates = candidate_df[candidate_df["label_seed"] == label_seed]
        raw_operational = _select_row(
            seed_candidates,
            [
                "weighted_validation_nrmse_radial",
                "unweighted_validation_nrmse_radial",
            ],
        )
        raw_hidden = _select_row(
            seed_candidates,
            ["weighted_validation_nrmse_3d", "weighted_validation_nrmse_radial"],
        )

        practical_rows = []
        for loss_name in LOSS_ORDER:
            basis = practical_basis_by_loss[loss_name]
            row = within_basis_df[
                (within_basis_df["label_seed"] == label_seed)
                & (within_basis_df["loss_name"] == loss_name)
                & (within_basis_df["n_basis"] == basis)
            ].iloc[0].to_dict()
            practical_rows.append(row)
            family_practical_records.append(
                {"selector": "family_practical_configuration", **row}
            )
        practical_frame = pd.DataFrame(practical_rows)
        practical_operational = _select_row(
            practical_frame,
            [
                "weighted_validation_nrmse_radial",
                "unweighted_validation_nrmse_radial",
            ],
        )
        practical_hidden = _select_row(
            practical_frame,
            ["weighted_validation_nrmse_3d", "weighted_validation_nrmse_radial"],
        )

        for selector_name, row in [
            ("raw_operational_radial_validation", raw_operational),
            ("raw_hidden_3d_validation_diagnostic", raw_hidden),
            ("practical_operational_radial_validation", practical_operational),
            ("practical_hidden_3d_validation_diagnostic", practical_hidden),
        ]:
            selector_records.append(
                {
                    "selector": selector_name,
                    "replicate": int(replicate),
                    "label_seed": int(label_seed),
                    **row,
                }
            )

        unweighted = seed_candidates[seed_candidates["loss_name"] == "unweighted"]
        unweighted_radial = _select_row(
            unweighted,
            [
                "weighted_validation_nrmse_radial",
                "unweighted_validation_nrmse_radial",
            ],
        )
        unweighted_hidden = _select_row(
            unweighted,
            ["weighted_validation_nrmse_3d", "weighted_validation_nrmse_radial"],
        )
        unweighted_selector_records.extend(
            [
                {
                    "selector": "unweighted_raw_radial_validation",
                    "replicate": int(replicate),
                    "label_seed": int(label_seed),
                    **unweighted_radial,
                },
                {
                    "selector": "unweighted_hidden_3d_validation_diagnostic",
                    "replicate": int(replicate),
                    "label_seed": int(label_seed),
                    **unweighted_hidden,
                },
            ]
        )

    selector_df = pd.DataFrame(selector_records)
    family_practical_df = pd.DataFrame(family_practical_records)
    unweighted_selector_df = pd.DataFrame(unweighted_selector_records)

    agreement_records = []
    for policy in ["raw", "practical"]:
        operational = selector_df[
            selector_df["selector"]
            == f"{policy}_operational_radial_validation"
        ].set_index("label_seed")
        hidden = selector_df[
            selector_df["selector"]
            == f"{policy}_hidden_3d_validation_diagnostic"
        ].set_index("label_seed")
        same_candidate = []
        for seed in LABEL_SPLIT_SEEDS:
            same_candidate.append(
                _same_configuration(
                    operational.loc[seed].to_dict(), hidden.loc[seed].to_dict()
                )
            )
        agreement_records.append(
            {
                "policy": policy,
                "n_seeds": len(LABEL_SPLIT_SEEDS),
                "same_candidate_count": int(np.sum(same_candidate)),
                "same_loss_count": int(
                    np.sum(
                        operational["loss_name"].to_numpy()
                        == hidden["loss_name"].to_numpy()
                    )
                ),
                "operational_selects_unweighted_count": int(
                    np.sum(operational["loss_name"].to_numpy() == "unweighted")
                ),
                "hidden_selects_unweighted_count": int(
                    np.sum(hidden["loss_name"].to_numpy() == "unweighted")
                ),
                "mean_operational_minus_hidden_basis": float(
                    np.mean(
                        operational["n_basis"].to_numpy(dtype=float)
                        - hidden["n_basis"].to_numpy(dtype=float)
                    )
                ),
            }
        )

    unweighted_radial = unweighted_selector_df[
        unweighted_selector_df["selector"] == "unweighted_raw_radial_validation"
    ].set_index("label_seed")
    unweighted_hidden = unweighted_selector_df[
        unweighted_selector_df["selector"]
        == "unweighted_hidden_3d_validation_diagnostic"
    ].set_index("label_seed")
    agreement_records.append(
        {
            "policy": "within_unweighted_raw",
            "n_seeds": len(LABEL_SPLIT_SEEDS),
            "same_candidate_count": int(
                np.sum(
                    [
                        _same_configuration(
                            unweighted_radial.loc[seed].to_dict(),
                            unweighted_hidden.loc[seed].to_dict(),
                        )
                        for seed in LABEL_SPLIT_SEEDS
                    ]
                )
            ),
            "same_loss_count": len(LABEL_SPLIT_SEEDS),
            "operational_selects_unweighted_count": len(LABEL_SPLIT_SEEDS),
            "hidden_selects_unweighted_count": len(LABEL_SPLIT_SEEDS),
            "mean_operational_minus_hidden_basis": float(
                np.mean(
                    unweighted_radial["n_basis"].to_numpy(dtype=float)
                    - unweighted_hidden["n_basis"].to_numpy(dtype=float)
                )
            ),
        }
    )
    return (
        selector_df,
        family_practical_df,
        unweighted_selector_df,
        pd.DataFrame(agreement_records),
    )


def _metric_record_extended(
    *,
    request: dict[str, Any],
    loss_name: str,
    subset: str,
    true,
    predicted,
    line_of_sight,
    weights=None,
    region_ids=None,
    summarize_octants: bool = False,
    region_id: int | None = None,
    m_lim: float | None = None,
    extra_cholesky_jitter: float = 0.0,
) -> dict[str, Any]:
    metrics = _radial_vector_metrics(
        true,
        predicted,
        line_of_sight,
        weights=weights,
    )
    record = {
        "model_id": request["model_id"],
        "model_role": request["model_role"],
        "protocol": request.get("protocol", "selector"),
        "selector": request.get("selector", ""),
        "loss_name": loss_name,
        "loss_label": LOSS_LABELS[loss_name],
        "subset": subset,
        "region_id": math.nan if region_id is None else int(region_id),
        "m_lim": math.nan if m_lim is None else float(m_lim),
        "n_basis": int(request["n_basis"]),
        "n_diffusion_modes": int(request["n_basis"] - N_AFFINE_MODES),
        "regularization_strength": float(request["regularization_strength"]),
        "penalty_power": float(request["penalty_power"]),
        "extra_cholesky_jitter": float(extra_cholesky_jitter),
        "n_octants": math.nan,
        "octant_nrmse_3d_mean": math.nan,
        "octant_nrmse_3d_std": math.nan,
        "worst_octant_nrmse_3d": math.nan,
        **metrics,
    }
    if summarize_octants:
        if region_ids is None:
            raise ValueError("region_ids are required for octant summaries.")
        summary = _octant_nrmse_summary(
            true,
            predicted,
            line_of_sight,
            region_ids,
            weights=weights,
        )
        record.update(
            {
                "n_octants": int(summary["n_octants"]),
                "octant_nrmse_3d_mean": float(summary["octant_nrmse_3d_mean"]),
                "octant_nrmse_3d_std": float(summary["octant_nrmse_3d_std"]),
                "worst_octant_nrmse_3d": float(summary["worst_octant_nrmse_3d"]),
            }
        )
    return record


def _solve_model_requests(
    *,
    requests: list[dict[str, Any]],
    loss_name: str,
    fit_indices: np.ndarray,
    test_indices: np.ndarray,
    radial_truth_survey: np.ndarray,
    truth_survey: np.ndarray,
    truth_query_primary: np.ndarray,
    line_of_sight_survey: np.ndarray,
    line_of_sight_query_primary: np.ndarray,
    survey_regions: np.ndarray,
    query_region_primary: np.ndarray,
    mlim_per_region: np.ndarray,
    common_primary: np.ndarray,
    near_primary: np.ndarray,
    far_primary: np.ndarray,
    radial_design_observed,
    observed_nonconstant_gradient,
    query_primary_nonconstant_gradient,
    potential_generator_values: np.ndarray,
    affine_mode_scale: float,
    loss_weights: np.ndarray,
    primary_metric_weights: np.ndarray,
    global_metric_weights: np.ndarray,
):
    if not requests:
        return {}, []
    unique_specs: dict[tuple[int, float, float], dict[str, Any]] = {}
    for request in requests:
        key = (
            int(request["n_basis"]),
            float(request["regularization_strength"]),
            float(request["penalty_power"]),
        )
        unique_specs[key] = request

    A_fit = np.asarray(
        radial_design_observed[fit_indices, :ANALYSIS_MAX_BASIS],
        dtype=np.float64,
    )
    gram, rhs = _weighted_crossproducts(
        A_fit,
        radial_truth_survey[fit_indices],
        loss_weights[fit_indices],
    )
    del A_fit
    A_test = np.asarray(
        radial_design_observed[test_indices, :ANALYSIS_MAX_BASIS],
        dtype=np.float64,
    )
    gradient_test = np.asarray(
        observed_nonconstant_gradient[
            test_indices, :ANALYSIS_MAX_NONCONSTANT, :
        ],
        dtype=np.float64,
    )
    gradient_query = np.asarray(
        query_primary_nonconstant_gradient[:, :ANALYSIS_MAX_NONCONSTANT, :],
        dtype=np.float64,
    )

    grouped = defaultdict(list)
    for key, request in unique_specs.items():
        grouped[(key[2], key[1])].append((key, request))
    fitted: dict[tuple[int, float, float], dict[str, Any]] = {}
    for (power, strength), items in grouped.items():
        regularization = _regularization_diagonal(
            potential_generator_values, power
        )
        system = gram.copy()
        system[np.diag_indices_from(system)] += (
            strength * regularization + NUMERICAL_RIDGE
        )
        factor, extra_jitter = _cholesky_with_jitter(system)
        for key, request in items:
            n_basis = key[0]
            coefficients = _solve_leading_cholesky(factor, rhs, n_basis)
            padded = np.zeros(ANALYSIS_MAX_BASIS, dtype=np.float64)
            padded[:n_basis] = coefficients
            radial_test = A_test[:, :n_basis] @ coefficients
            velocity_test = _single_velocity_prediction(
                gradient_test,
                padded,
                n_basis,
                affine_mode_scale,
            )
            velocity_query = _single_velocity_prediction(
                gradient_query,
                padded,
                n_basis,
                affine_mode_scale,
            )
            fitted[key] = {
                "coefficients": coefficients,
                "radial_test": radial_test,
                "velocity_test": velocity_test,
                "velocity_query": velocity_query,
                "extra_cholesky_jitter": extra_jitter,
            }
        del system, factor
    del gram, rhs, A_test, gradient_test, gradient_query
    gc.collect()

    evaluation_records: list[dict[str, Any]] = []
    test_truth = truth_survey[test_indices]
    test_los = line_of_sight_survey[test_indices]
    test_regions = survey_regions[test_indices]
    for request in requests:
        key = (
            int(request["n_basis"]),
            float(request["regularization_strength"]),
            float(request["penalty_power"]),
        )
        model = fitted[key]
        jitter = float(model["extra_cholesky_jitter"])
        evaluation_records.extend(
            [
                _metric_record_extended(
                    request=request,
                    loss_name=loss_name,
                    subset="observed_test_octant_parent_weighted",
                    true=test_truth,
                    predicted=model["velocity_test"],
                    line_of_sight=test_los,
                    weights=primary_metric_weights[test_indices],
                    region_ids=test_regions,
                    summarize_octants=True,
                    extra_cholesky_jitter=jitter,
                ),
                _metric_record_extended(
                    request=request,
                    loss_name=loss_name,
                    subset="observed_test_global_parent_weighted",
                    true=test_truth,
                    predicted=model["velocity_test"],
                    line_of_sight=test_los,
                    weights=global_metric_weights[test_indices],
                    region_ids=test_regions,
                    summarize_octants=True,
                    extra_cholesky_jitter=jitter,
                ),
                _metric_record_extended(
                    request=request,
                    loss_name=loss_name,
                    subset="observed_test_unweighted",
                    true=test_truth,
                    predicted=model["velocity_test"],
                    line_of_sight=test_los,
                    region_ids=test_regions,
                    summarize_octants=True,
                    extra_cholesky_jitter=jitter,
                ),
                _metric_record_extended(
                    request=request,
                    loss_name=loss_name,
                    subset="query_region_primary_support",
                    true=truth_query_primary,
                    predicted=model["velocity_query"],
                    line_of_sight=line_of_sight_query_primary,
                    region_ids=query_region_primary,
                    summarize_octants=True,
                    extra_cholesky_jitter=jitter,
                ),
            ]
        )
        for subset_name, mask in [
            ("query_common_strict_support", common_primary),
            ("query_region_primary_near_octant_boundary", near_primary),
            ("query_region_primary_far_from_octant_boundary", far_primary),
        ]:
            mask = np.asarray(mask, dtype=bool)
            if np.sum(mask) == 0:
                continue
            evaluation_records.append(
                _metric_record_extended(
                    request=request,
                    loss_name=loss_name,
                    subset=subset_name,
                    true=truth_query_primary[mask],
                    predicted=model["velocity_query"][mask],
                    line_of_sight=line_of_sight_query_primary[mask],
                    region_ids=query_region_primary[mask],
                    summarize_octants=True,
                    extra_cholesky_jitter=jitter,
                )
            )

        for region in range(8):
            observed_mask = test_regions == region
            if np.any(observed_mask):
                evaluation_records.append(
                    _metric_record_extended(
                        request=request,
                        loss_name=loss_name,
                        subset=f"observed_test_octant_{region}",
                        true=test_truth[observed_mask],
                        predicted=model["velocity_test"][observed_mask],
                        line_of_sight=test_los[observed_mask],
                        weights=primary_metric_weights[test_indices][observed_mask],
                        region_id=region,
                        m_lim=float(mlim_per_region[region]),
                        extra_cholesky_jitter=jitter,
                    )
                )
            query_mask = query_region_primary == region
            if np.any(query_mask):
                evaluation_records.append(
                    _metric_record_extended(
                        request=request,
                        loss_name=loss_name,
                        subset=f"query_region_primary_octant_{region}",
                        true=truth_query_primary[query_mask],
                        predicted=model["velocity_query"][query_mask],
                        line_of_sight=line_of_sight_query_primary[query_mask],
                        region_id=region,
                        m_lim=float(mlim_per_region[region]),
                        extra_cholesky_jitter=jitter,
                    )
                )
    return fitted, evaluation_records


def _final_seed_evaluation(
    *,
    replicate: int,
    label_seed: int,
    candidate_seed_df: pd.DataFrame,
    within_basis_df: pd.DataFrame,
    selector_df: pd.DataFrame,
    practical_basis_by_loss: dict[str, int],
    global_practical_loss: str,
    radial_truth_survey: np.ndarray,
    truth_survey: np.ndarray,
    truth_query_primary: np.ndarray,
    line_of_sight_survey: np.ndarray,
    line_of_sight_query_primary: np.ndarray,
    survey_regions: np.ndarray,
    query_region_primary: np.ndarray,
    mlim_per_region: np.ndarray,
    common_primary: np.ndarray,
    near_primary: np.ndarray,
    far_primary: np.ndarray,
    radial_design_observed,
    observed_nonconstant_gradient,
    query_primary_nonconstant_gradient,
    potential_generator_values: np.ndarray,
    affine_mode_scale: float,
    loss_weight_arrays: dict[str, np.ndarray],
    primary_metric_weights: np.ndarray,
    global_metric_weights: np.ndarray,
    practical_signature: str,
):
    split = _make_octant_stratified_split(
        survey_regions,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        int(label_seed),
    )
    fit_indices = np.sort(np.concatenate([split["train"], split["validation"]]))
    test_indices = split["test"]

    final_dir = CHECKPOINT_DIR / f"seed_{int(label_seed)}" / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    signature = _stable_signature(
        {
            "label_seed": int(label_seed),
            "practical_signature": practical_signature,
            "global_practical_loss": global_practical_loss,
            "common_primary_hash": _hash_array(np.asarray(common_primary, dtype=np.uint8)),
            "near_primary_hash": _hash_array(np.asarray(near_primary, dtype=np.uint8)),
            "far_primary_hash": _hash_array(np.asarray(far_primary, dtype=np.uint8)),
            "version": 1,
        }
    )
    evaluation_file = final_dir / f"evaluation_{signature[:12]}.csv"
    selected_file = final_dir / f"selected_{signature[:12]}.csv"
    predictions_file = final_dir / f"predictions_{signature[:12]}.npz"
    if (
        RESUME
        and not FORCE_RECOMPUTE
        and evaluation_file.is_file()
        and selected_file.is_file()
    ):
        print(f"  [Final checkpoint] reuse: seed={label_seed}", flush=True)
        return pd.read_csv(evaluation_file), pd.read_csv(selected_file)

    raw_by_loss: dict[str, dict[str, Any]] = {}
    practical_by_loss: dict[str, dict[str, Any]] = {}
    for loss_name in LOSS_ORDER:
        family_candidates = candidate_seed_df[
            candidate_seed_df["loss_name"] == loss_name
        ]
        raw_by_loss[loss_name] = _select_row(
            family_candidates,
            [
                "weighted_validation_nrmse_radial",
                "unweighted_validation_nrmse_radial",
            ],
        )
        practical_by_loss[loss_name] = within_basis_df[
            (within_basis_df["label_seed"] == label_seed)
            & (within_basis_df["loss_name"] == loss_name)
            & (within_basis_df["n_basis"] == practical_basis_by_loss[loss_name])
        ].iloc[0].to_dict()

    def selector_row(name: str) -> dict[str, Any]:
        return selector_df[
            (selector_df["label_seed"] == label_seed)
            & (selector_df["selector"] == name)
        ].iloc[0].to_dict()

    raw_operational = selector_row("raw_operational_radial_validation")
    raw_hidden = selector_row("raw_hidden_3d_validation_diagnostic")
    practical_operational = selector_row("practical_operational_radial_validation")
    practical_hidden = selector_row("practical_hidden_3d_validation_diagnostic")

    unweighted_practical = practical_by_loss["unweighted"]
    protocol_specs = {
        "retuned_practical": practical_by_loss,
        "retuned_raw": raw_by_loss,
        "matched_unweighted_practical": {
            loss_name: dict(unweighted_practical) for loss_name in LOSS_ORDER
        },
        "matched_raw_operational_structure": {
            loss_name: dict(raw_operational) for loss_name in LOSS_ORDER
        },
    }

    requests_by_loss: dict[str, list[dict[str, Any]]] = {
        loss_name: [] for loss_name in LOSS_ORDER
    }
    selected_records: list[dict[str, Any]] = []
    for protocol, mapping in protocol_specs.items():
        for loss_name in LOSS_ORDER:
            spec = dict(mapping[loss_name])
            request = {
                "model_id": f"protocol::{protocol}::{loss_name}",
                "model_role": "protocol",
                "protocol": protocol,
                "selector": "",
                "loss_name": loss_name,
                "n_basis": int(spec["n_basis"]),
                "regularization_strength": float(spec["regularization_strength"]),
                "penalty_power": float(spec["penalty_power"]),
            }
            requests_by_loss[loss_name].append(request)
            selected_records.append(
                {
                    "replicate": int(replicate),
                    "label_seed": int(label_seed),
                    **request,
                }
            )

    selector_specs = [
        ("raw_operational_radial_validation", raw_operational),
        ("raw_hidden_3d_validation_diagnostic", raw_hidden),
        ("practical_operational_radial_validation", practical_operational),
        ("practical_hidden_3d_validation_diagnostic", practical_hidden),
        ("fixed_global_practical_policy", practical_by_loss[global_practical_loss]),
    ]
    for selector_name, spec in selector_specs:
        loss_name = str(spec["loss_name"])
        request = {
            "model_id": f"selector::{selector_name}",
            "model_role": "selector",
            "protocol": "selector",
            "selector": selector_name,
            "loss_name": loss_name,
            "n_basis": int(spec["n_basis"]),
            "regularization_strength": float(spec["regularization_strength"]),
            "penalty_power": float(spec["penalty_power"]),
        }
        requests_by_loss[loss_name].append(request)
        selected_records.append(
            {
                "replicate": int(replicate),
                "label_seed": int(label_seed),
                **request,
            }
        )

    all_evaluations: list[dict[str, Any]] = []
    prediction_payload = {
        "label_seed": np.array(label_seed, dtype=np.int64),
        "test_indices": test_indices,
        "truth_observed_test": truth_survey[test_indices],
        "truth_query_primary": truth_query_primary,
        "query_region_primary": query_region_primary,
        "common_primary": common_primary,
        "near_primary": near_primary,
        "far_primary": far_primary,
    }
    for loss_name in LOSS_ORDER:
        print(
            f"    final fit seed={label_seed}, loss={loss_name}, "
            f"requests={len(requests_by_loss[loss_name])}",
            flush=True,
        )
        stop_event, heartbeat, _ = _start_heartbeat(
            f"Final fit seed={label_seed}, loss={loss_name}"
        )
        try:
            fitted, records = _solve_model_requests(
                requests=requests_by_loss[loss_name],
                loss_name=loss_name,
                fit_indices=fit_indices,
                test_indices=test_indices,
                radial_truth_survey=radial_truth_survey,
                truth_survey=truth_survey,
                truth_query_primary=truth_query_primary,
                line_of_sight_survey=line_of_sight_survey,
                line_of_sight_query_primary=line_of_sight_query_primary,
                survey_regions=survey_regions,
                query_region_primary=query_region_primary,
                mlim_per_region=mlim_per_region,
                common_primary=common_primary,
                near_primary=near_primary,
                far_primary=far_primary,
                radial_design_observed=radial_design_observed,
                observed_nonconstant_gradient=observed_nonconstant_gradient,
                query_primary_nonconstant_gradient=query_primary_nonconstant_gradient,
                potential_generator_values=potential_generator_values,
                affine_mode_scale=affine_mode_scale,
                loss_weights=loss_weight_arrays[loss_name],
                primary_metric_weights=primary_metric_weights,
                global_metric_weights=global_metric_weights,
            )
        finally:
            _stop_heartbeat(stop_event, heartbeat)
        for record in records:
            record["replicate"] = int(replicate)
            record["label_seed"] = int(label_seed)
        all_evaluations.extend(records)
        if SAVE_PER_SEED_PREDICTIONS:
            for request in requests_by_loss[loss_name]:
                if request["protocol"] != "retuned_practical":
                    continue
                key = (
                    int(request["n_basis"]),
                    float(request["regularization_strength"]),
                    float(request["penalty_power"]),
                )
                model = fitted[key]
                prediction_payload[f"{loss_name}_observed_test"] = model[
                    "velocity_test"
                ]
                prediction_payload[f"{loss_name}_query_primary"] = model[
                    "velocity_query"
                ]
                prediction_payload[f"{loss_name}_coefficients"] = model[
                    "coefficients"
                ]
        del fitted
        gc.collect()

    evaluation_df = pd.DataFrame(all_evaluations)
    selected_df = pd.DataFrame(selected_records)
    _atomic_to_csv(evaluation_df, evaluation_file)
    _atomic_to_csv(selected_df, selected_file)
    if SAVE_PER_SEED_PREDICTIONS:
        _atomic_save_npz(predictions_file, **prediction_payload)
    return evaluation_df, selected_df


def _continuity_audit(
    current_candidate_df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = [
        "octant_weighted_validation_nrmse_radial",
        "global_weighted_validation_nrmse_radial",
        "unweighted_validation_nrmse_radial",
        "octant_weighted_validation_nrmse_tangential",
        "octant_weighted_validation_nrmse_3d",
        "global_weighted_validation_nrmse_3d",
    ]
    keys = [
        "label_seed",
        "loss_name",
        "n_basis",
        "regularization_strength",
        "penalty_power",
    ]
    h2_records: list[dict[str, Any]] = []
    if H2_CANDIDATE_GRID_FILE.is_file():
        previous = pd.read_csv(H2_CANDIDATE_GRID_FILE)
        if "label_seed" not in previous.columns:
            previous["label_seed"] = 20261203
        current = current_candidate_df[
            current_candidate_df["label_seed"] == 20261203
        ]
        merged = current.merge(previous, on=keys, suffixes=("_current", "_h2"))
        record: dict[str, Any] = {
            "audit_available": True,
            "n_common_rows": int(merged.shape[0]),
            "expected_rows": int(previous.shape[0]),
        }
        for metric in metrics:
            left = f"{metric}_current"
            right = f"{metric}_h2"
            if left in merged.columns and right in merged.columns:
                record[f"{metric}_max_abs_difference"] = float(
                    np.max(
                        np.abs(
                            merged[left].to_numpy(dtype=float)
                            - merged[right].to_numpy(dtype=float)
                        )
                    )
                )
        h2_records.append(record)
    else:
        h2_records.append({"audit_available": False, "n_common_rows": 0})

    legacy_records: list[dict[str, Any]] = []
    if LEGACY_10SEED_VALIDATION_FILE.is_file():
        previous = pd.read_csv(LEGACY_10SEED_VALIDATION_FILE)
        current = current_candidate_df[current_candidate_df["n_basis"] <= 514]
        merged = current.merge(previous, on=keys, suffixes=("_current", "_legacy"))
        record = {
            "audit_available": True,
            "n_common_rows": int(merged.shape[0]),
            "expected_current_rows": int(current.shape[0]),
        }
        for metric in metrics:
            left = f"{metric}_current"
            right = f"{metric}_legacy"
            if left in merged.columns and right in merged.columns:
                record[f"{metric}_max_abs_difference"] = float(
                    np.max(
                        np.abs(
                            merged[left].to_numpy(dtype=float)
                            - merged[right].to_numpy(dtype=float)
                        )
                    )
                )
        legacy_records.append(record)
    else:
        legacy_records.append({"audit_available": False, "n_common_rows": 0})
    return pd.DataFrame(h2_records), pd.DataFrame(legacy_records)


def _paired_tables(protocol_evaluation_df: pd.DataFrame):
    comparison_definitions = [
        ("global_minus_unweighted", "global_g0p75_cap20", "unweighted"),
        ("octant_minus_unweighted", "octant_g1p00_cap20", "unweighted"),
        ("octant_minus_global", "octant_g1p00_cap20", "global_g0p75_cap20"),
    ]
    metric_columns = [
        "nrmse_radial",
        "nrmse_tangential",
        "nrmse_3d",
        "direction_error_median_deg",
        "octant_nrmse_3d_mean",
        "octant_nrmse_3d_std",
        "worst_octant_nrmse_3d",
    ]
    primary_subsets = [
        "observed_test_octant_parent_weighted",
        "observed_test_global_parent_weighted",
        "observed_test_unweighted",
        "query_region_primary_support",
        "query_common_strict_support",
        "query_region_primary_near_octant_boundary",
        "query_region_primary_far_from_octant_boundary",
    ]
    paired_records: list[dict[str, Any]] = []
    for protocol in sorted(protocol_evaluation_df["protocol"].unique()):
        for subset in primary_subsets:
            frame = protocol_evaluation_df[
                (protocol_evaluation_df["protocol"] == protocol)
                & (protocol_evaluation_df["subset"] == subset)
            ]
            if frame.empty:
                continue
            for comparison, model_a, model_b in comparison_definitions:
                first = frame[frame["loss_name"] == model_a].set_index("label_seed")
                second = frame[frame["loss_name"] == model_b].set_index("label_seed")
                for seed in first.index.intersection(second.index):
                    record = {
                        "protocol": protocol,
                        "subset": subset,
                        "comparison": comparison,
                        "model_a": model_a,
                        "model_b": model_b,
                        "label_seed": int(seed),
                    }
                    for metric in metric_columns:
                        a = float(first.loc[seed, metric])
                        b = float(second.loc[seed, metric])
                        record[f"delta_{metric}"] = a - b
                    paired_records.append(record)
    paired_df = pd.DataFrame(paired_records)

    summary_records: list[dict[str, Any]] = []
    for (protocol, subset, comparison), group in paired_df.groupby(
        ["protocol", "subset", "comparison"], sort=False
    ):
        record: dict[str, Any] = {
            "protocol": protocol,
            "subset": subset,
            "comparison": comparison,
            "n_seeds": int(group.shape[0]),
        }
        for metric in metric_columns:
            summary = _bootstrap_mean_summary(
                group[f"delta_{metric}"].to_numpy(dtype=float),
                seed=_stable_int_from_text(
                    f"{SEED_BOOTSTRAP_SEED}|{protocol}|{subset}|{comparison}|{metric}"
                ),
            )
            for key, value in summary.items():
                record[f"delta_{metric}_{key}"] = value
        summary_records.append(record)
    return paired_df, pd.DataFrame(summary_records)


def _model_summary(protocol_evaluation_df: pd.DataFrame) -> pd.DataFrame:
    summary_metrics = [
        "nrmse_radial",
        "nrmse_tangential",
        "nrmse_3d",
        "direction_error_median_deg",
        "direction_error_p90_deg",
        "vector_gain_through_origin",
        "octant_nrmse_3d_mean",
        "octant_nrmse_3d_std",
        "worst_octant_nrmse_3d",
    ]
    records: list[dict[str, Any]] = []
    overall = protocol_evaluation_df[
        ~protocol_evaluation_df["subset"].str.match(
            r"^(observed_test_octant_\d+|query_region_primary_octant_\d+)$"
        )
    ]
    for (protocol, subset, loss_name), group in overall.groupby(
        ["protocol", "subset", "loss_name"], sort=False
    ):
        record: dict[str, Any] = {
            "protocol": protocol,
            "subset": subset,
            "loss_name": loss_name,
            "loss_label": LOSS_LABELS[loss_name],
            "n_seeds": int(group["label_seed"].nunique()),
        }
        for metric in summary_metrics:
            values = group[metric].to_numpy(dtype=float)
            finite = values[np.isfinite(values)]
            record[f"{metric}_mean"] = (
                float(np.mean(finite)) if finite.size else math.nan
            )
            record[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
            record[f"{metric}_sem"] = (
                float(np.std(finite, ddof=1) / np.sqrt(finite.size))
                if finite.size > 1
                else 0.0
            )
        records.append(record)
    return pd.DataFrame(records)


def _octant_summary_tables(protocol_evaluation_df: pd.DataFrame):
    octant_rows = protocol_evaluation_df[
        np.isfinite(protocol_evaluation_df["region_id"].to_numpy(dtype=float))
    ].copy()
    if octant_rows.empty:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    octant_rows["region_id"] = octant_rows["region_id"].astype(int)
    octant_rows["octant_subset"] = np.select(
        [
            octant_rows["subset"].str.startswith("observed_test_octant_"),
            octant_rows["subset"].str.startswith("query_region_primary_octant_"),
        ],
        ["observed_test_octant", "query_region_primary_octant"],
        default="other",
    )
    octant_rows = octant_rows[octant_rows["octant_subset"] != "other"]

    seed_records: list[dict[str, Any]] = []
    for (protocol, label_seed, loss_name, octant_subset), group in octant_rows.groupby(
        ["protocol", "label_seed", "loss_name", "octant_subset"], sort=False
    ):
        group = group.drop_duplicates(subset=["region_id"])
        nrmse = group["nrmse_3d"].to_numpy(dtype=float)
        ids = sorted(group["region_id"].astype(int).tolist())
        seed_records.append(
            {
                "protocol": protocol,
                "label_seed": int(label_seed),
                "loss_name": loss_name,
                "loss_label": LOSS_LABELS[loss_name],
                "octant_subset": octant_subset,
                "n_octants": int(group.shape[0]),
                "octant_ids": ",".join(map(str, ids)),
                "missing_octant_ids": ",".join(
                    map(str, sorted(set(range(8)).difference(ids)))
                ),
                "nrmse_3d_mean": float(np.mean(nrmse)),
                "nrmse_3d_std": float(np.std(nrmse, ddof=0)),
                "nrmse_3d_worst": float(np.max(nrmse)),
                "nrmse_3d_best": float(np.min(nrmse)),
                "nrmse_3d_range": float(np.max(nrmse) - np.min(nrmse)),
            }
        )
    seed_df = pd.DataFrame(seed_records)

    robust_records: list[dict[str, Any]] = []
    for (protocol, loss_name, octant_subset), group in seed_df.groupby(
        ["protocol", "loss_name", "octant_subset"], sort=False
    ):
        record: dict[str, Any] = {
            "protocol": protocol,
            "loss_name": loss_name,
            "loss_label": LOSS_LABELS[loss_name],
            "octant_subset": octant_subset,
            "n_seeds": int(group.shape[0]),
            "n_octants_min": int(group["n_octants"].min()),
            "n_octants_max": int(group["n_octants"].max()),
        }
        for metric in [
            "nrmse_3d_mean",
            "nrmse_3d_std",
            "nrmse_3d_worst",
            "nrmse_3d_best",
            "nrmse_3d_range",
        ]:
            values = group[metric].to_numpy(dtype=float)
            record[f"{metric}_mean_over_seeds"] = float(np.mean(values))
            record[f"{metric}_std_over_seeds"] = (
                float(np.std(values, ddof=1)) if values.size > 1 else 0.0
            )
        robust_records.append(record)
    robust_df = pd.DataFrame(robust_records)

    per_region_records: list[dict[str, Any]] = []
    for (protocol, loss_name, octant_subset, region_id), group in octant_rows.groupby(
        ["protocol", "loss_name", "octant_subset", "region_id"], sort=False
    ):
        per_region_records.append(
            {
                "protocol": protocol,
                "loss_name": loss_name,
                "loss_label": LOSS_LABELS[loss_name],
                "octant_subset": octant_subset,
                "region_id": int(region_id),
                "m_lim": float(group["m_lim"].iloc[0]),
                "n_seeds": int(group["label_seed"].nunique()),
                "nrmse_3d_mean": float(group["nrmse_3d"].mean()),
                "nrmse_3d_std": (
                    float(group["nrmse_3d"].std(ddof=1))
                    if group.shape[0] > 1
                    else 0.0
                ),
            }
        )
    return seed_df, robust_df, pd.DataFrame(per_region_records)


# ==================================================================================================
# 8. Main analysis
# ==================================================================================================


def main():
    global PREDICTIONS_FILE, LEGACY_10SEED_VALIDATION_FILE

    start_time = time.perf_counter()
    print("=" * 132)
    print("Radial Mock-3 high-mode 10-label-seed robustness")
    print("=" * 132)
    print(
        f"Python / NumPy / SciPy : {platform.python_version()} / "
        f"{np.__version__} / {scipy.__version__}"
    )
    source_dir = _discover_mock3_source()
    legacy = next(
        (path for path in LEGACY_10SEED_VALIDATION_CANDIDATES if path.is_file()),
        None,
    )
    if legacy is not None:
        LEGACY_10SEED_VALIDATION_FILE = legacy

    print(f"Predictions             : {PREDICTIONS_FILE}")
    print(f"H2 source               : {H2_OUTPUT_DIR}")
    print(f"Output                  : {OUTPUT_DIR}")
    print(f"Label seeds             : {LABEL_SPLIT_SEEDS}")
    print(f"Potential parameters    : {ANALYSIS_BASIS_SIZES}")
    print(f"Lambda grid             : {POTENTIAL_REGULARIZATION_STRENGTHS}")
    print(f"Penalty powers          : {POTENTIAL_REGULARIZATION_POWERS}")
    print(f"Resume                  : {RESUME}")
    _update_progress("initializing")

    for path in (MOCK1_FILE, MOCK3_FILE, PREDICTIONS_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)

    survey_meta = _load_mock3_metadata(MOCK3_FILE)
    survey_regions = survey_meta["region_id"]
    mlim_per_region = survey_meta["mlim_per_region"]
    survey_positions = survey_meta["pos"]

    with np.load(PREDICTIONS_FILE, allow_pickle=False) as data:
        required = {
            "truth_survey",
            "truth_query",
            "line_of_sight_survey",
            "line_of_sight_query",
            "query_region",
            "probability_region_survey",
            "probability_global_survey",
            "probability_region_query",
            "probability_global_query",
            "region_primary_support",
            "common_strict_support",
            "octant_boundary_distance_query",
            "query_global_indices",
        }
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"Predictions file is missing keys: {sorted(missing)}")
        truth_survey = np.asarray(data["truth_survey"], dtype=np.float64)
        truth_query = np.asarray(data["truth_query"], dtype=np.float64)
        line_of_sight_survey = np.asarray(
            data["line_of_sight_survey"], dtype=np.float64
        )
        line_of_sight_query = np.asarray(
            data["line_of_sight_query"], dtype=np.float64
        )
        query_region = np.asarray(data["query_region"], dtype=np.int64)
        probability_region_survey = np.asarray(
            data["probability_region_survey"], dtype=np.float64
        )
        probability_global_survey = np.asarray(
            data["probability_global_survey"], dtype=np.float64
        )
        probability_region_query = np.asarray(
            data["probability_region_query"], dtype=np.float64
        )
        probability_global_query = np.asarray(
            data["probability_global_query"], dtype=np.float64
        )
        region_primary_support = np.asarray(
            data["region_primary_support"], dtype=bool
        )
        common_strict_support = np.asarray(
            data["common_strict_support"], dtype=bool
        )
        octant_boundary_distance_query = np.asarray(
            data["octant_boundary_distance_query"], dtype=np.float64
        )
        query_global_indices = np.asarray(
            data["query_global_indices"], dtype=np.int64
        )

    n_survey = truth_survey.shape[0]
    n_query = truth_query.shape[0]
    if survey_positions.shape[0] != n_survey:
        raise ValueError("Mock-3 catalog size does not match truth_survey.")
    if truth_survey.shape != line_of_sight_survey.shape:
        raise ValueError("Survey truth and line-of-sight arrays are inconsistent.")
    if truth_query.shape != line_of_sight_query.shape:
        raise ValueError("Query truth and line-of-sight arrays are inconsistent.")
    if query_region.shape != (n_query,):
        raise ValueError("query_region has an inconsistent shape.")
    if region_primary_support.shape != (n_query,):
        raise ValueError("region_primary_support has an inconsistent shape.")

    primary_query_indices = np.flatnonzero(region_primary_support)
    n_query_primary = int(primary_query_indices.size)
    if n_query_primary < 5:
        raise ValueError("Region-primary query subset contains too few objects.")
    truth_query_primary = truth_query[primary_query_indices]
    line_of_sight_query_primary = line_of_sight_query[primary_query_indices]
    query_region_primary = query_region[primary_query_indices]
    probability_region_query_primary = probability_region_query[
        primary_query_indices
    ]
    probability_global_query_primary = probability_global_query[
        primary_query_indices
    ]
    common_primary = common_strict_support[primary_query_indices]

    tree = cKDTree(survey_positions)
    try:
        neighbor_distances, _ = tree.query(
            survey_positions,
            k=GRAPH_BANDWIDTH_NEIGHBOR + 1,
            workers=-1,
        )
    except TypeError:
        neighbor_distances, _ = tree.query(
            survey_positions,
            k=GRAPH_BANDWIDTH_NEIGHBOR + 1,
        )
    boundary_scale = float(
        np.median(
            GRAPH_BANDWIDTH_MULTIPLIER
            * np.asarray(
                neighbor_distances[:, GRAPH_BANDWIDTH_NEIGHBOR],
                dtype=np.float64,
            )
        )
    )
    primary_boundary_distance = octant_boundary_distance_query[
        primary_query_indices
    ]
    near_primary = primary_boundary_distance <= boundary_scale
    far_primary = primary_boundary_distance >= 2.0 * boundary_scale

    basis = _load_h2_basis_cache(n_survey, n_query_primary)
    radial_design_observed = basis["radial_design"]
    observed_nonconstant_gradient = basis["observed_gradient"]
    query_primary_nonconstant_gradient = basis["query_gradient"]
    potential_generator_values = basis["potential_generator_values"]
    affine_mode_scale = basis["affine_mode_scale"]
    basis_signature = str(basis["metadata"].get("signature", ""))

    radial_truth_survey = np.einsum(
        "ij,ij->i", line_of_sight_survey, truth_survey
    )

    loss_weight_arrays: dict[str, np.ndarray] = {}
    weight_records: list[dict[str, Any]] = []
    for config in LOSS_CONFIGS:
        probability = _loss_probability(
            config,
            probability_global_survey,
            probability_region_survey,
        )
        weights, info = _tempered_inverse_selection_weights(
            probability,
            config["gamma"],
            config["cap"],
        )
        loss_weight_arrays[config["name"]] = weights
        weight_records.append(
            {
                "loss_name": config["name"],
                "loss_label": config["label"],
                "loss_mode": config["mode"],
                **info,
            }
        )
    primary_metric_weights, primary_metric_info = _tempered_inverse_selection_weights(
        probability_region_survey,
        PRIMARY_METRIC_GAMMA,
        PRIMARY_METRIC_CAP,
    )
    global_metric_weights, global_metric_info = _tempered_inverse_selection_weights(
        probability_global_survey,
        GLOBAL_METRIC_GAMMA,
        GLOBAL_METRIC_CAP,
    )
    weight_diagnostics_df = pd.DataFrame(weight_records)
    _atomic_to_csv(
        weight_diagnostics_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_weight_diagnostics.csv",
    )

    print(f"Source directory        : {source_dir}")
    print(f"Survey/query/primary    : {n_survey:,}/{n_query:,}/{n_query_primary:,}")
    print(f"Common-strict primary   : {int(np.sum(common_primary)):,}")
    print(f"Near/far boundary       : {int(np.sum(near_primary)):,}/{int(np.sum(far_primary)):,}")
    print(f"Boundary scale          : {boundary_scale:.8g}")
    print(f"Maximum potential basis : {ANALYSIS_MAX_BASIS}")
    print(f"Affine mode scale       : {affine_mode_scale:.8g}")
    print(
        "Octant-parent metric    : "
        f"gamma={PRIMARY_METRIC_GAMMA:g}, cap={PRIMARY_METRIC_CAP:g}, "
        f"ESS={primary_metric_info['effective_sample_size']:.1f}"
    )
    print(
        "Global-radial metric   : "
        f"gamma={GLOBAL_METRIC_GAMMA:g}, cap={GLOBAL_METRIC_CAP:g}, "
        f"ESS={global_metric_info['effective_sample_size']:.1f}"
    )

    split_records: list[dict[str, Any]] = []
    candidate_frames: list[pd.DataFrame] = []
    for replicate, label_seed in enumerate(LABEL_SPLIT_SEEDS, start=1):
        print("-" * 132)
        print(
            f"Candidate scan seed {replicate}/{len(LABEL_SPLIT_SEEDS)}: "
            f"octant-stratified label_seed={label_seed}",
            flush=True,
        )
        _update_progress(
            "candidate_seed",
            replicate=int(replicate),
            label_seed=int(label_seed),
        )
        split = _make_octant_stratified_split(
            survey_regions,
            TRAIN_FRACTION,
            VALIDATION_FRACTION,
            int(label_seed),
        )
        for region in range(8):
            for subset_name, indices in split.items():
                split_records.append(
                    {
                        "replicate": int(replicate),
                        "label_seed": int(label_seed),
                        "region_id": int(region),
                        "m_lim": float(mlim_per_region[region]),
                        "subset": subset_name,
                        "n": int(np.sum(survey_regions[indices] == region)),
                    }
                )
        seed_frames: list[pd.DataFrame] = []
        for config in LOSS_CONFIGS:
            frame = _scan_seed_loss(
                label_seed=int(label_seed),
                replicate=int(replicate),
                config=config,
                train_indices=split["train"],
                validation_indices=split["validation"],
                radial_truth_survey=radial_truth_survey,
                truth_survey=truth_survey,
                line_of_sight_survey=line_of_sight_survey,
                radial_design_observed=radial_design_observed,
                observed_nonconstant_gradient=observed_nonconstant_gradient,
                potential_generator_values=potential_generator_values,
                affine_mode_scale=affine_mode_scale,
                loss_weights=loss_weight_arrays[config["name"]],
                primary_metric_weights=primary_metric_weights,
                global_metric_weights=global_metric_weights,
                basis_cache_signature=basis_signature,
            )
            seed_frames.append(frame)
            candidate_frames.append(frame)
            best = _select_row(
                frame,
                [
                    "weighted_validation_nrmse_radial",
                    "unweighted_validation_nrmse_radial",
                ],
            )
            print(
                f"  {config['name']}: raw radial minimum "
                f"n={int(best['n_basis'])}, "
                f"lambda={float(best['regularization_strength']):.0e}, "
                f"p={float(best['penalty_power']):g}, "
                f"NRMSE={float(best['weighted_validation_nrmse_radial']):.6f}",
                flush=True,
            )
        _atomic_to_csv(
            pd.concat(seed_frames, ignore_index=True),
            CHECKPOINT_DIR / f"seed_{int(label_seed)}" / "candidate_complete.csv",
        )
        _update_progress(
            "candidate_seed_completed",
            replicate=int(replicate),
            label_seed=int(label_seed),
        )

    candidate_df = pd.concat(candidate_frames, ignore_index=True)
    split_df = pd.DataFrame(split_records)
    _atomic_to_csv(candidate_df, OUTPUT_DIR / f"{OUTPUT_PREFIX}_candidate_grid.csv")
    _atomic_to_csv(
        split_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_octant_stratified_split.csv",
    )

    h2_continuity_df, legacy_continuity_df = _continuity_audit(candidate_df)
    _atomic_to_csv(
        h2_continuity_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_h2_representative_continuity.csv",
    )
    _atomic_to_csv(
        legacy_continuity_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_legacy_514_continuity.csv",
    )
    print("-" * 132)
    print("H2 representative-seed continuity audit")
    print(h2_continuity_df.to_string(index=False))
    print("Legacy 514-parameter ten-seed continuity audit")
    print(legacy_continuity_df.to_string(index=False))

    within_basis_records: list[dict[str, Any]] = []
    for (label_seed, loss_name, n_basis), group in candidate_df.groupby(
        ["label_seed", "loss_name", "n_basis"], sort=False
    ):
        within_basis_records.append(
            _select_row(
                group,
                [
                    "weighted_validation_nrmse_radial",
                    "unweighted_validation_nrmse_radial",
                ],
            )
        )
    within_basis_df = pd.DataFrame(within_basis_records).sort_values(
        ["label_seed", "loss_name", "n_basis"]
    )
    _atomic_to_csv(
        within_basis_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_within_basis_selected.csv",
    )

    (
        capacity_aggregate_df,
        capacity_selection_df,
        plateau_diagnostic_df,
        raw_per_seed_loss_df,
        practical_basis_by_loss,
    ) = _aggregate_capacity(within_basis_df)
    _atomic_to_csv(
        capacity_aggregate_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_capacity_aggregate.csv",
    )
    _atomic_to_csv(
        capacity_selection_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_capacity_selection.csv",
    )
    _atomic_to_csv(
        plateau_diagnostic_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_plateau_diagnostic.csv",
    )
    _atomic_to_csv(
        raw_per_seed_loss_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_raw_per_seed_loss_selectors.csv",
    )

    practical_policy_rows: list[dict[str, Any]] = []
    for loss_name in LOSS_ORDER:
        basis_value = practical_basis_by_loss[loss_name]
        row = capacity_aggregate_df[
            (capacity_aggregate_df["loss_name"] == loss_name)
            & (capacity_aggregate_df["n_basis"] == basis_value)
        ].iloc[0]
        practical_policy_rows.append(
            {
                "loss_name": loss_name,
                "loss_label": LOSS_LABELS[loss_name],
                "combined_practical_basis": int(basis_value),
                "validation_mean": float(
                    row["weighted_validation_nrmse_radial_mean"]
                ),
                "validation_sem": float(
                    row["weighted_validation_nrmse_radial_sem"]
                ),
            }
        )
    practical_policy_df = pd.DataFrame(practical_policy_rows).sort_values(
        ["validation_mean", "combined_practical_basis", "loss_name"],
        kind="mergesort",
    )
    global_practical_loss = str(practical_policy_df.iloc[0]["loss_name"])
    practical_policy_df["selected_global_practical_loss"] = (
        practical_policy_df["loss_name"] == global_practical_loss
    )
    _atomic_to_csv(
        practical_policy_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_global_practical_policy.csv",
    )

    (
        selector_df,
        family_practical_df,
        unweighted_selector_df,
        selector_agreement_df,
    ) = _build_selector_tables(
        candidate_df,
        within_basis_df,
        practical_basis_by_loss,
    )
    _atomic_to_csv(selector_df, OUTPUT_DIR / f"{OUTPUT_PREFIX}_selectors.csv")
    _atomic_to_csv(
        family_practical_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_family_practical_configurations.csv",
    )
    _atomic_to_csv(
        unweighted_selector_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_unweighted_selector_diagnostic.csv",
    )
    _atomic_to_csv(
        selector_agreement_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_agreement.csv",
    )

    practical_signature = _stable_signature(
        {
            "practical_basis_by_loss": practical_basis_by_loss,
            "global_practical_loss": global_practical_loss,
            "basis_signature": basis_signature,
            "version": 1,
        }
    )

    evaluation_frames: list[pd.DataFrame] = []
    selected_frames: list[pd.DataFrame] = []
    for replicate, label_seed in enumerate(LABEL_SPLIT_SEEDS, start=1):
        print("-" * 132)
        print(
            f"Final evaluation seed {replicate}/{len(LABEL_SPLIT_SEEDS)}: "
            f"label_seed={label_seed}",
            flush=True,
        )
        _update_progress(
            "final_evaluation_seed",
            replicate=int(replicate),
            label_seed=int(label_seed),
        )
        evaluation_frame, selected_frame = _final_seed_evaluation(
            replicate=int(replicate),
            label_seed=int(label_seed),
            candidate_seed_df=candidate_df[candidate_df["label_seed"] == label_seed],
            within_basis_df=within_basis_df,
            selector_df=selector_df,
            practical_basis_by_loss=practical_basis_by_loss,
            global_practical_loss=global_practical_loss,
            radial_truth_survey=radial_truth_survey,
            truth_survey=truth_survey,
            truth_query_primary=truth_query_primary,
            line_of_sight_survey=line_of_sight_survey,
            line_of_sight_query_primary=line_of_sight_query_primary,
            survey_regions=survey_regions,
            query_region_primary=query_region_primary,
            mlim_per_region=mlim_per_region,
            common_primary=common_primary,
            near_primary=near_primary,
            far_primary=far_primary,
            radial_design_observed=radial_design_observed,
            observed_nonconstant_gradient=observed_nonconstant_gradient,
            query_primary_nonconstant_gradient=query_primary_nonconstant_gradient,
            potential_generator_values=potential_generator_values,
            affine_mode_scale=affine_mode_scale,
            loss_weight_arrays=loss_weight_arrays,
            primary_metric_weights=primary_metric_weights,
            global_metric_weights=global_metric_weights,
            practical_signature=practical_signature,
        )
        evaluation_frames.append(evaluation_frame)
        selected_frames.append(selected_frame)
        _update_progress(
            "final_evaluation_seed_completed",
            replicate=int(replicate),
            label_seed=int(label_seed),
        )

    evaluation_df = pd.concat(evaluation_frames, ignore_index=True)
    final_selected_df = pd.concat(selected_frames, ignore_index=True)
    _atomic_to_csv(evaluation_df, OUTPUT_DIR / f"{OUTPUT_PREFIX}_evaluation.csv")
    _atomic_to_csv(
        final_selected_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_final_selected_configurations.csv",
    )

    protocol_evaluation_df = evaluation_df[
        evaluation_df["model_role"] == "protocol"
    ].copy()
    selector_evaluation_df = evaluation_df[
        evaluation_df["model_role"] == "selector"
    ].copy()
    _atomic_to_csv(
        selector_evaluation_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_evaluation.csv",
    )

    paired_df, paired_summary_df = _paired_tables(protocol_evaluation_df)
    model_summary_df = _model_summary(protocol_evaluation_df)
    (
        octant_seed_summary_df,
        octant_robustness_df,
        per_octant_summary_df,
    ) = _octant_summary_tables(protocol_evaluation_df)
    _atomic_to_csv(
        paired_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_paired_differences.csv",
    )
    _atomic_to_csv(
        paired_summary_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_paired_summary.csv",
    )
    _atomic_to_csv(
        model_summary_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_model_summary.csv",
    )
    _atomic_to_csv(
        octant_seed_summary_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_octant_seed_summary.csv",
    )
    _atomic_to_csv(
        octant_robustness_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_octant_robustness.csv",
    )
    _atomic_to_csv(
        per_octant_summary_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_per_octant_summary.csv",
    )

    practical_hyperparameter_frequency_df = (
        final_selected_df[
            (final_selected_df["model_role"] == "protocol")
            & (final_selected_df["protocol"] == "retuned_practical")
        ]
        .groupby(
            ["loss_name", "n_basis", "regularization_strength", "penalty_power"],
            dropna=False,
        )
        .size()
        .reset_index(name="count")
        .sort_values(["loss_name", "count"], ascending=[True, False])
    )
    _atomic_to_csv(
        practical_hyperparameter_frequency_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_practical_hyperparameter_frequency.csv",
    )

    # ------------------------------------------------------------------------------------------------
    # Figures
    # ------------------------------------------------------------------------------------------------
    legacy_ceiling = 514
    h2_ceiling = ANALYSIS_MAX_BASIS

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    for loss_name in LOSS_ORDER:
        frame = capacity_aggregate_df[
            capacity_aggregate_df["loss_name"] == loss_name
        ].sort_values("n_basis")
        axes[0, 0].errorbar(
            frame["n_basis"],
            frame["weighted_validation_nrmse_radial_mean"],
            yerr=frame["weighted_validation_nrmse_radial_sem"],
            marker="o",
            capsize=3,
            label=LOSS_LABELS[loss_name],
        )
        axes[0, 1].errorbar(
            frame["n_basis"],
            frame["weighted_validation_nrmse_3d_mean"],
            yerr=frame["weighted_validation_nrmse_3d_sem"],
            marker="o",
            capsize=3,
            label=LOSS_LABELS[loss_name],
        )
    axes[0, 0].set_title("(a) Seed-aggregated observable octant-parent radial risk")
    axes[0, 1].set_title("(b) Seed-aggregated hidden validation 3D risk")
    axes[0, 0].set_ylabel("Octant-parent radial NRMSE")
    axes[0, 1].set_ylabel("Hidden validation 3D NRMSE")

    baseline = within_basis_df[
        within_basis_df["loss_name"] == "unweighted"
    ][["label_seed", "n_basis", "weighted_validation_nrmse_radial"]].rename(
        columns={"weighted_validation_nrmse_radial": "baseline"}
    )
    for loss_name in LOSS_ORDER[1:]:
        weighted = within_basis_df[
            within_basis_df["loss_name"] == loss_name
        ][["label_seed", "n_basis", "weighted_validation_nrmse_radial"]]
        merged = weighted.merge(baseline, on=["label_seed", "n_basis"])
        merged["delta"] = merged["weighted_validation_nrmse_radial"] - merged["baseline"]
        summary = (
            merged.groupby("n_basis")["delta"]
            .agg(["mean", "std", "count"])
            .reset_index()
        )
        summary["sem"] = summary["std"] / np.sqrt(summary["count"])
        axes[1, 0].errorbar(
            summary["n_basis"],
            summary["mean"],
            yerr=summary["sem"],
            marker="o",
            capsize=3,
            label=f"{LOSS_LABELS[loss_name]} minus unweighted",
        )
    axes[1, 0].axhline(0.0, linestyle="--", linewidth=1.0)
    axes[1, 0].set_title("(c) Observable weighting contrast across seeds")
    axes[1, 0].set_ylabel("Weighted minus unweighted radial NRMSE")

    rule_labels = ["Raw minimum", "One-SE", "Plateau", "Combined practical"]
    x = np.arange(len(LOSS_ORDER))
    offsets = [-0.24, -0.08, 0.08, 0.24]
    columns = [
        "validation_minimum_basis",
        "one_standard_error_basis",
        "practical_plateau_basis",
        "combined_practical_basis",
    ]
    for offset, column, label in zip(offsets, columns, rule_labels):
        values = [
            float(
                capacity_selection_df[
                    capacity_selection_df["loss_name"] == loss_name
                ].iloc[0][column]
            )
            for loss_name in LOSS_ORDER
        ]
        axes[1, 1].scatter(x + offset, values, label=label)

    # Panel (d) has a categorical horizontal axis.  Keep it independent of
    # the numerical potential-parameter axes used in panels (a)--(c).
    capacity_tick_labels = [
        LOSS_LABELS[name].replace(" (", "\n(")
        for name in LOSS_ORDER
    ]
    axes[1, 1].set_xticks(x)
    axes[1, 1].set_xticklabels(capacity_tick_labels)
    axes[1, 1].set_xlim(-0.55, len(LOSS_ORDER) - 0.45)
    axes[1, 1].set_ylabel("Selected potential parameters")
    axes[1, 1].set_title("(d) Capacity-selection rules")

    # The legacy-capacity reference line belongs only to the three numerical
    # capacity panels.  Applying x=legacy_ceiling to the categorical panel
    # compresses its x axis and collapses its tick labels.
    for axis in (axes[0, 0], axes[0, 1], axes[1, 0]):
        axis.axvline(legacy_ceiling, linestyle=":", linewidth=1.2)
        axis.grid(alpha=0.25)
    axes[1, 1].grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    axes[0, 1].legend(loc="best")
    axes[1, 0].legend(loc="best")
    axes[1, 1].legend(loc="best")
    for axis in axes[0, :]:
        axis.set_xlabel("Potential parameter count")
    axes[1, 0].set_xlabel("Potential parameter count")
    fig.tight_layout()
    _save_figure(fig, "capacity_seed_robustness_composite")

    practical_eval = protocol_evaluation_df[
        protocol_evaluation_df["protocol"] == "retuned_practical"
    ]
    seed_x = np.arange(len(LABEL_SPLIT_SEEDS))
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    plot_specs = [
        (
            "observed_test_octant_parent_weighted",
            "nrmse_radial",
            "(a) Observed octant-parent radial risk",
            "Radial NRMSE",
        ),
        (
            "query_region_primary_support",
            "nrmse_3d",
            "(b) Region-primary query recovery",
            "3D NRMSE",
        ),
        (
            "query_region_primary_near_octant_boundary",
            "nrmse_3d",
            "(c) Near-boundary query recovery",
            "3D NRMSE",
        ),
        (
            "query_region_primary_support",
            "octant_nrmse_3d_std",
            "(d) Query octant dispersion",
            "Octant NRMSE dispersion",
        ),
    ]
    for axis, (subset, metric, title, ylabel) in zip(axes.ravel(), plot_specs):
        frame = practical_eval[practical_eval["subset"] == subset]
        for loss_name in LOSS_ORDER:
            local = frame[frame["loss_name"] == loss_name].set_index("label_seed")
            values = [
                float(local.loc[seed, metric]) if seed in local.index else math.nan
                for seed in LABEL_SPLIT_SEEDS
            ]
            axis.plot(seed_x, values, marker="o", label=LOSS_LABELS[loss_name])
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(seed_x, [str(seed) for seed in LABEL_SPLIT_SEEDS], rotation=45)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, "practical_policy_risk_by_seed")

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.0))
    contrast_specs = [
        (
            "retuned_practical",
            "observed_test_octant_parent_weighted",
            "delta_nrmse_radial",
            "(a) Retuned parent-radial contrast",
            "Weighted minus unweighted radial NRMSE",
        ),
        (
            "retuned_practical",
            "query_region_primary_support",
            "delta_nrmse_3d",
            "(b) Retuned region-primary query contrast",
            "Weighted minus unweighted 3D NRMSE",
        ),
        (
            "matched_unweighted_practical",
            "query_region_primary_support",
            "delta_octant_nrmse_3d_std",
            "(c) Matched query octant-dispersion contrast",
            "Weighted minus unweighted octant dispersion",
        ),
        (
            "matched_unweighted_practical",
            "query_region_primary_near_octant_boundary",
            "delta_nrmse_3d",
            "(d) Matched near-boundary query contrast",
            "Weighted minus unweighted 3D NRMSE",
        ),
    ]
    comparison_labels = {
        "global_minus_unweighted": "Global radial minus unweighted",
        "octant_minus_unweighted": "Octant-aware minus unweighted",
    }
    for axis, (protocol, subset, metric, title, ylabel) in zip(
        axes.ravel(), contrast_specs
    ):
        frame = paired_df[
            (paired_df["protocol"] == protocol)
            & (paired_df["subset"] == subset)
            & (paired_df["comparison"].isin(comparison_labels))
        ]
        for comparison, label in comparison_labels.items():
            local = frame[frame["comparison"] == comparison].set_index("label_seed")
            values = [
                float(local.loc[seed, metric]) if seed in local.index else math.nan
                for seed in LABEL_SPLIT_SEEDS
            ]
            axis.plot(seed_x, values, marker="o", label=label)
        axis.axhline(0.0, linestyle="--", linewidth=1.0)
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xticks(seed_x, [str(seed) for seed in LABEL_SPLIT_SEEDS], rotation=45)
        axis.grid(alpha=0.25)
    axes[0, 0].legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, "paired_weighting_contrasts_by_seed")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.4))
    for axis, policy, title in [
        (axes[0], "raw", "(a) Raw selector capacity"),
        (axes[1], "practical", "(b) Practical selector capacity"),
    ]:
        operational = selector_df[
            selector_df["selector"] == f"{policy}_operational_radial_validation"
        ].set_index("label_seed")
        hidden = selector_df[
            selector_df["selector"] == f"{policy}_hidden_3d_validation_diagnostic"
        ].set_index("label_seed")
        axis.plot(
            seed_x,
            [float(operational.loc[seed, "n_basis"]) for seed in LABEL_SPLIT_SEEDS],
            marker="o",
            label="Observable radial selector",
        )
        axis.plot(
            seed_x,
            [float(hidden.loc[seed, "n_basis"]) for seed in LABEL_SPLIT_SEEDS],
            marker="x",
            label="Hidden 3D diagnostic selector",
        )
        axis.axhline(legacy_ceiling, linestyle=":", linewidth=1.2)
        axis.axhline(h2_ceiling, linestyle="-.", linewidth=1.2)
        axis.set_title(title)
        axis.set_ylabel("Selected potential parameters")
        axis.set_xticks(seed_x, [str(seed) for seed in LABEL_SPLIT_SEEDS], rotation=45)
        axis.grid(alpha=0.25)
    axes[0].legend(loc="best")
    fig.tight_layout()
    _save_figure(fig, "selector_capacity_mismatch")

    # ------------------------------------------------------------------------------------------------
    # Decision and compact summary
    # ------------------------------------------------------------------------------------------------
    all_practical_converged = bool(
        capacity_selection_df["practical_converged"].astype(bool).all()
    )
    needs_larger_eigensystem = bool(
        capacity_selection_df["needs_larger_eigensystem"].astype(bool).any()
    )
    global_practical_is_unweighted = global_practical_loss == "unweighted"
    policy_lookup = practical_policy_df.set_index("loss_name")
    octant_beats_global = bool(
        float(policy_lookup.loc["octant_g1p00_cap20", "validation_mean"])
        < float(policy_lookup.loc["global_g0p75_cap20", "validation_mean"])
    )

    dispersion_row = paired_summary_df[
        (paired_summary_df["protocol"] == "matched_unweighted_practical")
        & (paired_summary_df["subset"] == "query_region_primary_support")
        & (paired_summary_df["comparison"] == "octant_minus_unweighted")
    ]
    if dispersion_row.empty:
        octant_dispersion_mean = math.nan
        octant_dispersion_ci_high = math.nan
        octant_dispersion_improves = False
    else:
        octant_dispersion_mean = float(
            dispersion_row.iloc[0]["delta_octant_nrmse_3d_std_mean"]
        )
        octant_dispersion_ci_high = float(
            dispersion_row.iloc[0]["delta_octant_nrmse_3d_std_ci_high"]
        )
        octant_dispersion_improves = bool(
            octant_dispersion_mean < 0.0 and octant_dispersion_ci_high < 0.0
        )

    if not all_practical_converged or needs_larger_eigensystem:
        recommended_next_step = "larger_eigensystem_before_text_update"
    elif global_practical_is_unweighted:
        recommended_next_step = "finalize_mock3_highmode_text_and_update_paper"
    else:
        recommended_next_step = "review_mock3_loss_policy_before_text_update"

    decision_df = pd.DataFrame(
        [
            {
                "all_loss_families_practical_converged": all_practical_converged,
                "global_practical_loss": global_practical_loss,
                "global_practical_loss_is_unweighted": global_practical_is_unweighted,
                "octant_aware_practical_validation_better_than_global": octant_beats_global,
                "octant_aware_matched_query_dispersion_delta_mean": octant_dispersion_mean,
                "octant_aware_matched_query_dispersion_delta_ci_high": octant_dispersion_ci_high,
                "octant_aware_query_dispersion_improves_significantly": octant_dispersion_improves,
                "needs_larger_eigensystem": needs_larger_eigensystem,
                "recommended_next_step": recommended_next_step,
            }
        ]
    )
    _atomic_to_csv(
        decision_df,
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_decision.csv",
    )

    summary_payload = {
        "analysis": "Radial Mock-3 high-mode 10-label-seed robustness",
        "software": {
            "python": platform.python_version(),
            "numpy": np.__version__,
            "scipy": scipy.__version__,
        },
        "inputs": {
            "predictions": str(PREDICTIONS_FILE),
            "h2_output": str(H2_OUTPUT_DIR),
            "h2_geometry": str(H2_GEOMETRY_FILE),
        },
        "label_seeds": LABEL_SPLIT_SEEDS,
        "potential_parameters": ANALYSIS_BASIS_SIZES,
        "loss_families": LOSS_CONFIGS,
        "support": {
            "survey": int(n_survey),
            "query": int(n_query),
            "region_primary": int(n_query_primary),
            "common_strict_primary": int(np.sum(common_primary)),
            "near_boundary": int(np.sum(near_primary)),
            "far_boundary": int(np.sum(far_primary)),
            "boundary_scale": boundary_scale,
        },
        "capacity_selection": capacity_selection_df.to_dict(orient="records"),
        "global_practical_policy": practical_policy_df.to_dict(orient="records"),
        "selector_agreement": selector_agreement_df.to_dict(orient="records"),
        "decision": decision_df.to_dict(orient="records")[0],
    }
    _atomic_write_text(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json",
        json.dumps(_json_ready(summary_payload), ensure_ascii=False, indent=2),
    )

    elapsed_minutes = (time.perf_counter() - start_time) / 60.0
    _update_progress(
        "completed",
        elapsed_minutes=float(elapsed_minutes),
        recommended_next_step=recommended_next_step,
    )

    print("\n")
    print("=" * 132)
    print("Radial Mock-3 high-mode 10-label-seed robustness completed")
    print("=" * 132)
    print(f"Elapsed time: {elapsed_minutes:.2f} min")
    print("Capacity selection across ten seeds")
    display(capacity_selection_df)
    print("Global practical loss policy")
    display(practical_policy_df)
    print("Selector agreement")
    display(selector_agreement_df)
    print("Primary paired summary")
    primary_rows = paired_summary_df[
        (paired_summary_df["protocol"].isin(
            ["retuned_practical", "matched_unweighted_practical"]
        ))
        & (paired_summary_df["subset"].isin(
            [
                "observed_test_octant_parent_weighted",
                "query_region_primary_support",
                "query_region_primary_near_octant_boundary",
            ]
        ))
    ]
    display(primary_rows)
    print("Octant robustness")
    display(octant_robustness_df)
    print("Decision")
    display(decision_df)
    print("Saved files")
    for path in sorted(OUTPUT_DIR.glob(f"{OUTPUT_PREFIX}_*")):
        if path.is_file():
            print(f"  {path.name}")
    print("=" * 132)


if __name__ == "__main__":
    main()
