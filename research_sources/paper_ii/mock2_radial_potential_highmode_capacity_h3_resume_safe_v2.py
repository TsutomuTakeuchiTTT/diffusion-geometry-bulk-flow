# ==================================================================================================
# Radial Mock-2 high-mode capacity audit H3, resume-safe v2
# ==================================================================================================
#
# 目的
# ----
# Radial Mock-2 の selection-loss comparison は、これまで最大 514 potential parameters
# （3 affine modes + 511 nonconstant diffusion-gradient modes）で検証されていた。
# Mock-1 measurement-noise analysis では 514 parameters が under-capacity となり得ることが
# 判明したため、本scriptでは代表label splitを保持したまま、Mock-2 potential-flow familyを
# high-mode側へ拡張し、loss-family rankingとobservable/hidden selector mismatchのcapacity
# dependenceを監査する。
#
# Primary scientific protocol
# ---------------------------
#   * operator-side unweighted alpha_DM=0 geometryを固定する。
#   * loss familiesは事前登録する。
#       1. unweighted
#       2. gamma=0.75, cap=20
#       3. gamma=1.00, cap=50
#   * 各loss family・各basis内でlambdaとpenalty powerをparent-weighted radial validationから
#     選択する。
#   * hidden tangential/3D truthとcomplete-query truthは診断にのみ用いる。
#   * practical plateauはvalidation curveのみから判定する。
#
# H3 design
# ---------
#   * representative label seed: 20261202
#   * 4096-mode H2 geometryを保持し、最小限の256-mode continuationを追加する。
#   * default maximum: 4351 nonconstant modes + 3 affine modes = 4354 parameters
#   * H2の全candidate gridを保持し、4098から4354 parametersへの一段階を追加する。
#   * H3でpractical convergenceが得られれば、次に10 label seedsへ拡張する。
#   * 4354 parametersでも二段階連続のplateau規準を満たさない場合は、さらに大きな
#     eigensystemが必要であることを明示する。
#
# Jupyter execution
# -----------------
#   %run mock2_radial_potential_highmode_capacity_h3_v1.py
#
# Smoke test
# ----------
#   import os
#   os.environ["RADIAL_MOCK2_HIGHMODE_H3_SMOKE_TEST"] = "1"
#   %run mock2_radial_potential_highmode_capacity_h3_v1.py
#   os.environ.pop("RADIAL_MOCK2_HIGHMODE_H3_SMOKE_TEST", None)
#
# The script is standalone. No common notebook cell is required.

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
MOCK2_FILE = ROOT_DIR / "mock2_schechter_selection" / "mock2.npz"

SOURCE_DIR = ROOT_DIR / "mock2_radial_potential_selection_v1"
PREDICTIONS_FILE = SOURCE_DIR / "mock2_radial_potential_selection_predictions.npz"
LEGACY_GRADIENT_CACHE_FILE = (
    SOURCE_DIR / "cache" / "potential_gradient_basis_observed_query_m512.npz"
)

LEGACY_GEOMETRY_CANDIDATES = [
    ROOT_DIR
    / "mock2_selection_correction_optimization_v4"
    / "final_geometry_cache"
    / "geometry_a0p00_g0p00_c1p0_m512.npz",
    SOURCE_DIR / "cache" / "unweighted_alpha0_geometry_m512.npz",
]

_extended_geometry_override = os.environ.get(
    "RADIAL_MOCK2_HIGHMODE_H3_GEOMETRY",
    "",
).strip()
REFERENCE_GEOMETRY_CANDIDATES = [
    ROOT_DIR
    / "mock2_loss_weight_mode_convergence_to_4096_v2"
    / "cache"
    / "mock2_unweighted_alpha0_geometry_m4096.npz",
    ROOT_DIR
    / "mock2_loss_weight_mode_convergence_to_4096_v1"
    / "cache"
    / "mock2_unweighted_alpha0_geometry_m4096.npz",
    ROOT_DIR
    / "mock2_loss_weight_mode_convergence_to_4096"
    / "cache"
    / "mock2_unweighted_alpha0_geometry_m4096.npz",
]

LEGACY_V2_DIR = ROOT_DIR / "mock2_radial_potential_loss_seed_robustness_v2"
LEGACY_V2_VALIDATION_FILE = (
    LEGACY_V2_DIR / "mock2_radial_potential_loss_seed_robustness_validation_grid.csv"
)

# H2 output is optional and is used only for explicit continuation audits.
_previous_h2_output_override = os.environ.get(
    "RADIAL_MOCK2_HIGHMODE_H3_PREVIOUS_H2_OUTPUT",
    "",
).strip()
PREVIOUS_H2_OUTPUT_DIR = Path(
    _previous_h2_output_override
    if _previous_h2_output_override
    else ROOT_DIR / "mock2_radial_potential_highmode_capacity_h2_v1"
)
PREVIOUS_H2_PREFIX = "mock2_radial_potential_highmode_capacity_h2"
PREVIOUS_H2_CACHE_DIR = PREVIOUS_H2_OUTPUT_DIR / "cache"
PREVIOUS_H2_BASIS_METADATA_FILE = PREVIOUS_H2_CACHE_DIR / "highmode_basis_metadata.json"
PREVIOUS_H2_OBSERVED_GRADIENT_FILE = PREVIOUS_H2_CACHE_DIR / "observed_nonconstant_gradient.npy"
PREVIOUS_H2_QUERY_PRIMARY_GRADIENT_FILE = PREVIOUS_H2_CACHE_DIR / "query_primary_nonconstant_gradient.npy"
PREVIOUS_H2_RADIAL_DESIGN_FILE = PREVIOUS_H2_CACHE_DIR / "observed_radial_design.npy"
PREVIOUS_H2_COLUMN_RMS_FILE = PREVIOUS_H2_CACHE_DIR / "nonconstant_column_rms.npy"
PREVIOUS_H2_CANDIDATE_GRID_FILE = PREVIOUS_H2_OUTPUT_DIR / f"{PREVIOUS_H2_PREFIX}_candidate_grid.csv"
PREVIOUS_H2_GLOBAL_SELECTORS_FILE = PREVIOUS_H2_OUTPUT_DIR / f"{PREVIOUS_H2_PREFIX}_global_selectors.csv"
PREVIOUS_H2_MAX_NONCONSTANT_MODES = 4095
PREVIOUS_H2_MAX_BASIS = 4098

SMOKE_TEST = os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_SMOKE_TEST", "0") == "1"
FORCE_REBUILD_CACHE = (
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_FORCE_REBUILD_CACHE", "0") == "1"
)
FORCE_REBUILD_GEOMETRY = (
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_FORCE_REBUILD_GEOMETRY", "0") == "1"
)
RESUME = os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_RESUME", "1") == "1"

_output_default = "mock2_radial_potential_highmode_capacity_h3_resume_safe_v2"
if SMOKE_TEST:
    _output_default += "_smoke"
OUTPUT_DIR = Path(
    os.environ.get(
        "RADIAL_MOCK2_HIGHMODE_H3_OUTPUT",
        str(ROOT_DIR / _output_default),
    )
)
CACHE_DIR = OUTPUT_DIR / "cache"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
for _directory in (OUTPUT_DIR, CACHE_DIR, CHECKPOINT_DIR):
    _directory.mkdir(parents=True, exist_ok=True)

OUTPUT_PREFIX = "mock2_radial_potential_highmode_capacity_h3_resume_safe"

EIGEN_SOLVER_SEED = int(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_EIGEN_SEED", "20260820")
)
GEOMETRY_SOLVER = os.environ.get(
    "RADIAL_MOCK2_HIGHMODE_H3_GEOMETRY_SOLVER",
    "eigsh",
).strip().lower()
if GEOMETRY_SOLVER not in {"eigsh", "dense_eigh"}:
    raise ValueError(
        "RADIAL_MOCK2_HIGHMODE_H3_GEOMETRY_SOLVER must be 'eigsh' or 'dense_eigh'."
    )
GEOMETRY_HEARTBEAT_SECONDS = float(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_HEARTBEAT_SECONDS", "60")
)
CONTINUATION_OVERLAP_MODES = int(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_CONTINUATION_OVERLAP", "64")
)
_EIGSH_NCV_TEXT = os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_EIGSH_NCV", "").strip()
EIGSH_NCV = int(_EIGSH_NCV_TEXT) if _EIGSH_NCV_TEXT else None


# Resume-safe continuation configuration.  The old H3 implementation requested all
# 4352 eigenpairs in one ARPACK call.  SciPy then selected a very large default ncv,
# which could exhaust the Jupyter kernel memory, and no checkpoint existed inside
# that monolithic call.  Version 2 computes only the 256 modes beyond H2, in small
# deflated blocks, and writes an atomic checkpoint after every completed block.
CONTINUATION_BLOCK_SIZE = int(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_BLOCK_SIZE", "64")
)
CONTINUATION_EIGSH_TOL = float(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_BLOCK_TOL", "1e-8")
)
CONTINUATION_EIGSH_MAXITER = int(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_BLOCK_MAXITER", "120000")
)
CONTINUATION_DEFLATION_SHIFT = float(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_DEFLATION_SHIFT", "2.5")
)
_CONTINUATION_NCV_TEXT = os.environ.get(
    "RADIAL_MOCK2_HIGHMODE_H3_BLOCK_NCV", ""
).strip()
CONTINUATION_EIGSH_NCV = (
    int(_CONTINUATION_NCV_TEXT) if _CONTINUATION_NCV_TEXT else None
)
CONTINUATION_RESIDUAL_WARN = float(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_RESIDUAL_WARN", "1e-6")
)
CONTINUATION_RESIDUAL_FAIL = float(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_RESIDUAL_FAIL", "5e-4")
)
CONTINUATION_CHECKPOINT_DIR = CACHE_DIR / "continuation_blocks"
CONTINUATION_CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
BASIS_PROGRESS_FILE = CACHE_DIR / "highmode_basis_progress.json"

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)
LABEL_SPLIT_SEED = int(
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_LABEL_SEED", "20261202")
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
    os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_GRADIENT_CHUNK", "32")
)

MIN_SELECTION_PROBABILITY = 1.0e-4
WEIGHT_CLIP_QUANTILE = 0.995
EVALUATION_WEIGHT_GAMMA = 1.0
EVALUATION_WEIGHT_CAP = 50.0

LOSS_CONFIGS = [
    {
        "name": "unweighted",
        "label": "Unweighted",
        "gamma": 0.0,
        "cap": 1.0,
    },
    {
        "name": "optimized_gamma0p75_cap20",
        "label": r"Radial-CV weighted ($\gamma=0.75$, cap 20)",
        "gamma": 0.75,
        "cap": 20.0,
    },
    {
        "name": "previous_mock2_gamma1p00_cap50",
        "label": r"Previous Mock-2 ($\gamma=1$, cap 50)",
        "gamma": 1.0,
        "cap": 50.0,
    },
]
LOSS_ORDER = [item["name"] for item in LOSS_CONFIGS]
LOSS_LABELS = {item["name"]: item["label"] for item in LOSS_CONFIGS}

N_AFFINE_MODES = 3
GENERATOR_REFERENCE_NONCONSTANT_MODES = 511

_DEFAULT_MAX_NONCONSTANT = 4351
MAX_NONCONSTANT_MODES = int(
    os.environ.get(
        "RADIAL_MOCK2_HIGHMODE_H3_MAX_NONCONSTANT",
        str(96 if SMOKE_TEST else _DEFAULT_MAX_NONCONSTANT),
    )
)

_DEFAULT_DIFFUSION_MODE_COUNTS = [
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
if SMOKE_TEST:
    _DEFAULT_DIFFUSION_MODE_COUNTS = [0, 8, 32, 64, 96]

_basis_override = os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_MODE_COUNTS", "").strip()
if _basis_override:
    POTENTIAL_DIFFUSION_MODE_COUNTS = [
        int(value)
        for value in _basis_override.split(",")
        if value.strip()
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
    N_AFFINE_MODES + count
    for count in POTENTIAL_DIFFUSION_MODE_COUNTS
]
MAX_BASIS = int(max(POTENTIAL_BASIS_SIZES))
TARGET_TOTAL_GEOMETRY_MODES = MAX_NONCONSTANT_MODES + 1
REFERENCE_TOTAL_GEOMETRY_MODES = PREVIOUS_H2_MAX_NONCONSTANT_MODES + 1
EXTENDED_GEOMETRY_FILE = (
    Path(_extended_geometry_override)
    if _extended_geometry_override
    else CACHE_DIR
    / f"mock2_unweighted_alpha0_geometry_m{TARGET_TOTAL_GEOMETRY_MODES}.npz"
)
EXTENDED_GEOMETRY_METADATA_FILE = CACHE_DIR / (
    f"mock2_unweighted_alpha0_geometry_m{TARGET_TOTAL_GEOMETRY_MODES}_metadata.json"
)

if SMOKE_TEST:
    POTENTIAL_REGULARIZATION_STRENGTHS = [1.0e-6, 1.0e-4]
else:
    POTENTIAL_REGULARIZATION_STRENGTHS = [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0]
POTENTIAL_REGULARIZATION_POWERS = [1.0, 2.0]
NUMERICAL_RIDGE = 1.0e-12

PRACTICAL_PLATEAU_RELATIVE_THRESHOLD = 0.02
PRACTICAL_PLATEAU_CONSECUTIVE_STEPS = 2

SHOW_PLOTS = os.environ.get("RADIAL_MOCK2_HIGHMODE_H3_SHOW_PLOTS", "1") == "1"
SAVE_PDF = True
SAVE_PNG = True

BASIS_CACHE_METADATA_FILE = CACHE_DIR / "highmode_basis_metadata.json"
OBSERVED_GRADIENT_FILE = CACHE_DIR / "observed_nonconstant_gradient.npy"
QUERY_PRIMARY_GRADIENT_FILE = CACHE_DIR / "query_primary_nonconstant_gradient.npy"
RADIAL_DESIGN_FILE = CACHE_DIR / "observed_radial_design.npy"
COLUMN_RMS_FILE = CACHE_DIR / "nonconstant_column_rms.npy"


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
            raise KeyError(f"Mock-2 is missing keys: {sorted(missing)}")
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
            "mock2": _file_signature(MOCK2_FILE),
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
        raise ValueError("Non-positive kernel density in H3 geometry construction.")

    q_factor = np.power(q, -float(ALPHA_DM))
    kernel_alpha = (
        sparse.diags(q_factor) @ kernel @ sparse.diags(q_factor)
    ).tocsr()
    d = np.asarray(kernel_alpha @ quadrature_weights, dtype=np.float64).ravel()
    if np.any(d <= 0.0):
        raise ValueError("Non-positive row degree in H3 geometry construction.")

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
            "H3 and H2 stationary measures are inconsistent: "
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
            "Largest continuation eigenvalue exceeds the retained H2 boundary "
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
        warnings.warn(f"H3 extended geometry cache could not be reused: {exc}")
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
            raise ValueError("External H3 geometry has the wrong survey size.")
        if geometry["eigenfunctions"].shape[1] < TARGET_TOTAL_GEOMETRY_MODES:
            raise ValueError("External H3 geometry contains too few eigenfunctions.")
        if abs(float(geometry["alpha"]) - ALPHA_DM) > 1.0e-12:
            raise ValueError("External H3 geometry is not the required alpha=0 geometry.")
        return EXTENDED_GEOMETRY_FILE, "external override"

    if _extended_geometry_cache_valid(EXTENDED_GEOMETRY_FILE, signature):
        print(f"[Geometry cache] reuse: {EXTENDED_GEOMETRY_FILE}", flush=True)
        return EXTENDED_GEOMETRY_FILE, "cache"

    reference_geometry = _load_geometry(reference_geometry_file)
    if reference_geometry["eigenfunctions"].shape[0] != centered_survey.shape[0]:
        raise ValueError("H2 reference geometry has the wrong survey size.")
    if abs(float(reference_geometry["alpha"]) - ALPHA_DM) > 1.0e-12:
        raise ValueError("H2 reference geometry is not the required alpha=0 geometry.")

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
            "mock2": _file_signature(MOCK2_FILE),
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
        raise ValueError("High-mode geometry does not match the Mock-2 survey size.")
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

    # Preserve exact nesting by copying the complete H2 basis whenever its cache is available.
    if not progress_valid:
        previous_required = [
            PREVIOUS_H2_BASIS_METADATA_FILE,
            PREVIOUS_H2_OBSERVED_GRADIENT_FILE,
            PREVIOUS_H2_QUERY_PRIMARY_GRADIENT_FILE,
            PREVIOUS_H2_RADIAL_DESIGN_FILE,
            PREVIOUS_H2_COLUMN_RMS_FILE,
        ]
        if all(path.is_file() for path in previous_required):
            try:
                previous_metadata = json.loads(
                    PREVIOUS_H2_BASIS_METADATA_FILE.read_text(encoding="utf-8")
                )
                previous_observed = np.load(
                    PREVIOUS_H2_OBSERVED_GRADIENT_FILE, mmap_mode="r"
                )
                previous_query = np.load(
                    PREVIOUS_H2_QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r"
                )
                previous_radial = np.load(
                    PREVIOUS_H2_RADIAL_DESIGN_FILE, mmap_mode="r"
                )
                previous_column_rms = np.load(
                    PREVIOUS_H2_COLUMN_RMS_FILE, mmap_mode="r"
                )
                requested_reuse = min(
                    PREVIOUS_H2_MAX_NONCONSTANT_MODES,
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
                        f"[High-mode basis] copied {reuse_modes} H2 nonconstant modes; "
                        f"computing only {MAX_NONCONSTANT_MODES - reuse_modes} new modes.",
                        flush=True,
                    )
                else:
                    warnings.warn(
                        "H2 basis cache was found but could not be reused because its "
                        "shape or active-mode ordering did not match the nested H3 geometry."
                    )
            except Exception as exc:
                warnings.warn(f"H2 basis-cache reuse failed; rebuilding all modes: {exc}")

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


def _previous_h2_basis_continuity_audit(high_basis):
    required = [
        PREVIOUS_H2_BASIS_METADATA_FILE,
        PREVIOUS_H2_OBSERVED_GRADIENT_FILE,
        PREVIOUS_H2_QUERY_PRIMARY_GRADIENT_FILE,
        PREVIOUS_H2_RADIAL_DESIGN_FILE,
        PREVIOUS_H2_COLUMN_RMS_FILE,
    ]
    if not all(path.is_file() for path in required):
        return pd.DataFrame(
            [{"quantity": "previous_h2_basis_cache_available", "value": 0.0}]
        )
    try:
        metadata = json.loads(PREVIOUS_H2_BASIS_METADATA_FILE.read_text(encoding="utf-8"))
        previous_h2_observed = np.load(PREVIOUS_H2_OBSERVED_GRADIENT_FILE, mmap_mode="r")
        previous_h2_query = np.load(PREVIOUS_H2_QUERY_PRIMARY_GRADIENT_FILE, mmap_mode="r")
        previous_h2_radial = np.load(PREVIOUS_H2_RADIAL_DESIGN_FILE, mmap_mode="r")
        previous_h2_column_rms = np.load(PREVIOUS_H2_COLUMN_RMS_FILE, mmap_mode="r")

        n_modes = min(
            PREVIOUS_H2_MAX_NONCONSTANT_MODES,
            int(previous_h2_observed.shape[1]),
            int(high_basis["observed_nonconstant_gradient"].shape[1]),
        )
        if previous_h2_observed.shape[0] != high_basis["observed_nonconstant_gradient"].shape[0]:
            raise ValueError("H2 observed gradient cache has a different survey size.")
        if previous_h2_query.shape[0] != high_basis["query_primary_nonconstant_gradient"].shape[0]:
            raise ValueError("H2 query gradient cache has a different primary-query size.")

        observed_max, observed_relative = _chunked_mode_difference(
            high_basis["observed_nonconstant_gradient"], previous_h2_observed, n_modes
        )
        query_max, query_relative = _chunked_mode_difference(
            high_basis["query_primary_nonconstant_gradient"], previous_h2_query, n_modes
        )
        radial_max, radial_relative = _chunked_column_difference(
            high_basis["radial_design_observed"],
            previous_h2_radial,
            N_AFFINE_MODES + n_modes,
        )
        current_rms = np.asarray(high_basis["column_rms"][:n_modes], dtype=np.float64)
        reference_rms = np.asarray(previous_h2_column_rms[:n_modes], dtype=np.float64)
        current_modes = np.asarray(
            high_basis["metadata"].get("active_mode_indices", [])[:n_modes],
            dtype=np.int64,
        )
        previous_h2_modes = np.asarray(
            metadata.get("active_mode_indices", [])[:n_modes],
            dtype=np.int64,
        )
        active_equal = bool(
            current_modes.size == n_modes
            and previous_h2_modes.size == n_modes
            and np.array_equal(current_modes, previous_h2_modes)
        )
        return pd.DataFrame(
            [
                {"quantity": "previous_h2_basis_cache_available", "value": 1.0},
                {"quantity": "previous_h2_audited_nonconstant_modes", "value": float(n_modes)},
                {"quantity": "previous_h2_active_mode_indices_equal", "value": float(active_equal)},
                {
                    "quantity": "previous_h2_affine_scale_abs_difference",
                    "value": abs(
                        float(high_basis["metadata"]["affine_mode_scale"])
                        - float(metadata.get("affine_mode_scale", math.nan))
                    ),
                },
                {"quantity": "previous_h2_observed_gradient_max_abs_difference", "value": observed_max},
                {
                    "quantity": "previous_h2_observed_gradient_relative_frobenius_difference",
                    "value": observed_relative,
                },
                {"quantity": "previous_h2_query_gradient_max_abs_difference", "value": query_max},
                {
                    "quantity": "previous_h2_query_gradient_relative_frobenius_difference",
                    "value": query_relative,
                },
                {"quantity": "previous_h2_radial_design_max_abs_difference", "value": radial_max},
                {
                    "quantity": "previous_h2_radial_design_relative_frobenius_difference",
                    "value": radial_relative,
                },
                {
                    "quantity": "previous_h2_column_rms_max_abs_difference",
                    "value": float(np.max(np.abs(current_rms - reference_rms))),
                },
                {
                    "quantity": "previous_h2_column_rms_relative_frobenius_difference",
                    "value": _matrix_relative_difference(current_rms, reference_rms),
                },
            ]
        )
    except Exception as exc:
        warnings.warn(f"H2 basis continuity audit failed: {exc}")
        return pd.DataFrame(
            [
                {"quantity": "previous_h2_basis_cache_available", "value": 1.0},
                {"quantity": "previous_h2_basis_audit_completed", "value": 0.0},
            ]
        )


def _previous_h2_candidate_grid_audit(candidate_df):
    if not PREVIOUS_H2_CANDIDATE_GRID_FILE.is_file():
        return pd.DataFrame(
            [{"quantity": "previous_h2_candidate_grid_available", "value": 0.0}]
        )
    previous_h2 = pd.read_csv(PREVIOUS_H2_CANDIDATE_GRID_FILE)
    current = candidate_df[candidate_df["n_basis"] <= PREVIOUS_H2_MAX_BASIS].copy()
    previous_h2 = previous_h2[previous_h2["n_basis"] <= PREVIOUS_H2_MAX_BASIS].copy()
    keys = [
        "loss_name",
        "n_basis",
        "regularization_strength",
        "penalty_power",
    ]
    common = current.merge(
        previous_h2, on=keys, suffixes=("_h3", "_previous_h2")
    )
    expected_rows = int(
        len([basis for basis in POTENTIAL_BASIS_SIZES if basis <= PREVIOUS_H2_MAX_BASIS])
        * len(POTENTIAL_REGULARIZATION_STRENGTHS)
        * len(POTENTIAL_REGULARIZATION_POWERS)
        * len(LOSS_CONFIGS)
    )
    records = [
        {"quantity": "previous_h2_candidate_grid_available", "value": 1.0},
        {"quantity": "expected_common_candidate_rows", "value": float(expected_rows)},
        {"quantity": "n_common_candidate_rows", "value": float(common.shape[0])},
        {
            "quantity": "previous_h2_candidate_grid_complete",
            "value": float(common.shape[0] == expected_rows),
        },
    ]
    for column in [
        "weighted_validation_nrmse_radial",
        "unweighted_validation_nrmse_radial",
        "weighted_validation_nrmse_tangential",
        "weighted_validation_nrmse_3d",
    ]:
        left = f"{column}_h3"
        right = f"{column}_previous_h2"
        if left in common.columns and right in common.columns and not common.empty:
            difference = np.abs(
                common[left].to_numpy(dtype=np.float64)
                - common[right].to_numpy(dtype=np.float64)
            )
            records.extend(
                [
                    {
                        "quantity": f"previous_h2_{column}_max_abs_difference",
                        "value": float(np.max(difference)),
                    },
                    {
                        "quantity": f"previous_h2_{column}_rms_difference",
                        "value": float(np.sqrt(np.mean(np.square(difference)))),
                    },
                ]
            )
    return pd.DataFrame(records)


def _previous_h2_selector_continuity_audit(candidate_df):
    if not PREVIOUS_H2_GLOBAL_SELECTORS_FILE.is_file():
        return pd.DataFrame(
            [{"quantity": "previous_h2_global_selectors_available", "value": 0.0}]
        )
    restricted = candidate_df[candidate_df["n_basis"] <= PREVIOUS_H2_MAX_BASIS].copy()
    current_rows = {
        "operational_radial_validation": _select_row(
            restricted,
            ["weighted_validation_nrmse_radial", "unweighted_validation_nrmse_radial"],
        ),
        "hidden_3d_validation_diagnostic": _select_row(
            restricted,
            ["weighted_validation_nrmse_3d", "weighted_validation_nrmse_radial"],
        ),
    }
    previous_frame = pd.read_csv(PREVIOUS_H2_GLOBAL_SELECTORS_FILE)
    records = [{"quantity": "previous_h2_global_selectors_available", "value": 1.0}]
    for selector, current in current_rows.items():
        previous_rows = previous_frame[previous_frame["selector"] == selector]
        if previous_rows.empty:
            records.append({"quantity": f"previous_h2_{selector}_available", "value": 0.0})
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
                {"quantity": f"previous_h2_{selector}_available", "value": 1.0},
                {
                    "quantity": f"previous_h2_{selector}_configuration_equal",
                    "value": float(same_configuration),
                },
                {
                    "quantity": f"previous_h2_{selector}_radial_nrmse_abs_difference",
                    "value": abs(
                        float(current["weighted_validation_nrmse_radial"])
                        - float(previous["weighted_validation_nrmse_radial"])
                    ),
                },
                {
                    "quantity": f"previous_h2_{selector}_hidden_3d_nrmse_abs_difference",
                    "value": abs(
                        float(current["weighted_validation_nrmse_3d"])
                        - float(previous["weighted_validation_nrmse_3d"])
                    ),
                },
            ]
        )
    return pd.DataFrame(records)


# ==================================================================================================
# 6. Main analysis
# ==================================================================================================


def main():
    start_time = time.perf_counter()
    print("=" * 128)
    print("Radial Mock-2 potential-flow high-mode capacity audit: H3 resume-safe v2")
    print("=" * 128)
    print(f"Python / NumPy / SciPy : {platform.python_version()} / {np.__version__} / {scipy.__version__}")
    print(f"Mock-1                  : {MOCK1_FILE}")
    print(f"Mock-2                  : {MOCK2_FILE}")
    print(f"Source predictions      : {PREDICTIONS_FILE}")
    print(f"Output                  : {OUTPUT_DIR}")
    print(f"Label split seed        : {LABEL_SPLIT_SEED}")
    print(f"Nonconstant mode counts : {POTENTIAL_DIFFUSION_MODE_COUNTS}")
    print(f"Potential parameters    : {POTENTIAL_BASIS_SIZES}")
    print(f"Lambda grid             : {POTENTIAL_REGULARIZATION_STRENGTHS}")
    print(f"Penalty powers          : {POTENTIAL_REGULARIZATION_POWERS}")
    print(f"Resume                  : {RESUME}")

    for path in (MOCK1_FILE, MOCK2_FILE, PREDICTIONS_FILE):
        if not path.is_file():
            raise FileNotFoundError(path)

    legacy_geometry_file = _find_existing(
        LEGACY_GEOMETRY_CANDIDATES,
        "legacy Mock-2 unweighted alpha=0 geometry",
    )
    reference_geometry_file = _find_existing(
        REFERENCE_GEOMETRY_CANDIDATES,
        "Mock-2 4096-mode H2 reference geometry cache",
    )

    complete = _load_mock1(MOCK1_FILE)
    survey = _load_survey(MOCK2_FILE)
    centered_survey = survey["pos"] - SPHERE_CENTER[None, :]

    high_geometry_file, geometry_source = _build_or_load_extended_geometry(
        centered_survey,
        reference_geometry_file,
    )
    print(f"Legacy geometry         : {legacy_geometry_file}")
    print(f"H2 reference geometry   : {reference_geometry_file}")
    print(f"H3 extended geometry    : {high_geometry_file}")
    print(f"Geometry source         : {geometry_source}")

    with np.load(PREDICTIONS_FILE, allow_pickle=False) as data:
        required = {
            "truth_survey",
            "truth_query",
            "line_of_sight_survey",
            "line_of_sight_query",
            "selection_probability_survey",
            "selection_probability_query",
            "primary_query_mask",
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
        probability_survey = np.asarray(
            data["selection_probability_survey"], dtype=np.float64
        )
        probability_query = np.asarray(
            data["selection_probability_query"], dtype=np.float64
        )
        primary_query_mask = np.asarray(data["primary_query_mask"], dtype=bool)
        query_global_indices = np.asarray(data["query_global_indices"], dtype=np.int64)

    centered_query_all = complete["pos"][query_global_indices] - SPHERE_CENTER[None, :]
    centered_query_primary = centered_query_all[primary_query_mask]
    truth_query_primary = truth_query[primary_query_mask]
    line_of_sight_query_primary = line_of_sight_query[primary_query_mask]

    if truth_survey.shape != line_of_sight_survey.shape:
        raise ValueError("Survey truth and line-of-sight arrays have inconsistent shapes.")
    if truth_query.shape != line_of_sight_query.shape:
        raise ValueError("Query truth and line-of-sight arrays have inconsistent shapes.")
    if survey["pos"].shape[0] != truth_survey.shape[0]:
        raise ValueError("Mock-2 catalog and source predictions have different survey sizes.")
    if query_global_indices.size != truth_query.shape[0]:
        raise ValueError("Query indices and query truth have different sizes.")
    if np.sum(primary_query_mask) < 5:
        raise ValueError("Primary complete-query subset contains too few objects.")

    high_basis = _build_or_load_highmode_basis(
        centered_survey,
        centered_query_primary,
        line_of_sight_survey,
        high_geometry_file,
        legacy_geometry_file,
        query_global_indices,
        primary_query_mask,
    )
    affine_mode_scale = float(high_basis["metadata"]["affine_mode_scale"])
    active_generator_values = np.asarray(
        high_basis["metadata"]["active_generator_values"],
        dtype=np.float64,
    )
    potential_generator_values = np.concatenate(
        [np.zeros(N_AFFINE_MODES), active_generator_values]
    )
    if potential_generator_values.size != MAX_BASIS:
        raise RuntimeError("Potential generator spectrum has an unexpected size.")

    radial_design_observed = high_basis["radial_design_observed"]
    observed_nonconstant_gradient = high_basis["observed_nonconstant_gradient"]
    query_primary_nonconstant_gradient = high_basis[
        "query_primary_nonconstant_gradient"
    ]

    print(f"Survey/query/primary    : {truth_survey.shape[0]:,}/{truth_query.shape[0]:,}/{truth_query_primary.shape[0]:,}")
    print(f"Maximum potential basis : {MAX_BASIS}")
    print(f"Affine mode scale       : {affine_mode_scale:.8g}")
    print(f"Basis cache source      : {'cache' if high_basis['loaded_from_cache'] else 'new build'}")

    geometry_continuity_file = OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_continuity.csv"
    if geometry_continuity_file.is_file():
        geometry_continuity_df = pd.read_csv(geometry_continuity_file)
        print("-" * 128)
        print("High-mode versus legacy geometry continuity audit")
        print(geometry_continuity_df.to_string(index=False))

    basis_continuity_df = _basis_continuity_audit(
        high_basis,
        legacy_geometry_file,
        line_of_sight_survey,
        primary_query_mask,
    )
    basis_continuity_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_basis_continuity.csv",
        index=False,
    )
    print("-" * 128)
    print("High-mode versus legacy potential-basis continuity audit")
    print(basis_continuity_df.to_string(index=False))

    previous_h2_basis_continuity_df = _previous_h2_basis_continuity_audit(high_basis)
    previous_h2_basis_continuity_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_previous_h2_basis_continuity.csv",
        index=False,
    )
    print("-" * 128)
    print("H3 versus H2 potential-basis continuation audit")
    print(previous_h2_basis_continuity_df.to_string(index=False))
    gc.collect()

    split = _make_split(
        truth_survey.shape[0],
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        LABEL_SPLIT_SEED,
    )
    train_indices = split["train"]
    validation_indices = split["validation"]
    test_indices = split["test"]
    fit_indices = np.sort(np.concatenate([train_indices, validation_indices]))

    radial_truth_survey = np.einsum(
        "ij,ij->i",
        line_of_sight_survey,
        truth_survey,
    )

    loss_weight_arrays = {}
    weight_records = []
    for config in LOSS_CONFIGS:
        weights, info = _tempered_inverse_selection_weights(
            probability_survey,
            config["gamma"],
            config["cap"],
        )
        loss_weight_arrays[config["name"]] = weights
        weight_records.append(
            {
                "loss_name": config["name"],
                "loss_label": config["label"],
                **info,
            }
        )
    weight_diagnostics_df = pd.DataFrame(weight_records)
    evaluation_weights, evaluation_weight_info = _tempered_inverse_selection_weights(
        probability_survey,
        EVALUATION_WEIGHT_GAMMA,
        EVALUATION_WEIGHT_CAP,
    )
    weight_diagnostics_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_weight_diagnostics.csv",
        index=False,
    )

    A_train = np.asarray(radial_design_observed[train_indices, :MAX_BASIS], dtype=np.float64)
    A_validation = np.asarray(
        radial_design_observed[validation_indices, :MAX_BASIS], dtype=np.float64
    )
    gradient_validation = np.asarray(
        observed_nonconstant_gradient[validation_indices, :MAX_NONCONSTANT_MODES, :],
        dtype=np.float64,
    )

    candidate_frames = []
    candidate_coefficients_by_loss = {}
    candidate_signature = _stable_signature(
        {
            "basis_cache_signature": high_basis["metadata"]["signature"],
            "label_seed": LABEL_SPLIT_SEED,
            "basis_sizes": POTENTIAL_BASIS_SIZES,
            "strengths": POTENTIAL_REGULARIZATION_STRENGTHS,
            "powers": POTENTIAL_REGULARIZATION_POWERS,
            "loss_configs": LOSS_CONFIGS,
            "version": 2,
        }
    )

    for config in LOSS_CONFIGS:
        loss_name = config["name"]
        checkpoint_csv = CHECKPOINT_DIR / f"candidate_{loss_name}.csv"
        checkpoint_coefficients = CHECKPOINT_DIR / f"candidate_{loss_name}_coefficients.npy"
        checkpoint_metadata = CHECKPOINT_DIR / f"candidate_{loss_name}_metadata.json"
        can_resume = False
        if RESUME and all(
            path.is_file()
            for path in (checkpoint_csv, checkpoint_coefficients, checkpoint_metadata)
        ):
            try:
                metadata = json.loads(checkpoint_metadata.read_text(encoding="utf-8"))
                can_resume = metadata.get("signature") == candidate_signature
            except Exception:
                can_resume = False

        if can_resume:
            print(f"[Candidate checkpoint] reuse: {loss_name}")
            frame = pd.read_csv(checkpoint_csv)
            coefficient_matrix = np.load(checkpoint_coefficients)
        else:
            print("-" * 128)
            print(f"Candidate scan: {loss_name}")
            loss_weights = loss_weight_arrays[loss_name]
            gram, rhs = _weighted_crossproducts(
                A_train,
                radial_truth_survey[train_indices],
                loss_weights[train_indices],
            )
            coefficient_matrix, metadata_frame = _candidate_grid_coefficients(
                gram,
                rhs,
                potential_generator_values,
                POTENTIAL_BASIS_SIZES,
                POTENTIAL_REGULARIZATION_STRENGTHS,
                POTENTIAL_REGULARIZATION_POWERS,
            )
            radial_prediction_many = A_validation @ coefficient_matrix
            hidden_prediction_many = _many_velocity_predictions(
                gradient_validation,
                coefficient_matrix,
                affine_mode_scale,
            )
            hidden_metrics = _many_hidden_validation_metrics(
                truth_survey[validation_indices],
                hidden_prediction_many,
                line_of_sight_survey[validation_indices],
                evaluation_weights[validation_indices],
            )
            weighted_radial = []
            unweighted_radial = []
            for column in range(coefficient_matrix.shape[1]):
                weighted_radial.append(
                    _scalar_nrmse(
                        radial_truth_survey[validation_indices],
                        radial_prediction_many[:, column],
                        weights=evaluation_weights[validation_indices],
                    )
                )
                unweighted_radial.append(
                    _scalar_nrmse(
                        radial_truth_survey[validation_indices],
                        radial_prediction_many[:, column],
                    )
                )
            frame = metadata_frame.copy()
            frame["label_seed"] = LABEL_SPLIT_SEED
            frame["loss_name"] = loss_name
            frame["loss_label"] = config["label"]
            frame["loss_gamma"] = float(config["gamma"])
            frame["loss_cap"] = float(config["cap"])
            frame["loss_ess_train"] = _effective_sample_size(
                loss_weights[train_indices]
            )
            frame["weighted_validation_nrmse_radial"] = weighted_radial
            frame["unweighted_validation_nrmse_radial"] = unweighted_radial
            frame["weighted_validation_nrmse_tangential"] = hidden_metrics[
                "nrmse_tangential"
            ]
            frame["weighted_validation_nrmse_3d"] = hidden_metrics["nrmse_3d"]
            frame["candidate_id"] = np.arange(frame.shape[0], dtype=int)
            frame.to_csv(checkpoint_csv, index=False)
            np.save(checkpoint_coefficients, coefficient_matrix)
            checkpoint_metadata.write_text(
                json.dumps(
                    {"signature": candidate_signature, "loss_name": loss_name},
                    ensure_ascii=False,
                    indent=2,
                ),
                encoding="utf-8",
            )
            del gram, rhs, radial_prediction_many, hidden_prediction_many
            gc.collect()

        if coefficient_matrix.shape[0] != MAX_BASIS:
            raise RuntimeError(f"Candidate coefficient checkpoint for {loss_name} has wrong shape.")
        candidate_frames.append(frame)
        candidate_coefficients_by_loss[loss_name] = coefficient_matrix

    candidate_df = pd.concat(candidate_frames, ignore_index=True)
    candidate_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_candidate_grid.csv",
        index=False,
    )
    del A_train, A_validation, gradient_validation
    candidate_coefficients_by_loss.clear()
    gc.collect()

    legacy_grid_audit_df = _legacy_candidate_grid_audit(candidate_df)
    legacy_grid_audit_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_legacy_candidate_grid_continuity.csv",
        index=False,
    )
    print("-" * 128)
    print("Legacy 514-parameter candidate-grid continuity audit")
    print(legacy_grid_audit_df.to_string(index=False))

    previous_h2_candidate_audit_df = _previous_h2_candidate_grid_audit(candidate_df)
    previous_h2_candidate_audit_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_previous_h2_candidate_grid_continuity.csv",
        index=False,
    )
    previous_h2_selector_audit_df = _previous_h2_selector_continuity_audit(candidate_df)
    previous_h2_selector_audit_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_previous_h2_selector_continuity.csv",
        index=False,
    )
    print("-" * 128)
    print("H3 versus H2 common candidate-grid continuity audit")
    print(previous_h2_candidate_audit_df.to_string(index=False))
    print("-" * 128)
    print("H3 restricted-to-H2 selector continuity audit")
    print(previous_h2_selector_audit_df.to_string(index=False))

    within_basis_records = []
    capacity_selection_records = []
    practical_diagnostic_records = []
    per_loss_selected = {}

    for config in LOSS_CONFIGS:
        loss_name = config["name"]
        frame = candidate_df[candidate_df["loss_name"] == loss_name]
        selected_rows = []
        for n_basis in POTENTIAL_BASIS_SIZES:
            selected = _select_row(
                frame[frame["n_basis"] == n_basis],
                [
                    "weighted_validation_nrmse_radial",
                    "unweighted_validation_nrmse_radial",
                ],
            )
            selected_rows.append(selected)
            within_basis_records.append(selected)
        selected_frame = pd.DataFrame(selected_rows).sort_values("n_basis")
        per_loss_selected[loss_name] = selected_frame

        plateau = _practical_plateau(
            selected_frame["n_basis"].to_numpy(),
            selected_frame["weighted_validation_nrmse_radial"].to_numpy(),
        )
        minimum_row = selected_frame.sort_values(
            ["weighted_validation_nrmse_radial", "n_basis"],
            kind="mergesort",
        ).iloc[0]
        last_improvement = float(plateau["adjacent_improvement"][-1])
        strict_argmin_boundary = bool(int(minimum_row["n_basis"]) == MAX_BASIS)
        plateau_found = bool(plateau["plateau_found"])
        final_increment_small = bool(
            np.isfinite(last_improvement)
            and last_improvement < PRACTICAL_PLATEAU_RELATIVE_THRESHOLD
        )
        practical_converged = bool(
            (plateau_found or not strict_argmin_boundary)
            and final_increment_small
        )
        needs_larger_eigensystem = bool(
            strict_argmin_boundary
            and not practical_converged
        )
        if needs_larger_eigensystem:
            convergence_status = "geometry_limit_unresolved"
        elif practical_converged and strict_argmin_boundary:
            convergence_status = "boundary_argmin_with_practical_plateau"
        elif practical_converged:
            convergence_status = "practical_converged"
        elif not strict_argmin_boundary:
            convergence_status = "interior_minimum_requires_review"
        else:
            convergence_status = "boundary_requires_review"

        practical_basis = (
            int(plateau["plateau_basis"])
            if plateau_found
            else int(minimum_row["n_basis"])
        )
        maximum_row = selected_frame[selected_frame["n_basis"] == MAX_BASIS].iloc[0]
        h2_rows = selected_frame[selected_frame["n_basis"] == PREVIOUS_H2_MAX_BASIS]
        if h2_rows.empty:
            h2_validation_nrmse = math.nan
            relative_h2_to_maximum_improvement = math.nan
        else:
            h2_validation_nrmse = float(
                h2_rows.iloc[0]["weighted_validation_nrmse_radial"]
            )
            relative_h2_to_maximum_improvement = float(
                (
                    h2_validation_nrmse
                    - float(maximum_row["weighted_validation_nrmse_radial"])
                )
                / h2_validation_nrmse
            )
        capacity_selection_records.append(
            {
                "loss_name": loss_name,
                "validation_minimum_basis": int(minimum_row["n_basis"]),
                "validation_minimum_nrmse": float(
                    minimum_row["weighted_validation_nrmse_radial"]
                ),
                "practical_plateau_basis": plateau["plateau_basis"],
                "plateau_found": plateau_found,
                "practical_basis_h3": practical_basis,
                "validation_nrmse_at_h2_ceiling": h2_validation_nrmse,
                "validation_nrmse_at_maximum_basis": float(
                    maximum_row["weighted_validation_nrmse_radial"]
                ),
                "relative_validation_improvement_h2_to_maximum": (
                    relative_h2_to_maximum_improvement
                ),
                "last_adjacent_validation_improvement": last_improvement,
                "maximum_basis": MAX_BASIS,
                "strict_argmin_boundary": strict_argmin_boundary,
                "practical_converged": practical_converged,
                "convergence_status": convergence_status,
                "needs_larger_eigensystem": needs_larger_eigensystem,
                "upper_boundary_warning": bool(not practical_converged),
            }
        )
        for basis, value, improvement in zip(
            plateau["basis_values"],
            plateau["validation_values"],
            plateau["adjacent_improvement"],
        ):
            practical_diagnostic_records.append(
                {
                    "loss_name": loss_name,
                    "n_basis": int(basis),
                    "validation_nrmse": float(value),
                    "adjacent_validation_improvement": float(improvement),
                }
            )

    within_basis_df = pd.DataFrame(within_basis_records)
    capacity_selection_df = pd.DataFrame(capacity_selection_records)
    practical_diagnostic_df = pd.DataFrame(practical_diagnostic_records)
    within_basis_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_within_basis_selected.csv",
        index=False,
    )
    capacity_selection_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_capacity_selection.csv",
        index=False,
    )
    practical_diagnostic_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_plateau_diagnostic.csv",
        index=False,
    )

    full_geometry_practical_converged = bool(
        capacity_selection_df["practical_converged"].all()
    )
    any_larger_eigensystem_required = bool(
        capacity_selection_df["needs_larger_eigensystem"].any()
    )
    if full_geometry_practical_converged:
        recommended_next_step = "ten_seed_robustness"
    elif any_larger_eigensystem_required:
        recommended_next_step = "extend_to_4608_or_5120_before_seed_robustness"
    else:
        recommended_next_step = "manual_review_before_seed_robustness"
    decision_df = pd.DataFrame(
        [
            {
                "full_geometry_practical_converged": (
                    full_geometry_practical_converged
                ),
                "any_larger_eigensystem_required": (
                    any_larger_eigensystem_required
                ),
                "recommended_next_step": recommended_next_step,
            }
        ]
    )
    decision_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_decision.csv",
        index=False,
    )

    global_operational = _select_row(
        candidate_df,
        ["weighted_validation_nrmse_radial", "unweighted_validation_nrmse_radial"],
    )
    global_hidden = _select_row(
        candidate_df,
        ["weighted_validation_nrmse_3d", "weighted_validation_nrmse_radial"],
    )
    global_selector_df = pd.DataFrame(
        [
            {"selector": "operational_radial_validation", **global_operational},
            {"selector": "hidden_3d_validation_diagnostic", **global_hidden},
        ]
    )
    global_selector_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_global_selectors.csv",
        index=False,
    )

    # Final post-selection fits at every basis, using train+validation labels.
    A_fit = np.asarray(radial_design_observed[fit_indices, :MAX_BASIS], dtype=np.float64)
    A_test = np.asarray(radial_design_observed[test_indices, :MAX_BASIS], dtype=np.float64)
    gradient_test = np.asarray(
        observed_nonconstant_gradient[test_indices, :MAX_NONCONSTANT_MODES, :],
        dtype=np.float64,
    )
    gradient_query_primary = np.asarray(
        query_primary_nonconstant_gradient[:, :MAX_NONCONSTANT_MODES, :],
        dtype=np.float64,
    )

    basis_evaluation_records = []
    final_coefficient_payload = {}
    protocols = ["retuned", "matched_unweighted_structure"]

    for protocol in protocols:
        for config in LOSS_CONFIGS:
            loss_name = config["name"]
            loss_weights = loss_weight_arrays[loss_name]
            gram, rhs = _weighted_crossproducts(
                A_fit,
                radial_truth_survey[fit_indices],
                loss_weights[fit_indices],
            )
            if protocol == "retuned":
                source_frame = per_loss_selected[loss_name]
            else:
                source_frame = per_loss_selected["unweighted"]
            configurations = [row.to_dict() for _, row in source_frame.iterrows()]
            solutions = _solve_selected_configurations(
                gram,
                rhs,
                potential_generator_values,
                configurations,
            )
            coefficient_matrix = np.column_stack(
                [solutions[basis]["coefficients"] for basis in POTENTIAL_BASIS_SIZES]
            )
            radial_test_many = A_test @ coefficient_matrix
            velocity_test_many = _many_velocity_predictions(
                gradient_test,
                coefficient_matrix,
                affine_mode_scale,
            )
            velocity_query_many = _many_velocity_predictions(
                gradient_query_primary,
                coefficient_matrix,
                affine_mode_scale,
            )

            for column, n_basis in enumerate(POTENTIAL_BASIS_SIZES):
                specification = solutions[n_basis]
                test_parent = _radial_vector_metrics(
                    truth_survey[test_indices],
                    velocity_test_many[:, :, column],
                    line_of_sight_survey[test_indices],
                    weights=evaluation_weights[test_indices],
                )
                test_unweighted = _radial_vector_metrics(
                    truth_survey[test_indices],
                    velocity_test_many[:, :, column],
                    line_of_sight_survey[test_indices],
                    weights=None,
                )
                query_metrics = _radial_vector_metrics(
                    truth_query_primary,
                    velocity_query_many[:, :, column],
                    line_of_sight_query_primary,
                    weights=None,
                )
                for subset, metrics in [
                    ("observed_test_parent_weighted", test_parent),
                    ("observed_test_unweighted", test_unweighted),
                    ("query_primary_boundary_safe", query_metrics),
                ]:
                    basis_evaluation_records.append(
                        {
                            "protocol": protocol,
                            "loss_name": loss_name,
                            "loss_label": config["label"],
                            "subset": subset,
                            "n_basis": int(n_basis),
                            "n_diffusion_modes": int(n_basis - N_AFFINE_MODES),
                            "regularization_strength": float(
                                specification["regularization_strength"]
                            ),
                            "penalty_power": float(specification["penalty_power"]),
                            **metrics,
                        }
                    )
                final_coefficient_payload[
                    f"{protocol}_{loss_name}_basis{n_basis}_coefficients"
                ] = coefficient_matrix[:n_basis, column]

            del gram, rhs, coefficient_matrix
            del radial_test_many, velocity_test_many, velocity_query_many
            gc.collect()

    basis_evaluation_df = pd.DataFrame(basis_evaluation_records)
    basis_evaluation_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_basis_evaluation.csv",
        index=False,
    )
    np.savez_compressed(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_selected_coefficients.npz",
        **final_coefficient_payload,
    )

    weighting_contrast_records = []
    for protocol in protocols:
        for subset in [
            "observed_test_parent_weighted",
            "observed_test_unweighted",
            "query_primary_boundary_safe",
        ]:
            frame = basis_evaluation_df[
                (basis_evaluation_df["protocol"] == protocol)
                & (basis_evaluation_df["subset"] == subset)
            ]
            unweighted = frame[frame["loss_name"] == "unweighted"].set_index(
                "n_basis"
            )
            for loss_name in LOSS_ORDER[1:]:
                weighted = frame[frame["loss_name"] == loss_name].set_index(
                    "n_basis"
                )
                for n_basis in POTENTIAL_BASIS_SIZES:
                    record = {
                        "protocol": protocol,
                        "subset": subset,
                        "loss_name": loss_name,
                        "comparison": f"{loss_name}_minus_unweighted",
                        "n_basis": int(n_basis),
                    }
                    for metric in [
                        "nrmse_radial",
                        "nrmse_tangential",
                        "nrmse_3d",
                        "direction_error_median_deg",
                    ]:
                        record[f"delta_{metric}"] = float(
                            weighted.loc[n_basis, metric]
                            - unweighted.loc[n_basis, metric]
                        )
                    weighting_contrast_records.append(record)
    weighting_contrast_df = pd.DataFrame(weighting_contrast_records)
    weighting_contrast_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_weighting_contrasts.csv",
        index=False,
    )

    # Maximum-basis closure using the operational train+validation design.
    closure_rng = np.random.default_rng(20270829)
    A_closure = A_fit
    column_scale = np.sqrt(np.mean(np.square(A_closure), axis=0))
    coefficient_true = closure_rng.normal(size=MAX_BASIS) / np.maximum(
        column_scale,
        1.0e-8,
    )
    radial_labels = A_closure @ coefficient_true
    closure_gram = A_closure.T @ A_closure
    closure_rhs = A_closure.T @ radial_labels
    closure_solver = "cholesky"
    closure_rank = MAX_BASIS
    try:
        closure_factor, closure_jitter = _cholesky_with_jitter(closure_gram)
        coefficient_recovered = _solve_leading_cholesky(
            closure_factor,
            closure_rhs,
            MAX_BASIS,
        )
    except linalg.LinAlgError as exc:
        warnings.warn(
            "Unregularized maximum-basis closure Cholesky failed; "
            f"falling back to QR least squares: {exc}"
        )
        closure_solver = "gelsy"
        closure_jitter = math.nan
        coefficient_recovered, _, closure_rank, _ = linalg.lstsq(
            A_closure,
            radial_labels,
            cond=None,
            lapack_driver="gelsy",
            check_finite=False,
        )
    coefficient_relative_error = float(
        np.linalg.norm(coefficient_recovered - coefficient_true)
        / np.linalg.norm(coefficient_true)
    )
    radial_closure = float(
        np.linalg.norm(A_closure @ coefficient_recovered - radial_labels)
        / np.linalg.norm(radial_labels)
    )
    query_true_closure = _single_velocity_prediction(
        gradient_query_primary,
        coefficient_true,
        MAX_BASIS,
        affine_mode_scale,
    )
    query_recovered_closure = _single_velocity_prediction(
        gradient_query_primary,
        coefficient_recovered,
        MAX_BASIS,
        affine_mode_scale,
    )
    query_closure = float(
        np.linalg.norm(query_recovered_closure - query_true_closure)
        / np.linalg.norm(query_true_closure)
    )
    closure_df = pd.DataFrame(
        [
            {
                "n_basis": MAX_BASIS,
                "n_fit": fit_indices.size,
                "numerical_rank": int(closure_rank),
                "closure_solver": closure_solver,
                "coefficient_relative_error": coefficient_relative_error,
                "radial_fit_relative_error": radial_closure,
                "query_3d_relative_error": query_closure,
                "extra_cholesky_jitter": closure_jitter,
                "closure_status": (
                    "pass"
                    if max(
                        coefficient_relative_error,
                        radial_closure,
                        query_closure,
                    )
                    < 1.0e-8
                    else "review"
                ),
            }
        ]
    )
    closure_df.to_csv(
        OUTPUT_DIR / f"{OUTPUT_PREFIX}_closure.csv",
        index=False,
    )

    # ----------------------------------------------------------------------------------------------
    # Figures
    # ----------------------------------------------------------------------------------------------
    legacy_ceiling = 514
    h2_ceiling = PREVIOUS_H2_MAX_BASIS

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.0))
    panels = [
        (
            axes[0, 0],
            within_basis_df,
            "weighted_validation_nrmse_radial",
            "(a) Observable radial validation risk",
            "Parent-weighted validation radial NRMSE",
        ),
        (
            axes[0, 1],
            within_basis_df,
            "weighted_validation_nrmse_3d",
            "(b) Hidden validation field recovery",
            "Hidden validation 3D NRMSE",
        ),
        (
            axes[1, 0],
            basis_evaluation_df[
                (basis_evaluation_df["protocol"] == "retuned")
                & (
                    basis_evaluation_df["subset"]
                    == "observed_test_parent_weighted"
                )
            ],
            "nrmse_3d",
            "(c) Held-out observed field recovery",
            "Parent-weighted observed-test 3D NRMSE",
        ),
        (
            axes[1, 1],
            basis_evaluation_df[
                (basis_evaluation_df["protocol"] == "retuned")
                & (
                    basis_evaluation_df["subset"]
                    == "query_primary_boundary_safe"
                )
            ],
            "nrmse_3d",
            "(d) Independent complete-query recovery",
            "Primary complete-query 3D NRMSE",
        ),
    ]
    for ax, frame, metric, title, ylabel in panels:
        for loss_name in LOSS_ORDER:
            selected = frame[frame["loss_name"] == loss_name].sort_values("n_basis")
            ax.plot(
                selected["n_basis"],
                selected[metric],
                marker="o",
                label=LOSS_LABELS[loss_name],
            )
        ax.axvline(legacy_ceiling, linestyle=":", linewidth=1.2, label="Legacy ceiling = 514")
        ax.axvline(h2_ceiling, linestyle="-.", linewidth=1.2, label="H2 ceiling = 4098")
        ax.set_title(title)
        ax.set_xlabel("Potential parameter count")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save_figure(fig, "capacity_composite")

    fig, axes = plt.subplots(2, 2, figsize=(14.0, 9.5))
    contrast_panels = [
        (
            axes[0, 0],
            within_basis_df,
            "weighted_validation_nrmse_radial",
            "(a) Retuned observable validation contrast",
            "Weighted minus unweighted radial NRMSE",
        ),
        (
            axes[0, 1],
            within_basis_df,
            "weighted_validation_nrmse_3d",
            "(b) Retuned hidden validation contrast",
            "Weighted minus unweighted hidden 3D NRMSE",
        ),
    ]
    for ax, frame, metric, title, ylabel in contrast_panels:
        baseline = frame[frame["loss_name"] == "unweighted"].set_index("n_basis")
        for loss_name in LOSS_ORDER[1:]:
            selected = frame[frame["loss_name"] == loss_name].set_index("n_basis")
            values = selected.loc[POTENTIAL_BASIS_SIZES, metric].to_numpy() - baseline.loc[
                POTENTIAL_BASIS_SIZES, metric
            ].to_numpy()
            ax.plot(
                POTENTIAL_BASIS_SIZES,
                values,
                marker="o",
                label=f"{LOSS_LABELS[loss_name]} minus unweighted",
            )
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.axvline(legacy_ceiling, linestyle=":", linewidth=1.2)
        ax.axvline(h2_ceiling, linestyle="-.", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Potential parameter count")
        ax.set_ylabel(ylabel)
        ax.grid(alpha=0.2)

    for ax, subset, title in [
        (
            axes[1, 0],
            "observed_test_parent_weighted",
            "(c) Matched-structure observed-test contrast",
        ),
        (
            axes[1, 1],
            "query_primary_boundary_safe",
            "(d) Matched-structure complete-query contrast",
        ),
    ]:
        frame = weighting_contrast_df[
            (weighting_contrast_df["protocol"] == "matched_unweighted_structure")
            & (weighting_contrast_df["subset"] == subset)
        ]
        for loss_name in LOSS_ORDER[1:]:
            selected = frame[frame["loss_name"] == loss_name].sort_values("n_basis")
            ax.plot(
                selected["n_basis"],
                selected["delta_nrmse_3d"],
                marker="o",
                label=f"{LOSS_LABELS[loss_name]} minus unweighted",
            )
        ax.axhline(0.0, linestyle="--", linewidth=1.0)
        ax.axvline(legacy_ceiling, linestyle=":", linewidth=1.2)
        ax.axvline(h2_ceiling, linestyle="-.", linewidth=1.2)
        ax.set_title(title)
        ax.set_xlabel("Potential parameter count")
        ax.set_ylabel("Weighted minus unweighted 3D NRMSE")
        ax.grid(alpha=0.2)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    by_label = dict(zip(labels, handles))
    fig.legend(by_label.values(), by_label.keys(), loc="upper center", ncol=2)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.93))
    _save_figure(fig, "weighting_capacity_interaction")

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    for loss_name in LOSS_ORDER:
        selected = practical_diagnostic_df[
            practical_diagnostic_df["loss_name"] == loss_name
        ].sort_values("n_basis")
        ax.plot(
            selected["n_basis"],
            100.0 * selected["adjacent_validation_improvement"],
            marker="o",
            label=LOSS_LABELS[loss_name],
        )
    ax.axhline(
        100.0 * PRACTICAL_PLATEAU_RELATIVE_THRESHOLD,
        linestyle="--",
        linewidth=1.1,
        label="2% practical threshold",
    )
    ax.axhline(0.0, linestyle=":", linewidth=1.0)
    ax.axvline(legacy_ceiling, linestyle=":", linewidth=1.2, label="Legacy ceiling = 514")
    ax.axvline(h2_ceiling, linestyle="-.", linewidth=1.2, label="H2 ceiling = 4098")
    ax.set_xlabel("Potential parameter count")
    ax.set_ylabel("Adjacent validation improvement [%]")
    ax.set_title("Radial Mock-2 H3 practical-plateau diagnostic")
    ax.legend(fontsize="small")
    ax.grid(alpha=0.2)
    fig.tight_layout()
    _save_figure(fig, "plateau_diagnostic")

    fig, ax = plt.subplots(figsize=(9.5, 5.8))
    selector_positions = np.arange(2)
    selector_labels = ["Operational radial", "Hidden 3D diagnostic"]
    selector_basis = [
        int(global_operational["n_basis"]),
        int(global_hidden["n_basis"]),
    ]
    ax.scatter(selector_positions, selector_basis, s=100)
    for index, row in enumerate([global_operational, global_hidden]):
        ax.annotate(
            f"{row['loss_name']}\n$\\lambda$={float(row['regularization_strength']):.0e}, p={float(row['penalty_power']):g}",
            (selector_positions[index], selector_basis[index]),
            xytext=(0, 12),
            textcoords="offset points",
            ha="center",
        )
    ax.axhline(legacy_ceiling, linestyle=":", linewidth=1.2, label="Legacy ceiling = 514")
    ax.axhline(h2_ceiling, linestyle="-.", linewidth=1.2, label="H2 ceiling = 4098")
    ax.set_xticks(selector_positions)
    ax.set_xticklabels(selector_labels)
    ax.set_ylabel("Selected potential parameter count")
    ax.set_title("Operational versus hidden selector after the H3 continuation")
    ax.legend()
    fig.tight_layout()
    _save_figure(fig, "global_selector_comparison")

    summary = {
        "experiment": "Radial Mock-2 potential-flow high-mode capacity audit H3 resume-safe v2",
        "label_seed": LABEL_SPLIT_SEED,
        "input": {
            "mock1": str(MOCK1_FILE),
            "mock2": str(MOCK2_FILE),
            "source_predictions": str(PREDICTIONS_FILE),
            "legacy_geometry": str(legacy_geometry_file),
            "h2_reference_geometry": str(reference_geometry_file),
            "highmode_geometry": str(high_geometry_file),
            "geometry_source": geometry_source,
            "previous_h2_output": str(PREVIOUS_H2_OUTPUT_DIR),
        },
        "basis": {
            "diffusion_mode_counts": POTENTIAL_DIFFUSION_MODE_COUNTS,
            "potential_basis_sizes": POTENTIAL_BASIS_SIZES,
            "max_nonconstant_modes": MAX_NONCONSTANT_MODES,
            "max_basis": MAX_BASIS,
            "affine_mode_scale": affine_mode_scale,
        },
        "loss_configs": LOSS_CONFIGS,
        "regularization_strengths": POTENTIAL_REGULARIZATION_STRENGTHS,
        "penalty_powers": POTENTIAL_REGULARIZATION_POWERS,
        "capacity_selection": capacity_selection_df.to_dict(orient="records"),
        "decision": decision_df.to_dict(orient="records"),
        "global_selectors": global_selector_df.to_dict(orient="records"),
        "closure": closure_df.to_dict(orient="records"),
        "continuity_audits": {
            "legacy_geometry": (
                geometry_continuity_df.to_dict(orient="records")
                if "geometry_continuity_df" in locals()
                else []
            ),
            "legacy_basis": basis_continuity_df.to_dict(orient="records"),
            "previous_h2_basis": previous_h2_basis_continuity_df.to_dict(orient="records"),
            "legacy_514_candidate_grid": legacy_grid_audit_df.to_dict(orient="records"),
            "previous_h2_candidate_grid": previous_h2_candidate_audit_df.to_dict(orient="records"),
            "previous_h2_selectors": previous_h2_selector_audit_df.to_dict(orient="records"),
        },
        "basis_cache_metadata": high_basis["metadata"],
        "outputs": {
            "candidate_grid_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_candidate_grid.csv"
            ),
            "within_basis_selected_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_within_basis_selected.csv"
            ),
            "capacity_selection_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_capacity_selection.csv"
            ),
            "decision_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_decision.csv"
            ),
            "basis_evaluation_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_basis_evaluation.csv"
            ),
            "weighting_contrasts_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_weighting_contrasts.csv"
            ),
            "previous_h2_basis_continuity_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_previous_h2_basis_continuity.csv"
            ),
            "previous_h2_candidate_grid_continuity_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_previous_h2_candidate_grid_continuity.csv"
            ),
            "previous_h2_selector_continuity_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_previous_h2_selector_continuity.csv"
            ),
            "eigensystem_continuation_csv": str(
                OUTPUT_DIR / f"{OUTPUT_PREFIX}_eigensystem_continuation.csv"
            ),
        },
    }
    summary_file = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json"
    summary_file.write_text(
        json.dumps(_json_ready(summary), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    elapsed = (time.perf_counter() - start_time) / 60.0
    print("=" * 128)
    print("Radial Mock-2 high-mode capacity audit H3 resume-safe v2 completed")
    print("=" * 128)
    print(f"Elapsed time: {elapsed:.2f} min")
    print("Capacity selection")
    display(capacity_selection_df)
    print("H3 decision")
    display(decision_df)
    print("Global selectors")
    display(
        global_selector_df[
            [
                "selector",
                "loss_name",
                "n_basis",
                "regularization_strength",
                "penalty_power",
                "weighted_validation_nrmse_radial",
                "weighted_validation_nrmse_3d",
            ]
        ]
    )
    print("Maximum-basis closure")
    display(closure_df)
    print("Saved files")
    for path in sorted(OUTPUT_DIR.glob(f"{OUTPUT_PREFIX}_*")):
        print(f"  {path.name}")
    print("=" * 128)


if __name__ == "__main__":
    main()
