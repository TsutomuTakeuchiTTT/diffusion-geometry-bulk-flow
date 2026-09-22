# Paper I: Mock-3 primary-unweighted mode-number convergence to 4096
#
# 実行順序:
#   1. mock23_current_samples_common_single_cell_v2.py
#   2. このファイル全体
#
# 目的:
#   Mock-3 の primary estimator（unweighted alpha_DM=0 geometry + unweighted loss）
#   について、まず spectral capacity だけを 4096 modes まで監査する。
#
# 重要:
#   この第一段階では lambda=1e-2 を固定する。
#   これは v5 の全3 loss families が n_b=512 で lambda=1e-2 を選択し、
#   mode-capacity dependence を最小計算量で先に確定するためである。
#   plateau が現れた後、その近傍で lambda grid と3 loss familiesを再評価する。
#
# 出力:
#   CSV / JSON / PDF / PNG
#   各 seed 終了時に checkpoint を保存し、再実行時に完了 seed を再利用する。

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import matplotlib
SHOW_PLOTS = os.environ.get("PAPER1_MOCK3_HIGHMODE_SHOW_PLOTS", "1") != "0"
if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import linalg
from scipy.sparse import diags
from scipy.sparse.linalg import ArpackNoConvergence, eigsh

# ============================================================
# 0. 共通セルの確認
# ============================================================

_REQUIRED_COMMON_SYMBOLS = [
    "_load_mock1",
    "_load_survey",
    "_build_base_kernel",
    "_nystrom_extend",
    "_effective_sample_size",
    "velocity_metrics",
]

_missing_common = [name for name in _REQUIRED_COMMON_SYMBOLS if name not in globals()]
if _missing_common:
    raise RuntimeError(
        "共通セルが未実行です。先に\n"
        "  mock23_current_samples_common_single_cell_v2.py\n"
        "を実行してください。\n"
        f"不足している関数: {_missing_common}"
    )

# ============================================================
# 1. パス
# ============================================================

ROOT_DIR = Path(
    os.environ.get(
        "COSMIC_DIPOLE_MOCK_DATA_ROOT",
        'mock_data',
    )
)

MOCK1_FILE = ROOT_DIR / "mock1_complete_sphere" / "mock1.npz"
MOCK3_FILE = ROOT_DIR / "mock3_inhomogeneous_survey" / "mock3.npz"

SOURCE_DIR = ROOT_DIR / "mock3_octant_aware_loss_optimization_v4"
SOURCE_PREDICTIONS_FILE = SOURCE_DIR / "mock3_octant_aware_predictions.npz"
LEGACY_GEOMETRY_FILE = SOURCE_DIR / "cache" / "unweighted_alpha0_geometry_m512.npz"

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_MOCK3_HIGHMODE_OUTPUT",
        str(ROOT_DIR / "mock3_primary_mode_convergence_to_4096_v1"),
    )
)
CACHE_DIR = OUTPUT_DIR / "cache"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
for directory in (OUTPUT_DIR, CACHE_DIR, CHECKPOINT_DIR):
    directory.mkdir(parents=True, exist_ok=True)

GEOMETRY_CACHE_FILE = CACHE_DIR / "mock3_unweighted_alpha0_geometry_m4096.npz"
OUTPUT_PREFIX = "mock3_primary_mode_convergence_to_4096"

# ============================================================
# 2. 設定
# ============================================================

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)

GRAPH_NEIGHBORS = 48
GRAPH_BANDWIDTH_NEIGHBOR = 16
GRAPH_BANDWIDTH_MULTIPLIER = 1.0
ALPHA_DM = 0.0

MAX_BASIS = 4096
BASIS_SIZES = [
    512,
    1024,
    1536,
    2048,
    2560,
    3072,
    3584,
    4096,
]

# 第一段階では固定。
REGULARIZATION_STRENGTH = 1.0e-2
NUMERICAL_RIDGE = 1.0e-10

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

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

PRIMARY_METRIC_GAMMA = 1.0
PRIMARY_METRIC_CAP = 50.0
MIN_SELECTION_PROBABILITY = 1.0e-4
WEIGHT_CLIP_QUANTILE = 0.995

PRACTICAL_RELATIVE_THRESHOLD = 0.02
PRACTICAL_CONSECUTIVE_STEPS = 2

SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 180

# ============================================================
# 3. Utility
# ============================================================

def configure_font():
    matplotlib.rcParams.update(
        {
            "font.family": "DejaVu Sans",
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
    return "DejaVu Sans"


def stable_hash(payload) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:16]


def effective_sample_size(weights):
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights * weights))
    if denom <= 0.0:
        return math.nan
    return float(np.sum(weights) ** 2 / denom)


def tempered_inverse_selection_weights(
    probability,
    gamma,
    floor,
    max_weight,
    quantile,
):
    probability = np.asarray(probability, dtype=np.float64)
    safe = np.maximum(probability, float(floor))

    if float(gamma) == 0.0:
        weights = np.ones_like(safe)
        cap = 1.0
        raw_max = 1.0
    else:
        raw = safe ** (-float(gamma))
        cap = min(float(max_weight), float(np.quantile(raw, float(quantile))))
        weights = np.minimum(raw, cap)
        weights /= np.mean(weights)
        raw_max = float(np.max(raw))

    return weights, {
        "gamma": float(gamma),
        "cap": float(cap),
        "raw_weight_max": raw_max,
        "normalized_weight_max": float(np.max(weights)),
        "effective_sample_size": effective_sample_size(weights),
        "effective_sample_fraction": float(
            effective_sample_size(weights) / weights.size
        ),
    }


def make_octant_stratified_split(
    region_id,
    train_fraction,
    validation_fraction,
    seed,
):
    region_id = np.asarray(region_id, dtype=np.int64)
    rng = np.random.default_rng(int(seed))

    train_parts = []
    validation_parts = []
    test_parts = []

    for region in range(8):
        idx = np.flatnonzero(region_id == region)
        idx = rng.permutation(idx)

        n_train = int(round(float(train_fraction) * idx.size))
        n_validation = int(round(float(validation_fraction) * idx.size))

        if n_train + n_validation >= idx.size:
            n_validation = max(1, idx.size - n_train - 1)

        train_parts.append(idx[:n_train])
        validation_parts.append(idx[n_train:n_train + n_validation])
        test_parts.append(idx[n_train + n_validation:])

    return {
        "train": np.sort(np.concatenate(train_parts)),
        "validation": np.sort(np.concatenate(validation_parts)),
        "test": np.sort(np.concatenate(test_parts)),
    }


def save_figure(fig, stem):
    pdf_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.pdf"
    png_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.png"

    if SAVE_PDF:
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.05)
    if SAVE_PNG:
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


# ============================================================
# 4. Geometry
# ============================================================

def build_geometry(base_kernel, max_basis):
    kernel = base_kernel["kernel"]
    n = kernel.shape[0]
    quadrature = np.ones(n, dtype=np.float64)

    q = np.asarray(kernel @ quadrature).ravel()
    if np.any(q <= 0.0):
        raise ValueError("Non-positive q in geometry construction.")

    q_factor = np.power(q, -ALPHA_DM)
    kernel_alpha = (
        diags(q_factor) @ kernel @ diags(q_factor)
    ).tocsr()

    d = np.asarray(kernel_alpha @ quadrature).ravel()
    if np.any(d <= 0.0):
        raise ValueError("Non-positive d in geometry construction.")

    stationary = quadrature * d
    stationary /= np.sum(stationary)

    symmetric_factor = np.sqrt(quadrature / d)
    symmetric_operator = (
        diags(symmetric_factor)
        @ kernel_alpha
        @ diags(symmetric_factor)
    )
    symmetric_operator = (
        0.5 * (symmetric_operator + symmetric_operator.T)
    ).tocsr()

    k = min(int(max_basis), n - 2)

    print(
        f"[Geometry] computing {k:,} eigenfunctions for {n:,} survey points...",
        flush=True,
    )
    start = time.perf_counter()

    try:
        eigenvalues, eigenvectors = eigsh(
            symmetric_operator,
            k=k,
            which="LA",
            tol=1.0e-8,
            maxiter=max(5000, 20 * n),
        )
    except ArpackNoConvergence as exc:
        n_converged = 0 if exc.eigenvalues is None else len(exc.eigenvalues)
        raise RuntimeError(
            f"ARPACK did not converge: {n_converged}/{k} eigenpairs."
        ) from exc

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)

    eigenfunctions = eigenvectors / np.sqrt(stationary)[:, None]

    for col in range(eigenfunctions.shape[1]):
        pivot = int(np.argmax(np.abs(eigenfunctions[:, col])))
        if eigenfunctions[pivot, col] < 0.0:
            eigenfunctions[:, col] *= -1.0

    generator_raw = np.maximum(1.0 - eigenvalues, 0.0)

    # 512-mode legacy analysisとの penalty scale の連続性を保つ。
    anchor = generator_raw[1:min(512, generator_raw.size)]
    anchor = anchor[anchor > 0.0]
    generator_scale = float(np.median(anchor))
    if not np.isfinite(generator_scale) or generator_scale <= 0.0:
        raise ValueError("Invalid generator normalization scale.")
    generator = generator_raw / generator_scale

    elapsed = time.perf_counter() - start
    print(
        f"[Geometry] completed in {elapsed:.1f} s; "
        f"generator scale={generator_scale:.6e}",
        flush=True,
    )

    return {
        "eigenvalues": eigenvalues,
        "generator_eigenvalues": generator,
        "eigenfunctions": eigenfunctions,
        "quadrature_weights": quadrature,
        "q": q,
        "d": d,
        "stationary": stationary,
        "alpha": float(ALPHA_DM),
        "generator_scale": generator_scale,
    }


def save_geometry(path, geometry):
    np.savez_compressed(
        path,
        eigenvalues=geometry["eigenvalues"],
        generator_eigenvalues=geometry["generator_eigenvalues"],
        eigenfunctions=geometry["eigenfunctions"],
        quadrature_weights=geometry["quadrature_weights"],
        q=geometry["q"],
        d=geometry["d"],
        stationary=geometry["stationary"],
        alpha=np.array(geometry["alpha"]),
        generator_scale=np.array(geometry["generator_scale"]),
    )


def load_geometry(path):
    with np.load(path, allow_pickle=False) as data:
        return {
            "eigenvalues": np.asarray(data["eigenvalues"], dtype=np.float64),
            "generator_eigenvalues": np.asarray(
                data["generator_eigenvalues"], dtype=np.float64
            ),
            "eigenfunctions": np.asarray(
                data["eigenfunctions"], dtype=np.float64
            ),
            "quadrature_weights": np.asarray(
                data["quadrature_weights"], dtype=np.float64
            ),
            "q": np.asarray(data["q"], dtype=np.float64),
            "d": np.asarray(data["d"], dtype=np.float64),
            "stationary": np.asarray(data["stationary"], dtype=np.float64),
            "alpha": float(np.asarray(data["alpha"]).item()),
            "generator_scale": (
                float(np.asarray(data["generator_scale"]).item())
                if "generator_scale" in data.files
                else math.nan
            ),
        }


def geometry_continuity_audit(new_geometry, legacy_path):
    if not legacy_path.is_file():
        return pd.DataFrame(
            [{"quantity": "legacy_geometry_available", "value": 0.0}]
        )

    with np.load(legacy_path, allow_pickle=False) as old:
        old_eigenvalues = np.asarray(old["eigenvalues"], dtype=np.float64)
        old_eigenfunctions = np.asarray(old["eigenfunctions"], dtype=np.float64)
        old_stationary = np.asarray(old["stationary"], dtype=np.float64)

    m = min(512, old_eigenvalues.size, new_geometry["eigenvalues"].size)
    m_corr = min(64, m)

    stationary_diff = float(
        np.max(np.abs(old_stationary - new_geometry["stationary"]))
    )
    eigenvalue_diff = float(
        np.max(
            np.abs(
                old_eigenvalues[:m]
                - new_geometry["eigenvalues"][:m]
            )
        )
    )

    correlations = []
    weights = new_geometry["stationary"]
    for col in range(m_corr):
        a = old_eigenfunctions[:, col]
        b = new_geometry["eigenfunctions"][:, col]
        numerator = float(np.sum(weights * a * b))
        denom = math.sqrt(
            float(np.sum(weights * a * a))
            * float(np.sum(weights * b * b))
        )
        correlations.append(abs(numerator / denom))

    return pd.DataFrame(
        [
            {"quantity": "legacy_geometry_available", "value": 1.0},
            {"quantity": "stationary_max_abs_difference", "value": stationary_diff},
            {"quantity": "first_512_markov_eigenvalue_max_abs_difference", "value": eigenvalue_diff},
            {"quantity": "first_64_mode_correlation_median", "value": float(np.median(correlations))},
            {"quantity": "first_64_mode_correlation_minimum", "value": float(np.min(correlations))},
        ]
    )


# ============================================================
# 5. Regression
# ============================================================

def weighted_gram_and_rhs(phi, velocity, weights):
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.mean(weights)
    sqrt_w = np.sqrt(weights)

    weighted_phi = phi * sqrt_w[:, None]
    weighted_velocity = velocity * sqrt_w[:, None]

    gram = (weighted_phi.T @ weighted_phi) / phi.shape[0]
    rhs = (weighted_phi.T @ weighted_velocity) / phi.shape[0]
    return gram, rhs


def solve_from_normal_equations(
    gram,
    rhs,
    generator_eigenvalues,
    regularization_strength,
):
    system = np.array(gram, copy=True)
    diagonal = np.diag_indices_from(system)
    system[diagonal] += (
        float(regularization_strength)
        * np.maximum(
            generator_eigenvalues[: system.shape[0]],
            NUMERICAL_RIDGE,
        )
        + NUMERICAL_RIDGE
    )

    try:
        factor = linalg.cho_factor(
            system,
            lower=True,
            check_finite=False,
        )
        return linalg.cho_solve(
            factor,
            rhs,
            check_finite=False,
        )
    except linalg.LinAlgError:
        return linalg.solve(
            system,
            rhs,
            assume_a="sym",
            check_finite=False,
        )


def nrmse(true, pred, weights=None):
    return float(
        velocity_metrics(
            true,
            pred,
            weights=weights,
        )["normalized_rmse"]
    )


def octant_nrmse_summary(
    true,
    pred,
    regions,
    weights=None,
):
    values = []

    for region in range(8):
        mask = regions == region
        if np.sum(mask) < 5:
            continue

        local_weights = None if weights is None else weights[mask]
        values.append(
            nrmse(
                true[mask],
                pred[mask],
                local_weights,
            )
        )

    values = np.asarray(values, dtype=np.float64)

    if values.size == 0:
        return {
            "octant_nrmse_std": math.nan,
            "octant_nrmse_worst": math.nan,
            "octant_nrmse_range": math.nan,
        }

    return {
        "octant_nrmse_std": float(np.std(values, ddof=0)),
        "octant_nrmse_worst": float(np.max(values)),
        "octant_nrmse_range": float(np.max(values) - np.min(values)),
    }


# ============================================================
# 6. Practical selection
# ============================================================

def practical_selection(aggregate_df):
    selected = aggregate_df.sort_values("n_basis").reset_index(drop=True)

    minimum_index = int(selected["validation_nrmse_mean"].idxmin())
    minimum_basis = int(selected.loc[minimum_index, "n_basis"])
    minimum_mean = float(selected.loc[minimum_index, "validation_nrmse_mean"])
    minimum_sem = float(selected.loc[minimum_index, "validation_nrmse_sem"])
    threshold = minimum_mean + minimum_sem

    one_se_candidates = selected.loc[
        selected["validation_nrmse_mean"] <= threshold,
        "n_basis",
    ]
    one_se_basis = int(one_se_candidates.min())

    plateau_basis = None
    values = selected["validation_nrmse_mean"].to_numpy(dtype=float)
    bases = selected["n_basis"].to_numpy(dtype=int)

    reductions = np.full(values.size, np.nan)
    reductions[1:] = (values[:-1] - values[1:]) / values[:-1]

    for index in range(values.size):
        future = reductions[index + 1:index + 1 + PRACTICAL_CONSECUTIVE_STEPS]
        if future.size < PRACTICAL_CONSECUTIVE_STEPS:
            continue
        if np.all(future < PRACTICAL_RELATIVE_THRESHOLD):
            plateau_basis = int(bases[index])
            break

    return {
        "minimum_validation_basis": minimum_basis,
        "minimum_validation_mean": minimum_mean,
        "minimum_validation_sem": minimum_sem,
        "one_standard_error_threshold": threshold,
        "one_standard_error_basis": one_se_basis,
        "practical_plateau_basis": (
            int(plateau_basis) if plateau_basis is not None else math.nan
        ),
        "plateau_found": plateau_basis is not None,
    }


# ============================================================
# 7. Input
# ============================================================

font_name = configure_font()

print("=" * 118)
print("Paper I: Mock-3 primary-unweighted mode-number convergence to 4096")
print("=" * 118)
print(f"Python / NumPy / SciPy : {platform.python_version()} / {np.__version__} / {scipy.__version__}")
print(f"Matplotlib font         : {font_name}")
print(f"Mock-1                  : {MOCK1_FILE}")
print(f"Mock-3                  : {MOCK3_FILE}")
print(f"Source predictions      : {SOURCE_PREDICTIONS_FILE}")
print(f"Output                   : {OUTPUT_DIR}")
print(f"Basis sizes              : {BASIS_SIZES}")
print(f"Fixed lambda             : {REGULARIZATION_STRENGTH:g}")
print(f"Label split seeds        : {len(LABEL_SPLIT_SEEDS)}")
print("=" * 118)

complete = _load_mock1(MOCK1_FILE)
survey = _load_survey(MOCK3_FILE, "mock3")

with np.load(SOURCE_PREDICTIONS_FILE, allow_pickle=False) as data:
    required = {
        "query_global_indices",
        "truth_survey",
        "truth_query",
        "selection_probability_region_survey",
        "selection_probability_global_survey",
        "selection_probability_region_query",
        "selection_probability_global_query",
        "query_region",
        "region_primary_support",
        "common_strict_support",
        "octant_boundary_distance_query",
    }
    missing = required.difference(data.files)
    if missing:
        raise KeyError(
            "Source prediction NPZ is missing keys: "
            f"{sorted(missing)}"
        )

    query_global = np.asarray(data["query_global_indices"], dtype=np.int64)
    truth_survey = np.asarray(data["truth_survey"], dtype=np.float64)
    truth_query = np.asarray(data["truth_query"], dtype=np.float64)
    probability_region_survey = np.asarray(
        data["selection_probability_region_survey"],
        dtype=np.float64,
    )
    probability_global_survey = np.asarray(
        data["selection_probability_global_survey"],
        dtype=np.float64,
    )
    query_region = np.asarray(data["query_region"], dtype=np.int64)
    region_primary_support = np.asarray(
        data["region_primary_support"],
        dtype=bool,
    )
    common_strict_support = np.asarray(
        data["common_strict_support"],
        dtype=bool,
    )
    octant_boundary_distance_query = np.asarray(
        data["octant_boundary_distance_query"],
        dtype=np.float64,
    )

n_survey = survey["pos"].shape[0]
if truth_survey.shape != (n_survey, 3):
    raise ValueError(f"truth_survey shape mismatch: {truth_survey.shape}")

centered_survey = survey["pos"] - SPHERE_CENTER[None, :]
centered_query = complete["pos"][query_global] - SPHERE_CENTER[None, :]

primary_metric_weights, primary_weight_info = (
    tempered_inverse_selection_weights(
        probability_region_survey,
        PRIMARY_METRIC_GAMMA,
        MIN_SELECTION_PROBABILITY,
        PRIMARY_METRIC_CAP,
        WEIGHT_CLIP_QUANTILE,
    )
)

print(
    "Primary octant-aware evaluation ESS: "
    f"{primary_weight_info['effective_sample_size']:.1f}",
    flush=True,
)
print(
    f"Survey/query/region-primary: "
    f"{n_survey:,}/{query_global.size:,}/{int(np.sum(region_primary_support)):,}",
    flush=True,
)

# ============================================================
# 8. Geometry and Nyström
# ============================================================

base_kernel = _build_base_kernel(
    centered_survey,
    GRAPH_NEIGHBORS,
    GRAPH_BANDWIDTH_NEIGHBOR,
    GRAPH_BANDWIDTH_MULTIPLIER,
)

if GEOMETRY_CACHE_FILE.is_file():
    geometry = load_geometry(GEOMETRY_CACHE_FILE)
    geometry_from_cache = True
    print(f"[Geometry] loaded cache: {GEOMETRY_CACHE_FILE}", flush=True)
else:
    geometry = build_geometry(base_kernel, MAX_BASIS)
    save_geometry(GEOMETRY_CACHE_FILE, geometry)
    geometry_from_cache = False
    print(f"[Geometry] saved cache: {GEOMETRY_CACHE_FILE}", flush=True)

if geometry["eigenfunctions"].shape[1] < max(BASIS_SIZES):
    raise ValueError("Geometry cache has too few modes.")

audit_df = geometry_continuity_audit(
    geometry,
    LEGACY_GEOMETRY_FILE,
)
audit_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_continuity_audit.csv",
    index=False,
)
print("-" * 118)
print("4096-vs-legacy geometry continuity audit")
print(audit_df.to_string(index=False))
print("-" * 118)

print(
    f"[Nyström] extending {centered_query.shape[0]:,} query points...",
    flush=True,
)
query_extension = _nystrom_extend(
    centered_query,
    centered_survey,
    base_kernel,
    geometry,
)
query_eigenfunctions = np.asarray(
    query_extension["eigenfunctions"],
    dtype=np.float64,
)
print("[Nyström] completed.", flush=True)

boundary_scale = float(np.median(base_kernel["rho"]))
near_boundary_mask = (
    region_primary_support
    & (
        octant_boundary_distance_query
        <= boundary_scale
    )
)
far_boundary_mask = (
    region_primary_support
    & (
        octant_boundary_distance_query
        >= 2.0 * boundary_scale
    )
)

# ============================================================
# 9. Seed loop with checkpoints
# ============================================================

run_token = stable_hash(
    {
        "basis_sizes": BASIS_SIZES,
        "lambda": REGULARIZATION_STRENGTH,
        "seeds": LABEL_SPLIT_SEEDS,
        "graph_neighbors": GRAPH_NEIGHBORS,
        "graph_bandwidth_neighbor": GRAPH_BANDWIDTH_NEIGHBOR,
        "alpha_dm": ALPHA_DM,
        "source_predictions": str(SOURCE_PREDICTIONS_FILE),
    }
)

all_seed_frames = []

for replicate, seed in enumerate(LABEL_SPLIT_SEEDS, start=1):
    checkpoint = (
        CHECKPOINT_DIR
        / f"{OUTPUT_PREFIX}_{run_token}_seed_{seed}.csv"
    )

    if checkpoint.is_file():
        print(
            f"[Checkpoint] replicate={replicate}, seed={seed}",
            flush=True,
        )
        all_seed_frames.append(pd.read_csv(checkpoint))
        continue

    print("-" * 118)
    print(
        f"Replicate {replicate}/{len(LABEL_SPLIT_SEEDS)}: "
        f"label_seed={seed}",
        flush=True,
    )
    start_seed = time.perf_counter()

    split = make_octant_stratified_split(
        survey["region_id"],
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        seed,
    )

    train_indices = split["train"]
    validation_indices = split["validation"]
    test_indices = split["test"]
    fit_indices = np.sort(
        np.concatenate([train_indices, validation_indices])
    )

    max_basis = max(BASIS_SIZES)

    # Validation equations: train only.
    train_phi_max = geometry["eigenfunctions"][
        train_indices,
        :max_basis,
    ]
    train_gram, train_rhs = weighted_gram_and_rhs(
        train_phi_max,
        truth_survey[train_indices],
        np.ones(train_indices.size),
    )

    # Final fit equations: train + validation.
    fit_phi_max = geometry["eigenfunctions"][
        fit_indices,
        :max_basis,
    ]
    fit_gram, fit_rhs = weighted_gram_and_rhs(
        fit_phi_max,
        truth_survey[fit_indices],
        np.ones(fit_indices.size),
    )

    seed_records = []

    for n_basis in BASIS_SIZES:
        p = int(n_basis)
        start_basis = time.perf_counter()

        # Validation fit: training labels only.
        coefficients_validation = solve_from_normal_equations(
            train_gram[:p, :p],
            train_rhs[:p],
            geometry["generator_eigenvalues"],
            REGULARIZATION_STRENGTH,
        )

        validation_prediction = (
            geometry["eigenfunctions"][
                validation_indices,
                :p,
            ]
            @ coefficients_validation
        )

        validation_parent_nrmse = nrmse(
            truth_survey[validation_indices],
            validation_prediction,
            primary_metric_weights[validation_indices],
        )
        validation_unweighted_nrmse = nrmse(
            truth_survey[validation_indices],
            validation_prediction,
        )

        # Configuration固定後の final fit。
        coefficients_final = solve_from_normal_equations(
            fit_gram[:p, :p],
            fit_rhs[:p],
            geometry["generator_eigenvalues"],
            REGULARIZATION_STRENGTH,
        )

        test_prediction = (
            geometry["eigenfunctions"][
                test_indices,
                :p,
            ]
            @ coefficients_final
        )
        query_prediction = (
            query_eigenfunctions[:, :p]
            @ coefficients_final
        )

        test_parent_metrics = velocity_metrics(
            truth_survey[test_indices],
            test_prediction,
            weights=primary_metric_weights[test_indices],
        )
        test_unweighted_metrics = velocity_metrics(
            truth_survey[test_indices],
            test_prediction,
        )

        observed_octant = octant_nrmse_summary(
            truth_survey[test_indices],
            test_prediction,
            survey["region_id"][test_indices],
            primary_metric_weights[test_indices],
        )

        query_primary_metrics = velocity_metrics(
            truth_query[region_primary_support],
            query_prediction[region_primary_support],
        )
        query_octant = octant_nrmse_summary(
            truth_query[region_primary_support],
            query_prediction[region_primary_support],
            query_region[region_primary_support],
            None,
        )

        query_common_nrmse = (
            nrmse(
                truth_query[common_strict_support],
                query_prediction[common_strict_support],
            )
            if np.sum(common_strict_support) >= 5
            else math.nan
        )
        query_near_nrmse = (
            nrmse(
                truth_query[near_boundary_mask],
                query_prediction[near_boundary_mask],
            )
            if np.sum(near_boundary_mask) >= 5
            else math.nan
        )
        query_far_nrmse = (
            nrmse(
                truth_query[far_boundary_mask],
                query_prediction[far_boundary_mask],
            )
            if np.sum(far_boundary_mask) >= 5
            else math.nan
        )

        seed_records.append(
            {
                "replicate": int(replicate),
                "label_seed": int(seed),
                "n_basis": int(n_basis),
                "regularization_strength": float(REGULARIZATION_STRENGTH),
                "n_train": int(train_indices.size),
                "n_validation": int(validation_indices.size),
                "n_test": int(test_indices.size),
                "validation_parent_nrmse": float(validation_parent_nrmse),
                "validation_unweighted_nrmse": float(validation_unweighted_nrmse),
                "parent_test_nrmse": float(
                    test_parent_metrics["normalized_rmse"]
                ),
                "unweighted_test_nrmse": float(
                    test_unweighted_metrics["normalized_rmse"]
                ),
                "test_median_direction_error_deg": float(
                    test_unweighted_metrics["median_direction_error_deg"]
                ),
                "test_vector_gain": float(
                    test_unweighted_metrics["vector_gain_through_origin"]
                ),
                "observed_octant_nrmse_std": float(
                    observed_octant["octant_nrmse_std"]
                ),
                "observed_worst_octant_nrmse": float(
                    observed_octant["octant_nrmse_worst"]
                ),
                "observed_octant_nrmse_range": float(
                    observed_octant["octant_nrmse_range"]
                ),
                "query_region_nrmse": float(
                    query_primary_metrics["normalized_rmse"]
                ),
                "query_region_median_direction_error_deg": float(
                    query_primary_metrics["median_direction_error_deg"]
                ),
                "query_region_vector_gain": float(
                    query_primary_metrics["vector_gain_through_origin"]
                ),
                "query_octant_nrmse_std": float(
                    query_octant["octant_nrmse_std"]
                ),
                "query_worst_octant_nrmse": float(
                    query_octant["octant_nrmse_worst"]
                ),
                "query_octant_nrmse_range": float(
                    query_octant["octant_nrmse_range"]
                ),
                "query_common_nrmse": float(query_common_nrmse),
                "query_near_boundary_nrmse": float(query_near_nrmse),
                "query_far_boundary_nrmse": float(query_far_nrmse),
            }
        )

        print(
            f"  n_basis={n_basis:4d}: "
            f"val_parent={validation_parent_nrmse:.6f}, "
            f"test_parent={test_parent_metrics['normalized_rmse']:.6f}, "
            f"query={query_primary_metrics['normalized_rmse']:.6f}, "
            f"elapsed={time.perf_counter() - start_basis:.1f} s",
            flush=True,
        )

    seed_df = pd.DataFrame(seed_records)
    seed_df.to_csv(checkpoint, index=False)
    all_seed_frames.append(seed_df)

    print(
        f"[Seed completed] elapsed={time.perf_counter() - start_seed:.1f} s",
        flush=True,
    )

runs_df = pd.concat(all_seed_frames, ignore_index=True)
runs_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_runs.csv",
    index=False,
)

# ============================================================
# 10. Aggregate and practical selection
# ============================================================

aggregate_records = []

for n_basis, group in runs_df.groupby("n_basis", sort=True):
    val = group["validation_parent_nrmse"].to_numpy(dtype=float)
    aggregate_records.append(
        {
            "n_basis": int(n_basis),
            "n_seeds": int(group.shape[0]),
            "validation_nrmse_mean": float(np.mean(val)),
            "validation_nrmse_std": float(np.std(val, ddof=1)),
            "validation_nrmse_sem": float(
                np.std(val, ddof=1) / math.sqrt(group.shape[0])
            ),
            "parent_test_nrmse_mean": float(group["parent_test_nrmse"].mean()),
            "parent_test_nrmse_std": float(group["parent_test_nrmse"].std(ddof=1)),
            "unweighted_test_nrmse_mean": float(group["unweighted_test_nrmse"].mean()),
            "unweighted_test_nrmse_std": float(group["unweighted_test_nrmse"].std(ddof=1)),
            "query_region_nrmse_mean": float(group["query_region_nrmse"].mean()),
            "query_region_nrmse_std": float(group["query_region_nrmse"].std(ddof=1)),
            "observed_octant_std_mean": float(group["observed_octant_nrmse_std"].mean()),
            "observed_octant_std_std": float(group["observed_octant_nrmse_std"].std(ddof=1)),
            "query_octant_std_mean": float(group["query_octant_nrmse_std"].mean()),
            "query_octant_std_std": float(group["query_octant_nrmse_std"].std(ddof=1)),
            "query_worst_octant_mean": float(group["query_worst_octant_nrmse"].mean()),
            "query_worst_octant_std": float(group["query_worst_octant_nrmse"].std(ddof=1)),
        }
    )

aggregate_df = pd.DataFrame(aggregate_records).sort_values("n_basis").reset_index(drop=True)
aggregate_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_aggregate.csv",
    index=False,
)

selection = practical_selection(aggregate_df)
selection_df = pd.DataFrame([selection])
selection_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_practical_selection.csv",
    index=False,
)

# Plateau diagnostics
plateau_rows = []
for index in range(1, aggregate_df.shape[0]):
    prev = aggregate_df.iloc[index - 1]
    cur = aggregate_df.iloc[index]
    reduction = (
        float(prev["validation_nrmse_mean"])
        - float(cur["validation_nrmse_mean"])
    ) / float(prev["validation_nrmse_mean"])

    plateau_rows.append(
        {
            "previous_basis": int(prev["n_basis"]),
            "n_basis": int(cur["n_basis"]),
            "adjacent_validation_relative_reduction": float(reduction),
            "adjacent_validation_reduction_percent": float(100.0 * reduction),
        }
    )

plateau_df = pd.DataFrame(plateau_rows)
plateau_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_plateau_diagnostics.csv",
    index=False,
)

# ============================================================
# 11. Figures
# ============================================================

x = aggregate_df["n_basis"].to_numpy()

fig, axes = plt.subplots(
    2,
    2,
    figsize=(11.6, 8.6),
    constrained_layout=True,
)

ax = axes[0, 0]
ax.errorbar(
    x,
    aggregate_df["validation_nrmse_mean"],
    yerr=aggregate_df["validation_nrmse_std"],
    marker="o",
    capsize=3,
    label="Validation parent-weighted",
)
ax.errorbar(
    x,
    aggregate_df["parent_test_nrmse_mean"],
    yerr=aggregate_df["parent_test_nrmse_std"],
    marker="s",
    capsize=3,
    label="Held-out parent-weighted",
)
ax.set_xlabel(r"Retained modes $n_{\rm b}$")
ax.set_ylabel(r"$\mathrm{NRMSE}_{3\mathrm D}$")
ax.set_title("(a) Observed-sample convergence")
ax.legend()

ax = axes[0, 1]
ax.errorbar(
    x,
    aggregate_df["query_region_nrmse_mean"],
    yerr=aggregate_df["query_region_nrmse_std"],
    marker="o",
    capsize=3,
)
ax.set_xlabel(r"Retained modes $n_{\rm b}$")
ax.set_ylabel(r"Region-primary query $\mathrm{NRMSE}_{3\mathrm D}$")
ax.set_title("(b) Complete-query convergence")

ax = axes[1, 0]
ax.errorbar(
    x,
    aggregate_df["observed_octant_std_mean"],
    yerr=aggregate_df["observed_octant_std_std"],
    marker="o",
    capsize=3,
    label="Observed octant dispersion",
)
ax.errorbar(
    x,
    aggregate_df["query_octant_std_mean"],
    yerr=aggregate_df["query_octant_std_std"],
    marker="s",
    capsize=3,
    label="Query octant dispersion",
)
ax.set_xlabel(r"Retained modes $n_{\rm b}$")
ax.set_ylabel(r"Octant dispersion in $\mathrm{NRMSE}_{3\mathrm D}$")
ax.set_title("(c) Directional heterogeneity")
ax.legend()

ax = axes[1, 1]
if not plateau_df.empty:
    ax.plot(
        plateau_df["n_basis"],
        plateau_df["adjacent_validation_reduction_percent"],
        marker="o",
    )
ax.axhline(
    100.0 * PRACTICAL_RELATIVE_THRESHOLD,
    linestyle="--",
    linewidth=1.2,
    label="2% practical threshold",
)
ax.axhline(0.0, linestyle=":", linewidth=1.0)
ax.set_xlabel(r"Retained modes $n_{\rm b}$")
ax.set_ylabel("Adjacent validation improvement [%]")
ax.set_title("(d) Practical-plateau diagnostic")
ax.legend()

composite_pdf, composite_png = save_figure(
    fig,
    "composite",
)

# Separate plateau figure
fig, ax = plt.subplots(figsize=(7.5, 4.8))
if not plateau_df.empty:
    ax.plot(
        plateau_df["n_basis"],
        plateau_df["adjacent_validation_reduction_percent"],
        marker="o",
    )
ax.axhline(
    100.0 * PRACTICAL_RELATIVE_THRESHOLD,
    linestyle="--",
    linewidth=1.2,
    label="2% practical threshold",
)
ax.axhline(0.0, linestyle=":", linewidth=1.0)

if bool(selection["plateau_found"]):
    selected_basis = int(selection["practical_plateau_basis"])
    ax.axvline(
        selected_basis,
        linestyle="-.",
        linewidth=1.2,
        label=f"Practical basis = {selected_basis}",
    )

ax.set_xlabel(r"Retained modes $n_{\rm b}$")
ax.set_ylabel("Adjacent validation improvement [%]")
ax.legend()
plateau_pdf, plateau_png = save_figure(
    fig,
    "plateau_diagnostic",
)

# ============================================================
# 12. Summary
# ============================================================

summary = {
    "scope": (
        "Mock-3 first-stage high-mode convergence for the primary "
        "unweighted loss with fixed unweighted alpha_DM=0 geometry"
    ),
    "fixed_regularization_strength": REGULARIZATION_STRENGTH,
    "basis_sizes": BASIS_SIZES,
    "label_split_seeds": LABEL_SPLIT_SEEDS,
    "primary_metric_weight": primary_weight_info,
    "practical_selection": selection,
    "geometry_cache": str(GEOMETRY_CACHE_FILE),
    "geometry_loaded_from_cache": bool(geometry_from_cache),
    "geometry_continuity_audit": audit_df.to_dict(orient="records"),
    "outputs": {
        "runs_csv": str(
            OUTPUT_DIR / f"{OUTPUT_PREFIX}_runs.csv"
        ),
        "aggregate_csv": str(
            OUTPUT_DIR / f"{OUTPUT_PREFIX}_aggregate.csv"
        ),
        "practical_selection_csv": str(
            OUTPUT_DIR / f"{OUTPUT_PREFIX}_practical_selection.csv"
        ),
        "plateau_diagnostics_csv": str(
            OUTPUT_DIR / f"{OUTPUT_PREFIX}_plateau_diagnostics.csv"
        ),
        "composite_pdf": str(composite_pdf),
        "composite_png": str(composite_png),
        "plateau_pdf": str(plateau_pdf),
        "plateau_png": str(plateau_png),
    },
}

summary_path = OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json"
summary_path.write_text(
    json.dumps(summary, ensure_ascii=False, indent=2),
    encoding="utf-8",
)

print("=" * 118)
print("Mock-3 primary-unweighted mode-number convergence completed")
print("=" * 118)
print("Practical selection")
print(selection_df.to_string(index=False))
print("-" * 118)
print("Aggregate")
print(aggregate_df.to_string(index=False))
print("-" * 118)
print(f"runs_csv              : {OUTPUT_DIR / f'{OUTPUT_PREFIX}_runs.csv'}")
print(f"aggregate_csv         : {OUTPUT_DIR / f'{OUTPUT_PREFIX}_aggregate.csv'}")
print(f"practical_selection   : {OUTPUT_DIR / f'{OUTPUT_PREFIX}_practical_selection.csv'}")
print(f"composite_pdf         : {composite_pdf}")
print(f"composite_png         : {composite_png}")
print(f"plateau_pdf           : {plateau_pdf}")
print(f"plateau_png           : {plateau_png}")
print(f"summary_json          : {summary_path}")
print("=" * 118)
