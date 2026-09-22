# Mock-2: continuation of mode-number convergence to 4096 modes
#
# このファイル全体を、次の共通セルを実行した後、Jupyter Notebookの
# 新しい空のコードセルへ貼り付けて実行してください。
#
#   mock23_current_samples_common_single_cell_v2.py
#
# 目的:
#   1. 2048 modesまで改善が継続したMock-2を、4096 modesまで追計算する。
#   2. n_b=2048を接続点として旧v1出力と連結し、512--4096 modesの
#      validation/test/query convergenceを一つの系列として評価する。
#   3. unweighted lossとinverse-selection-weighted lossを、同じparent-weighted
#      validation criterionで10通りのlabel splitについて再選択する。
#   4. matched hyperparametersにより、loss weighting自体の効果を分離する。
#   5. one-standard-error ruleと2% practical-plateau criterionを出力する。
#
# 重要:
#   - query labelsはmodel selectionに使用しません。
#   - geometryは全モデルで同一のunweighted alpha=0 constructionです。
#   - generator spectrumの全体scaleは、旧512-mode解析との連続性を保つため、
#     先頭511個の非定数modesのraw generator eigenvaluesの中央値で固定します。

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import warnings
from pathlib import Path

import matplotlib

SHOW_PLOTS = os.environ.get(
    "PAPER1_MOCK2_HIGHMODE_SHOW_PLOTS",
    "1",
) != "0"

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.font_manager as fm
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
    "_make_split",
    "velocity_metrics",
    "_file_signature",
    "_stable_signature",
]

_missing_common = [
    name
    for name in _REQUIRED_COMMON_SYMBOLS
    if name not in globals()
]

if _missing_common:
    raise RuntimeError(
        "共通セルが未実行です。先に\n"
        "  mock23_current_samples_common_single_cell_v2.py\n"
        "を実行してください。\n"
        f"不足している関数: {_missing_common}"
    )


# ============================================================
# 1. 固定パス
# ============================================================

ROOT_DIR = Path(
    os.environ.get(
        "COSMIC_DIPOLE_MOCK_DATA_ROOT",
        'mock_data',
    )
)

MOCK1_FILE = (
    ROOT_DIR
    / "mock1_complete_sphere"
    / "mock1.npz"
)

MOCK2_FILE = (
    ROOT_DIR
    / "mock2_schechter_selection"
    / "mock2.npz"
)

OPTIMIZATION_DIR = (
    ROOT_DIR
    / "mock2_selection_correction_optimization_v4"
)

OPTIMIZATION_PREDICTIONS_FILE = (
    OPTIMIZATION_DIR
    / "mock2_selection_optimization_predictions.npz"
)

OPTIMIZATION_EVALUATION_FILE = (
    OPTIMIZATION_DIR
    / "mock2_selection_optimization_evaluation.csv"
)

LEGACY_GEOMETRY_FILE = (
    OPTIMIZATION_DIR
    / "final_geometry_cache"
    / "geometry_a0p00_g0p00_c1p0_m512.npz"
)

LEGACY_ROBUSTNESS_DIR = (
    ROOT_DIR
    / "mock2_loss_weight_seed_robustness_v5"
)

LEGACY_RUNS_FILE = (
    LEGACY_ROBUSTNESS_DIR
    / "mock2_loss_weight_seed_robustness_runs.csv"
)

# 2048-modeまでの前段計算。追計算では2048を重複して再評価し、
# それ未満のrowsはこのdirectoryから引き継ぐ。
PREVIOUS_OUTPUT_DIR = (
    ROOT_DIR
    / "mock2_loss_weight_high_mode_revalidation_v1"
)
PREVIOUS_VALIDATION_FILE = (
    PREVIOUS_OUTPUT_DIR
    / "mock2_loss_weight_high_mode_validation.csv"
)
PREVIOUS_BASIS_RUNS_FILE = (
    PREVIOUS_OUTPUT_DIR
    / "mock2_loss_weight_high_mode_basis_runs.csv"
)
PREVIOUS_MATCHED_FILE = (
    PREVIOUS_OUTPUT_DIR
    / "mock2_loss_weight_high_mode_matched_grid.csv"
)
PREVIOUS_GEOMETRY_FILE = (
    PREVIOUS_OUTPUT_DIR
    / "cache"
    / "mock2_unweighted_alpha0_geometry_m2048.npz"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_MOCK2_HIGHMODE_OUTPUT",
        str(
            ROOT_DIR
            / "mock2_loss_weight_mode_convergence_to_4096_v2"
        ),
    )
)

CACHE_DIR = OUTPUT_DIR / "cache"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for directory in [
    OUTPUT_DIR,
    CACHE_DIR,
    CHECKPOINT_DIR,
]:
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

GEOMETRY_CACHE_FILE = (
    CACHE_DIR
    / "mock2_unweighted_alpha0_geometry_m4096.npz"
)

OUTPUT_PREFIX = "mock2_loss_weight_mode_convergence_to_4096"


# ============================================================
# 2. Controlled-experiment設定
# ============================================================

SPHERE_CENTER = np.array(
    [250.0, 250.0, 250.0],
    dtype=np.float64,
)

GRAPH_NEIGHBORS = 48
GRAPH_BANDWIDTH_NEIGHBOR = 16
GRAPH_BANDWIDTH_MULTIPLIER = 1.0
ALPHA_DM = 0.0
MAX_BASIS = int(
    os.environ.get(
        "PAPER1_MOCK2_MAX_BASIS",
        "4096",
    )
)
EIGEN_SOLVER_SEED = 20260820

# 旧512-mode penalty scaleと揃える。
GENERATOR_REFERENCE_NONCONSTANT_MODES = 511

_DEFAULT_BASIS_SIZES = [
    2048,
    2560,
    3072,
    3584,
    4096,
]
BASIS_SIZES = [
    int(value)
    for value in os.environ.get(
        "PAPER1_MOCK2_BASIS_SIZES",
        ",".join(str(value) for value in _DEFAULT_BASIS_SIZES),
    ).split(",")
    if value.strip()
]
BASIS_SIZES = sorted(
    set(
        value
        for value in BASIS_SIZES
        if 1 <= value <= MAX_BASIS
    )
)
if not BASIS_SIZES:
    raise ValueError("BASIS_SIZES is empty after applying MAX_BASIS.")
if 2048 <= MAX_BASIS and 2048 not in BASIS_SIZES:
    BASIS_SIZES.append(2048)
    BASIS_SIZES.sort()
if BASIS_SIZES[0] != 2048:
    raise ValueError(
        "The continuation grid must start at n_basis=2048 so that "
        "the v1 and v2 calculations have an explicit overlap point."
    )

# 旧候補を保持しつつ、10^{-2}近傍を精密化する。
_DEFAULT_REGULARIZATION_STRENGTHS = [
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
    1.0,
]
REGULARIZATION_STRENGTHS = [
    float(value)
    for value in os.environ.get(
        "PAPER1_MOCK2_LAMBDA_GRID",
        ",".join(
            f"{value:.12g}"
            for value in _DEFAULT_REGULARIZATION_STRENGTHS
        ),
    ).split(",")
    if value.strip()
]

# 旧解析のcontinuity auditでのみ使用する強正則化候補。
LEGACY_REGULARIZATION_STRENGTHS = [
    1.0e-4,
    1.0e-2,
    1.0,
]

NUMERICAL_RIDGE = 1.0e-10
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

_DEFAULT_LABEL_SPLIT_SEEDS = [
    20261202,
    20261212,
    20261222,
    20270102,
    20270112,
    20270122,
    20270202,
    20270212,
    20270222,
    20270302,
]
_N_SEEDS = int(
    os.environ.get(
        "PAPER1_MOCK2_N_SEEDS",
        str(len(_DEFAULT_LABEL_SPLIT_SEEDS)),
    )
)
LABEL_SPLIT_SEEDS = _DEFAULT_LABEL_SPLIT_SEEDS[:_N_SEEDS]

MIN_SELECTION_PROBABILITY = 1.0e-4
WEIGHT_CLIP_QUANTILE = 0.995
LOSS_WEIGHT_GAMMA = 1.0
LOSS_WEIGHT_CAP = 50.0
EVALUATION_WEIGHT_GAMMA = 1.0
EVALUATION_WEIGHT_CAP = 50.0

PRACTICAL_RELATIVE_THRESHOLD = 0.02
PRACTICAL_CONSECUTIVE_STEPS = 2

# Mock-1 fiducial targetでvalidationから選ばれたbasisを、外部に事前指定した
# matched comparisonとして使う。
PRIMARY_MATCHED_BASIS = int(
    os.environ.get(
        "PAPER1_MOCK2_PRIMARY_MATCHED_BASIS",
        "1280",
    )
)
if PRIMARY_MATCHED_BASIS > MAX_BASIS:
    PRIMARY_MATCHED_BASIS = BASIS_SIZES[-1]
PRIMARY_MATCHED_REGULARIZATION = 1.0e-2

_DEFAULT_MATCHED_BASIS_SIZES = [
    2048,
    2560,
    3072,
    3584,
    4096,
]
MATCHED_BASIS_SIZES = [
    int(value)
    for value in os.environ.get(
        "PAPER1_MOCK2_MATCHED_BASIS_SIZES",
        ",".join(
            str(value)
            for value in _DEFAULT_MATCHED_BASIS_SIZES
        ),
    ).split(",")
    if value.strip()
]
MATCHED_BASIS_SIZES = sorted(
    set(
        value
        for value in MATCHED_BASIS_SIZES
        if 1 <= value <= MAX_BASIS
    )
)
if (
    PRIMARY_MATCHED_BASIS not in MATCHED_BASIS_SIZES
    and not PREVIOUS_MATCHED_FILE.is_file()
):
    MATCHED_BASIS_SIZES.append(PRIMARY_MATCHED_BASIS)
    MATCHED_BASIS_SIZES.sort()
MATCHED_REGULARIZATION = 1.0e-2

SEED_BOOTSTRAP_REPLICATES = int(
    os.environ.get(
        "PAPER1_MOCK2_BOOTSTRAP_REPLICATES",
        "50000",
    )
)
SEED_BOOTSTRAP_SEED = 20270421

SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 300

RUN_TOKEN = hashlib.sha256(
    json.dumps(
        {
            "max_basis": int(MAX_BASIS),
            "basis_sizes": BASIS_SIZES,
            "regularization_strengths": REGULARIZATION_STRENGTHS,
            "matched_basis_sizes": MATCHED_BASIS_SIZES,
            "matched_regularization": float(MATCHED_REGULARIZATION),
            "label_split_seeds": LABEL_SPLIT_SEEDS,
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()[:12]


# ============================================================
# 3. 補助関数
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



def save_figure(
    figure: plt.Figure,
    stem: str,
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
        figure.savefig(
            pdf_path,
            bbox_inches="tight",
            pad_inches=0.05,
        )
    if png_path is not None:
        figure.savefig(
            png_path,
            dpi=PNG_DPI,
            bbox_inches="tight",
            pad_inches=0.05,
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(figure)

    return pdf_path, png_path



def tempered_inverse_selection_weights(
    probability,
    gamma: float,
    floor: float,
    max_weight: float,
    quantile: float,
) -> tuple[np.ndarray, dict]:
    probability = np.asarray(
        probability,
        dtype=np.float64,
    )
    probability_safe = np.maximum(
        probability,
        float(floor),
    )

    if float(gamma) == 0.0:
        normalized = np.ones_like(
            probability_safe,
            dtype=np.float64,
        )
    else:
        raw = np.power(
            probability_safe,
            -float(gamma),
        )
        empirical_cap = float(
            np.quantile(
                raw,
                float(quantile),
            )
        )
        cap = min(
            float(max_weight),
            empirical_cap,
        )
        normalized = np.minimum(
            raw,
            cap,
        )
        normalized /= np.mean(normalized)

    denominator = float(
        np.sum(
            np.square(normalized)
        )
    )
    effective_sample_size = (
        float(
            np.square(
                np.sum(normalized)
            )
            / denominator
        )
        if denominator > 0.0
        else math.nan
    )

    return normalized, {
        "gamma": float(gamma),
        "probability_min": float(
            np.min(probability_safe)
        ),
        "probability_median": float(
            np.median(probability_safe)
        ),
        "normalized_weight_max": float(
            np.max(normalized)
        ),
        "effective_sample_size": effective_sample_size,
        "effective_sample_fraction": float(
            effective_sample_size
            / normalized.size
        ),
    }



def geometry_signature() -> str:
    return _stable_signature(
        {
            "survey": _file_signature(
                MOCK2_FILE
            ),
            "graph_neighbors": int(
                GRAPH_NEIGHBORS
            ),
            "graph_bandwidth_neighbor": int(
                GRAPH_BANDWIDTH_NEIGHBOR
            ),
            "graph_bandwidth_multiplier": float(
                GRAPH_BANDWIDTH_MULTIPLIER
            ),
            "alpha_dm": float(ALPHA_DM),
            "max_basis": int(MAX_BASIS),
            "eigen_solver_seed": int(
                EIGEN_SOLVER_SEED
            ),
            "generator_reference_nonconstant_modes": int(
                GENERATOR_REFERENCE_NONCONSTANT_MODES
            ),
            "version": 2,
        }
    )



def build_or_load_geometry(
    base_kernel,
) -> tuple[dict, bool]:
    signature = geometry_signature()

    if GEOMETRY_CACHE_FILE.is_file():
        try:
            with np.load(
                GEOMETRY_CACHE_FILE,
                allow_pickle=False,
            ) as data:
                if (
                    str(
                        data["signature"].item()
                    )
                    == signature
                ):
                    geometry = {
                        "eigenvalues": np.asarray(
                            data["eigenvalues"],
                            dtype=np.float64,
                        ),
                        "generator_eigenvalues": np.asarray(
                            data["generator_eigenvalues"],
                            dtype=np.float64,
                        ),
                        "eigenfunctions": np.asarray(
                            data["eigenfunctions"],
                            dtype=np.float64,
                        ),
                        "quadrature_weights": np.asarray(
                            data["quadrature_weights"],
                            dtype=np.float64,
                        ),
                        "q": np.asarray(
                            data["q"],
                            dtype=np.float64,
                        ),
                        "d": np.asarray(
                            data["d"],
                            dtype=np.float64,
                        ),
                        "stationary": np.asarray(
                            data["stationary"],
                            dtype=np.float64,
                        ),
                        "alpha": float(
                            data["alpha"].item()
                        ),
                        "generator_scale": float(
                            data["generator_scale"].item()
                        ),
                    }
                    return geometry, True
        except Exception as exc:
            warnings.warn(
                "High-mode geometry cache could not be used; "
                f"rebuilding: {exc}"
            )

    kernel = base_kernel["kernel"]
    n = kernel.shape[0]
    quadrature_weights = np.ones(
        n,
        dtype=np.float64,
    )

    q = np.asarray(
        kernel @ quadrature_weights
    ).ravel()
    if np.any(q <= 0.0):
        raise ValueError(
            "Non-positive kernel density q."
        )

    q_factor = np.power(
        q,
        -float(ALPHA_DM),
    )
    kernel_alpha = (
        diags(q_factor)
        @ kernel
        @ diags(q_factor)
    ).tocsr()

    d = np.asarray(
        kernel_alpha @ quadrature_weights
    ).ravel()
    if np.any(d <= 0.0):
        raise ValueError(
            "Non-positive row normalization d."
        )

    stationary = quadrature_weights * d
    stationary /= np.sum(stationary)

    symmetric_factor = np.sqrt(
        quadrature_weights / d
    )
    symmetric_operator = (
        diags(symmetric_factor)
        @ kernel_alpha
        @ diags(symmetric_factor)
    )
    symmetric_operator = (
        0.5
        * (
            symmetric_operator
            + symmetric_operator.T
        )
    ).tocsr()

    k = min(
        int(MAX_BASIS),
        n - 2,
    )
    rng = np.random.default_rng(
        EIGEN_SOLVER_SEED
    )

    print(
        f"[Geometry] Computing {k} diffusion eigenfunctions..."
    )
    start = time.perf_counter()

    try:
        eigenvalues, eigenvectors = eigsh(
            symmetric_operator,
            k=k,
            which="LA",
            v0=rng.normal(size=n),
            tol=1.0e-8,
            maxiter=max(
                10000,
                30 * n,
            ),
        )
    except ArpackNoConvergence as exc:
        n_converged = (
            0
            if exc.eigenvalues is None
            else len(exc.eigenvalues)
        )
        raise RuntimeError(
            "ARPACK did not converge: "
            f"{n_converged}/{k} eigenpairs."
        ) from exc

    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(
        eigenvalues[order],
        dtype=np.float64,
    )
    eigenvectors = np.asarray(
        eigenvectors[:, order],
        dtype=np.float64,
    )

    eigenfunctions = (
        eigenvectors
        / np.sqrt(stationary)[:, None]
    )

    for column in range(
        eigenfunctions.shape[1]
    ):
        pivot = int(
            np.argmax(
                np.abs(
                    eigenfunctions[:, column]
                )
            )
        )
        if eigenfunctions[pivot, column] < 0.0:
            eigenfunctions[:, column] *= -1.0

    raw_generator = np.maximum(
        1.0 - eigenvalues,
        0.0,
    )
    stop = min(
        1
        + GENERATOR_REFERENCE_NONCONSTANT_MODES,
        raw_generator.size,
    )
    reference_positive = raw_generator[1:stop]
    reference_positive = reference_positive[
        reference_positive > 0.0
    ]
    if reference_positive.size == 0:
        raise ValueError(
            "Could not define the fixed generator scale."
        )
    generator_scale = float(
        np.median(reference_positive)
    )
    generator_eigenvalues = (
        raw_generator
        / generator_scale
    )

    audit_modes = min(
        128,
        eigenfunctions.shape[1],
    )
    gram = (
        eigenfunctions[:, :audit_modes].T
        @ (
            stationary[:, None]
            * eigenfunctions[:, :audit_modes]
        )
    )
    orthogonality_rms = float(
        np.sqrt(
            np.mean(
                np.square(
                    gram
                    - np.eye(audit_modes)
                )
            )
        )
    )

    elapsed = time.perf_counter() - start
    print(
        "[Geometry] completed: "
        f"elapsed={elapsed:.1f} s, "
        f"low-mode orthogonality RMS={orthogonality_rms:.3e}, "
        f"generator scale={generator_scale:.6e}"
    )

    geometry = {
        "eigenvalues": eigenvalues,
        "generator_eigenvalues": generator_eigenvalues,
        "eigenfunctions": eigenfunctions,
        "quadrature_weights": quadrature_weights,
        "q": q,
        "d": d,
        "stationary": stationary,
        "alpha": float(ALPHA_DM),
        "generator_scale": generator_scale,
    }

    # 巨大配列の保存時間を抑えるため非圧縮NPZを用いる。
    np.savez(
        GEOMETRY_CACHE_FILE,
        signature=np.array(signature),
        eigenvalues=eigenvalues,
        generator_eigenvalues=(
            generator_eigenvalues
        ),
        eigenfunctions=eigenfunctions,
        quadrature_weights=(
            quadrature_weights
        ),
        q=q,
        d=d,
        stationary=stationary,
        alpha=np.array(ALPHA_DM),
        generator_scale=np.array(
            generator_scale
        ),
    )

    return geometry, False



def audit_against_legacy(
    geometry: dict,
) -> pd.DataFrame:
    records = []

    if not PREVIOUS_GEOMETRY_FILE.is_file():
        records.append(
            {
                "quantity": "legacy_geometry_available",
                "value": 0.0,
            }
        )
        return pd.DataFrame(records)

    with np.load(
        PREVIOUS_GEOMETRY_FILE,
        allow_pickle=False,
    ) as data:
        legacy_eigenvalues = np.asarray(
            data["eigenvalues"],
            dtype=np.float64,
        )
        legacy_generator = np.asarray(
            data["generator_eigenvalues"],
            dtype=np.float64,
        )
        legacy_eigenfunctions = np.asarray(
            data["eigenfunctions"],
            dtype=np.float64,
        )
        legacy_stationary = np.asarray(
            data["stationary"],
            dtype=np.float64,
        )

    n_compare = min(
        legacy_eigenvalues.size,
        geometry["eigenvalues"].size,
    )
    n_modes = min(
        64,
        legacy_eigenfunctions.shape[1],
        geometry["eigenfunctions"].shape[1],
    )

    weighted_cross = (
        legacy_eigenfunctions[:, :n_modes].T
        @ (
            legacy_stationary[:, None]
            * geometry["eigenfunctions"][:, :n_modes]
        )
    )
    diagonal_correlations = np.abs(
        np.diag(weighted_cross)
    )
    singular_values = np.linalg.svd(
        weighted_cross,
        compute_uv=False,
    )

    records.extend(
        [
            {
                "quantity": "legacy_geometry_available",
                "value": 1.0,
            },
            {
                "quantity": "stationary_max_abs_difference",
                "value": float(
                    np.max(
                        np.abs(
                            geometry["stationary"]
                            - legacy_stationary
                        )
                    )
                ),
            },
            {
                "quantity": "markov_eigenvalue_max_abs_difference",
                "value": float(
                    np.max(
                        np.abs(
                            geometry["eigenvalues"][:n_compare]
                            - legacy_eigenvalues[:n_compare]
                        )
                    )
                ),
            },
            {
                "quantity": "generator_eigenvalue_max_abs_difference",
                "value": float(
                    np.max(
                        np.abs(
                            geometry["generator_eigenvalues"][:n_compare]
                            - legacy_generator[:n_compare]
                        )
                    )
                ),
            },
            {
                "quantity": "first_64_mode_correlation_median",
                "value": float(
                    np.median(
                        diagonal_correlations
                    )
                ),
            },
            {
                "quantity": "first_64_mode_correlation_minimum",
                "value": float(
                    np.min(
                        diagonal_correlations
                    )
                ),
            },
            {
                "quantity": "first_64_subspace_minimum_singular_value",
                "value": float(
                    np.min(singular_values)
                ),
            },
        ]
    )

    return pd.DataFrame(records)



def prepare_normal_equations(
    phi: np.ndarray,
    truth: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )
    weights = weights / np.mean(weights)
    sqrt_weight = np.sqrt(weights)
    weighted_phi = phi * sqrt_weight[:, None]
    weighted_truth = truth * sqrt_weight[:, None]

    gram = (
        weighted_phi.T
        @ weighted_phi
    ) / phi.shape[0]
    rhs = (
        weighted_phi.T
        @ weighted_truth
    ) / phi.shape[0]

    return gram, rhs



def solve_from_normal_equations(
    gram: np.ndarray,
    rhs: np.ndarray,
    generator_eigenvalues: np.ndarray,
    regularization_strength: float,
) -> np.ndarray:
    system = np.array(
        gram,
        copy=True,
    )
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
        warnings.warn(
            "Cholesky factorization failed; using a symmetric direct solve."
        )
        return linalg.solve(
            system,
            rhs,
            assume_a="sym",
            check_finite=False,
        )



def candidate_lambdas(
    n_basis: int,
) -> list[float]:
    values = list(REGULARIZATION_STRENGTHS)
    if int(n_basis) == 512:
        values.extend(
            LEGACY_REGULARIZATION_STRENGTHS
        )
    return sorted(set(float(value) for value in values))



def evaluate_prediction(
    truth: np.ndarray,
    prediction: np.ndarray,
    weights: np.ndarray | None = None,
) -> dict:
    metrics = velocity_metrics(
        truth,
        prediction,
        weights=weights,
    )
    return {
        key: float(value)
        if np.isscalar(value)
        else value
        for key, value in metrics.items()
    }



def bootstrap_mean_difference(
    values,
    n_bootstrap: int,
    seed: int,
) -> dict:
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    if values.ndim != 1 or values.size == 0:
        raise ValueError(
            "Bootstrap values must be a nonempty one-dimensional array."
        )

    rng = np.random.default_rng(seed)
    sampled_indices = rng.integers(
        0,
        values.size,
        size=(
            int(n_bootstrap),
            values.size,
        ),
    )
    sampled_means = np.mean(
        values[sampled_indices],
        axis=1,
    )

    return {
        "mean_difference": float(
            np.mean(values)
        ),
        "std_difference": float(
            np.std(values, ddof=1)
        ),
        "ci_low": float(
            np.quantile(
                sampled_means,
                0.025,
            )
        ),
        "ci_high": float(
            np.quantile(
                sampled_means,
                0.975,
            )
        ),
        "wins": int(
            np.sum(values < 0.0)
        ),
        "ties": int(
            np.sum(values == 0.0)
        ),
        "losses": int(
            np.sum(values > 0.0)
        ),
    }



def practical_selection_for_model(
    aggregate: pd.DataFrame,
    model_name: str,
) -> dict:
    selected = (
        aggregate.loc[
            aggregate["model"] == model_name
        ]
        .sort_values("n_basis")
        .reset_index(drop=True)
    )

    if selected.empty:
        raise ValueError(
            f"No aggregate rows for model={model_name}."
        )

    minimum_index = int(
        selected[
            "validation_nrmse_mean"
        ].idxmin()
    )
    minimum_row = selected.loc[
        minimum_index
    ]
    one_se_threshold = float(
        minimum_row[
            "validation_nrmse_mean"
        ]
        + minimum_row[
            "validation_nrmse_sem"
        ]
    )
    one_se_candidates = selected.loc[
        selected[
            "validation_nrmse_mean"
        ]
        <= one_se_threshold
    ]
    one_se_basis = int(
        one_se_candidates[
            "n_basis"
        ].min()
    )

    means = selected[
        "validation_nrmse_mean"
    ].to_numpy(dtype=np.float64)
    bases = selected[
        "n_basis"
    ].to_numpy(dtype=np.int64)

    plateau_basis = int(bases[-1])
    plateau_found = False

    for index in range(
        0,
        len(bases)
        - PRACTICAL_CONSECUTIVE_STEPS,
    ):
        improvements = []
        for step in range(
            1,
            PRACTICAL_CONSECUTIVE_STEPS
            + 1,
        ):
            previous = means[
                index + step - 1
            ]
            current = means[
                index + step
            ]
            improvements.append(
                float(
                    (previous - current)
                    / previous
                )
            )
        if all(
            improvement
            < PRACTICAL_RELATIVE_THRESHOLD
            for improvement in improvements
        ):
            plateau_basis = int(
                bases[index]
            )
            plateau_found = True
            break

    return {
        "model": model_name,
        "minimum_validation_basis": int(
            minimum_row["n_basis"]
        ),
        "minimum_validation_mean": float(
            minimum_row[
                "validation_nrmse_mean"
            ]
        ),
        "minimum_validation_sem": float(
            minimum_row[
                "validation_nrmse_sem"
            ]
        ),
        "one_standard_error_threshold": one_se_threshold,
        "one_standard_error_basis": one_se_basis,
        "practical_plateau_basis": plateau_basis,
        "plateau_found": bool(
            plateau_found
        ),
        "plateau_relative_threshold": float(
            PRACTICAL_RELATIVE_THRESHOLD
        ),
        "plateau_consecutive_steps": int(
            PRACTICAL_CONSECUTIVE_STEPS
        ),
    }



def checkpoint_paths(
    seed: int,
) -> dict[str, Path]:
    prefix = (
        CHECKPOINT_DIR
        / f"seed_{int(seed)}_{RUN_TOKEN}"
    )
    return {
        "validation": Path(
            f"{prefix}_validation.csv"
        ),
        "basis_runs": Path(
            f"{prefix}_basis_runs.csv"
        ),
        "matched": Path(
            f"{prefix}_matched.csv"
        ),
        "legacy_audit": Path(
            f"{prefix}_legacy_audit.csv"
        ),
    }


# ============================================================
# 4. 入力と固定geometry
# ============================================================

font_name = configure_font()

print("=" * 116)
print("Paper I: Mock-2 mode-number convergence to 4096 modes")
print("=" * 116)
print(f"Python / NumPy / SciPy : {platform.python_version()} / {np.__version__} / {scipy.__version__}")
print(f"Matplotlib font         : {font_name}")
print(f"Mock-1                  : {MOCK1_FILE}")
print(f"Mock-2                  : {MOCK2_FILE}")
print(f"Optimization predictions: {OPTIMIZATION_PREDICTIONS_FILE}")
print(f"Previous output         : {PREVIOUS_OUTPUT_DIR}")
print(f"Output                  : {OUTPUT_DIR}")
print(f"Basis sizes             : {BASIS_SIZES}")
print(f"Lambda grid             : {REGULARIZATION_STRENGTHS}")
print(f"Label split seeds       : {len(LABEL_SPLIT_SEEDS)}")

for required_path in [
    MOCK1_FILE,
    MOCK2_FILE,
    OPTIMIZATION_PREDICTIONS_FILE,
    PREVIOUS_VALIDATION_FILE,
    PREVIOUS_BASIS_RUNS_FILE,
    PREVIOUS_MATCHED_FILE,
    PREVIOUS_GEOMETRY_FILE,
]:
    if not required_path.is_file():
        raise FileNotFoundError(required_path)

complete = _load_mock1(MOCK1_FILE)
survey = _load_survey(
    MOCK2_FILE,
    "mock2",
)

with np.load(
    OPTIMIZATION_PREDICTIONS_FILE,
    allow_pickle=False,
) as data:
    required_keys = {
        "query_global_indices",
        "truth_query",
        "truth_survey",
        "selection_probability_survey",
        "selection_probability_query",
        "primary_query_mask",
    }
    missing_keys = required_keys.difference(
        data.files
    )
    if missing_keys:
        raise KeyError(
            "Optimization prediction NPZ is missing keys: "
            f"{sorted(missing_keys)}"
        )

    query_global = np.asarray(
        data["query_global_indices"],
        dtype=np.int64,
    )
    truth_query = np.asarray(
        data["truth_query"],
        dtype=np.float64,
    )
    truth_survey = np.asarray(
        data["truth_survey"],
        dtype=np.float64,
    )
    probability_survey = np.asarray(
        data["selection_probability_survey"],
        dtype=np.float64,
    )
    probability_query = np.asarray(
        data["selection_probability_query"],
        dtype=np.float64,
    )
    primary_query_mask = np.asarray(
        data["primary_query_mask"],
        dtype=bool,
    )

n_survey = survey["pos"].shape[0]
n_query = query_global.size

if truth_survey.shape != (n_survey, 3):
    raise ValueError(
        f"truth_survey shape mismatch: {truth_survey.shape}"
    )
if probability_survey.shape != (n_survey,):
    raise ValueError(
        f"probability_survey shape mismatch: {probability_survey.shape}"
    )
if truth_query.shape != (n_query, 3):
    raise ValueError(
        f"truth_query shape mismatch: {truth_query.shape}"
    )
if probability_query.shape != (n_query,):
    raise ValueError(
        f"probability_query shape mismatch: {probability_query.shape}"
    )
if primary_query_mask.shape != (n_query,):
    raise ValueError(
        f"primary_query_mask shape mismatch: {primary_query_mask.shape}"
    )
if np.sum(primary_query_mask) < 5:
    raise ValueError(
        "Primary query mask contains fewer than five points."
    )

centered_survey = (
    survey["pos"]
    - SPHERE_CENTER[None, :]
)
centered_query = (
    complete["pos"][query_global]
    - SPHERE_CENTER[None, :]
)

naive_loss_weights = np.ones(
    n_survey,
    dtype=np.float64,
)
loss_weights, loss_weight_info = (
    tempered_inverse_selection_weights(
        probability_survey,
        LOSS_WEIGHT_GAMMA,
        MIN_SELECTION_PROBABILITY,
        LOSS_WEIGHT_CAP,
        WEIGHT_CLIP_QUANTILE,
    )
)
evaluation_weights, evaluation_weight_info = (
    tempered_inverse_selection_weights(
        probability_survey,
        EVALUATION_WEIGHT_GAMMA,
        MIN_SELECTION_PROBABILITY,
        EVALUATION_WEIGHT_CAP,
        WEIGHT_CLIP_QUANTILE,
    )
)

print(
    "Selection-weighted loss: "
    f"gamma={LOSS_WEIGHT_GAMMA:g}, "
    f"cap={LOSS_WEIGHT_CAP:g}, "
    f"ESS={loss_weight_info['effective_sample_size']:.1f}"
)
print(
    "Parent-weighted evaluation: "
    f"gamma={EVALUATION_WEIGHT_GAMMA:g}, "
    f"cap={EVALUATION_WEIGHT_CAP:g}, "
    f"ESS={evaluation_weight_info['effective_sample_size']:.1f}"
)
print(
    f"Survey/query/primary-query: {n_survey:,}/{n_query:,}/{int(np.sum(primary_query_mask)):,}"
)

base_kernel = _build_base_kernel(
    centered_survey,
    GRAPH_NEIGHBORS,
    GRAPH_BANDWIDTH_NEIGHBOR,
    GRAPH_BANDWIDTH_MULTIPLIER,
)

geometry, geometry_from_cache = (
    build_or_load_geometry(
        base_kernel
    )
)
print(
    "Geometry source          : "
    + (
        "cache"
        if geometry_from_cache
        else "new eigensystem"
    )
)

geometry_audit_df = audit_against_legacy(
    geometry
)
print("-" * 116)
print("4096-vs-2048 geometry continuity audit")
print(geometry_audit_df.to_string(index=False))

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

if query_eigenfunctions.shape != (
    n_query,
    geometry["eigenfunctions"].shape[1],
):
    raise ValueError(
        "Nyström extension shape mismatch."
    )

finite_by_mode = np.all(
    np.isfinite(query_eigenfunctions),
    axis=0,
)
if not np.all(
    finite_by_mode[: max(BASIS_SIZES)]
):
    first_invalid = int(
        np.flatnonzero(
            ~finite_by_mode
        )[0]
    )
    raise FloatingPointError(
        "Nyström extension became non-finite at mode "
        f"{first_invalid}."
    )


# ============================================================
# 5. Seed反復
# ============================================================

model_definitions = {
    "naive": naive_loss_weights,
    "loss_weighted": loss_weights,
}

validation_frames = []
basis_run_frames = []
matched_frames = []
legacy_audit_frames = []

legacy_runs_df = (
    pd.read_csv(LEGACY_RUNS_FILE)
    if LEGACY_RUNS_FILE.is_file()
    else None
)

for replicate, seed in enumerate(
    LABEL_SPLIT_SEEDS,
    start=1,
):
    paths = checkpoint_paths(seed)

    if all(path.is_file() for path in paths.values()):
        print(
            f"[Checkpoint] replicate={replicate}, seed={seed}"
        )
        validation_frames.append(
            pd.read_csv(paths["validation"])
        )
        basis_run_frames.append(
            pd.read_csv(paths["basis_runs"])
        )
        matched_frames.append(
            pd.read_csv(paths["matched"])
        )
        legacy_audit_frames.append(
            pd.read_csv(paths["legacy_audit"])
        )
        continue

    print("-" * 116)
    print(
        f"Replicate {replicate}/{len(LABEL_SPLIT_SEEDS)}: label_seed={seed}"
    )

    split = _make_split(
        n_survey,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        int(seed),
    )
    train_indices = split["train"]
    validation_indices = split["validation"]
    test_indices = split["test"]
    fit_indices = np.sort(
        np.concatenate(
            [
                train_indices,
                validation_indices,
            ]
        )
    )

    seed_validation_records = []
    seed_basis_run_records = []
    seed_matched_records = []
    seed_legacy_audit_records = []

    fit_equations = {}

    for model_name, model_loss_weights in (
        model_definitions.items()
    ):
        start = time.perf_counter()

        phi_train_max = geometry[
            "eigenfunctions"
        ][
            train_indices,
            :MAX_BASIS,
        ]
        train_gram, train_rhs = (
            prepare_normal_equations(
                phi_train_max,
                truth_survey[train_indices],
                model_loss_weights[train_indices],
            )
        )

        phi_fit_max = geometry[
            "eigenfunctions"
        ][
            fit_indices,
            :MAX_BASIS,
        ]
        fit_gram, fit_rhs = (
            prepare_normal_equations(
                phi_fit_max,
                truth_survey[fit_indices],
                model_loss_weights[fit_indices],
            )
        )
        fit_equations[model_name] = (
            fit_gram,
            fit_rhs,
        )

        phi_validation_max = geometry[
            "eigenfunctions"
        ][
            validation_indices,
            :MAX_BASIS,
        ]
        phi_test_max = geometry[
            "eigenfunctions"
        ][
            test_indices,
            :MAX_BASIS,
        ]

        model_validation_rows = []

        for n_basis in BASIS_SIZES:
            p = int(n_basis)
            gram = train_gram[:p, :p]
            rhs = train_rhs[:p]
            phi_validation = (
                phi_validation_max[:, :p]
            )

            for regularization in (
                candidate_lambdas(n_basis)
            ):
                coefficients = (
                    solve_from_normal_equations(
                        gram,
                        rhs,
                        geometry[
                            "generator_eigenvalues"
                        ],
                        regularization,
                    )
                )
                prediction = (
                    phi_validation
                    @ coefficients
                )
                parent_metrics = (
                    evaluate_prediction(
                        truth_survey[
                            validation_indices
                        ],
                        prediction,
                        evaluation_weights[
                            validation_indices
                        ],
                    )
                )
                unweighted_metrics = (
                    evaluate_prediction(
                        truth_survey[
                            validation_indices
                        ],
                        prediction,
                    )
                )

                row = {
                    "replicate": int(replicate),
                    "label_seed": int(seed),
                    "model": model_name,
                    "n_basis": int(n_basis),
                    "regularization_strength": float(
                        regularization
                    ),
                    "parent_validation_nrmse": float(
                        parent_metrics[
                            "normalized_rmse"
                        ]
                    ),
                    "unweighted_validation_nrmse": float(
                        unweighted_metrics[
                            "normalized_rmse"
                        ]
                    ),
                    "parent_validation_direction_deg": float(
                        parent_metrics[
                            "median_direction_error_deg"
                        ]
                    ),
                    "parent_validation_vector_gain": float(
                        parent_metrics[
                            "vector_gain_through_origin"
                        ]
                    ),
                }
                seed_validation_records.append(row)
                model_validation_rows.append(row)

        model_validation_df = pd.DataFrame(
            model_validation_rows
        )

        # 各basisでvalidationによりlambdaを選び、test/queryを一度だけ評価する。
        for n_basis in BASIS_SIZES:
            basis_rows = model_validation_df.loc[
                model_validation_df[
                    "n_basis"
                ] == int(n_basis)
            ].sort_values(
                [
                    "parent_validation_nrmse",
                    "unweighted_validation_nrmse",
                    "regularization_strength",
                ]
            )
            best_row = basis_rows.iloc[0]
            regularization = float(
                best_row[
                    "regularization_strength"
                ]
            )
            p = int(n_basis)
            coefficients = (
                solve_from_normal_equations(
                    fit_gram[:p, :p],
                    fit_rhs[:p],
                    geometry[
                        "generator_eigenvalues"
                    ],
                    regularization,
                )
            )
            observed_prediction = (
                phi_test_max[:, :p]
                @ coefficients
            )
            query_prediction = (
                query_eigenfunctions[:, :p]
                @ coefficients
            )

            parent_test = evaluate_prediction(
                truth_survey[test_indices],
                observed_prediction,
                evaluation_weights[test_indices],
            )
            unweighted_test = evaluate_prediction(
                truth_survey[test_indices],
                observed_prediction,
            )
            query_primary = evaluate_prediction(
                truth_query[primary_query_mask],
                query_prediction[primary_query_mask],
            )

            seed_basis_run_records.append(
                {
                    "replicate": int(replicate),
                    "label_seed": int(seed),
                    "model": model_name,
                    "n_basis": int(n_basis),
                    "selected_regularization": regularization,
                    "parent_validation_nrmse": float(
                        best_row[
                            "parent_validation_nrmse"
                        ]
                    ),
                    "unweighted_validation_nrmse": float(
                        best_row[
                            "unweighted_validation_nrmse"
                        ]
                    ),
                    **{
                        f"parent_{key}": value
                        for key, value in parent_test.items()
                    },
                    **{
                        f"unweighted_{key}": value
                        for key, value in unweighted_test.items()
                    },
                    **{
                        f"query_primary_{key}": value
                        for key, value in query_primary.items()
                    },
                }
            )

        # 旧v5 resultと、同一basis/lambdaにおける512-mode continuityを監査する。
        if legacy_runs_df is not None:
            legacy_row_df = legacy_runs_df.loc[
                (
                    legacy_runs_df[
                        "protocol"
                    ] == "retuned"
                )
                & (
                    legacy_runs_df[
                        "replicate"
                    ] == int(replicate)
                )
                & (
                    legacy_runs_df[
                        "model"
                    ] == model_name
                )
            ]
            if legacy_row_df.shape[0] == 1:
                legacy_row = legacy_row_df.iloc[0]
                legacy_basis = int(
                    legacy_row[
                        "selected_basis"
                    ]
                )
                legacy_lambda = float(
                    legacy_row[
                        "selected_regularization"
                    ]
                )
                coefficients = (
                    solve_from_normal_equations(
                        fit_gram[
                            :legacy_basis,
                            :legacy_basis,
                        ],
                        fit_rhs[:legacy_basis],
                        geometry[
                            "generator_eigenvalues"
                        ],
                        legacy_lambda,
                    )
                )
                prediction = (
                    phi_test_max[
                        :, :legacy_basis
                    ]
                    @ coefficients
                )
                parent_metrics = evaluate_prediction(
                    truth_survey[test_indices],
                    prediction,
                    evaluation_weights[test_indices],
                )
                unweighted_metrics = evaluate_prediction(
                    truth_survey[test_indices],
                    prediction,
                )

                seed_legacy_audit_records.append(
                    {
                        "replicate": int(replicate),
                        "label_seed": int(seed),
                        "model": model_name,
                        "n_basis": legacy_basis,
                        "regularization_strength": legacy_lambda,
                        "new_parent_nrmse": float(
                            parent_metrics[
                                "normalized_rmse"
                            ]
                        ),
                        "legacy_parent_nrmse": float(
                            legacy_row[
                                "parent_normalized_rmse"
                            ]
                        ),
                        "parent_nrmse_difference": float(
                            parent_metrics[
                                "normalized_rmse"
                            ]
                            - legacy_row[
                                "parent_normalized_rmse"
                            ]
                        ),
                        "new_unweighted_nrmse": float(
                            unweighted_metrics[
                                "normalized_rmse"
                            ]
                        ),
                        "legacy_unweighted_nrmse": float(
                            legacy_row[
                                "unweighted_normalized_rmse"
                            ]
                        ),
                        "unweighted_nrmse_difference": float(
                            unweighted_metrics[
                                "normalized_rmse"
                            ]
                            - legacy_row[
                                "unweighted_normalized_rmse"
                            ]
                        ),
                    }
                )

        print(
            f"  model={model_name:13s}: validation/test/query grids completed "
            f"in {time.perf_counter() - start:.1f} s"
        )

    # Matched-hyperparameter comparison: only loss weighting differs.
    for n_basis in MATCHED_BASIS_SIZES:
        p = int(n_basis)
        for model_name, model_loss_weights in (
            model_definitions.items()
        ):
            fit_gram, fit_rhs = fit_equations[
                model_name
            ]
            coefficients = (
                solve_from_normal_equations(
                    fit_gram[:p, :p],
                    fit_rhs[:p],
                    geometry[
                        "generator_eigenvalues"
                    ],
                    MATCHED_REGULARIZATION,
                )
            )
            observed_prediction = (
                geometry[
                    "eigenfunctions"
                ][
                    test_indices,
                    :p,
                ]
                @ coefficients
            )
            query_prediction = (
                query_eigenfunctions[:, :p]
                @ coefficients
            )

            parent_test = evaluate_prediction(
                truth_survey[test_indices],
                observed_prediction,
                evaluation_weights[test_indices],
            )
            unweighted_test = evaluate_prediction(
                truth_survey[test_indices],
                observed_prediction,
            )
            query_primary = evaluate_prediction(
                truth_query[primary_query_mask],
                query_prediction[primary_query_mask],
            )

            seed_matched_records.append(
                {
                    "replicate": int(replicate),
                    "label_seed": int(seed),
                    "model": model_name,
                    "n_basis": int(n_basis),
                    "regularization_strength": float(
                        MATCHED_REGULARIZATION
                    ),
                    **{
                        f"parent_{key}": value
                        for key, value in parent_test.items()
                    },
                    **{
                        f"unweighted_{key}": value
                        for key, value in unweighted_test.items()
                    },
                    **{
                        f"query_primary_{key}": value
                        for key, value in query_primary.items()
                    },
                }
            )

    seed_validation_df = pd.DataFrame(
        seed_validation_records
    )
    seed_basis_runs_df = pd.DataFrame(
        seed_basis_run_records
    )
    seed_matched_df = pd.DataFrame(
        seed_matched_records
    )
    seed_legacy_audit_df = pd.DataFrame(
        seed_legacy_audit_records,
        columns=[
            "replicate",
            "label_seed",
            "model",
            "n_basis",
            "regularization_strength",
            "new_parent_nrmse",
            "legacy_parent_nrmse",
            "parent_nrmse_difference",
            "new_unweighted_nrmse",
            "legacy_unweighted_nrmse",
            "unweighted_nrmse_difference",
        ],
    )

    seed_validation_df.to_csv(
        paths["validation"],
        index=False,
    )
    seed_basis_runs_df.to_csv(
        paths["basis_runs"],
        index=False,
    )
    seed_matched_df.to_csv(
        paths["matched"],
        index=False,
    )
    seed_legacy_audit_df.to_csv(
        paths["legacy_audit"],
        index=False,
    )

    validation_frames.append(
        seed_validation_df
    )
    basis_run_frames.append(
        seed_basis_runs_df
    )
    matched_frames.append(
        seed_matched_df
    )
    legacy_audit_frames.append(
        seed_legacy_audit_df
    )


# ============================================================
# 6. 集約とpractical selection
# ============================================================

new_validation_df = pd.concat(
    validation_frames,
    ignore_index=True,
)
new_basis_runs_df = pd.concat(
    basis_run_frames,
    ignore_index=True,
)
new_matched_df = pd.concat(
    matched_frames,
    ignore_index=True,
)
legacy_audit_df = pd.concat(
    legacy_audit_frames,
    ignore_index=True,
)

previous_validation_df = pd.read_csv(
    PREVIOUS_VALIDATION_FILE
)
previous_basis_runs_df = pd.read_csv(
    PREVIOUS_BASIS_RUNS_FILE
)
previous_matched_df = pd.read_csv(
    PREVIOUS_MATCHED_FILE
)

# Optional short runs use a prefix of the ten predefined label seeds.  Filter the
# inherited rows as well, so that new and previous segments always contain the
# same replicate set.
_seed_set = set(int(value) for value in LABEL_SPLIT_SEEDS)
for _frame_name, _frame in [
    ("previous_validation", previous_validation_df),
    ("previous_basis_runs", previous_basis_runs_df),
    ("previous_matched", previous_matched_df),
]:
    if "label_seed" not in _frame.columns:
        raise KeyError(f"{_frame_name} is missing label_seed.")
previous_validation_df = previous_validation_df.loc[
    previous_validation_df["label_seed"].astype(int).isin(_seed_set)
].copy()
previous_basis_runs_df = previous_basis_runs_df.loc[
    previous_basis_runs_df["label_seed"].astype(int).isin(_seed_set)
].copy()
previous_matched_df = previous_matched_df.loc[
    previous_matched_df["label_seed"].astype(int).isin(_seed_set)
].copy()

# The v1 basis-run table predates storage of the secondary validation metric.
# Recover it from the validation grid, so all model selection and tie breaking
# remain strictly validation-only.
if "unweighted_validation_nrmse" not in previous_basis_runs_df.columns:
    _selected_validation = previous_validation_df[[
        "replicate",
        "label_seed",
        "model",
        "n_basis",
        "regularization_strength",
        "unweighted_validation_nrmse",
    ]].copy()
    _selected_validation = _selected_validation.rename(
        columns={"regularization_strength": "selected_regularization"}
    )
    previous_basis_runs_df = previous_basis_runs_df.merge(
        _selected_validation,
        on=[
            "replicate",
            "label_seed",
            "model",
            "n_basis",
            "selected_regularization",
        ],
        how="left",
        validate="one_to_one",
    )
    if previous_basis_runs_df["unweighted_validation_nrmse"].isna().any():
        raise ValueError(
            "Could not recover unweighted validation NRMSE for all inherited "
            "basis-run rows."
        )

# n_b=2048は新しいlambda gridで再評価するため、前段結果からは
# 2048未満のみを引き継ぐ。
continuation_basis = int(min(BASIS_SIZES))
validation_df = pd.concat(
    [
        previous_validation_df.loc[
            previous_validation_df["n_basis"]
            < continuation_basis
        ],
        new_validation_df,
    ],
    ignore_index=True,
)
basis_runs_df = pd.concat(
    [
        previous_basis_runs_df.loc[
            previous_basis_runs_df["n_basis"]
            < continuation_basis
        ],
        new_basis_runs_df,
    ],
    ignore_index=True,
)
matched_df = pd.concat(
    [
        previous_matched_df.loc[
            previous_matched_df["n_basis"]
            < continuation_basis
        ],
        new_matched_df,
    ],
    ignore_index=True,
)

# 2048-mode接続点のfixed-lambda metricsを独立に監査する。
overlap_columns = [
    "parent_normalized_rmse",
    "unweighted_normalized_rmse",
    "query_primary_normalized_rmse",
]
previous_overlap = previous_matched_df.loc[
    previous_matched_df["n_basis"]
    == continuation_basis
].copy()
new_overlap = new_matched_df.loc[
    new_matched_df["n_basis"]
    == continuation_basis
].copy()
overlap_records = []
if (
    not previous_overlap.empty
    and not new_overlap.empty
):
    overlap_merged = previous_overlap.merge(
        new_overlap,
        on=[
            "replicate",
            "label_seed",
            "model",
            "n_basis",
            "regularization_strength",
        ],
        suffixes=("_previous", "_new"),
        validate="one_to_one",
    )
    for column in overlap_columns:
        differences = (
            overlap_merged[f"{column}_new"]
            - overlap_merged[f"{column}_previous"]
        ).to_numpy(dtype=np.float64)
        overlap_records.append(
            {
                "metric": column,
                "n_rows": int(differences.size),
                "max_abs_difference": float(
                    np.max(np.abs(differences))
                ),
                "mean_difference": float(
                    np.mean(differences)
                ),
            }
        )
overlap_audit_df = pd.DataFrame(
    overlap_records
)

# 各seed/model/basisでlambda選択済みのbasis_runs_dfからvalidation curveを集約。
aggregate_records = []

for (
    model_name,
    n_basis,
), group in basis_runs_df.groupby(
    [
        "model",
        "n_basis",
    ],
    sort=False,
):
    n_replicates = int(group.shape[0])
    validation_values = group[
        "parent_validation_nrmse"
    ].to_numpy(dtype=np.float64)
    aggregate_records.append(
        {
            "model": model_name,
            "n_basis": int(n_basis),
            "n_replicates": n_replicates,
            "validation_nrmse_mean": float(
                np.mean(validation_values)
            ),
            "validation_nrmse_std": float(
                np.std(
                    validation_values,
                    ddof=1,
                )
            ),
            "validation_nrmse_sem": float(
                np.std(
                    validation_values,
                    ddof=1,
                )
                / np.sqrt(n_replicates)
            ),
            "parent_test_nrmse_mean": float(
                group[
                    "parent_normalized_rmse"
                ].mean()
            ),
            "parent_test_nrmse_std": float(
                group[
                    "parent_normalized_rmse"
                ].std(ddof=1)
            ),
            "unweighted_test_nrmse_mean": float(
                group[
                    "unweighted_normalized_rmse"
                ].mean()
            ),
            "unweighted_test_nrmse_std": float(
                group[
                    "unweighted_normalized_rmse"
                ].std(ddof=1)
            ),
            "query_primary_nrmse_mean": float(
                group[
                    "query_primary_normalized_rmse"
                ].mean()
            ),
            "query_primary_nrmse_std": float(
                group[
                    "query_primary_normalized_rmse"
                ].std(ddof=1)
            ),
        }
    )

aggregate_df = pd.DataFrame(
    aggregate_records
).sort_values(
    [
        "model",
        "n_basis",
    ]
).reset_index(drop=True)

practical_selection_df = pd.DataFrame(
    [
        practical_selection_for_model(
            aggregate_df,
            model_name,
        )
        for model_name in model_definitions
    ]
)

plateau_records = []
for model_name, group in aggregate_df.groupby(
    "model",
    sort=False,
):
    selected = group.sort_values(
        "n_basis"
    ).reset_index(drop=True)
    for index in range(1, selected.shape[0]):
        previous_row = selected.iloc[index - 1]
        current_row = selected.iloc[index]
        previous_value = float(
            previous_row["validation_nrmse_mean"]
        )
        current_value = float(
            current_row["validation_nrmse_mean"]
        )
        plateau_records.append(
            {
                "model": model_name,
                "previous_basis": int(
                    previous_row["n_basis"]
                ),
                "n_basis": int(
                    current_row["n_basis"]
                ),
                "adjacent_validation_relative_reduction": float(
                    (previous_value - current_value)
                    / previous_value
                ),
                "adjacent_validation_reduction_percent": float(
                    100.0
                    * (previous_value - current_value)
                    / previous_value
                ),
            }
        )
plateau_diagnostics_df = pd.DataFrame(
    plateau_records
)

lambda_frequency_df = (
    basis_runs_df.groupby(
        [
            "model",
            "n_basis",
            "selected_regularization",
        ],
        as_index=False,
    )
    .size()
    .rename(columns={"size": "count"})
    .sort_values(
        [
            "model",
            "n_basis",
            "count",
            "selected_regularization",
        ],
        ascending=[True, True, False, True],
    )
)

# Per-seed retuned protocol: basis and lambda both selected by validation.
retuned_rows = []
for (
    replicate,
    model_name,
), group in basis_runs_df.groupby(
    [
        "replicate",
        "model",
    ]
):
    selected = group.sort_values(
        [
            "parent_validation_nrmse",
            "unweighted_validation_nrmse",
            "n_basis",
            "selected_regularization",
        ]
    ).iloc[0]
    row = selected.to_dict()
    row["protocol"] = "retuned"
    retuned_rows.append(row)
retuned_df = pd.DataFrame(retuned_rows)

# Model-specific practical basis selected from aggregate validation only.
practical_rows = []
for model_name in model_definitions:
    selection_row = practical_selection_df.loc[
        practical_selection_df[
            "model"
        ] == model_name
    ].iloc[0]
    practical_basis = int(
        selection_row[
            "one_standard_error_basis"
        ]
    )
    selected = basis_runs_df.loc[
        (
            basis_runs_df["model"]
            == model_name
        )
        & (
            basis_runs_df["n_basis"]
            == practical_basis
        )
    ].copy()
    selected["protocol"] = (
        "practical_model_specific"
    )
    practical_rows.append(selected)
practical_df = pd.concat(
    practical_rows,
    ignore_index=True,
)

# Pre-specified matched protocol at Mock-1 fiducial practical truncation.
matched_primary_df = matched_df.loc[
    (
        matched_df["n_basis"]
        == PRIMARY_MATCHED_BASIS
    )
    & np.isclose(
        matched_df[
            "regularization_strength"
        ].to_numpy(dtype=np.float64),
        PRIMARY_MATCHED_REGULARIZATION,
        rtol=0.0,
        atol=1.0e-12,
    )
].copy()
matched_primary_df["protocol"] = (
    "matched_mock1_fiducial"
)

protocol_frames = {
    "retuned": retuned_df,
    "practical_model_specific": practical_df,
    "matched_mock1_fiducial": matched_primary_df,
}

paired_records = []
paired_summary_records = []

metric_columns = {
    "parent": "parent_normalized_rmse",
    "observed_unweighted": "unweighted_normalized_rmse",
    "query_primary": "query_primary_normalized_rmse",
}

for protocol_name, frame in protocol_frames.items():
    pivot = frame.pivot(
        index="replicate",
        columns="model",
        values=list(metric_columns.values()),
    )

    protocol_paired = pd.DataFrame(
        {
            "protocol": protocol_name,
            "replicate": pivot.index.to_numpy(
                dtype=np.int64
            ),
        }
    )

    for metric_name, column_name in (
        metric_columns.items()
    ):
        differences = (
            pivot[column_name][
                "loss_weighted"
            ]
            - pivot[column_name][
                "naive"
            ]
        )
        protocol_paired[
            f"{metric_name}_difference_weighted_minus_naive"
        ] = differences.to_numpy(
            dtype=np.float64
        )

        inference = bootstrap_mean_difference(
            differences.to_numpy(
                dtype=np.float64
            ),
            SEED_BOOTSTRAP_REPLICATES,
            SEED_BOOTSTRAP_SEED
            + len(paired_summary_records)
            + len(metric_name),
        )
        paired_summary_records.append(
            {
                "protocol": protocol_name,
                "metric": metric_name,
                "n_replicates": int(
                    differences.size
                ),
                **inference,
            }
        )

    paired_records.append(
        protocol_paired
    )

paired_df = pd.concat(
    paired_records,
    ignore_index=True,
)
paired_summary_df = pd.DataFrame(
    paired_summary_records
)

# Matched weighting effect as a function of basis at lambda=10^-2.
matched_difference_records = []
for n_basis, group in matched_df.groupby(
    "n_basis"
):
    pivot = group.pivot(
        index="replicate",
        columns="model",
        values=list(metric_columns.values()),
    )
    record = {
        "n_basis": int(n_basis),
        "regularization_strength": float(
            MATCHED_REGULARIZATION
        ),
    }
    for metric_name, column_name in (
        metric_columns.items()
    ):
        differences = (
            pivot[column_name][
                "loss_weighted"
            ]
            - pivot[column_name][
                "naive"
            ]
        ).to_numpy(dtype=np.float64)
        record[
            f"{metric_name}_mean_difference"
        ] = float(np.mean(differences))
        record[
            f"{metric_name}_std_difference"
        ] = float(
            np.std(differences, ddof=1)
        )
        record[
            f"{metric_name}_wins"
        ] = int(np.sum(differences < 0.0))
    matched_difference_records.append(record)

matched_summary_df = pd.DataFrame(
    matched_difference_records
).sort_values("n_basis")

# 旧512-mode resultとのcontinuity summary。
legacy_audit_summary_df = pd.DataFrame()
if not legacy_audit_df.empty:
    legacy_audit_summary_df = (
        legacy_audit_df.groupby(
            "model",
            as_index=False,
        )
        .agg(
            n_rows=(
                "replicate",
                "size",
            ),
            parent_max_abs_difference=(
                "parent_nrmse_difference",
                lambda values: float(
                    np.max(
                        np.abs(values)
                    )
                ),
            ),
            unweighted_max_abs_difference=(
                "unweighted_nrmse_difference",
                lambda values: float(
                    np.max(
                        np.abs(values)
                    )
                ),
            ),
        )
    )


# ============================================================
# 7. 図
# ============================================================

model_labels = {
    "naive": "Unweighted loss",
    "loss_weighted": "Selection-weighted loss",
}

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13.0, 10.0),
    constrained_layout=True,
)

# (a) Parent-weighted validation convergence.
ax = axes[0, 0]
for model_name, model_label in model_labels.items():
    selected = aggregate_df.loc[
        aggregate_df["model"] == model_name
    ].sort_values("n_basis")
    ax.errorbar(
        selected["n_basis"],
        selected["validation_nrmse_mean"],
        yerr=selected["validation_nrmse_std"],
        marker="o",
        capsize=4,
        label=model_label,
    )
    selection_row = practical_selection_df.loc[
        practical_selection_df["model"]
        == model_name
    ].iloc[0]
    selected_basis = int(
        selection_row["one_standard_error_basis"]
    )
    marker_row = selected.loc[
        selected["n_basis"] == selected_basis
    ]
    if not marker_row.empty:
        ax.plot(
            marker_row["n_basis"],
            marker_row["validation_nrmse_mean"],
            marker="D",
            markersize=9,
            markerfacecolor="none",
            markeredgewidth=1.8,
            linestyle="none",
        )
ax.set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
ax.set_ylabel(r"Parent-weighted validation $\mathrm{NRMSE}_{3\mathrm{D}}$")
ax.set_title("High-mode validation convergence")
ax.legend(frameon=True)
ax.text(
    -0.10,
    1.04,
    "(a)",
    transform=ax.transAxes,
    fontweight="bold",
    fontsize=14,
    va="top",
)

# (b) Parent-weighted test convergence.
ax = axes[0, 1]
for model_name, model_label in model_labels.items():
    selected = aggregate_df.loc[
        aggregate_df["model"] == model_name
    ].sort_values("n_basis")
    ax.errorbar(
        selected["n_basis"],
        selected["parent_test_nrmse_mean"],
        yerr=selected["parent_test_nrmse_std"],
        marker="o",
        capsize=4,
        label=model_label,
    )
ax.set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
ax.set_ylabel(r"Parent-weighted test $\mathrm{NRMSE}_{3\mathrm{D}}$")
ax.set_title("Observed parent-risk convergence")
ax.legend(frameon=True)
ax.text(
    -0.10,
    1.04,
    "(b)",
    transform=ax.transAxes,
    fontweight="bold",
    fontsize=14,
    va="top",
)

# (c) Primary complete-query convergence.
ax = axes[1, 0]
for model_name, model_label in model_labels.items():
    selected = aggregate_df.loc[
        aggregate_df["model"] == model_name
    ].sort_values("n_basis")
    ax.errorbar(
        selected["n_basis"],
        selected["query_primary_nrmse_mean"],
        yerr=selected["query_primary_nrmse_std"],
        marker="o",
        capsize=4,
        label=model_label,
    )
ax.set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
ax.set_ylabel(r"Primary query $\mathrm{NRMSE}_{3\mathrm{D}}$")
ax.set_title("Complete-query convergence")
ax.legend(frameon=True)
ax.text(
    -0.10,
    1.04,
    "(c)",
    transform=ax.transAxes,
    fontweight="bold",
    fontsize=14,
    va="top",
)

# (d) Practical-protocol paired effects.
ax = axes[1, 1]
summary_selected = paired_summary_df.loc[
    paired_summary_df["protocol"]
    == "practical_model_specific"
].copy()
metric_order = [
    "parent",
    "observed_unweighted",
    "query_primary",
]
metric_labels = {
    "parent": "Observed parent risk",
    "observed_unweighted": "Observed-unweighted risk",
    "query_primary": "Primary complete-query risk",
}
positions = np.arange(len(metric_order))
means = []
lower = []
upper = []
for metric_name in metric_order:
    row = summary_selected.loc[
        summary_selected["metric"]
        == metric_name
    ].iloc[0]
    means.append(float(row["mean_difference"]))
    lower.append(
        float(row["mean_difference"] - row["ci_low"])
    )
    upper.append(
        float(row["ci_high"] - row["mean_difference"])
    )
ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)
ax.errorbar(
    positions,
    means,
    yerr=np.vstack([lower, upper]),
    marker="o",
    linestyle="none",
    capsize=5,
)
ax.set_xticks(
    positions,
    [metric_labels[name] for name in metric_order],
    rotation=12,
    ha="right",
)
ax.set_ylabel(
    r"Selection-weighted minus unweighted loss $\mathrm{NRMSE}_{3\mathrm{D}}$"
)
all_plateaus_found = bool(
    practical_selection_df["plateau_found"].all()
)
selection_rules_agree = bool(
    np.all(
        practical_selection_df["one_standard_error_basis"].to_numpy(
            dtype=np.int64
        )
        == practical_selection_df["practical_plateau_basis"].to_numpy(
            dtype=np.int64
        )
    )
)
ax.set_title(
    "Paired weighting effect at practical truncation"
    if all_plateaus_found and selection_rules_agree
    else "Paired weighting effect at the one-standard-error selection"
)
ax.text(
    0.03,
    0.04,
    "Error bars: 95% seed-bootstrap interval",
    transform=ax.transAxes,
    fontsize=9.0,
    ha="left",
    va="bottom",
)
ax.text(
    -0.10,
    1.04,
    "(d)",
    transform=ax.transAxes,
    fontweight="bold",
    fontsize=14,
    va="top",
)

composite_pdf, composite_png = save_figure(
    fig,
    "composite",
)

# Matched effect versus basis.
fig, ax = plt.subplots(
    figsize=(7.2, 5.2),
    constrained_layout=True,
)
for metric_name, label, marker in [
    (
        "parent",
        "Observed parent risk",
        "o",
    ),
    (
        "observed_unweighted",
        "Observed-unweighted risk",
        "s",
    ),
    (
        "query_primary",
        "Primary complete-query risk",
        "D",
    ),
]:
    ax.plot(
        matched_summary_df["n_basis"],
        matched_summary_df[
            f"{metric_name}_mean_difference"
        ],
        marker=marker,
        label=label,
    )
ax.axhline(
    0.0,
    linestyle="--",
    linewidth=1.0,
)
ax.set_xlabel(r"Number of retained modes $n_{\mathrm{b}}$")
ax.set_ylabel(
    r"Selection-weighted minus unweighted loss $\mathrm{NRMSE}_{3\mathrm{D}}$"
)
ax.set_title(
    r"Matched-hyperparameter weighting effect at $\lambda=10^{-2}$"
)
ax.legend(frameon=True)
matched_pdf, matched_png = save_figure(
    fig,
    "matched_effect_vs_basis",
)

# Practical plateau diagnostic.
fig, ax = plt.subplots(
    figsize=(8.2, 5.6),
    constrained_layout=True,
)
for model_name, model_label in model_labels.items():
    selected = plateau_diagnostics_df.loc[
        plateau_diagnostics_df["model"]
        == model_name
    ].sort_values("n_basis")
    ax.plot(
        selected["n_basis"],
        selected["adjacent_validation_reduction_percent"],
        marker="o",
        label=model_label,
    )

    selection_row = practical_selection_df.loc[
        practical_selection_df["model"]
        == model_name
    ].iloc[0]
    marker_basis = int(
        selection_row["one_standard_error_basis"]
    )
    marker_row = selected.loc[
        selected["n_basis"] == marker_basis
    ]
    if not marker_row.empty:
        ax.plot(
            marker_row["n_basis"],
            marker_row["adjacent_validation_reduction_percent"],
            marker="D",
            markersize=10,
            markerfacecolor="none",
            markeredgewidth=2.0,
            linestyle="none",
        )
ax.axhline(
    100.0 * PRACTICAL_RELATIVE_THRESHOLD,
    linestyle="--",
    linewidth=1.2,
    label=(
        f"{100.0 * PRACTICAL_RELATIVE_THRESHOLD:.1f}% "
        "practical threshold"
    ),
)
ax.axhline(
    0.0,
    linestyle=":",
    linewidth=1.2,
)
ax.set_xticks(
    sorted(
        plateau_diagnostics_df["n_basis"].unique()
    )
)
ax.tick_params(
    axis="x",
    rotation=30,
)
ax.set_xlabel(
    r"Number of retained modes $n_{\mathrm{b}}$"
)
ax.set_ylabel(
    "Adjacent validation-error reduction [per cent]"
)
ax.set_title(
    "Mock-2 practical plateau diagnostic"
)
ax.legend(
    loc="upper right",
    frameon=True,
)
ax.text(
    0.03,
    0.05,
    "Open diamonds mark the one-standard-error selection\n"
    "Negative values indicate validation-error increase",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9.2,
)
plateau_pdf, plateau_png = save_figure(
    fig,
    "plateau_diagnostic",
)


# ============================================================
# 8. 既存local baselineの読込み
# ============================================================

local_baseline_df = pd.DataFrame()
if OPTIMIZATION_EVALUATION_FILE.is_file():
    optimization_evaluation_df = pd.read_csv(
        OPTIMIZATION_EVALUATION_FILE
    )
    required_columns = {
        "model",
        "subset",
        "normalized_rmse",
    }
    if required_columns.issubset(
        optimization_evaluation_df.columns
    ):
        local_baseline_df = (
            optimization_evaluation_df.loc[
                optimization_evaluation_df[
                    "model"
                ].astype(str).str.startswith(
                    "local_kernel"
                )
                & optimization_evaluation_df[
                    "subset"
                ].isin(
                    [
                        "observed_test_parent_weighted",
                        "observed_test_unweighted",
                        "query_primary_boundary_safe",
                    ]
                )
            ]
            .copy()
        )


# ============================================================
# 9. 保存
# ============================================================

paths = {
    "new_validation_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_new_validation.csv",
    "new_basis_runs_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_new_basis_runs.csv",
    "new_matched_grid_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_new_matched_grid.csv",
    "validation_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_validation_to_4096.csv",
    "basis_runs_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_basis_runs_to_4096.csv",
    "aggregate_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_aggregate_to_4096.csv",
    "practical_selection_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_practical_selection.csv",
    "retuned_runs_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_retuned_runs.csv",
    "practical_runs_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_practical_runs.csv",
    "matched_grid_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_matched_grid.csv",
    "matched_summary_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_matched_summary.csv",
    "paired_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_paired.csv",
    "paired_summary_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_paired_summary.csv",
    "geometry_audit_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_audit.csv",
    "legacy_audit_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_legacy_512_audit.csv",
    "legacy_audit_summary_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_legacy_512_audit_summary.csv",
    "local_baseline_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_local_baseline.csv",
    "plateau_diagnostics_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_plateau_diagnostics.csv",
    "lambda_frequency_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_lambda_frequency.csv",
    "overlap_2048_audit_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_overlap_2048_audit.csv",
    "summary_json": OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary_to_4096.json",
}

new_validation_df.to_csv(
    paths["new_validation_csv"],
    index=False,
)
new_basis_runs_df.to_csv(
    paths["new_basis_runs_csv"],
    index=False,
)
new_matched_df.to_csv(
    paths["new_matched_grid_csv"],
    index=False,
)
validation_df.to_csv(
    paths["validation_csv"],
    index=False,
)
basis_runs_df.to_csv(
    paths["basis_runs_csv"],
    index=False,
)
aggregate_df.to_csv(
    paths["aggregate_csv"],
    index=False,
)
practical_selection_df.to_csv(
    paths["practical_selection_csv"],
    index=False,
)
retuned_df.to_csv(
    paths["retuned_runs_csv"],
    index=False,
)
practical_df.to_csv(
    paths["practical_runs_csv"],
    index=False,
)
matched_df.to_csv(
    paths["matched_grid_csv"],
    index=False,
)
matched_summary_df.to_csv(
    paths["matched_summary_csv"],
    index=False,
)
paired_df.to_csv(
    paths["paired_csv"],
    index=False,
)
paired_summary_df.to_csv(
    paths["paired_summary_csv"],
    index=False,
)
geometry_audit_df.to_csv(
    paths["geometry_audit_csv"],
    index=False,
)
legacy_audit_df.to_csv(
    paths["legacy_audit_csv"],
    index=False,
)
legacy_audit_summary_df.to_csv(
    paths[
        "legacy_audit_summary_csv"
    ],
    index=False,
)
local_baseline_df.to_csv(
    paths["local_baseline_csv"],
    index=False,
)
plateau_diagnostics_df.to_csv(
    paths["plateau_diagnostics_csv"],
    index=False,
)
lambda_frequency_df.to_csv(
    paths["lambda_frequency_csv"],
    index=False,
)
overlap_audit_df.to_csv(
    paths["overlap_2048_audit_csv"],
    index=False,
)

summary = {
    "scope": (
        "Mock-2 continuation of mode-number convergence to 4096 modes "
        "for unweighted and selection-weighted losses with fixed "
        "unweighted alpha=0 geometry"
    ),
    "input": {
        "mock1": str(MOCK1_FILE),
        "mock2": str(MOCK2_FILE),
        "optimization_predictions": str(
            OPTIMIZATION_PREDICTIONS_FILE
        ),
        "previous_output": str(
            PREVIOUS_OUTPUT_DIR
        ),
    },
    "geometry": {
        "max_basis": int(MAX_BASIS),
        "graph_neighbors": int(
            GRAPH_NEIGHBORS
        ),
        "graph_bandwidth_neighbor": int(
            GRAPH_BANDWIDTH_NEIGHBOR
        ),
        "alpha_dm": float(ALPHA_DM),
        "generator_scale": float(
            geometry["generator_scale"]
        ),
        "cache": str(GEOMETRY_CACHE_FILE),
        "loaded_from_cache": bool(
            geometry_from_cache
        ),
        "reference_geometry": str(
            PREVIOUS_GEOMETRY_FILE
        ),
    },
    "run_token": RUN_TOKEN,
    "continuation_basis": int(continuation_basis),
    "selection_weight": loss_weight_info,
    "evaluation_weight": evaluation_weight_info,
    "label_split_seeds": [
        int(seed)
        for seed in LABEL_SPLIT_SEEDS
    ],
    "basis_sizes": [
        int(value)
        for value in BASIS_SIZES
    ],
    "regularization_strengths": [
        float(value)
        for value in REGULARIZATION_STRENGTHS
    ],
    "practical_selection": (
        practical_selection_df.to_dict(
            orient="records"
        )
    ),
    "paired_summary": (
        paired_summary_df.to_dict(
            orient="records"
        )
    ),
    "matched_summary": (
        matched_summary_df.to_dict(
            orient="records"
        )
    ),
    "plateau_diagnostics": (
        plateau_diagnostics_df.to_dict(
            orient="records"
        )
    ),
    "lambda_frequency": (
        lambda_frequency_df.to_dict(
            orient="records"
        )
    ),
    "overlap_2048_audit": (
        overlap_audit_df.to_dict(
            orient="records"
        )
    ),
    "outputs": {
        **{
            name: str(path)
            for name, path in paths.items()
        },
        "composite_pdf": str(
            composite_pdf
        ),
        "composite_png": str(
            composite_png
        ),
        "matched_effect_pdf": str(
            matched_pdf
        ),
        "matched_effect_png": str(
            matched_png
        ),
        "plateau_pdf": str(
            plateau_pdf
        ),
        "plateau_png": str(
            plateau_png
        ),
    },
}

with paths["summary_json"].open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        summary,
        handle,
        ensure_ascii=False,
        indent=2,
    )

print("=" * 116)
print("Mock-2 mode-number convergence to 4096 completed")
print("=" * 116)
print("Practical selection")
print(practical_selection_df.to_string(index=False))
print("-" * 116)
print("Paired weighting effects")
print(paired_summary_df.to_string(index=False))
print("-" * 116)
if not overlap_audit_df.empty:
    print("2048-mode overlap audit")
    print(overlap_audit_df.to_string(index=False))
    print("-" * 116)
if not legacy_audit_summary_df.empty:
    print("Legacy 512-mode continuity audit")
    print(legacy_audit_summary_df.to_string(index=False))
    print("-" * 116)
for name, path in paths.items():
    print(f"{name:34s}: {path}")
print(f"composite_pdf                     : {composite_pdf}")
print(f"composite_png                     : {composite_png}")
print(f"matched_effect_pdf                : {matched_pdf}")
print(f"matched_effect_png                : {matched_png}")
print(f"plateau_pdf                       : {plateau_pdf}")
print(f"plateau_png                       : {plateau_png}")
print("=" * 116)
