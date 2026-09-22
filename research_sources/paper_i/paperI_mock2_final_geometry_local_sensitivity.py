# Mock-2: final restricted geometry-side sensitivity and 10-split local-kernel comparison
#
# このファイル全体を、次の共通セルを実行した後、Jupyter Notebookの
# 新しい空のコードセルへ貼り付けて実行してください。
#
#   mock23_current_samples_common_single_cell_v2.py
#
# 目的:
#   1. Mock-2でvalidation-selected practical truncationとなったn_b=2560を固定し、
#      operator-side unweighted geometry、旧最良geometry-weighted候補、
#      およびfull inverse-selection geometryをrestricted sensitivity testとして比較する。
#   2. geometry weightingとloss weightingを分離し、同じ10通りのlabel split、
#      同じparent-weighted validation criterionでlambdaを再選択する。
#   3. adaptive local-kernel regressionを同じ10 splitで再調整し、
#      final diffusion-spectral estimatorとのpaired comparisonを構成する。
#   4. observed parent-weighted risk、observed-unweighted risk、
#      primary complete-query riskを別々に保存し、target-risk trade-offを評価する。
#
# 重要:
#   - query labelsはhyperparameter selectionに使用しません。
#   - held-out observed testもhyperparameter selectionに使用しません。
#   - geometry candidatesは事前に限定したrestricted setのみです。
#   - 大きなNyström配列はchunk処理し、メモリ使用量を抑えます。
#   - 各geometryと各seedの進行状況をflush付きで表示します。

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

import matplotlib

SHOW_PLOTS = os.environ.get(
    "PAPER1_MOCK2_FINAL_SHOW_PLOTS",
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
    "_make_split",
    "_query_neighbors",
    "_local_kernel_from_neighbors",
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
# 1. 入出力
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

FINAL_ROLES_FILE = (
    OPTIMIZATION_DIR
    / "mock2_selection_optimization_final_roles.csv"
)

STAGE3_CANDIDATES_FILE = (
    OPTIMIZATION_DIR
    / "mock2_selection_optimization_stage3_candidates.csv"
)

CONVERGENCE_DIR = (
    ROOT_DIR
    / "mock2_loss_weight_mode_convergence_to_4096_v2"
)

PRACTICAL_SELECTION_FILE = (
    CONVERGENCE_DIR
    / "mock2_loss_weight_mode_convergence_to_4096_practical_selection.csv"
)

PRACTICAL_RUNS_FILE = (
    CONVERGENCE_DIR
    / "mock2_loss_weight_mode_convergence_to_4096_practical_runs.csv"
)

UNWEIGHTED_GEOMETRY_4096_FILE = (
    CONVERGENCE_DIR
    / "cache"
    / "mock2_unweighted_alpha0_geometry_m4096.npz"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_MOCK2_FINAL_OUTPUT",
        str(
            ROOT_DIR
            / "mock2_final_geometry_local_sensitivity_v1"
        ),
    )
)

CACHE_DIR = OUTPUT_DIR / "geometry_cache"
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

OUTPUT_PREFIX = "mock2_final_geometry_local_sensitivity"


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
EIGEN_SOLVER_SEED = 20260820
GENERATOR_REFERENCE_NONCONSTANT_MODES = 511

FINAL_BASIS_OVERRIDE = os.environ.get(
    "PAPER1_MOCK2_FINAL_BASIS",
    "",
).strip()

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
        "PAPER1_MOCK2_FINAL_LAMBDAS",
        ",".join(
            str(value)
            for value in _DEFAULT_REGULARIZATION_STRENGTHS
        ),
    ).split(",")
    if value.strip()
]
REGULARIZATION_STRENGTHS = sorted(
    set(REGULARIZATION_STRENGTHS)
)

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
N_SEEDS = int(
    os.environ.get(
        "PAPER1_MOCK2_FINAL_N_SEEDS",
        str(len(_DEFAULT_LABEL_SPLIT_SEEDS)),
    )
)
if not 1 <= N_SEEDS <= len(_DEFAULT_LABEL_SPLIT_SEEDS):
    raise ValueError(
        "PAPER1_MOCK2_FINAL_N_SEEDS must be between 1 and 10."
    )
LABEL_SPLIT_SEEDS = _DEFAULT_LABEL_SPLIT_SEEDS[:N_SEEDS]

MIN_SELECTION_PROBABILITY = 1.0e-4
WEIGHT_CLIP_QUANTILE = 0.995
PRIMARY_LOSS_GAMMA = 1.0
PRIMARY_LOSS_CAP = 50.0
EVALUATION_WEIGHT_GAMMA = 1.0
EVALUATION_WEIGHT_CAP = 50.0

NUMERICAL_RIDGE = 1.0e-10

# Restricted geometry set.
FULL_INVERSE_ALPHA = 0.0
FULL_INVERSE_GEOMETRY_GAMMA = 1.0
FULL_INVERSE_GEOMETRY_CAP = 50.0

# Local-kernel grid: legacy v4と同じ範囲。
_DEFAULT_LOCAL_NEIGHBORS = [
    8,
    16,
    32,
    64,
]
LOCAL_KERNEL_NEIGHBORS = [
    int(value)
    for value in os.environ.get(
        "PAPER1_MOCK2_LOCAL_NEIGHBORS",
        ",".join(str(value) for value in _DEFAULT_LOCAL_NEIGHBORS),
    ).split(",")
    if value.strip()
]

_DEFAULT_LOCAL_MULTIPLIERS = [
    0.50,
    0.75,
    1.00,
    1.50,
]
LOCAL_KERNEL_MULTIPLIERS = [
    float(value)
    for value in os.environ.get(
        "PAPER1_MOCK2_LOCAL_MULTIPLIERS",
        ",".join(str(value) for value in _DEFAULT_LOCAL_MULTIPLIERS),
    ).split(",")
    if value.strip()
]
LOCAL_KERNEL_BANDWIDTH_FRACTION = 3.0 / 8.0

_DEFAULT_LOCAL_SOURCE_GAMMAS = [
    0.0,
    0.25,
    0.50,
    0.75,
    1.0,
]
LOCAL_SOURCE_GAMMAS = [
    float(value)
    for value in os.environ.get(
        "PAPER1_MOCK2_LOCAL_SOURCE_GAMMAS",
        ",".join(str(value) for value in _DEFAULT_LOCAL_SOURCE_GAMMAS),
    ).split(",")
    if value.strip()
]

_DEFAULT_LOCAL_SOURCE_CAPS = [
    5.0,
    10.0,
    20.0,
    50.0,
]
LOCAL_SOURCE_CAPS = [
    float(value)
    for value in os.environ.get(
        "PAPER1_MOCK2_LOCAL_SOURCE_CAPS",
        ",".join(str(value) for value in _DEFAULT_LOCAL_SOURCE_CAPS),
    ).split(",")
    if value.strip()
]

NYSTROM_QUERY_CHUNK = int(
    os.environ.get(
        "PAPER1_MOCK2_NYSTROM_CHUNK",
        "128",
    )
)
if NYSTROM_QUERY_CHUNK < 1:
    raise ValueError("NYSTROM_QUERY_CHUNK must be positive.")

SEED_BOOTSTRAP_REPLICATES = int(
    os.environ.get(
        "PAPER1_MOCK2_FINAL_BOOTSTRAP_REPLICATES",
        "50000",
    )
)
SEED_BOOTSTRAP_SEED = 20270501

SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 300


# ============================================================
# 3. 一般補助関数
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



def canonical_cap(
    gamma: float,
    cap: float,
) -> float:
    return 1.0 if float(gamma) == 0.0 else float(cap)



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
        empirical_cap = 1.0
        actual_cap = 1.0
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
        actual_cap = min(
            float(max_weight),
            empirical_cap,
        )
        normalized = np.minimum(
            raw,
            actual_cap,
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
        "requested_cap": float(max_weight),
        "empirical_cap": float(empirical_cap),
        "actual_raw_cap": float(actual_cap),
        "normalized_weight_max": float(
            np.max(normalized)
        ),
        "effective_sample_size": effective_sample_size,
        "effective_sample_fraction": float(
            effective_sample_size
            / normalized.size
        ),
    }



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
        ) if values.size > 1 else 0.0,
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



def clean_float(
    value,
    default: float,
) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    return number if np.isfinite(number) else float(default)



def config_token(
    alpha: float,
    geometry_gamma: float,
    geometry_cap: float,
) -> str:
    cap = canonical_cap(
        geometry_gamma,
        geometry_cap,
    )
    return (
        f"a{float(alpha):.2f}"
        f"_g{float(geometry_gamma):.2f}"
        f"_c{float(cap):.1f}"
        f"_m{int(FINAL_BASIS)}"
    ).replace(".", "p")


# ============================================================
# 4. Practical basisとrestricted candidates
# ============================================================


def read_practical_basis() -> int:
    if FINAL_BASIS_OVERRIDE:
        return int(FINAL_BASIS_OVERRIDE)

    if not PRACTICAL_SELECTION_FILE.is_file():
        raise FileNotFoundError(
            "Practical-selection CSVがありません。"
            f"\n{PRACTICAL_SELECTION_FILE}"
        )

    frame = pd.read_csv(
        PRACTICAL_SELECTION_FILE
    )
    required = {
        "model",
        "one_standard_error_basis",
        "practical_plateau_basis",
        "plateau_found",
    }
    missing = required.difference(frame.columns)
    if missing:
        raise KeyError(
            "Practical-selection CSV is missing columns: "
            f"{sorted(missing)}"
        )

    selected = frame.loc[
        frame["model"].isin(
            ["naive", "loss_weighted"]
        )
    ].copy()
    if selected.shape[0] != 2:
        raise ValueError(
            "Expected practical-selection rows for naive and loss_weighted."
        )

    one_se_values = set(
        selected[
            "one_standard_error_basis"
        ].astype(int)
    )
    plateau_values = set(
        selected[
            "practical_plateau_basis"
        ].astype(int)
    )
    plateau_flags = selected[
        "plateau_found"
    ].astype(bool)

    if len(one_se_values) != 1:
        raise ValueError(
            "The two loss models do not share one practical basis."
        )
    if one_se_values != plateau_values:
        raise ValueError(
            "One-standard-error and practical-plateau bases disagree."
        )
    if not bool(np.all(plateau_flags)):
        raise ValueError(
            "Practical plateau has not been confirmed for both loss models."
        )

    return int(next(iter(one_se_values)))


FINAL_BASIS = read_practical_basis()


def extract_legacy_geometry_candidate() -> dict:
    candidate = None

    if FINAL_ROLES_FILE.is_file():
        roles = pd.read_csv(
            FINAL_ROLES_FILE
        )
        if "role" in roles.columns:
            selected = roles.loc[
                roles["role"]
                == "optimized_geometry_only"
            ]
            if selected.shape[0] == 1:
                row = selected.iloc[0]
                candidate = {
                    "alpha": clean_float(
                        row.get("alpha"),
                        0.0,
                    ),
                    "geometry_gamma": clean_float(
                        row.get("geometry_gamma"),
                        0.0,
                    ),
                    "geometry_cap": clean_float(
                        row.get("geometry_cap"),
                        1.0,
                    ),
                    "source": "optimized_geometry_only role",
                }

    def is_nontrivial(config: dict | None) -> bool:
        if config is None:
            return False
        return not (
            np.isclose(config["alpha"], 0.0)
            and np.isclose(
                config["geometry_gamma"],
                0.0,
            )
        )

    if is_nontrivial(candidate):
        return candidate

    if not STAGE3_CANDIDATES_FILE.is_file():
        raise FileNotFoundError(
            "Nontrivial legacy geometry candidate could not be found, and "
            "the stage-3 candidate CSV is unavailable.\n"
            f"{STAGE3_CANDIDATES_FILE}"
        )

    stage3 = pd.read_csv(
        STAGE3_CANDIDATES_FILE
    )
    required = {
        "alpha",
        "geometry_gamma",
        "geometry_cap",
        "loss_gamma",
        "weighted_validation_nrmse",
    }
    missing = required.difference(
        stage3.columns
    )
    if missing:
        raise KeyError(
            "Stage-3 candidate CSV is missing columns: "
            f"{sorted(missing)}"
        )

    geometry_only = stage3.loc[
        np.isclose(
            stage3["loss_gamma"].to_numpy(
                dtype=np.float64
            ),
            0.0,
        )
    ].copy()
    geometry_only = geometry_only.loc[
        ~(
            np.isclose(
                geometry_only["alpha"].to_numpy(
                    dtype=np.float64
                ),
                0.0,
            )
            & np.isclose(
                geometry_only[
                    "geometry_gamma"
                ].to_numpy(dtype=np.float64),
                0.0,
            )
        )
    ]
    if geometry_only.empty:
        raise ValueError(
            "No nontrivial geometry-only candidate exists in stage 3."
        )

    sort_columns = [
        "weighted_validation_nrmse",
    ]
    if "unweighted_validation_nrmse" in geometry_only.columns:
        sort_columns.append(
            "unweighted_validation_nrmse"
        )
    row = geometry_only.sort_values(
        sort_columns
    ).iloc[0]

    return {
        "alpha": float(row["alpha"]),
        "geometry_gamma": float(
            row["geometry_gamma"]
        ),
        "geometry_cap": float(
            row["geometry_cap"]
        ),
        "source": "best nontrivial geometry-only stage-3 candidate",
    }


LEGACY_GEOMETRY = extract_legacy_geometry_candidate()


def build_role_definitions() -> list[dict]:
    raw_roles = [
        {
            "role": "primary_unweighted",
            "label": "Primary unweighted",
            "alpha": 0.0,
            "geometry_gamma": 0.0,
            "geometry_cap": 1.0,
            "loss_gamma": 0.0,
            "loss_cap": 1.0,
            "source": "primary operator and unweighted loss",
        },
        {
            "role": "primary_loss_weighted",
            "label": "Primary loss weighted",
            "alpha": 0.0,
            "geometry_gamma": 0.0,
            "geometry_cap": 1.0,
            "loss_gamma": PRIMARY_LOSS_GAMMA,
            "loss_cap": PRIMARY_LOSS_CAP,
            "source": "primary operator and capped inverse-selection loss",
        },
        {
            "role": "legacy_geometry_only",
            "label": "Legacy geometry only",
            "alpha": LEGACY_GEOMETRY["alpha"],
            "geometry_gamma": LEGACY_GEOMETRY[
                "geometry_gamma"
            ],
            "geometry_cap": LEGACY_GEOMETRY[
                "geometry_cap"
            ],
            "loss_gamma": 0.0,
            "loss_cap": 1.0,
            "source": LEGACY_GEOMETRY["source"],
        },
        {
            "role": "legacy_geometry_plus_loss",
            "label": "Legacy geometry + loss",
            "alpha": LEGACY_GEOMETRY["alpha"],
            "geometry_gamma": LEGACY_GEOMETRY[
                "geometry_gamma"
            ],
            "geometry_cap": LEGACY_GEOMETRY[
                "geometry_cap"
            ],
            "loss_gamma": PRIMARY_LOSS_GAMMA,
            "loss_cap": PRIMARY_LOSS_CAP,
            "source": LEGACY_GEOMETRY["source"],
        },
        {
            "role": "full_inverse_geometry_only",
            "label": "Full-inverse geometry only",
            "alpha": FULL_INVERSE_ALPHA,
            "geometry_gamma": FULL_INVERSE_GEOMETRY_GAMMA,
            "geometry_cap": FULL_INVERSE_GEOMETRY_CAP,
            "loss_gamma": 0.0,
            "loss_cap": 1.0,
            "source": "pre-specified full inverse-selection geometry",
        },
        {
            "role": "full_inverse_joint",
            "label": "Full-inverse geometry + loss",
            "alpha": FULL_INVERSE_ALPHA,
            "geometry_gamma": FULL_INVERSE_GEOMETRY_GAMMA,
            "geometry_cap": FULL_INVERSE_GEOMETRY_CAP,
            "loss_gamma": PRIMARY_LOSS_GAMMA,
            "loss_cap": PRIMARY_LOSS_CAP,
            "source": "pre-specified full inverse-selection geometry and loss",
        },
    ]

    # Exact duplicate configurations are removed, while primary roles are kept.
    seen = set()
    output = []
    for role in raw_roles:
        key = (
            round(float(role["alpha"]), 12),
            round(float(role["geometry_gamma"]), 12),
            round(
                canonical_cap(
                    role["geometry_gamma"],
                    role["geometry_cap"],
                ),
                12,
            ),
            round(float(role["loss_gamma"]), 12),
            round(
                canonical_cap(
                    role["loss_gamma"],
                    role["loss_cap"],
                ),
                12,
            ),
        )
        if key in seen:
            continue
        seen.add(key)
        role["geometry_cap"] = canonical_cap(
            role["geometry_gamma"],
            role["geometry_cap"],
        )
        role["loss_cap"] = canonical_cap(
            role["loss_gamma"],
            role["loss_cap"],
        )
        output.append(role)

    required_roles = {
        "primary_unweighted",
        "primary_loss_weighted",
    }
    present = {
        role["role"]
        for role in output
    }
    if not required_roles.issubset(present):
        raise RuntimeError(
            "Primary configurations were unexpectedly deduplicated."
        )

    return output


ROLE_DEFINITIONS = build_role_definitions()
ROLE_BY_NAME = {
    role["role"]: role
    for role in ROLE_DEFINITIONS
}

GEOMETRY_GROUPS: dict[tuple[float, float, float], list[dict]] = {}
for role in ROLE_DEFINITIONS:
    key = (
        float(role["alpha"]),
        float(role["geometry_gamma"]),
        float(role["geometry_cap"]),
    )
    GEOMETRY_GROUPS.setdefault(
        key,
        [],
    ).append(role)

RUN_TOKEN = hashlib.sha256(
    json.dumps(
        {
            "final_basis": int(FINAL_BASIS),
            "regularization_strengths": REGULARIZATION_STRENGTHS,
            "label_split_seeds": LABEL_SPLIT_SEEDS,
            "role_definitions": ROLE_DEFINITIONS,
            "local_neighbors": LOCAL_KERNEL_NEIGHBORS,
            "local_multipliers": LOCAL_KERNEL_MULTIPLIERS,
            "local_source_gammas": LOCAL_SOURCE_GAMMAS,
            "local_source_caps": LOCAL_SOURCE_CAPS,
            "local_bandwidth_fraction": float(
                LOCAL_KERNEL_BANDWIDTH_FRACTION
            ),
        },
        sort_keys=True,
    ).encode("utf-8")
).hexdigest()[:12]


# ============================================================
# 5. Geometry construction and chunked Nyström extension
# ============================================================


def geometry_cache_file(
    alpha: float,
    geometry_gamma: float,
    geometry_cap: float,
) -> Path:
    token = config_token(
        alpha,
        geometry_gamma,
        geometry_cap,
    )
    return CACHE_DIR / f"geometry_{token}.npz"



def geometry_signature(
    alpha: float,
    geometry_gamma: float,
    geometry_cap: float,
) -> str:
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
            "alpha": float(alpha),
            "geometry_gamma": float(
                geometry_gamma
            ),
            "geometry_cap": float(
                canonical_cap(
                    geometry_gamma,
                    geometry_cap,
                )
            ),
            "max_basis": int(FINAL_BASIS),
            "eigen_solver_seed": int(
                EIGEN_SOLVER_SEED
            ),
            "generator_reference_nonconstant_modes": int(
                GENERATOR_REFERENCE_NONCONSTANT_MODES
            ),
            "version": 1,
        }
    )



def _load_geometry_npz(
    path: Path,
    max_basis: int,
) -> dict:
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        stored_eigenvalues = np.asarray(
            data["eigenvalues"],
            dtype=np.float64,
        )
        stored_eigenfunctions = np.asarray(
            data["eigenfunctions"],
            dtype=np.float64,
        )
        if (
            stored_eigenvalues.size < int(max_basis)
            or stored_eigenfunctions.shape[1] < int(max_basis)
        ):
            raise ValueError(
                f"Geometry cache {path} stores fewer than {max_basis} modes."
            )
        eigenvalues = stored_eigenvalues[:max_basis]
        eigenfunctions = stored_eigenfunctions[:, :max_basis]
        if "generator_eigenvalues" in data.files:
            generator = np.asarray(
                data["generator_eigenvalues"],
                dtype=np.float64,
            )[:max_basis]
        else:
            raw_generator = np.maximum(
                1.0 - eigenvalues,
                0.0,
            )
            stop = min(
                1 + GENERATOR_REFERENCE_NONCONSTANT_MODES,
                raw_generator.size,
            )
            positive = raw_generator[1:stop]
            positive = positive[positive > 0.0]
            scale = float(np.median(positive))
            generator = raw_generator / scale
        return {
            "eigenvalues": eigenvalues,
            "generator_eigenvalues": generator,
            "eigenfunctions": eigenfunctions,
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
            ) if "generator_scale" in data.files else math.nan,
        }



def build_or_load_geometry(
    base_kernel,
    probability_survey: np.ndarray,
    alpha: float,
    geometry_gamma: float,
    geometry_cap: float,
) -> tuple[dict, dict, bool]:
    is_primary_unweighted = (
        np.isclose(alpha, 0.0)
        and np.isclose(geometry_gamma, 0.0)
    )

    geometry_weights, weight_info = (
        tempered_inverse_selection_weights(
            probability_survey,
            geometry_gamma,
            MIN_SELECTION_PROBABILITY,
            canonical_cap(
                geometry_gamma,
                geometry_cap,
            ),
            WEIGHT_CLIP_QUANTILE,
        )
    )

    if (
        is_primary_unweighted
        and UNWEIGHTED_GEOMETRY_4096_FILE.is_file()
    ):
        geometry = _load_geometry_npz(
            UNWEIGHTED_GEOMETRY_4096_FILE,
            FINAL_BASIS,
        )
        if not np.isclose(
            geometry["alpha"],
            0.0,
        ):
            raise ValueError(
                "The reused 4096-mode geometry is not alpha=0."
            )
        if not np.allclose(
            geometry["quadrature_weights"],
            1.0,
            rtol=0.0,
            atol=1.0e-12,
        ):
            raise ValueError(
                "The reused 4096-mode geometry is not unweighted."
            )
        return geometry, weight_info, True

    cache_file = geometry_cache_file(
        alpha,
        geometry_gamma,
        geometry_cap,
    )
    signature = geometry_signature(
        alpha,
        geometry_gamma,
        geometry_cap,
    )

    if cache_file.is_file():
        try:
            with np.load(
                cache_file,
                allow_pickle=False,
            ) as data:
                cached_signature = (
                    str(data["signature"].item())
                    if "signature" in data.files
                    else ""
                )
            if cached_signature == signature:
                geometry = _load_geometry_npz(
                    cache_file,
                    FINAL_BASIS,
                )
                return geometry, weight_info, True
        except Exception as exc:
            warnings.warn(
                "Geometry cache could not be used; rebuilding: "
                f"{exc}"
            )

    kernel = base_kernel["kernel"]
    n = kernel.shape[0]
    q = np.asarray(
        kernel @ geometry_weights
    ).ravel()
    if np.any(q <= 0.0):
        raise ValueError(
            "Non-positive weighted kernel density q."
        )

    q_factor = np.power(
        q,
        -float(alpha),
    )
    kernel_alpha = (
        diags(q_factor)
        @ kernel
        @ diags(q_factor)
    ).tocsr()

    d = np.asarray(
        kernel_alpha @ geometry_weights
    ).ravel()
    if np.any(d <= 0.0):
        raise ValueError(
            "Non-positive row normalization d."
        )

    stationary = geometry_weights * d
    stationary /= np.sum(stationary)

    symmetric_factor = np.sqrt(
        geometry_weights / d
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
        int(FINAL_BASIS),
        n - 2,
    )
    rng = np.random.default_rng(
        EIGEN_SOLVER_SEED
    )

    print(
        "[Geometry] computing "
        f"{k} eigenfunctions: "
        f"alpha={alpha:g}, "
        f"gamma={geometry_gamma:g}, "
        f"cap={canonical_cap(geometry_gamma, geometry_cap):g}",
        flush=True,
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
        1 + GENERATOR_REFERENCE_NONCONSTANT_MODES,
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
        f"orthogonality RMS={orthogonality_rms:.3e}, "
        f"geometry ESS={weight_info['effective_sample_size']:.1f}",
        flush=True,
    )

    geometry = {
        "eigenvalues": eigenvalues,
        "generator_eigenvalues": generator_eigenvalues,
        "eigenfunctions": eigenfunctions,
        "quadrature_weights": geometry_weights,
        "q": q,
        "d": d,
        "stationary": stationary,
        "alpha": float(alpha),
        "generator_scale": generator_scale,
    }

    np.savez(
        cache_file,
        signature=np.array(signature),
        eigenvalues=eigenvalues,
        generator_eigenvalues=generator_eigenvalues,
        eigenfunctions=eigenfunctions,
        quadrature_weights=geometry_weights,
        q=q,
        d=d,
        stationary=stationary,
        alpha=np.array(alpha),
        generator_scale=np.array(generator_scale),
    )

    return geometry, weight_info, False



def nystrom_extend_chunked(
    query_positions: np.ndarray,
    observed_positions: np.ndarray,
    base_kernel,
    geometry: dict,
    chunk_size: int,
) -> dict:
    del observed_positions  # tree is already stored in base_kernel

    tree = base_kernel["tree"]
    n_neighbors = int(
        base_kernel["n_neighbors"]
    )
    bandwidth_neighbor = int(
        base_kernel["bandwidth_neighbor"]
    )
    rho_observed = np.asarray(
        base_kernel["rho"],
        dtype=np.float64,
    )

    try:
        distances, indices = tree.query(
            query_positions,
            k=n_neighbors,
            workers=-1,
        )
    except TypeError:
        distances, indices = tree.query(
            query_positions,
            k=n_neighbors,
        )

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    rho_query = (
        float(base_kernel["multiplier"])
        * distances[:, bandwidth_neighbor - 1]
    )
    positive = rho_query[rho_query > 0.0]
    floor = max(
        float(np.median(positive)) * 1.0e-8,
        np.finfo(np.float64).eps,
    )
    rho_query = np.maximum(
        rho_query,
        floor,
    )

    n_query = query_positions.shape[0]
    n_modes = geometry[
        "eigenfunctions"
    ].shape[1]
    extended = np.empty(
        (n_query, n_modes),
        dtype=np.float64,
    )
    row_normalization = np.empty(
        n_query,
        dtype=np.float64,
    )

    quadrature = geometry[
        "quadrature_weights"
    ]
    q_observed = geometry["q"]
    alpha = float(geometry["alpha"])
    eigenvalues = geometry["eigenvalues"]
    safe_eigenvalues = np.where(
        np.abs(eigenvalues) > 1.0e-10,
        eigenvalues,
        np.nan,
    )

    print(
        "[Nyström] extending "
        f"{n_query:,} query points in chunks of {chunk_size}...",
        flush=True,
    )
    start = time.perf_counter()

    for start_index in range(
        0,
        n_query,
        int(chunk_size),
    ):
        stop_index = min(
            start_index + int(chunk_size),
            n_query,
        )
        local_distances = distances[
            start_index:stop_index
        ]
        local_indices = indices[
            start_index:stop_index
        ]
        local_rho = rho_query[
            start_index:stop_index
        ]

        kernel_query = np.exp(
            -np.square(local_distances)
            / (
                local_rho[:, None]
                * rho_observed[local_indices]
            )
        )
        local_quadrature = quadrature[
            local_indices
        ]
        q_query = np.sum(
            kernel_query
            * local_quadrature,
            axis=1,
        )
        q_query = np.maximum(
            q_query,
            np.finfo(np.float64).eps,
        )
        kernel_alpha_query = kernel_query / (
            np.power(
                q_query[:, None],
                alpha,
            )
            * np.power(
                q_observed[local_indices],
                alpha,
            )
        )
        row_weight = (
            kernel_alpha_query
            * local_quadrature
        )
        row_sum = np.sum(
            row_weight,
            axis=1,
        )
        transition = (
            row_weight
            / row_sum[:, None]
        )

        extended[
            start_index:stop_index
        ] = (
            np.einsum(
                "ij,ijm->im",
                transition,
                geometry["eigenfunctions"][
                    local_indices,
                    :,
                ],
                optimize=True,
            )
            / safe_eigenvalues[None, :]
        )
        row_normalization[
            start_index:stop_index
        ] = row_sum

        if (
            stop_index == n_query
            or stop_index % max(512, chunk_size) == 0
        ):
            print(
                f"[Nyström] {stop_index:,}/{n_query:,}",
                flush=True,
            )

    elapsed = time.perf_counter() - start
    print(
        f"[Nyström] completed: elapsed={elapsed:.1f} s",
        flush=True,
    )

    if not np.all(
        np.isfinite(extended)
    ):
        invalid_modes = np.flatnonzero(
            ~np.all(
                np.isfinite(extended),
                axis=0,
            )
        )
        raise FloatingPointError(
            "Non-finite Nyström modes: "
            f"{invalid_modes[:10].tolist()}"
        )

    return {
        "eigenfunctions": extended,
        "rho_query": rho_query,
        "nearest_distance": distances[:, 0],
        "row_normalization": row_normalization,
    }


# ============================================================
# 6. Geometry-role regression
# ============================================================


def select_lambda_and_predict(
    geometry: dict,
    query_basis: np.ndarray,
    truth_survey: np.ndarray,
    truth_query: np.ndarray,
    primary_query_mask: np.ndarray,
    evaluation_weights: np.ndarray,
    loss_weights: np.ndarray,
    split: dict,
) -> tuple[dict, list[dict]]:
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

    phi = geometry[
        "eigenfunctions"
    ][:, :FINAL_BASIS]
    phi_train = phi[
        train_indices
    ]
    train_gram, train_rhs = (
        prepare_normal_equations(
            phi_train,
            truth_survey[train_indices],
            loss_weights[train_indices],
        )
    )

    validation_rows = []
    for regularization in REGULARIZATION_STRENGTHS:
        coefficients = solve_from_normal_equations(
            train_gram,
            train_rhs,
            geometry[
                "generator_eigenvalues"
            ],
            regularization,
        )
        prediction = (
            phi[validation_indices]
            @ coefficients
        )
        parent_metrics = evaluate_prediction(
            truth_survey[validation_indices],
            prediction,
            evaluation_weights[validation_indices],
        )
        unweighted_metrics = evaluate_prediction(
            truth_survey[validation_indices],
            prediction,
        )
        validation_rows.append(
            {
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
        )

    validation_df = pd.DataFrame(
        validation_rows
    ).sort_values(
        [
            "parent_validation_nrmse",
            "unweighted_validation_nrmse",
            "regularization_strength",
        ]
    )
    best = validation_df.iloc[0]
    selected_regularization = float(
        best["regularization_strength"]
    )

    phi_fit = phi[fit_indices]
    fit_gram, fit_rhs = prepare_normal_equations(
        phi_fit,
        truth_survey[fit_indices],
        loss_weights[fit_indices],
    )
    coefficients = solve_from_normal_equations(
        fit_gram,
        fit_rhs,
        geometry["generator_eigenvalues"],
        selected_regularization,
    )

    observed_prediction = (
        phi[test_indices]
        @ coefficients
    )
    query_prediction = (
        query_basis[:, :FINAL_BASIS]
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

    result = {
        "n_basis": int(FINAL_BASIS),
        "selected_regularization": selected_regularization,
        "parent_validation_nrmse": float(
            best["parent_validation_nrmse"]
        ),
        "unweighted_validation_nrmse": float(
            best["unweighted_validation_nrmse"]
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

    return result, validation_rows


# ============================================================
# 7. Local-kernel tuning and prediction
# ============================================================


def local_source_configs() -> list[tuple[float, float]]:
    configs = []
    for gamma in LOCAL_SOURCE_GAMMAS:
        if np.isclose(gamma, 0.0):
            configs.append((0.0, 1.0))
        else:
            for cap in LOCAL_SOURCE_CAPS:
                configs.append(
                    (
                        float(gamma),
                        float(cap),
                    )
                )
    # The two fixed local roles must remain available even when an
    # environment variable shortens the exploratory source-weight grid.
    configs.extend(
        [
            (0.0, 1.0),
            (1.0, 50.0),
        ]
    )
    return sorted(set(configs))


LOCAL_SOURCE_CONFIGS = local_source_configs()


def select_local_roles_and_predict(
    centered_survey: np.ndarray,
    centered_query: np.ndarray,
    probability_survey: np.ndarray,
    truth_survey: np.ndarray,
    truth_query: np.ndarray,
    primary_query_mask: np.ndarray,
    evaluation_weights: np.ndarray,
    split: dict,
) -> tuple[list[dict], list[dict]]:
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

    max_neighbors = min(
        max(LOCAL_KERNEL_NEIGHBORS),
        train_indices.size,
    )
    validation_distances, validation_neighbor_indices = (
        _query_neighbors(
            centered_survey[validation_indices],
            centered_survey[train_indices],
            max_neighbors,
        )
    )

    validation_records = []

    for source_gamma, source_cap in LOCAL_SOURCE_CONFIGS:
        source_weights, source_info = (
            tempered_inverse_selection_weights(
                probability_survey,
                source_gamma,
                MIN_SELECTION_PROBABILITY,
                source_cap,
                WEIGHT_CLIP_QUANTILE,
            )
        )
        train_source_weights = source_weights[
            train_indices
        ]

        for requested_neighbors in LOCAL_KERNEL_NEIGHBORS:
            n_neighbors = min(
                int(requested_neighbors),
                max_neighbors,
            )
            bandwidth_neighbor = min(
                n_neighbors,
                max(
                    1,
                    int(
                        round(
                            n_neighbors
                            * LOCAL_KERNEL_BANDWIDTH_FRACTION
                        )
                    ),
                ),
            )
            for multiplier in LOCAL_KERNEL_MULTIPLIERS:
                prediction = _local_kernel_from_neighbors(
                    validation_distances,
                    validation_neighbor_indices,
                    truth_survey[train_indices],
                    train_source_weights,
                    n_neighbors,
                    bandwidth_neighbor,
                    multiplier,
                )
                parent_metrics = evaluate_prediction(
                    truth_survey[validation_indices],
                    prediction,
                    evaluation_weights[validation_indices],
                )
                unweighted_metrics = evaluate_prediction(
                    truth_survey[validation_indices],
                    prediction,
                )
                validation_records.append(
                    {
                        "source_gamma": float(
                            source_gamma
                        ),
                        "source_cap": float(
                            source_cap
                        ),
                        "source_ess": float(
                            source_info[
                                "effective_sample_size"
                            ]
                        ),
                        "n_neighbors": int(
                            n_neighbors
                        ),
                        "bandwidth_neighbor": int(
                            bandwidth_neighbor
                        ),
                        "bandwidth_multiplier": float(
                            multiplier
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
                    }
                )

    validation_df = pd.DataFrame(
        validation_records
    )

    role_filters = {
        "local_unweighted": (
            np.isclose(
                validation_df[
                    "source_gamma"
                ].to_numpy(dtype=np.float64),
                0.0,
            )
        ),
        "local_full_inverse": (
            np.isclose(
                validation_df[
                    "source_gamma"
                ].to_numpy(dtype=np.float64),
                1.0,
            )
            & np.isclose(
                validation_df[
                    "source_cap"
                ].to_numpy(dtype=np.float64),
                50.0,
            )
        ),
        "local_selection_optimized": np.ones(
            validation_df.shape[0],
            dtype=bool,
        ),
    }

    selected_rows = []
    for role, mask in role_filters.items():
        candidates = validation_df.loc[
            mask
        ].sort_values(
            [
                "parent_validation_nrmse",
                "unweighted_validation_nrmse",
                "n_neighbors",
                "bandwidth_multiplier",
                "source_gamma",
                "source_cap",
            ]
        )
        if candidates.empty:
            raise ValueError(
                f"No local candidates for role={role}."
            )
        best = candidates.iloc[0].to_dict()
        best["role"] = role
        selected_rows.append(best)

    max_fit_neighbors = min(
        max(LOCAL_KERNEL_NEIGHBORS),
        fit_indices.size,
    )
    test_distances, test_neighbor_indices = (
        _query_neighbors(
            centered_survey[test_indices],
            centered_survey[fit_indices],
            max_fit_neighbors,
        )
    )
    query_distances, query_neighbor_indices = (
        _query_neighbors(
            centered_query,
            centered_survey[fit_indices],
            max_fit_neighbors,
        )
    )

    results = []
    for selected in selected_rows:
        source_weights, source_info = (
            tempered_inverse_selection_weights(
                probability_survey,
                float(selected["source_gamma"]),
                MIN_SELECTION_PROBABILITY,
                float(selected["source_cap"]),
                WEIGHT_CLIP_QUANTILE,
            )
        )
        n_neighbors = min(
            int(selected["n_neighbors"]),
            max_fit_neighbors,
        )
        bandwidth_neighbor = min(
            n_neighbors,
            int(selected["bandwidth_neighbor"]),
        )
        multiplier = float(
            selected["bandwidth_multiplier"]
        )

        observed_prediction = (
            _local_kernel_from_neighbors(
                test_distances,
                test_neighbor_indices,
                truth_survey[fit_indices],
                source_weights[fit_indices],
                n_neighbors,
                bandwidth_neighbor,
                multiplier,
            )
        )
        query_prediction = (
            _local_kernel_from_neighbors(
                query_distances,
                query_neighbor_indices,
                truth_survey[fit_indices],
                source_weights[fit_indices],
                n_neighbors,
                bandwidth_neighbor,
                multiplier,
            )
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

        results.append(
            {
                "role": selected["role"],
                "source_gamma": float(
                    selected["source_gamma"]
                ),
                "source_cap": float(
                    selected["source_cap"]
                ),
                "source_ess": float(
                    source_info[
                        "effective_sample_size"
                    ]
                ),
                "n_neighbors": int(
                    n_neighbors
                ),
                "bandwidth_neighbor": int(
                    bandwidth_neighbor
                ),
                "bandwidth_multiplier": float(
                    multiplier
                ),
                "parent_validation_nrmse": float(
                    selected[
                        "parent_validation_nrmse"
                    ]
                ),
                "unweighted_validation_nrmse": float(
                    selected[
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

    return results, validation_records


# ============================================================
# 8. Pairing and aggregation
# ============================================================


def paired_summary(
    runs: pd.DataFrame,
    contrasts: list[tuple[str, str, str]],
    family: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metric_columns = {
        "parent": "parent_normalized_rmse",
        "observed_unweighted": "unweighted_normalized_rmse",
        "query_primary": "query_primary_normalized_rmse",
    }

    paired_frames = []
    summary_records = []

    for contrast_index, (
        contrast_name,
        role_a,
        role_b,
    ) in enumerate(contrasts):
        subset = runs.loc[
            runs["role"].isin(
                [role_a, role_b]
            )
        ].copy()
        if subset["role"].nunique() != 2:
            continue

        pivot = subset.pivot(
            index="replicate",
            columns="role",
            values=list(
                metric_columns.values()
            ),
        )
        if (
            role_a not in pivot.columns.get_level_values(1)
            or role_b not in pivot.columns.get_level_values(1)
        ):
            continue

        frame = pd.DataFrame(
            {
                "family": family,
                "contrast": contrast_name,
                "role_a": role_a,
                "role_b": role_b,
                "replicate": pivot.index.to_numpy(
                    dtype=np.int64
                ),
            }
        )

        for metric_index, (
            metric_name,
            column,
        ) in enumerate(metric_columns.items()):
            differences = (
                pivot[column][role_a]
                - pivot[column][role_b]
            ).to_numpy(dtype=np.float64)
            frame[
                f"{metric_name}_difference_a_minus_b"
            ] = differences

            inference = bootstrap_mean_difference(
                differences,
                SEED_BOOTSTRAP_REPLICATES,
                SEED_BOOTSTRAP_SEED
                + 101 * contrast_index
                + 17 * metric_index
                + len(family),
            )
            summary_records.append(
                {
                    "family": family,
                    "contrast": contrast_name,
                    "role_a": role_a,
                    "role_b": role_b,
                    "metric": metric_name,
                    "n_replicates": int(
                        differences.size
                    ),
                    **inference,
                }
            )

        paired_frames.append(frame)

    paired_df = (
        pd.concat(
            paired_frames,
            ignore_index=True,
        )
        if paired_frames
        else pd.DataFrame()
    )
    summary_df = pd.DataFrame(
        summary_records
    )

    return paired_df, summary_df



def aggregate_runs(
    runs: pd.DataFrame,
    family: str,
) -> pd.DataFrame:
    metric_columns = [
        "parent_validation_nrmse",
        "unweighted_validation_nrmse",
        "parent_normalized_rmse",
        "unweighted_normalized_rmse",
        "query_primary_normalized_rmse",
        "parent_median_direction_error_deg",
        "query_primary_median_direction_error_deg",
        "parent_vector_gain_through_origin",
        "query_primary_vector_gain_through_origin",
    ]

    records = []
    for role, group in runs.groupby(
        "role",
        sort=False,
    ):
        record = {
            "family": family,
            "role": role,
            "n_replicates": int(
                group.shape[0]
            ),
        }
        for column in metric_columns:
            if column not in group.columns:
                continue
            values = group[column].to_numpy(
                dtype=np.float64
            )
            record[f"{column}_mean"] = float(
                np.mean(values)
            )
            record[f"{column}_std"] = float(
                np.std(values, ddof=1)
            ) if values.size > 1 else 0.0
        records.append(record)

    return pd.DataFrame(records)


# ============================================================
# 9. 入力読込み
# ============================================================

font_name = configure_font()

print("=" * 124, flush=True)
print(
    "Paper I: Mock-2 final restricted geometry sensitivity and local-kernel comparison",
    flush=True,
)
print("=" * 124, flush=True)
print(
    f"Python / NumPy / SciPy : {platform.python_version()} / {np.__version__} / {scipy.__version__}",
    flush=True,
)
print(f"Matplotlib font         : {font_name}", flush=True)
print(f"Mock-1                  : {MOCK1_FILE}", flush=True)
print(f"Mock-2                  : {MOCK2_FILE}", flush=True)
print(f"Practical basis         : {FINAL_BASIS}", flush=True)
print(f"Lambda grid             : {REGULARIZATION_STRENGTHS}", flush=True)
print(f"Label split seeds       : {len(LABEL_SPLIT_SEEDS)}", flush=True)
print(f"Nyström chunk           : {NYSTROM_QUERY_CHUNK}", flush=True)
print(f"Run token               : {RUN_TOKEN}", flush=True)
print(f"Output                   : {OUTPUT_DIR}", flush=True)

for required_path in [
    MOCK1_FILE,
    MOCK2_FILE,
    OPTIMIZATION_PREDICTIONS_FILE,
    PRACTICAL_SELECTION_FILE,
]:
    if not required_path.is_file():
        raise FileNotFoundError(required_path)

complete = _load_mock1(
    MOCK1_FILE
)
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

if FINAL_BASIS >= n_survey - 1:
    raise ValueError(
        "FINAL_BASIS must be smaller than the survey sample size minus one."
    )
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
if int(np.sum(primary_query_mask)) < 5:
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

naive_weights = np.ones(
    n_survey,
    dtype=np.float64,
)
primary_loss_weights, primary_loss_info = (
    tempered_inverse_selection_weights(
        probability_survey,
        PRIMARY_LOSS_GAMMA,
        MIN_SELECTION_PROBABILITY,
        PRIMARY_LOSS_CAP,
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
    "Survey/query/primary-query: "
    f"{n_survey:,}/{n_query:,}/{int(np.sum(primary_query_mask)):,}",
    flush=True,
)
print(
    "Selection-weighted loss ESS: "
    f"{primary_loss_info['effective_sample_size']:.1f}",
    flush=True,
)
print(
    "Parent-weighted evaluation ESS: "
    f"{evaluation_weight_info['effective_sample_size']:.1f}",
    flush=True,
)

role_config_df = pd.DataFrame(
    ROLE_DEFINITIONS
)
role_config_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_geometry_role_configurations.csv",
    index=False,
)
print("-" * 124, flush=True)
print("Restricted geometry/loss configurations", flush=True)
print(role_config_df.to_string(index=False), flush=True)

base_kernel = _build_base_kernel(
    centered_survey,
    GRAPH_NEIGHBORS,
    GRAPH_BANDWIDTH_NEIGHBOR,
    GRAPH_BANDWIDTH_MULTIPLIER,
)


# ============================================================
# 10. Restricted geometry sensitivity
# ============================================================

geometry_run_frames = []
geometry_validation_frames = []
geometry_info_records = []

for geometry_index, (
    geometry_key,
    grouped_roles,
) in enumerate(
    GEOMETRY_GROUPS.items(),
    start=1,
):
    alpha, geometry_gamma, geometry_cap = geometry_key

    print("=" * 124, flush=True)
    print(
        f"Geometry {geometry_index}/{len(GEOMETRY_GROUPS)}: "
        f"alpha={alpha:g}, gamma={geometry_gamma:g}, cap={geometry_cap:g}",
        flush=True,
    )
    print(
        "Roles: "
        + ", ".join(
            role["role"]
            for role in grouped_roles
        ),
        flush=True,
    )

    geometry, geometry_weight_info, cache_loaded = (
        build_or_load_geometry(
            base_kernel,
            probability_survey,
            alpha,
            geometry_gamma,
            geometry_cap,
        )
    )
    geometry_info_records.append(
        {
            "alpha": float(alpha),
            "geometry_gamma": float(
                geometry_gamma
            ),
            "geometry_cap": float(
                geometry_cap
            ),
            "geometry_ess": float(
                geometry_weight_info[
                    "effective_sample_size"
                ]
            ),
            "geometry_ess_fraction": float(
                geometry_weight_info[
                    "effective_sample_fraction"
                ]
            ),
            "geometry_weight_max": float(
                geometry_weight_info[
                    "normalized_weight_max"
                ]
            ),
            "cache_loaded": bool(cache_loaded),
        }
    )

    query_extension = nystrom_extend_chunked(
        centered_query,
        centered_survey,
        base_kernel,
        geometry,
        NYSTROM_QUERY_CHUNK,
    )
    query_basis = np.asarray(
        query_extension["eigenfunctions"],
        dtype=np.float64,
    )

    geometry_token = config_token(
        alpha,
        geometry_gamma,
        geometry_cap,
    )

    for replicate, seed in enumerate(
        LABEL_SPLIT_SEEDS,
        start=1,
    ):
        checkpoint_file = (
            CHECKPOINT_DIR
            / f"geometry_{geometry_token}_seed_{seed}_{RUN_TOKEN}.csv"
        )
        validation_checkpoint_file = (
            CHECKPOINT_DIR
            / f"geometry_{geometry_token}_seed_{seed}_{RUN_TOKEN}_validation.csv"
        )

        if (
            checkpoint_file.is_file()
            and validation_checkpoint_file.is_file()
        ):
            print(
                f"[Checkpoint] geometry={geometry_token}, seed={seed}",
                flush=True,
            )
            geometry_run_frames.append(
                pd.read_csv(checkpoint_file)
            )
            geometry_validation_frames.append(
                pd.read_csv(
                    validation_checkpoint_file
                )
            )
            continue

        print(
            f"[Geometry regression] replicate={replicate}/{len(LABEL_SPLIT_SEEDS)}, seed={seed}",
            flush=True,
        )
        split = _make_split(
            n_survey,
            TRAIN_FRACTION,
            VALIDATION_FRACTION,
            int(seed),
        )

        seed_run_records = []
        seed_validation_records = []

        for role in grouped_roles:
            loss_weights, loss_info = (
                tempered_inverse_selection_weights(
                    probability_survey,
                    float(role["loss_gamma"]),
                    MIN_SELECTION_PROBABILITY,
                    float(role["loss_cap"]),
                    WEIGHT_CLIP_QUANTILE,
                )
            )

            start = time.perf_counter()
            result, validation_rows = (
                select_lambda_and_predict(
                    geometry,
                    query_basis,
                    truth_survey,
                    truth_query,
                    primary_query_mask,
                    evaluation_weights,
                    loss_weights,
                    split,
                )
            )
            elapsed = time.perf_counter() - start

            seed_run_records.append(
                {
                    "replicate": int(replicate),
                    "label_seed": int(seed),
                    "role": role["role"],
                    "label": role["label"],
                    "alpha": float(alpha),
                    "geometry_gamma": float(
                        geometry_gamma
                    ),
                    "geometry_cap": float(
                        geometry_cap
                    ),
                    "geometry_ess": float(
                        geometry_weight_info[
                            "effective_sample_size"
                        ]
                    ),
                    "loss_gamma": float(
                        role["loss_gamma"]
                    ),
                    "loss_cap": float(
                        role["loss_cap"]
                    ),
                    "loss_ess": float(
                        loss_info[
                            "effective_sample_size"
                        ]
                    ),
                    "elapsed_seconds": float(
                        elapsed
                    ),
                    **result,
                }
            )
            for row in validation_rows:
                seed_validation_records.append(
                    {
                        "replicate": int(replicate),
                        "label_seed": int(seed),
                        "role": role["role"],
                        "alpha": float(alpha),
                        "geometry_gamma": float(
                            geometry_gamma
                        ),
                        "geometry_cap": float(
                            geometry_cap
                        ),
                        "loss_gamma": float(
                            role["loss_gamma"]
                        ),
                        "loss_cap": float(
                            role["loss_cap"]
                        ),
                        "n_basis": int(
                            FINAL_BASIS
                        ),
                        **row,
                    }
                )

            print(
                "  "
                f"{role['role']}: "
                f"lambda={result['selected_regularization']:g}, "
                f"parent={result['parent_normalized_rmse']:.6f}, "
                f"unweighted={result['unweighted_normalized_rmse']:.6f}, "
                f"query={result['query_primary_normalized_rmse']:.6f}, "
                f"elapsed={elapsed:.1f} s",
                flush=True,
            )

        seed_run_df = pd.DataFrame(
            seed_run_records
        )
        seed_validation_df = pd.DataFrame(
            seed_validation_records
        )
        seed_run_df.to_csv(
            checkpoint_file,
            index=False,
        )
        seed_validation_df.to_csv(
            validation_checkpoint_file,
            index=False,
        )
        geometry_run_frames.append(
            seed_run_df
        )
        geometry_validation_frames.append(
            seed_validation_df
        )

    del query_basis
    del query_extension
    del geometry
    gc.collect()

geometry_runs_df = pd.concat(
    geometry_run_frames,
    ignore_index=True,
)
geometry_validation_df = pd.concat(
    geometry_validation_frames,
    ignore_index=True,
)
geometry_info_df = pd.DataFrame(
    geometry_info_records
)


# ============================================================
# 11. Current primary-result audit
# ============================================================

primary_audit_records = []
if PRACTICAL_RUNS_FILE.is_file():
    current_practical = pd.read_csv(
        PRACTICAL_RUNS_FILE
    )
    model_to_role = {
        "naive": "primary_unweighted",
        "loss_weighted": "primary_loss_weighted",
    }
    for model_name, role_name in model_to_role.items():
        expected = current_practical.loc[
            current_practical["model"]
            == model_name
        ].copy()
        observed = geometry_runs_df.loc[
            geometry_runs_df["role"]
            == role_name
        ].copy()
        if expected.empty or observed.empty:
            continue
        merged = expected.merge(
            observed,
            on=[
                "replicate",
                "label_seed",
            ],
            suffixes=("_expected", "_new"),
            validate="one_to_one",
        )
        for metric in [
            "parent_normalized_rmse",
            "unweighted_normalized_rmse",
            "query_primary_normalized_rmse",
        ]:
            difference = (
                merged[f"{metric}_new"]
                - merged[f"{metric}_expected"]
            ).to_numpy(dtype=np.float64)
            primary_audit_records.append(
                {
                    "model": model_name,
                    "role": role_name,
                    "metric": metric,
                    "n_rows": int(
                        difference.size
                    ),
                    "max_abs_difference": float(
                        np.max(
                            np.abs(difference)
                        )
                    ),
                    "mean_difference": float(
                        np.mean(difference)
                    ),
                }
            )
primary_audit_df = pd.DataFrame(
    primary_audit_records
)


# ============================================================
# 12. Same-split local-kernel baseline
# ============================================================

local_run_frames = []
local_validation_frames = []

print("=" * 124, flush=True)
print(
    "Same-split adaptive local-kernel comparison",
    flush=True,
)
print("=" * 124, flush=True)

for replicate, seed in enumerate(
    LABEL_SPLIT_SEEDS,
    start=1,
):
    checkpoint_file = (
        CHECKPOINT_DIR
        / f"local_seed_{seed}_{RUN_TOKEN}.csv"
    )
    validation_checkpoint_file = (
        CHECKPOINT_DIR
        / f"local_seed_{seed}_{RUN_TOKEN}_validation.csv"
    )

    if (
        checkpoint_file.is_file()
        and validation_checkpoint_file.is_file()
    ):
        print(
            f"[Checkpoint] local seed={seed}",
            flush=True,
        )
        local_run_frames.append(
            pd.read_csv(checkpoint_file)
        )
        local_validation_frames.append(
            pd.read_csv(
                validation_checkpoint_file
            )
        )
        continue

    print(
        f"[Local regression] replicate={replicate}/{len(LABEL_SPLIT_SEEDS)}, seed={seed}",
        flush=True,
    )
    split = _make_split(
        n_survey,
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        int(seed),
    )
    start = time.perf_counter()
    results, validation_rows = (
        select_local_roles_and_predict(
            centered_survey,
            centered_query,
            probability_survey,
            truth_survey,
            truth_query,
            primary_query_mask,
            evaluation_weights,
            split,
        )
    )
    elapsed = time.perf_counter() - start

    for row in results:
        row["replicate"] = int(replicate)
        row["label_seed"] = int(seed)
        row["elapsed_seconds"] = float(
            elapsed
        )
    for row in validation_rows:
        row["replicate"] = int(replicate)
        row["label_seed"] = int(seed)

    seed_run_df = pd.DataFrame(results)
    seed_validation_df = pd.DataFrame(
        validation_rows
    )
    seed_run_df.to_csv(
        checkpoint_file,
        index=False,
    )
    seed_validation_df.to_csv(
        validation_checkpoint_file,
        index=False,
    )
    local_run_frames.append(seed_run_df)
    local_validation_frames.append(
        seed_validation_df
    )

    print(
        "  completed: "
        + "; ".join(
            f"{row['role']} parent={row['parent_normalized_rmse']:.6f}, "
            f"query={row['query_primary_normalized_rmse']:.6f}"
            for row in results
        )
        + f"; elapsed={elapsed:.1f} s",
        flush=True,
    )

local_runs_df = pd.concat(
    local_run_frames,
    ignore_index=True,
)
local_validation_df = pd.concat(
    local_validation_frames,
    ignore_index=True,
)


# ============================================================
# 13. Paired comparisons
# ============================================================

geometry_contrasts = [
    (
        "primary_loss_minus_primary_unweighted",
        "primary_loss_weighted",
        "primary_unweighted",
    ),
    (
        "legacy_geometry_only_minus_primary_unweighted",
        "legacy_geometry_only",
        "primary_unweighted",
    ),
    (
        "legacy_geometry_plus_loss_minus_primary_loss",
        "legacy_geometry_plus_loss",
        "primary_loss_weighted",
    ),
    (
        "full_inverse_geometry_only_minus_primary_unweighted",
        "full_inverse_geometry_only",
        "primary_unweighted",
    ),
    (
        "full_inverse_joint_minus_primary_loss",
        "full_inverse_joint",
        "primary_loss_weighted",
    ),
]

geometry_paired_df, geometry_paired_summary_df = (
    paired_summary(
        geometry_runs_df,
        geometry_contrasts,
        "geometry",
    )
)

# Local and primary diffusion methods are combined for paired comparison.
diffusion_primary_for_local = geometry_runs_df.loc[
    geometry_runs_df["role"].isin(
        [
            "primary_unweighted",
            "primary_loss_weighted",
        ]
    )
].copy()
local_comparison_runs_df = pd.concat(
    [
        diffusion_primary_for_local,
        local_runs_df,
    ],
    ignore_index=True,
    sort=False,
)

local_contrasts = [
    (
        "local_unweighted_minus_primary_unweighted",
        "local_unweighted",
        "primary_unweighted",
    ),
    (
        "local_full_inverse_minus_primary_unweighted",
        "local_full_inverse",
        "primary_unweighted",
    ),
    (
        "local_selection_optimized_minus_primary_unweighted",
        "local_selection_optimized",
        "primary_unweighted",
    ),
    (
        "primary_loss_minus_local_selection_optimized",
        "primary_loss_weighted",
        "local_selection_optimized",
    ),
]

local_paired_df, local_paired_summary_df = (
    paired_summary(
        local_comparison_runs_df,
        local_contrasts,
        "local",
    )
)

geometry_aggregate_df = aggregate_runs(
    geometry_runs_df,
    "geometry",
)
local_aggregate_df = aggregate_runs(
    local_runs_df,
    "local",
)
all_method_aggregate_df = pd.concat(
    [
        geometry_aggregate_df,
        local_aggregate_df,
    ],
    ignore_index=True,
)

local_selected_frequency_df = (
    local_runs_df.groupby(
        [
            "role",
            "source_gamma",
            "source_cap",
            "n_neighbors",
            "bandwidth_neighbor",
            "bandwidth_multiplier",
        ],
        as_index=False,
    )
    .size()
    .rename(columns={"size": "count"})
    .sort_values(
        [
            "role",
            "count",
            "source_gamma",
            "source_cap",
            "n_neighbors",
            "bandwidth_multiplier",
        ],
        ascending=[
            True,
            False,
            True,
            True,
            True,
            True,
        ],
    )
)

geometry_lambda_frequency_df = (
    geometry_runs_df.groupby(
        [
            "role",
            "selected_regularization",
        ],
        as_index=False,
    )
    .size()
    .rename(columns={"size": "count"})
    .sort_values(
        [
            "role",
            "count",
            "selected_regularization",
        ],
        ascending=[
            True,
            False,
            True,
        ],
    )
)


# ============================================================
# 14. Diagnostic figures
# ============================================================

role_labels = {
    row["role"]: row["label"]
    for row in ROLE_DEFINITIONS
}
role_labels.update(
    {
        "local_unweighted": "Local unweighted",
        "local_full_inverse": "Local full inverse",
        "local_selection_optimized": "Local selection optimized",
    }
)
plot_role_labels = {
    "primary_unweighted": "Primary unweighted",
    "primary_loss_weighted": "Primary weighted",
    "legacy_geometry_only": "Legacy geometry",
    "legacy_geometry_plus_loss": "Legacy geom. + loss",
    "full_inverse_geometry_only": "Full-inverse geometry",
    "full_inverse_joint": "Full-inverse joint",
    "local_unweighted": "Local unweighted",
    "local_full_inverse": "Local full inverse",
    "local_selection_optimized": "Local optimized",
}

# ---------- Figure A: restricted geometry sensitivity ----------
geometry_order = [
    role["role"]
    for role in ROLE_DEFINITIONS
    if role["role"] in set(
        geometry_aggregate_df["role"]
    )
]

fig, axes = plt.subplots(
    2,
    2,
    figsize=(13.0, 9.2),
    constrained_layout=True,
)

metric_specs = [
    (
        axes[0, 0],
        "parent_normalized_rmse",
        "Observed parent-weighted risk",
    ),
    (
        axes[0, 1],
        "unweighted_normalized_rmse",
        "Observed-unweighted risk",
    ),
    (
        axes[1, 0],
        "query_primary_normalized_rmse",
        "Primary complete-query risk",
    ),
]

for panel_index, (
    ax,
    metric,
    title,
) in enumerate(metric_specs):
    means = []
    errors = []
    labels = []
    for role in geometry_order:
        row = geometry_aggregate_df.loc[
            geometry_aggregate_df["role"]
            == role
        ].iloc[0]
        means.append(
            float(row[f"{metric}_mean"])
        )
        errors.append(
            float(row[f"{metric}_std"])
        )
        labels.append(
            plot_role_labels.get(role, role)
        )
    positions = np.arange(len(labels))
    ax.errorbar(
        positions,
        means,
        yerr=errors,
        marker="o",
        linestyle="none",
        capsize=4,
    )
    ax.set_xticks(positions)
    ax.set_xticklabels(
        labels,
        rotation=28,
        ha="right",
    )
    ax.set_ylabel(r"$\mathrm{NRMSE}_{3\mathrm{D}}$")
    ax.set_title(title, pad=12)
    ax.text(
        0.02,
        0.98,
        f"({chr(ord('a') + panel_index)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )

ax = axes[1, 1]
parent_geometry_summary = geometry_paired_summary_df.loc[
    geometry_paired_summary_df["metric"]
    == "parent"
].copy()
if not parent_geometry_summary.empty:
    positions = np.arange(
        parent_geometry_summary.shape[0]
    )
    means = parent_geometry_summary[
        "mean_difference"
    ].to_numpy(dtype=np.float64)
    low = parent_geometry_summary[
        "ci_low"
    ].to_numpy(dtype=np.float64)
    high = parent_geometry_summary[
        "ci_high"
    ].to_numpy(dtype=np.float64)
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )
    ax.errorbar(
        means,
        positions,
        xerr=np.vstack(
            [
                means - low,
                high - means,
            ]
        ),
        marker="o",
        linestyle="none",
        capsize=4,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(
        parent_geometry_summary[
            "contrast"
        ].str.replace(
            "_",
            " ",
            regex=False,
        )
    )
    ax.invert_yaxis()
ax.set_xlabel(
    "Mean paired parent-risk difference"
)
ax.set_title(
    "Restricted geometry contrasts",
    pad=12,
)
ax.text(
    0.03,
    0.04,
    "Negative values favor the first-named method",
    transform=ax.transAxes,
    ha="left",
    va="bottom",
    fontsize=9.0,
)
ax.text(
    0.02,
    0.98,
    "(d)",
    transform=ax.transAxes,
    ha="left",
    va="top",
    fontsize=14,
    fontweight="bold",
    clip_on=False,
)

geometry_figure_pdf, geometry_figure_png = save_figure(
    fig,
    "geometry_sensitivity_composite",
)

# ---------- Figure B: local-kernel comparison ----------
fig, axes = plt.subplots(
    2,
    2,
    figsize=(13.0, 9.2),
    constrained_layout=True,
)

local_plot_roles = [
    "primary_unweighted",
    "primary_loss_weighted",
    "local_unweighted",
    "local_full_inverse",
    "local_selection_optimized",
]
local_plot_roles = [
    role
    for role in local_plot_roles
    if role in set(
        local_comparison_runs_df["role"]
    )
]

for panel_index, (
    ax,
    metric,
    title,
) in enumerate(
    [
        (
            axes[0, 0],
            "parent_normalized_rmse",
            "Observed parent-weighted risk",
        ),
        (
            axes[0, 1],
            "query_primary_normalized_rmse",
            "Primary complete-query risk",
        ),
    ]
):
    for role in local_plot_roles:
        selected = local_comparison_runs_df.loc[
            local_comparison_runs_df["role"]
            == role
        ].sort_values("replicate")
        ax.plot(
            selected["replicate"],
            selected[metric],
            marker="o",
            label=plot_role_labels.get(role, role),
        )
    ax.set_xticks(
        np.arange(
            1,
            len(LABEL_SPLIT_SEEDS) + 1,
        )
    )
    ax.set_xlabel("Label-split replicate")
    ax.set_ylabel(r"$\mathrm{NRMSE}_{3\mathrm{D}}$")
    ax.set_title(title, pad=12)
    ax.legend(
        fontsize=8.3,
        loc="best",
    )
    ax.text(
        0.02,
        0.98,
        f"({chr(ord('a') + panel_index)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )

for panel_offset, (
    ax,
    metric,
    title,
) in enumerate(
    [
        (
            axes[1, 0],
            "parent",
            "Paired parent-risk differences",
        ),
        (
            axes[1, 1],
            "query_primary",
            "Paired complete-query differences",
        ),
    ],
    start=2,
):
    selected = local_paired_summary_df.loc[
        local_paired_summary_df["metric"]
        == metric
    ].copy()
    positions = np.arange(selected.shape[0])
    means = selected[
        "mean_difference"
    ].to_numpy(dtype=np.float64)
    low = selected[
        "ci_low"
    ].to_numpy(dtype=np.float64)
    high = selected[
        "ci_high"
    ].to_numpy(dtype=np.float64)
    ax.axvline(
        0.0,
        linestyle="--",
        linewidth=1.0,
    )
    ax.errorbar(
        means,
        positions,
        xerr=np.vstack(
            [
                means - low,
                high - means,
            ]
        ),
        marker="o",
        linestyle="none",
        capsize=4,
    )
    ax.set_yticks(positions)
    ax.set_yticklabels(
        selected["contrast"].str.replace(
            "_",
            " ",
            regex=False,
        )
    )
    ax.invert_yaxis()
    ax.set_xlabel(
        "Mean paired difference"
    )
    ax.set_title(title, pad=12)
    ax.text(
        0.02,
        0.98,
        f"({chr(ord('a') + panel_offset)})",
        transform=ax.transAxes,
        ha="left",
        va="top",
        fontsize=14,
        fontweight="bold",
        clip_on=False,
    )

local_figure_pdf, local_figure_png = save_figure(
    fig,
    "local_comparison_composite",
)


# ============================================================
# 15. Save outputs
# ============================================================

paths = {
    "geometry_roles_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_role_configurations.csv",
    "geometry_info_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_information.csv",
    "geometry_validation_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_validation.csv",
    "geometry_runs_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_runs.csv",
    "geometry_aggregate_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_aggregate.csv",
    "geometry_paired_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_paired.csv",
    "geometry_paired_summary_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_paired_summary.csv",
    "geometry_lambda_frequency_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_geometry_lambda_frequency.csv",
    "primary_audit_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_primary_result_audit.csv",
    "local_validation_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_local_validation.csv",
    "local_runs_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_local_runs.csv",
    "local_aggregate_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_local_aggregate.csv",
    "local_selected_frequency_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_local_selected_frequency.csv",
    "local_paired_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_local_paired.csv",
    "local_paired_summary_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_local_paired_summary.csv",
    "all_method_aggregate_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_all_method_aggregate.csv",
    "summary_json": OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json",
}

role_config_df.to_csv(
    paths["geometry_roles_csv"],
    index=False,
)
geometry_info_df.to_csv(
    paths["geometry_info_csv"],
    index=False,
)
geometry_validation_df.to_csv(
    paths["geometry_validation_csv"],
    index=False,
)
geometry_runs_df.to_csv(
    paths["geometry_runs_csv"],
    index=False,
)
geometry_aggregate_df.to_csv(
    paths["geometry_aggregate_csv"],
    index=False,
)
geometry_paired_df.to_csv(
    paths["geometry_paired_csv"],
    index=False,
)
geometry_paired_summary_df.to_csv(
    paths["geometry_paired_summary_csv"],
    index=False,
)
geometry_lambda_frequency_df.to_csv(
    paths["geometry_lambda_frequency_csv"],
    index=False,
)
primary_audit_df.to_csv(
    paths["primary_audit_csv"],
    index=False,
)
local_validation_df.to_csv(
    paths["local_validation_csv"],
    index=False,
)
local_runs_df.to_csv(
    paths["local_runs_csv"],
    index=False,
)
local_aggregate_df.to_csv(
    paths["local_aggregate_csv"],
    index=False,
)
local_selected_frequency_df.to_csv(
    paths["local_selected_frequency_csv"],
    index=False,
)
local_paired_df.to_csv(
    paths["local_paired_csv"],
    index=False,
)
local_paired_summary_df.to_csv(
    paths["local_paired_summary_csv"],
    index=False,
)
all_method_aggregate_df.to_csv(
    paths["all_method_aggregate_csv"],
    index=False,
)

summary_payload = {
    "environment": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "matplotlib_font": font_name,
    },
    "inputs": {
        "mock1": str(MOCK1_FILE),
        "mock2": str(MOCK2_FILE),
        "optimization_predictions": str(
            OPTIMIZATION_PREDICTIONS_FILE
        ),
        "practical_selection": str(
            PRACTICAL_SELECTION_FILE
        ),
        "practical_runs": str(
            PRACTICAL_RUNS_FILE
        ),
    },
    "configuration": {
        "final_basis": int(FINAL_BASIS),
        "regularization_strengths": REGULARIZATION_STRENGTHS,
        "label_split_seeds": LABEL_SPLIT_SEEDS,
        "nystrom_chunk": int(
            NYSTROM_QUERY_CHUNK
        ),
        "run_token": RUN_TOKEN,
        "geometry_roles": ROLE_DEFINITIONS,
        "local_neighbors": LOCAL_KERNEL_NEIGHBORS,
        "local_multipliers": LOCAL_KERNEL_MULTIPLIERS,
        "local_source_gammas": LOCAL_SOURCE_GAMMAS,
        "local_source_caps": LOCAL_SOURCE_CAPS,
    },
    "samples": {
        "survey": int(n_survey),
        "query": int(n_query),
        "primary_query": int(
            np.sum(primary_query_mask)
        ),
        "loss_weight_ess": float(
            primary_loss_info[
                "effective_sample_size"
            ]
        ),
        "evaluation_weight_ess": float(
            evaluation_weight_info[
                "effective_sample_size"
            ]
        ),
    },
    "geometry_aggregate": geometry_aggregate_df.to_dict(
        orient="records"
    ),
    "geometry_paired_summary": geometry_paired_summary_df.to_dict(
        orient="records"
    ),
    "local_aggregate": local_aggregate_df.to_dict(
        orient="records"
    ),
    "local_paired_summary": local_paired_summary_df.to_dict(
        orient="records"
    ),
    "outputs": {
        key: str(value)
        for key, value in paths.items()
    },
    "figures": {
        "geometry_pdf": str(
            geometry_figure_pdf
        ) if geometry_figure_pdf is not None else "",
        "geometry_png": str(
            geometry_figure_png
        ) if geometry_figure_png is not None else "",
        "local_pdf": str(
            local_figure_pdf
        ) if local_figure_pdf is not None else "",
        "local_png": str(
            local_figure_png
        ) if local_figure_png is not None else "",
    },
}

paths["summary_json"].write_text(
    json.dumps(
        summary_payload,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print("=" * 124, flush=True)
print(
    "Mock-2 final restricted sensitivity completed",
    flush=True,
)
print("=" * 124, flush=True)
print("Geometry aggregate", flush=True)
print(
    geometry_aggregate_df.to_string(
        index=False
    ),
    flush=True,
)
print("-" * 124, flush=True)
print("Geometry paired summary", flush=True)
print(
    geometry_paired_summary_df.to_string(
        index=False
    ),
    flush=True,
)
print("-" * 124, flush=True)
print("Local aggregate", flush=True)
print(
    local_aggregate_df.to_string(
        index=False
    ),
    flush=True,
)
print("-" * 124, flush=True)
print("Local paired summary", flush=True)
print(
    local_paired_summary_df.to_string(
        index=False
    ),
    flush=True,
)
print("-" * 124, flush=True)
if not primary_audit_df.empty:
    print("Primary-result continuity audit", flush=True)
    print(
        primary_audit_df.to_string(
            index=False
        ),
        flush=True,
    )
    print("-" * 124, flush=True)

for key, value in paths.items():
    print(
        f"{key:36s}: {value}",
        flush=True,
    )
print(
    f"geometry_figure_pdf                 : {geometry_figure_pdf}",
    flush=True,
)
print(
    f"geometry_figure_png                 : {geometry_figure_png}",
    flush=True,
)
print(
    f"local_figure_pdf                    : {local_figure_pdf}",
    flush=True,
)
print(
    f"local_figure_png                    : {local_figure_png}",
    flush=True,
)
print("=" * 124, flush=True)
