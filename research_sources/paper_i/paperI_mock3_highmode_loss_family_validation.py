# Paper I: Mock-3 high-mode lambda and loss-family validation
#
# 実行順序:
#   1. mock23_current_samples_common_single_cell_v2.py
#   2. paperI_mock3_primary_mode_convergence_to_4096.py
#      （少なくとも4096-mode geometry cacheが生成済みであること）
#   3. このファイル全体
#
# 目的:
#   Mock-3で確定したpractical spectral capacityの近傍において、
#   regularization strengthとloss familyを再評価する。
#
# 検証するloss families:
#   1. unweighted loss
#   2. global radial inverse-selection loss
#   3. octant-aware inverse-selection loss
#
# Protocols:
#   - retuned:
#       各loss familyがbasisとlambdaをparent-weighted validationで個別選択
#   - matched_primary:
#       unweighted familyがvalidationで選んだbasisとlambdaを全familyに共通適用
#   - fixed_practical:
#       n_b=practical basis, lambda=1e-2を全familyに固定
#   - matched_basis_grid:
#       lambda=1e-2を固定し、loss weighting effectのcapacity dependenceを診断
#
# Query truthとheld-out testはhyperparameter selectionに使用しない。
# 各seed終了時にcheckpointを保存し、再実行時には完了seedを再利用する。

from __future__ import annotations

import gc
import hashlib
import json
import math
import os
import platform
import time
from pathlib import Path

import matplotlib

SHOW_PLOTS = (
    os.environ.get(
        "PAPER1_MOCK3_FINAL_SHOW_PLOTS",
        "1",
    )
    != "0"
)

if not SHOW_PLOTS:
    matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from scipy import linalg


# ============================================================
# 0. Common-cell audit
# ============================================================

_REQUIRED_COMMON_SYMBOLS = [
    "_load_mock1",
    "_load_survey",
    "_build_base_kernel",
    "_nystrom_extend",
    "velocity_metrics",
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
# 1. Paths
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

MOCK3_FILE = (
    ROOT_DIR
    / "mock3_inhomogeneous_survey"
    / "mock3.npz"
)

SOURCE_DIR = (
    ROOT_DIR
    / "mock3_octant_aware_loss_optimization_v4"
)

SOURCE_PREDICTIONS_FILE = (
    SOURCE_DIR
    / "mock3_octant_aware_predictions.npz"
)

PRIMARY_CONVERGENCE_DIR = (
    ROOT_DIR
    / "mock3_primary_mode_convergence_to_4096_v1"
)

PRIMARY_GEOMETRY_FILE = (
    PRIMARY_CONVERGENCE_DIR
    / "cache"
    / "mock3_unweighted_alpha0_geometry_m4096.npz"
)

PRIMARY_CONVERGENCE_RUNS_FILE = (
    PRIMARY_CONVERGENCE_DIR
    / "mock3_primary_mode_convergence_to_4096_runs.csv"
)

PRIMARY_SELECTION_FILE = (
    PRIMARY_CONVERGENCE_DIR
    / "mock3_primary_mode_convergence_to_4096_practical_selection.csv"
)

LEGACY_ROBUSTNESS_DIR = (
    ROOT_DIR
    / "mock3_octant_loss_seed_robustness_v5"
)

LEGACY_ROBUSTNESS_RUNS_FILE = (
    LEGACY_ROBUSTNESS_DIR
    / "mock3_octant_loss_seed_robustness_runs.csv"
)

OUTPUT_DIR = Path(
    os.environ.get(
        "PAPER1_MOCK3_FINAL_OUTPUT",
        str(
            ROOT_DIR
            / "mock3_highmode_loss_family_validation_v1"
        ),
    )
)

CACHE_DIR = OUTPUT_DIR / "cache"
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"

for directory in (
    OUTPUT_DIR,
    CACHE_DIR,
    CHECKPOINT_DIR,
):
    directory.mkdir(
        parents=True,
        exist_ok=True,
    )

QUERY_EXTENSION_CACHE_FILE = (
    CACHE_DIR
    / "mock3_query_eigenfunctions_m4096.npz"
)

OUTPUT_PREFIX = (
    "mock3_highmode_loss_family_validation"
)


# ============================================================
# 2. Controlled-experiment settings
# ============================================================

SPHERE_CENTER = np.array(
    [250.0, 250.0, 250.0],
    dtype=np.float64,
)

GRAPH_NEIGHBORS = 48
GRAPH_BANDWIDTH_NEIGHBOR = 16
GRAPH_BANDWIDTH_MULTIPLIER = 1.0

TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

_DEFAULT_LABEL_SPLIT_SEEDS = [
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

_N_SEEDS = int(
    os.environ.get(
        "PAPER1_MOCK3_FINAL_N_SEEDS",
        str(
            len(
                _DEFAULT_LABEL_SPLIT_SEEDS
            )
        ),
    )
)

LABEL_SPLIT_SEEDS = (
    _DEFAULT_LABEL_SPLIT_SEEDS[
        :_N_SEEDS
    ]
)

# Practical-basis neighborhood for validation retuning.
SEARCH_BASIS_SIZES = [
    2048,
    2560,
    3072,
]

# Fixed-lambda capacity diagnostic.
MATCHED_BASIS_SIZES = [
    512,
    1024,
    1536,
    2048,
    2560,
    3072,
]

REGULARIZATION_STRENGTHS = [
    3.0e-3,
    1.0e-2,
    3.0e-2,
    1.0e-1,
    3.0e-1,
]

FIXED_MATCHED_REGULARIZATION = (
    1.0e-2
)

NUMERICAL_RIDGE = 1.0e-10

MODEL_CONFIGS = {
    "unweighted_loss": {
        "probability_mode": "none",
        "gamma": 0.0,
        "cap": 1.0,
        "label": "Unweighted loss",
    },
    "global_radial_loss": {
        "probability_mode": "global",
        "gamma": 0.75,
        "cap": 20.0,
        "label": "Global radial loss",
    },
    "octant_aware_loss": {
        "probability_mode": "region",
        "gamma": 1.0,
        "cap": 20.0,
        "label": "Octant-aware loss",
    },
}

MODEL_ORDER = list(
    MODEL_CONFIGS.keys()
)

PRIMARY_METRIC_GAMMA = 1.0
PRIMARY_METRIC_CAP = 50.0

GLOBAL_METRIC_GAMMA = 1.0
GLOBAL_METRIC_CAP = 50.0

MIN_SELECTION_PROBABILITY = 1.0e-4
WEIGHT_CLIP_QUANTILE = 0.995

SEED_BOOTSTRAP_REPLICATES = 50000
SEED_BOOTSTRAP_SEED = 20270413

SAVE_PDF = True
SAVE_PNG = True
PNG_DPI = 180


# ============================================================
# 3. Utility functions
# ============================================================

def configure_font():
    selected = "DejaVu Sans"

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


def stable_hash(
    payload,
) -> str:
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )

    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()[:16]


def file_signature(
    path: Path,
) -> dict:
    stat = path.stat()

    return {
        "path": str(
            path.resolve()
        ),
        "size": int(
            stat.st_size
        ),
        "mtime_ns": int(
            stat.st_mtime_ns
        ),
    }


def effective_sample_size(
    weights,
) -> float:
    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    denominator = float(
        np.sum(
            np.square(
                weights
            )
        )
    )

    if denominator <= 0.0:
        return math.nan

    return float(
        np.square(
            np.sum(
                weights
            )
        )
        / denominator
    )


def tempered_inverse_selection_weights(
    probability,
    gamma,
    floor,
    max_weight,
    quantile,
):
    probability = np.asarray(
        probability,
        dtype=np.float64,
    )

    safe_probability = np.maximum(
        probability,
        float(
            floor
        ),
    )

    if float(
        gamma
    ) == 0.0:
        weights = np.ones(
            safe_probability.shape,
            dtype=np.float64,
        )

        cap = 1.0
        raw_weight_max = 1.0
    else:
        raw = np.power(
            safe_probability,
            -float(
                gamma
            ),
        )

        empirical_cap = float(
            np.quantile(
                raw,
                float(
                    quantile
                ),
            )
        )

        cap = min(
            float(
                max_weight
            ),
            empirical_cap,
        )

        weights = np.minimum(
            raw,
            cap,
        )

        weights /= np.mean(
            weights
        )

        raw_weight_max = float(
            np.max(
                raw
            )
        )

    ess = effective_sample_size(
        weights
    )

    return weights, {
        "gamma": float(
            gamma
        ),
        "cap": float(
            cap
        ),
        "raw_weight_max": (
            raw_weight_max
        ),
        "normalized_weight_max": float(
            np.max(
                weights
            )
        ),
        "effective_sample_size": float(
            ess
        ),
        "effective_sample_fraction": float(
            ess
            / weights.size
        ),
    }


def make_octant_stratified_split(
    region_id,
    train_fraction,
    validation_fraction,
    seed,
):
    region_id = np.asarray(
        region_id,
        dtype=np.int64,
    )

    rng = np.random.default_rng(
        int(
            seed
        )
    )

    train_parts = []
    validation_parts = []
    test_parts = []

    for region in range(8):
        indices = np.flatnonzero(
            region_id
            == region
        )

        indices = rng.permutation(
            indices
        )

        n_train = int(
            round(
                float(
                    train_fraction
                )
                * indices.size
            )
        )

        n_validation = int(
            round(
                float(
                    validation_fraction
                )
                * indices.size
            )
        )

        if (
            n_train
            + n_validation
            >= indices.size
        ):
            n_validation = max(
                1,
                indices.size
                - n_train
                - 1,
            )

        train_parts.append(
            indices[
                :n_train
            ]
        )

        validation_parts.append(
            indices[
                n_train:
                n_train
                + n_validation
            ]
        )

        test_parts.append(
            indices[
                n_train
                + n_validation:
            ]
        )

    return {
        "train": np.sort(
            np.concatenate(
                train_parts
            )
        ),
        "validation": np.sort(
            np.concatenate(
                validation_parts
            )
        ),
        "test": np.sort(
            np.concatenate(
                test_parts
            )
        ),
    }


def save_figure(
    figure: plt.Figure,
    stem: str,
) -> tuple[
    Path | None,
    Path | None,
]:
    pdf_path = (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_{stem}.pdf"
        if SAVE_PDF
        else None
    )

    png_path = (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_{stem}.png"
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
        plt.close(
            figure
        )

    return (
        pdf_path,
        png_path,
    )


def seed_bootstrap_interval(
    values,
    n_bootstrap,
    seed,
):
    values = np.asarray(
        values,
        dtype=np.float64,
    )

    values = values[
        np.isfinite(
            values
        )
    ]

    if values.size == 0:
        return (
            math.nan,
            math.nan,
        )

    rng = np.random.default_rng(
        int(
            seed
        )
    )

    indices = rng.integers(
        0,
        values.size,
        size=(
            int(
                n_bootstrap
            ),
            values.size,
        ),
    )

    means = np.mean(
        values[
            indices
        ],
        axis=1,
    )

    return (
        float(
            np.quantile(
                means,
                0.025,
            )
        ),
        float(
            np.quantile(
                means,
                0.975,
            )
        ),
    )


# ============================================================
# 4. Geometry and query-extension loading
# ============================================================

def load_geometry(
    path: Path,
) -> dict:
    with np.load(
        path,
        allow_pickle=False,
    ) as data:
        geometry = {
            "eigenvalues": np.asarray(
                data[
                    "eigenvalues"
                ],
                dtype=np.float64,
            ),
            "generator_eigenvalues": np.asarray(
                data[
                    "generator_eigenvalues"
                ],
                dtype=np.float64,
            ),
            "eigenfunctions": np.asarray(
                data[
                    "eigenfunctions"
                ],
                dtype=np.float64,
            ),
            "quadrature_weights": np.asarray(
                data[
                    "quadrature_weights"
                ],
                dtype=np.float64,
            ),
            "q": np.asarray(
                data[
                    "q"
                ],
                dtype=np.float64,
            ),
            "d": np.asarray(
                data[
                    "d"
                ],
                dtype=np.float64,
            ),
            "stationary": np.asarray(
                data[
                    "stationary"
                ],
                dtype=np.float64,
            ),
            "alpha": float(
                np.asarray(
                    data[
                        "alpha"
                    ]
                ).item()
            ),
        }

        if (
            "generator_scale"
            in data.files
        ):
            geometry[
                "generator_scale"
            ] = float(
                np.asarray(
                    data[
                        "generator_scale"
                    ]
                ).item()
            )

    return geometry


def load_or_compute_query_eigenfunctions(
    centered_query,
    centered_survey,
    query_global,
    base_kernel,
    geometry,
):
    cache_signature = stable_hash(
        {
            "geometry": file_signature(
                PRIMARY_GEOMETRY_FILE
            ),
            "source_predictions": file_signature(
                SOURCE_PREDICTIONS_FILE
            ),
            "query_size": int(
                centered_query.shape[
                    0
                ]
            ),
            "n_modes": int(
                geometry[
                    "eigenfunctions"
                ].shape[
                    1
                ]
            ),
        }
    )

    if (
        QUERY_EXTENSION_CACHE_FILE.is_file()
    ):
        try:
            with np.load(
                QUERY_EXTENSION_CACHE_FILE,
                allow_pickle=False,
            ) as data:
                stored_signature = str(
                    data[
                        "signature"
                    ].item()
                )

                stored_query_global = np.asarray(
                    data[
                        "query_global_indices"
                    ],
                    dtype=np.int64,
                )

                stored_eigenfunctions = np.asarray(
                    data[
                        "query_eigenfunctions"
                    ],
                    dtype=np.float64,
                )

            if (
                stored_signature
                == cache_signature
                and np.array_equal(
                    stored_query_global,
                    query_global,
                )
                and stored_eigenfunctions.shape
                == (
                    centered_query.shape[
                        0
                    ],
                    geometry[
                        "eigenfunctions"
                    ].shape[
                        1
                    ],
                )
            ):
                print(
                    "[Nyström] loaded query-extension cache: "
                    f"{QUERY_EXTENSION_CACHE_FILE}",
                    flush=True,
                )

                return (
                    stored_eigenfunctions,
                    True,
                )
        except Exception as exc:
            print(
                "[Nyström] query-extension cache was not reused: "
                f"{exc}",
                flush=True,
            )

    print(
        "[Nyström] extending "
        f"{centered_query.shape[0]:,} query points...",
        flush=True,
    )

    start = time.perf_counter()

    query_extension = (
        _nystrom_extend(
            centered_query,
            centered_survey,
            base_kernel,
            geometry,
        )
    )

    query_eigenfunctions = np.asarray(
        query_extension[
            "eigenfunctions"
        ],
        dtype=np.float64,
    )

    np.savez(
        QUERY_EXTENSION_CACHE_FILE,
        signature=np.array(
            cache_signature
        ),
        query_global_indices=np.asarray(
            query_global,
            dtype=np.int64,
        ),
        query_eigenfunctions=(
            query_eigenfunctions
        ),
    )

    print(
        "[Nyström] completed in "
        f"{time.perf_counter() - start:.1f} s; "
        f"cache saved: {QUERY_EXTENSION_CACHE_FILE}",
        flush=True,
    )

    return (
        query_eigenfunctions,
        False,
    )


# ============================================================
# 5. Regression and metrics
# ============================================================

def weighted_gram_and_rhs(
    phi,
    velocity,
    weights,
):
    phi = np.asarray(
        phi,
        dtype=np.float64,
    )

    velocity = np.asarray(
        velocity,
        dtype=np.float64,
    )

    weights = np.asarray(
        weights,
        dtype=np.float64,
    )

    normalized_weights = (
        weights
        / np.mean(
            weights
        )
    )

    square_root_weights = np.sqrt(
        normalized_weights
    )

    weighted_phi = (
        phi
        * square_root_weights[
            :,
            None,
        ]
    )

    weighted_velocity = (
        velocity
        * square_root_weights[
            :,
            None,
        ]
    )

    gram = (
        weighted_phi.T
        @ weighted_phi
    ) / phi.shape[
        0
    ]

    rhs = (
        weighted_phi.T
        @ weighted_velocity
    ) / phi.shape[
        0
    ]

    return (
        gram,
        rhs,
    )


def solve_from_normal_equations(
    gram,
    rhs,
    generator_eigenvalues,
    regularization_strength,
):
    system = np.array(
        gram,
        copy=True,
    )

    diagonal = np.diag_indices_from(
        system
    )

    system[
        diagonal
    ] += (
        float(
            regularization_strength
        )
        * np.maximum(
            generator_eigenvalues[
                :system.shape[
                    0
                ]
            ],
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

        coefficients = linalg.cho_solve(
            factor,
            rhs,
            check_finite=False,
        )
    except linalg.LinAlgError:
        coefficients = linalg.solve(
            system,
            rhs,
            assume_a="sym",
            check_finite=False,
        )

    return coefficients


def normalized_rmse(
    true,
    predicted,
    weights=None,
) -> float:
    return float(
        velocity_metrics(
            true,
            predicted,
            weights=weights,
        )[
            "normalized_rmse"
        ]
    )


def octant_nrmse_summary(
    true,
    predicted,
    regions,
    weights=None,
):
    true = np.asarray(
        true,
        dtype=np.float64,
    )

    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    regions = np.asarray(
        regions,
        dtype=np.int64,
    )

    if weights is not None:
        weights = np.asarray(
            weights,
            dtype=np.float64,
        )

    values = []

    for region in range(8):
        mask = (
            regions
            == region
        )

        if np.sum(
            mask
        ) < 5:
            continue

        local_weights = (
            None
            if weights is None
            else weights[
                mask
            ]
        )

        values.append(
            normalized_rmse(
                true[
                    mask
                ],
                predicted[
                    mask
                ],
                local_weights,
            )
        )

    values = np.asarray(
        values,
        dtype=np.float64,
    )

    if values.size == 0:
        return {
            "octant_nrmse_std": math.nan,
            "octant_nrmse_worst": math.nan,
            "octant_nrmse_range": math.nan,
        }

    return {
        "octant_nrmse_std": float(
            np.std(
                values,
                ddof=0,
            )
        ),
        "octant_nrmse_worst": float(
            np.max(
                values
            )
        ),
        "octant_nrmse_range": float(
            np.max(
                values
            )
            - np.min(
                values
            )
        ),
    }


def evaluate_configuration(
    observed_true,
    observed_prediction,
    observed_regions,
    primary_metric_weights,
    global_metric_weights,
    query_true,
    query_prediction,
    query_regions,
    region_primary_support,
    common_strict_support,
    near_boundary_mask,
    far_boundary_mask,
):
    observed_primary_metrics = (
        velocity_metrics(
            observed_true,
            observed_prediction,
            weights=(
                primary_metric_weights
            ),
        )
    )

    observed_global_metrics = (
        velocity_metrics(
            observed_true,
            observed_prediction,
            weights=(
                global_metric_weights
            ),
        )
    )

    observed_unweighted_metrics = (
        velocity_metrics(
            observed_true,
            observed_prediction,
        )
    )

    observed_octant_summary = (
        octant_nrmse_summary(
            observed_true,
            observed_prediction,
            observed_regions,
            primary_metric_weights,
        )
    )

    query_region_metrics = (
        velocity_metrics(
            query_true[
                region_primary_support
            ],
            query_prediction[
                region_primary_support
            ],
        )
    )

    query_octant_summary = (
        octant_nrmse_summary(
            query_true[
                region_primary_support
            ],
            query_prediction[
                region_primary_support
            ],
            query_regions[
                region_primary_support
            ],
            None,
        )
    )

    query_common_nrmse = (
        normalized_rmse(
            query_true[
                common_strict_support
            ],
            query_prediction[
                common_strict_support
            ],
        )
        if np.sum(
            common_strict_support
        ) >= 5
        else math.nan
    )

    query_near_boundary_nrmse = (
        normalized_rmse(
            query_true[
                near_boundary_mask
            ],
            query_prediction[
                near_boundary_mask
            ],
        )
        if np.sum(
            near_boundary_mask
        ) >= 5
        else math.nan
    )

    query_far_boundary_nrmse = (
        normalized_rmse(
            query_true[
                far_boundary_mask
            ],
            query_prediction[
                far_boundary_mask
            ],
        )
        if np.sum(
            far_boundary_mask
        ) >= 5
        else math.nan
    )

    return {
        "observed_primary_nrmse": float(
            observed_primary_metrics[
                "normalized_rmse"
            ]
        ),
        "observed_primary_median_direction_error_deg": float(
            observed_primary_metrics[
                "median_direction_error_deg"
            ]
        ),
        "observed_primary_vector_gain": float(
            observed_primary_metrics[
                "vector_gain_through_origin"
            ]
        ),
        "observed_global_nrmse": float(
            observed_global_metrics[
                "normalized_rmse"
            ]
        ),
        "observed_unweighted_nrmse": float(
            observed_unweighted_metrics[
                "normalized_rmse"
            ]
        ),
        "observed_unweighted_median_direction_error_deg": float(
            observed_unweighted_metrics[
                "median_direction_error_deg"
            ]
        ),
        "observed_unweighted_vector_gain": float(
            observed_unweighted_metrics[
                "vector_gain_through_origin"
            ]
        ),
        "observed_octant_nrmse_std": float(
            observed_octant_summary[
                "octant_nrmse_std"
            ]
        ),
        "observed_worst_octant_nrmse": float(
            observed_octant_summary[
                "octant_nrmse_worst"
            ]
        ),
        "observed_octant_nrmse_range": float(
            observed_octant_summary[
                "octant_nrmse_range"
            ]
        ),
        "query_region_nrmse": float(
            query_region_metrics[
                "normalized_rmse"
            ]
        ),
        "query_region_median_direction_error_deg": float(
            query_region_metrics[
                "median_direction_error_deg"
            ]
        ),
        "query_region_vector_gain": float(
            query_region_metrics[
                "vector_gain_through_origin"
            ]
        ),
        "query_octant_nrmse_std": float(
            query_octant_summary[
                "octant_nrmse_std"
            ]
        ),
        "query_worst_octant_nrmse": float(
            query_octant_summary[
                "octant_nrmse_worst"
            ]
        ),
        "query_octant_nrmse_range": float(
            query_octant_summary[
                "octant_nrmse_range"
            ]
        ),
        "query_common_nrmse": float(
            query_common_nrmse
        ),
        "query_near_boundary_nrmse": float(
            query_near_boundary_nrmse
        ),
        "query_far_boundary_nrmse": float(
            query_far_boundary_nrmse
        ),
    }


def select_best_validation_row(
    frame,
):
    selected = frame.copy()

    selected[
        "lambda_distance"
    ] = np.abs(
        np.log10(
            selected[
                "regularization_strength"
            ]
        )
        - np.log10(
            FIXED_MATCHED_REGULARIZATION
        )
    )

    selected = selected.sort_values(
        [
            "parent_validation_nrmse",
            "unweighted_validation_nrmse",
            "n_basis",
            "lambda_distance",
            "regularization_strength",
        ],
        ascending=[
            True,
            True,
            True,
            True,
            True,
        ],
    )

    return selected.iloc[
        0
    ]


# ============================================================
# 6. Input loading
# ============================================================

font_name = configure_font()

required_files = [
    MOCK1_FILE,
    MOCK3_FILE,
    SOURCE_PREDICTIONS_FILE,
    PRIMARY_GEOMETRY_FILE,
    PRIMARY_CONVERGENCE_RUNS_FILE,
    PRIMARY_SELECTION_FILE,
]

missing_files = [
    path
    for path in required_files
    if not path.is_file()
]

if missing_files:
    raise FileNotFoundError(
        "必要な入力ファイルがありません:\n"
        + "\n".join(
            str(
                path
            )
            for path in missing_files
        )
    )

primary_selection_df = pd.read_csv(
    PRIMARY_SELECTION_FILE
)

if primary_selection_df.shape[
    0
] != 1:
    raise ValueError(
        "Primary practical-selection CSV must contain one row."
    )

practical_basis = int(
    primary_selection_df.iloc[
        0
    ][
        "practical_plateau_basis"
    ]
)

if not np.isfinite(
    practical_basis
):
    practical_basis = int(
        primary_selection_df.iloc[
            0
        ][
            "minimum_validation_basis"
        ]
    )

if practical_basis not in SEARCH_BASIS_SIZES:
    SEARCH_BASIS_SIZES = sorted(
        set(
            SEARCH_BASIS_SIZES
            + [
                practical_basis
            ]
        )
    )

max_required_basis = max(
    SEARCH_BASIS_SIZES
    + MATCHED_BASIS_SIZES
)

geometry = load_geometry(
    PRIMARY_GEOMETRY_FILE
)

if geometry[
    "eigenfunctions"
].shape[
    1
] < max_required_basis:
    raise ValueError(
        "4096-mode geometry cache has too few modes."
    )

if not np.allclose(
    geometry[
        "quadrature_weights"
    ],
    1.0,
    rtol=0.0,
    atol=1.0e-12,
):
    raise ValueError(
        "Loaded geometry is not the unweighted primary geometry."
    )

if abs(
    float(
        geometry[
            "alpha"
        ]
    )
) > 1.0e-12:
    raise ValueError(
        "Loaded geometry does not have alpha_DM=0."
    )

complete = _load_mock1(
    MOCK1_FILE
)

survey = _load_survey(
    MOCK3_FILE,
    "mock3",
)

with np.load(
    SOURCE_PREDICTIONS_FILE,
    allow_pickle=False,
) as data:
    required_keys = {
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

    missing_keys = required_keys.difference(
        data.files
    )

    if missing_keys:
        raise KeyError(
            "Source prediction NPZ is missing keys: "
            f"{sorted(missing_keys)}"
        )

    query_global = np.asarray(
        data[
            "query_global_indices"
        ],
        dtype=np.int64,
    )

    truth_survey = np.asarray(
        data[
            "truth_survey"
        ],
        dtype=np.float64,
    )

    truth_query = np.asarray(
        data[
            "truth_query"
        ],
        dtype=np.float64,
    )

    probability_region_survey = np.asarray(
        data[
            "selection_probability_region_survey"
        ],
        dtype=np.float64,
    )

    probability_global_survey = np.asarray(
        data[
            "selection_probability_global_survey"
        ],
        dtype=np.float64,
    )

    probability_region_query = np.asarray(
        data[
            "selection_probability_region_query"
        ],
        dtype=np.float64,
    )

    probability_global_query = np.asarray(
        data[
            "selection_probability_global_query"
        ],
        dtype=np.float64,
    )

    query_region = np.asarray(
        data[
            "query_region"
        ],
        dtype=np.int64,
    )

    region_primary_support = np.asarray(
        data[
            "region_primary_support"
        ],
        dtype=bool,
    )

    common_strict_support = np.asarray(
        data[
            "common_strict_support"
        ],
        dtype=bool,
    )

    octant_boundary_distance_query = np.asarray(
        data[
            "octant_boundary_distance_query"
        ],
        dtype=np.float64,
    )

n_survey = survey[
    "pos"
].shape[
    0
]

n_query = query_global.size

if truth_survey.shape != (
    n_survey,
    3,
):
    raise ValueError(
        "truth_survey shape mismatch: "
        f"{truth_survey.shape}"
    )

if truth_query.shape != (
    n_query,
    3,
):
    raise ValueError(
        "truth_query shape mismatch: "
        f"{truth_query.shape}"
    )

if survey[
    "region_id"
].shape != (
    n_survey,
):
    raise ValueError(
        "Mock-3 region_id shape mismatch."
    )

centered_survey = (
    survey[
        "pos"
    ]
    - SPHERE_CENTER[
        None,
        :,
    ]
)

centered_query = (
    complete[
        "pos"
    ][
        query_global
    ]
    - SPHERE_CENTER[
        None,
        :,
    ]
)

base_kernel = _build_base_kernel(
    centered_survey,
    GRAPH_NEIGHBORS,
    GRAPH_BANDWIDTH_NEIGHBOR,
    GRAPH_BANDWIDTH_MULTIPLIER,
)

(
    query_eigenfunctions,
    query_extension_from_cache,
) = load_or_compute_query_eigenfunctions(
    centered_query,
    centered_survey,
    query_global,
    base_kernel,
    geometry,
)

boundary_scale = float(
    np.median(
        base_kernel[
            "rho"
        ]
    )
)

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
        >= 2.0
        * boundary_scale
    )
)


# ============================================================
# 7. Weights
# ============================================================

primary_metric_weights, (
    primary_metric_info
) = tempered_inverse_selection_weights(
    probability_region_survey,
    PRIMARY_METRIC_GAMMA,
    MIN_SELECTION_PROBABILITY,
    PRIMARY_METRIC_CAP,
    WEIGHT_CLIP_QUANTILE,
)

global_metric_weights, (
    global_metric_info
) = tempered_inverse_selection_weights(
    probability_global_survey,
    GLOBAL_METRIC_GAMMA,
    MIN_SELECTION_PROBABILITY,
    GLOBAL_METRIC_CAP,
    WEIGHT_CLIP_QUANTILE,
)

model_loss_weights = {}
model_weight_records = []

for model_name, config in (
    MODEL_CONFIGS.items()
):
    mode = config[
        "probability_mode"
    ]

    if mode == "none":
        probability = np.ones(
            n_survey,
            dtype=np.float64,
        )
    elif mode == "global":
        probability = (
            probability_global_survey
        )
    elif mode == "region":
        probability = (
            probability_region_survey
        )
    else:
        raise ValueError(
            "Unknown probability mode: "
            f"{mode}"
        )

    weights, weight_info = (
        tempered_inverse_selection_weights(
            probability,
            config[
                "gamma"
            ],
            MIN_SELECTION_PROBABILITY,
            config[
                "cap"
            ],
            WEIGHT_CLIP_QUANTILE,
        )
    )

    model_loss_weights[
        model_name
    ] = weights

    model_weight_records.append(
        {
            "model": model_name,
            "label": config[
                "label"
            ],
            "probability_mode": mode,
            **weight_info,
        }
    )

weight_summary_df = pd.DataFrame(
    model_weight_records
)

weight_summary_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_weight_summary.csv",
    index=False,
)


# ============================================================
# 8. Configuration and progress header
# ============================================================

run_token = stable_hash(
    {
        "search_basis_sizes": (
            SEARCH_BASIS_SIZES
        ),
        "matched_basis_sizes": (
            MATCHED_BASIS_SIZES
        ),
        "regularization_strengths": (
            REGULARIZATION_STRENGTHS
        ),
        "fixed_matched_regularization": (
            FIXED_MATCHED_REGULARIZATION
        ),
        "practical_basis": int(
            practical_basis
        ),
        "label_split_seeds": (
            LABEL_SPLIT_SEEDS
        ),
        "model_configs": (
            MODEL_CONFIGS
        ),
        "geometry": file_signature(
            PRIMARY_GEOMETRY_FILE
        ),
        "source_predictions": file_signature(
            SOURCE_PREDICTIONS_FILE
        ),
    }
)

print(
    "="
    * 120
)

print(
    "Paper I: Mock-3 high-mode lambda and loss-family validation"
)

print(
    "="
    * 120
)

print(
    "Python / NumPy / SciPy : "
    f"{platform.python_version()} / "
    f"{np.__version__} / "
    f"{scipy.__version__}"
)

print(
    "Matplotlib font         : "
    f"{font_name}"
)

print(
    "Mock-3                  : "
    f"{MOCK3_FILE}"
)

print(
    "4096-mode geometry      : "
    f"{PRIMARY_GEOMETRY_FILE}"
)

print(
    "Output                  : "
    f"{OUTPUT_DIR}"
)

print(
    "Practical basis         : "
    f"{practical_basis}"
)

print(
    "Search basis sizes      : "
    f"{SEARCH_BASIS_SIZES}"
)

print(
    "Lambda grid             : "
    f"{REGULARIZATION_STRENGTHS}"
)

print(
    "Matched basis sizes     : "
    f"{MATCHED_BASIS_SIZES}"
)

print(
    "Label split seeds       : "
    f"{len(LABEL_SPLIT_SEEDS)}"
)

print(
    "Region-primary query    : "
    f"{int(np.sum(region_primary_support))}"
)

print(
    "Primary metric ESS      : "
    f"{primary_metric_info['effective_sample_size']:.1f}"
)

print(
    "Query extension source  : "
    + (
        "cache"
        if query_extension_from_cache
        else "new computation"
    )
)

print(
    "-"
    * 120
)

print(
    weight_summary_df.to_string(
        index=False
    )
)

print(
    "="
    * 120
)


# ============================================================
# 9. Seed-level computation
# ============================================================

all_validation_frames = []
all_protocol_frames = []
all_matched_basis_frames = []

for replicate, label_seed in enumerate(
    LABEL_SPLIT_SEEDS,
    start=1,
):
    validation_checkpoint = (
        CHECKPOINT_DIR
        / (
            f"{OUTPUT_PREFIX}_{run_token}_"
            f"seed_{label_seed}_validation.csv"
        )
    )

    protocol_checkpoint = (
        CHECKPOINT_DIR
        / (
            f"{OUTPUT_PREFIX}_{run_token}_"
            f"seed_{label_seed}_protocols.csv"
        )
    )

    matched_basis_checkpoint = (
        CHECKPOINT_DIR
        / (
            f"{OUTPUT_PREFIX}_{run_token}_"
            f"seed_{label_seed}_matched_basis.csv"
        )
    )

    if (
        validation_checkpoint.is_file()
        and protocol_checkpoint.is_file()
        and matched_basis_checkpoint.is_file()
    ):
        print(
            "[Checkpoint] "
            f"replicate={replicate}, "
            f"seed={label_seed}",
            flush=True,
        )

        all_validation_frames.append(
            pd.read_csv(
                validation_checkpoint
            )
        )

        all_protocol_frames.append(
            pd.read_csv(
                protocol_checkpoint
            )
        )

        all_matched_basis_frames.append(
            pd.read_csv(
                matched_basis_checkpoint
            )
        )

        continue

    print(
        "-"
        * 120
    )

    print(
        "Replicate "
        f"{replicate}/"
        f"{len(LABEL_SPLIT_SEEDS)}: "
        f"label_seed={label_seed}",
        flush=True,
    )

    seed_start = time.perf_counter()

    split = make_octant_stratified_split(
        survey[
            "region_id"
        ],
        TRAIN_FRACTION,
        VALIDATION_FRACTION,
        label_seed,
    )

    train_indices = split[
        "train"
    ]

    validation_indices = split[
        "validation"
    ]

    test_indices = split[
        "test"
    ]

    fit_indices = np.sort(
        np.concatenate(
            [
                train_indices,
                validation_indices,
            ]
        )
    )

    max_search_basis = max(
        SEARCH_BASIS_SIZES
    )

    validation_records = []
    best_configurations = {}

    # --------------------------------------------------------
    # 9.1 Validation search
    # --------------------------------------------------------

    for model_name in MODEL_ORDER:
        model_start = time.perf_counter()

        loss_weights = model_loss_weights[
            model_name
        ]

        train_phi = geometry[
            "eigenfunctions"
        ][
            train_indices,
            :max_search_basis,
        ]

        train_gram, train_rhs = (
            weighted_gram_and_rhs(
                train_phi,
                truth_survey[
                    train_indices
                ],
                loss_weights[
                    train_indices
                ],
            )
        )

        for n_basis in (
            SEARCH_BASIS_SIZES
        ):
            p = int(
                n_basis
            )

            for regularization_strength in (
                REGULARIZATION_STRENGTHS
            ):
                coefficients = (
                    solve_from_normal_equations(
                        train_gram[
                            :p,
                            :p,
                        ],
                        train_rhs[
                            :p
                        ],
                        geometry[
                            "generator_eigenvalues"
                        ],
                        regularization_strength,
                    )
                )

                validation_prediction = (
                    geometry[
                        "eigenfunctions"
                    ][
                        validation_indices,
                        :p,
                    ]
                    @ coefficients
                )

                parent_validation_nrmse = (
                    normalized_rmse(
                        truth_survey[
                            validation_indices
                        ],
                        validation_prediction,
                        primary_metric_weights[
                            validation_indices
                        ],
                    )
                )

                unweighted_validation_nrmse = (
                    normalized_rmse(
                        truth_survey[
                            validation_indices
                        ],
                        validation_prediction,
                    )
                )

                validation_records.append(
                    {
                        "replicate": int(
                            replicate
                        ),
                        "label_seed": int(
                            label_seed
                        ),
                        "model": model_name,
                        "n_basis": int(
                            n_basis
                        ),
                        "regularization_strength": float(
                            regularization_strength
                        ),
                        "parent_validation_nrmse": float(
                            parent_validation_nrmse
                        ),
                        "unweighted_validation_nrmse": float(
                            unweighted_validation_nrmse
                        ),
                    }
                )

        model_validation_df = pd.DataFrame(
            [
                record
                for record in validation_records
                if record[
                    "model"
                ]
                == model_name
            ]
        )

        best_row = select_best_validation_row(
            model_validation_df
        )

        best_configurations[
            model_name
        ] = {
            "n_basis": int(
                best_row[
                    "n_basis"
                ]
            ),
            "regularization_strength": float(
                best_row[
                    "regularization_strength"
                ]
            ),
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
        }

        print(
            "  validation "
            f"{model_name:20s}: "
            f"n_b={best_configurations[model_name]['n_basis']:4d}, "
            f"lambda={best_configurations[model_name]['regularization_strength']:.3g}, "
            f"parent={best_configurations[model_name]['parent_validation_nrmse']:.6f}, "
            f"elapsed={time.perf_counter() - model_start:.1f} s",
            flush=True,
        )

        del (
            train_phi,
            train_gram,
            train_rhs,
        )

        gc.collect()

    validation_df_seed = pd.DataFrame(
        validation_records
    )

    primary_configuration = (
        best_configurations[
            "unweighted_loss"
        ]
    )

    # --------------------------------------------------------
    # 9.2 Final fits and all protocols
    # --------------------------------------------------------

    protocol_records = []
    matched_basis_records = []

    for model_name in MODEL_ORDER:
        model_start = time.perf_counter()

        loss_weights = model_loss_weights[
            model_name
        ]

        max_fit_basis = max(
            SEARCH_BASIS_SIZES
            + MATCHED_BASIS_SIZES
        )

        fit_phi = geometry[
            "eigenfunctions"
        ][
            fit_indices,
            :max_fit_basis,
        ]

        fit_gram, fit_rhs = (
            weighted_gram_and_rhs(
                fit_phi,
                truth_survey[
                    fit_indices
                ],
                loss_weights[
                    fit_indices
                ],
            )
        )

        requested_configurations = set()

        requested_configurations.add(
            (
                int(
                    best_configurations[
                        model_name
                    ][
                        "n_basis"
                    ]
                ),
                float(
                    best_configurations[
                        model_name
                    ][
                        "regularization_strength"
                    ]
                ),
            )
        )

        requested_configurations.add(
            (
                int(
                    primary_configuration[
                        "n_basis"
                    ]
                ),
                float(
                    primary_configuration[
                        "regularization_strength"
                    ]
                ),
            )
        )

        requested_configurations.add(
            (
                int(
                    practical_basis
                ),
                float(
                    FIXED_MATCHED_REGULARIZATION
                ),
            )
        )

        for n_basis in MATCHED_BASIS_SIZES:
            requested_configurations.add(
                (
                    int(
                        n_basis
                    ),
                    float(
                        FIXED_MATCHED_REGULARIZATION
                    ),
                )
            )

        prediction_cache = {}

        for (
            n_basis,
            regularization_strength,
        ) in sorted(
            requested_configurations
        ):
            coefficients = (
                solve_from_normal_equations(
                    fit_gram[
                        :n_basis,
                        :n_basis,
                    ],
                    fit_rhs[
                        :n_basis
                    ],
                    geometry[
                        "generator_eigenvalues"
                    ],
                    regularization_strength,
                )
            )

            observed_prediction = (
                geometry[
                    "eigenfunctions"
                ][
                    test_indices,
                    :n_basis,
                ]
                @ coefficients
            )

            query_prediction = (
                query_eigenfunctions[
                    :,
                    :n_basis,
                ]
                @ coefficients
            )

            prediction_cache[
                (
                    int(
                        n_basis
                    ),
                    float(
                        regularization_strength
                    ),
                )
            ] = (
                observed_prediction,
                query_prediction,
            )

        protocol_configurations = {
            "retuned": (
                int(
                    best_configurations[
                        model_name
                    ][
                        "n_basis"
                    ]
                ),
                float(
                    best_configurations[
                        model_name
                    ][
                        "regularization_strength"
                    ]
                ),
            ),
            "matched_primary": (
                int(
                    primary_configuration[
                        "n_basis"
                    ]
                ),
                float(
                    primary_configuration[
                        "regularization_strength"
                    ]
                ),
            ),
            "fixed_practical": (
                int(
                    practical_basis
                ),
                float(
                    FIXED_MATCHED_REGULARIZATION
                ),
            ),
        }

        for protocol, (
            n_basis,
            regularization_strength,
        ) in protocol_configurations.items():
            (
                observed_prediction,
                query_prediction,
            ) = prediction_cache[
                (
                    int(
                        n_basis
                    ),
                    float(
                        regularization_strength
                    ),
                )
            ]

            metrics = evaluate_configuration(
                truth_survey[
                    test_indices
                ],
                observed_prediction,
                survey[
                    "region_id"
                ][
                    test_indices
                ],
                primary_metric_weights[
                    test_indices
                ],
                global_metric_weights[
                    test_indices
                ],
                truth_query,
                query_prediction,
                query_region,
                region_primary_support,
                common_strict_support,
                near_boundary_mask,
                far_boundary_mask,
            )

            protocol_records.append(
                {
                    "protocol": protocol,
                    "replicate": int(
                        replicate
                    ),
                    "label_seed": int(
                        label_seed
                    ),
                    "model": model_name,
                    "n_basis": int(
                        n_basis
                    ),
                    "regularization_strength": float(
                        regularization_strength
                    ),
                    "selected_parent_validation_nrmse": float(
                        best_configurations[
                            model_name
                        ][
                            "parent_validation_nrmse"
                        ]
                        if protocol
                        == "retuned"
                        else primary_configuration[
                            "parent_validation_nrmse"
                        ]
                        if protocol
                        == "matched_primary"
                        else math.nan
                    ),
                    **metrics,
                }
            )

        for n_basis in MATCHED_BASIS_SIZES:
            (
                observed_prediction,
                query_prediction,
            ) = prediction_cache[
                (
                    int(
                        n_basis
                    ),
                    float(
                        FIXED_MATCHED_REGULARIZATION
                    ),
                )
            ]

            metrics = evaluate_configuration(
                truth_survey[
                    test_indices
                ],
                observed_prediction,
                survey[
                    "region_id"
                ][
                    test_indices
                ],
                primary_metric_weights[
                    test_indices
                ],
                global_metric_weights[
                    test_indices
                ],
                truth_query,
                query_prediction,
                query_region,
                region_primary_support,
                common_strict_support,
                near_boundary_mask,
                far_boundary_mask,
            )

            matched_basis_records.append(
                {
                    "protocol": "matched_basis_grid",
                    "replicate": int(
                        replicate
                    ),
                    "label_seed": int(
                        label_seed
                    ),
                    "model": model_name,
                    "n_basis": int(
                        n_basis
                    ),
                    "regularization_strength": float(
                        FIXED_MATCHED_REGULARIZATION
                    ),
                    **metrics,
                }
            )

        print(
            "  final fits "
            f"{model_name:20s}: "
            f"{len(requested_configurations)} unique configurations, "
            f"elapsed={time.perf_counter() - model_start:.1f} s",
            flush=True,
        )

        del (
            fit_phi,
            fit_gram,
            fit_rhs,
            prediction_cache,
        )

        gc.collect()

    protocol_df_seed = pd.DataFrame(
        protocol_records
    )

    matched_basis_df_seed = pd.DataFrame(
        matched_basis_records
    )

    validation_df_seed.to_csv(
        validation_checkpoint,
        index=False,
    )

    protocol_df_seed.to_csv(
        protocol_checkpoint,
        index=False,
    )

    matched_basis_df_seed.to_csv(
        matched_basis_checkpoint,
        index=False,
    )

    all_validation_frames.append(
        validation_df_seed
    )

    all_protocol_frames.append(
        protocol_df_seed
    )

    all_matched_basis_frames.append(
        matched_basis_df_seed
    )

    print(
        "[Seed completed] "
        f"elapsed={time.perf_counter() - seed_start:.1f} s",
        flush=True,
    )


# ============================================================
# 10. Combined tables
# ============================================================

validation_df = pd.concat(
    all_validation_frames,
    ignore_index=True,
)

protocol_runs_df = pd.concat(
    all_protocol_frames,
    ignore_index=True,
)

matched_basis_df = pd.concat(
    all_matched_basis_frames,
    ignore_index=True,
)

validation_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_validation_grid.csv",
    index=False,
)

protocol_runs_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_protocol_runs.csv",
    index=False,
)

matched_basis_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_matched_basis_grid.csv",
    index=False,
)


# ============================================================
# 11. Aggregation and selection frequencies
# ============================================================

validation_aggregate_df = (
    validation_df.groupby(
        [
            "model",
            "n_basis",
            "regularization_strength",
        ],
        as_index=False,
    )
    .agg(
        parent_validation_nrmse_mean=(
            "parent_validation_nrmse",
            "mean",
        ),
        parent_validation_nrmse_std=(
            "parent_validation_nrmse",
            "std",
        ),
        parent_validation_nrmse_sem=(
            "parent_validation_nrmse",
            lambda series: (
                float(
                    series.std(
                        ddof=1
                    )
                    / math.sqrt(
                        series.shape[
                            0
                        ]
                    )
                )
            ),
        ),
        unweighted_validation_nrmse_mean=(
            "unweighted_validation_nrmse",
            "mean",
        ),
        unweighted_validation_nrmse_std=(
            "unweighted_validation_nrmse",
            "std",
        ),
    )
    .sort_values(
        [
            "model",
            "n_basis",
            "regularization_strength",
        ]
    )
    .reset_index(
        drop=True
    )
)

validation_aggregate_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_validation_aggregate.csv",
    index=False,
)

retuned_selection_frequency_df = (
    protocol_runs_df[
        protocol_runs_df[
            "protocol"
        ]
        == "retuned"
    ]
    .groupby(
        [
            "model",
            "n_basis",
            "regularization_strength",
        ],
        as_index=False,
    )
    .size()
    .rename(
        columns={
            "size": "count",
        }
    )
    .sort_values(
        [
            "model",
            "count",
            "n_basis",
            "regularization_strength",
        ],
        ascending=[
            True,
            False,
            True,
            True,
        ],
    )
)

retuned_selection_frequency_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_retuned_selection_frequency.csv",
    index=False,
)

metric_columns = [
    column
    for column in protocol_runs_df.columns
    if column.startswith(
        (
            "observed_",
            "query_",
        )
    )
]

protocol_summary_records = []

for (
    protocol,
    model,
), group in protocol_runs_df.groupby(
    [
        "protocol",
        "model",
    ],
    sort=False,
):
    record = {
        "protocol": protocol,
        "model": model,
        "n_seeds": int(
            group.shape[
                0
            ]
        ),
        "n_basis_mean": float(
            group[
                "n_basis"
            ].mean()
        ),
        "regularization_strength_median": float(
            group[
                "regularization_strength"
            ].median()
        ),
    }

    for metric in metric_columns:
        record[
            f"{metric}_mean"
        ] = float(
            group[
                metric
            ].mean()
        )

        record[
            f"{metric}_std"
        ] = float(
            group[
                metric
            ].std(
                ddof=1
            )
        )

    protocol_summary_records.append(
        record
    )

protocol_summary_df = pd.DataFrame(
    protocol_summary_records
)

protocol_summary_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_protocol_summary.csv",
    index=False,
)

matched_basis_summary_df = (
    matched_basis_df.groupby(
        [
            "model",
            "n_basis",
        ],
        as_index=False,
    )
    .agg(
        observed_primary_nrmse_mean=(
            "observed_primary_nrmse",
            "mean",
        ),
        observed_primary_nrmse_std=(
            "observed_primary_nrmse",
            "std",
        ),
        observed_unweighted_nrmse_mean=(
            "observed_unweighted_nrmse",
            "mean",
        ),
        observed_unweighted_nrmse_std=(
            "observed_unweighted_nrmse",
            "std",
        ),
        observed_octant_nrmse_std_mean=(
            "observed_octant_nrmse_std",
            "mean",
        ),
        observed_octant_nrmse_std_std=(
            "observed_octant_nrmse_std",
            "std",
        ),
        query_region_nrmse_mean=(
            "query_region_nrmse",
            "mean",
        ),
        query_region_nrmse_std=(
            "query_region_nrmse",
            "std",
        ),
        query_octant_nrmse_std_mean=(
            "query_octant_nrmse_std",
            "mean",
        ),
        query_octant_nrmse_std_std=(
            "query_octant_nrmse_std",
            "std",
        ),
        query_worst_octant_nrmse_mean=(
            "query_worst_octant_nrmse",
            "mean",
        ),
        query_worst_octant_nrmse_std=(
            "query_worst_octant_nrmse",
            "std",
        ),
    )
    .sort_values(
        [
            "model",
            "n_basis",
        ]
    )
    .reset_index(
        drop=True
    )
)

matched_basis_summary_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_matched_basis_summary.csv",
    index=False,
)


# ============================================================
# 12. Paired differences and seed-bootstrap intervals
# ============================================================

PAIR_DEFINITIONS = {
    "global_minus_unweighted": (
        "global_radial_loss",
        "unweighted_loss",
    ),
    "octant_minus_unweighted": (
        "octant_aware_loss",
        "unweighted_loss",
    ),
    "octant_minus_global": (
        "octant_aware_loss",
        "global_radial_loss",
    ),
}

PAIRED_METRICS = [
    "observed_primary_nrmse",
    "observed_unweighted_nrmse",
    "observed_octant_nrmse_std",
    "observed_worst_octant_nrmse",
    "query_region_nrmse",
    "query_octant_nrmse_std",
    "query_worst_octant_nrmse",
    "query_near_boundary_nrmse",
    "query_far_boundary_nrmse",
]

paired_records = []

for protocol in [
    "retuned",
    "matched_primary",
    "fixed_practical",
]:
    selected_protocol = protocol_runs_df[
        protocol_runs_df[
            "protocol"
        ]
        == protocol
    ]

    for replicate, group in selected_protocol.groupby(
        "replicate"
    ):
        lookup = group.set_index(
            "model"
        )

        for comparison, (
            model_a,
            model_b,
        ) in PAIR_DEFINITIONS.items():
            if (
                model_a
                not in lookup.index
                or model_b
                not in lookup.index
            ):
                continue

            row_a = lookup.loc[
                model_a
            ]

            row_b = lookup.loc[
                model_b
            ]

            record = {
                "protocol": protocol,
                "comparison": comparison,
                "replicate": int(
                    replicate
                ),
                "label_seed": int(
                    row_a[
                        "label_seed"
                    ]
                ),
            }

            for metric in PAIRED_METRICS:
                record[
                    f"{metric}_difference"
                ] = float(
                    row_a[
                        metric
                    ]
                    - row_b[
                        metric
                    ]
                )

            paired_records.append(
                record
            )

paired_df = pd.DataFrame(
    paired_records
)

paired_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_paired.csv",
    index=False,
)

matched_basis_paired_records = []

for n_basis, selected_basis in matched_basis_df.groupby(
    "n_basis"
):
    for replicate, group in selected_basis.groupby(
        "replicate"
    ):
        lookup = group.set_index(
            "model"
        )

        for comparison, (
            model_a,
            model_b,
        ) in PAIR_DEFINITIONS.items():
            row_a = lookup.loc[
                model_a
            ]

            row_b = lookup.loc[
                model_b
            ]

            record = {
                "protocol": "matched_basis_grid",
                "comparison": comparison,
                "n_basis": int(
                    n_basis
                ),
                "replicate": int(
                    replicate
                ),
                "label_seed": int(
                    row_a[
                        "label_seed"
                    ]
                ),
            }

            for metric in PAIRED_METRICS:
                record[
                    f"{metric}_difference"
                ] = float(
                    row_a[
                        metric
                    ]
                    - row_b[
                        metric
                    ]
                )

            matched_basis_paired_records.append(
                record
            )

matched_basis_paired_df = pd.DataFrame(
    matched_basis_paired_records
)

matched_basis_paired_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_matched_basis_paired.csv",
    index=False,
)

paired_summary_records = []

for (
    protocol,
    comparison,
), group in paired_df.groupby(
    [
        "protocol",
        "comparison",
    ],
    sort=False,
):
    for metric in PAIRED_METRICS:
        column = (
            f"{metric}_difference"
        )

        values = group[
            column
        ].to_numpy(
            dtype=float
        )

        ci_low, ci_high = (
            seed_bootstrap_interval(
                values,
                SEED_BOOTSTRAP_REPLICATES,
                SEED_BOOTSTRAP_SEED
                + len(
                    paired_summary_records
                ),
            )
        )

        paired_summary_records.append(
            {
                "protocol": protocol,
                "comparison": comparison,
                "metric": metric,
                "n_seeds": int(
                    np.sum(
                        np.isfinite(
                            values
                        )
                    )
                ),
                "mean_difference": float(
                    np.nanmean(
                        values
                    )
                ),
                "std_difference": float(
                    np.nanstd(
                        values,
                        ddof=1,
                    )
                ),
                "bootstrap_ci_low": float(
                    ci_low
                ),
                "bootstrap_ci_high": float(
                    ci_high
                ),
                "n_negative": int(
                    np.sum(
                        values
                        < 0.0
                    )
                ),
                "n_positive": int(
                    np.sum(
                        values
                        > 0.0
                    )
                ),
            }
        )

paired_summary_df = pd.DataFrame(
    paired_summary_records
)

paired_summary_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_paired_summary.csv",
    index=False,
)

matched_basis_paired_summary_records = []

for (
    n_basis,
    comparison,
), group in matched_basis_paired_df.groupby(
    [
        "n_basis",
        "comparison",
    ],
    sort=True,
):
    for metric in PAIRED_METRICS:
        column = (
            f"{metric}_difference"
        )

        values = group[
            column
        ].to_numpy(
            dtype=float
        )

        ci_low, ci_high = (
            seed_bootstrap_interval(
                values,
                SEED_BOOTSTRAP_REPLICATES,
                SEED_BOOTSTRAP_SEED
                + 10000
                + len(
                    matched_basis_paired_summary_records
                ),
            )
        )

        matched_basis_paired_summary_records.append(
            {
                "protocol": "matched_basis_grid",
                "n_basis": int(
                    n_basis
                ),
                "comparison": comparison,
                "metric": metric,
                "n_seeds": int(
                    np.sum(
                        np.isfinite(
                            values
                        )
                    )
                ),
                "mean_difference": float(
                    np.nanmean(
                        values
                    )
                ),
                "std_difference": float(
                    np.nanstd(
                        values,
                        ddof=1,
                    )
                ),
                "bootstrap_ci_low": float(
                    ci_low
                ),
                "bootstrap_ci_high": float(
                    ci_high
                ),
                "n_negative": int(
                    np.sum(
                        values
                        < 0.0
                    )
                ),
                "n_positive": int(
                    np.sum(
                        values
                        > 0.0
                    )
                ),
            }
        )

matched_basis_paired_summary_df = pd.DataFrame(
    matched_basis_paired_summary_records
)

matched_basis_paired_summary_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_matched_basis_paired_summary.csv",
    index=False,
)


# ============================================================
# 13. Continuity audits
# ============================================================

continuity_records = []

if PRIMARY_CONVERGENCE_RUNS_FILE.is_file():
    primary_runs_df = pd.read_csv(
        PRIMARY_CONVERGENCE_RUNS_FILE
    )

    current_primary = protocol_runs_df[
        (
            protocol_runs_df[
                "protocol"
            ]
            == "fixed_practical"
        )
        & (
            protocol_runs_df[
                "model"
            ]
            == "unweighted_loss"
        )
    ][
        [
            "replicate",
            "label_seed",
            "observed_primary_nrmse",
            "observed_unweighted_nrmse",
            "query_region_nrmse",
        ]
    ].copy()

    legacy_primary = primary_runs_df[
        primary_runs_df[
            "n_basis"
        ]
        == practical_basis
    ][
        [
            "replicate",
            "label_seed",
            "parent_test_nrmse",
            "unweighted_test_nrmse",
            "query_region_nrmse",
        ]
    ].copy()

    merged = current_primary.merge(
        legacy_primary,
        on=[
            "replicate",
            "label_seed",
        ],
        how="inner",
        suffixes=(
            "_new",
            "_legacy",
        ),
    )

    if not merged.empty:
        continuity_records.extend(
            [
                {
                    "audit": "primary_practical_basis",
                    "quantity": "parent_nrmse_max_abs_difference",
                    "value": float(
                        np.max(
                            np.abs(
                                merged[
                                    "observed_primary_nrmse"
                                ]
                                - merged[
                                    "parent_test_nrmse"
                                ]
                            )
                        )
                    ),
                },
                {
                    "audit": "primary_practical_basis",
                    "quantity": "unweighted_nrmse_max_abs_difference",
                    "value": float(
                        np.max(
                            np.abs(
                                merged[
                                    "observed_unweighted_nrmse"
                                ]
                                - merged[
                                    "unweighted_test_nrmse"
                                ]
                            )
                        )
                    ),
                },
                {
                    "audit": "primary_practical_basis",
                    "quantity": "query_nrmse_max_abs_difference",
                    "value": float(
                        np.max(
                            np.abs(
                                merged[
                                    "query_region_nrmse_new"
                                ]
                                - merged[
                                    "query_region_nrmse_legacy"
                                ]
                            )
                        )
                    ),
                },
            ]
        )

if LEGACY_ROBUSTNESS_RUNS_FILE.is_file():
    legacy_v5_df = pd.read_csv(
        LEGACY_ROBUSTNESS_RUNS_FILE
    )

    legacy_model_map = {
        "unweighted_loss": "unweighted_loss",
        "global_radial_loss": "global_radial_loss_optimized",
        "octant_aware_loss": "octant_aware_loss_optimized",
    }

    current_512 = matched_basis_df[
        matched_basis_df[
            "n_basis"
        ]
        == 512
    ]

    legacy_fixed = legacy_v5_df[
        legacy_v5_df[
            "protocol"
        ]
        == "fixed_v4"
    ]

    for current_model, legacy_model in (
        legacy_model_map.items()
    ):
        new_rows = current_512[
            current_512[
                "model"
            ]
            == current_model
        ][
            [
                "replicate",
                "label_seed",
                "observed_primary_nrmse",
                "query_region_nrmse",
            ]
        ]

        old_rows = legacy_fixed[
            legacy_fixed[
                "model"
            ]
            == legacy_model
        ][
            [
                "replicate",
                "label_seed",
                "observed_primary_normalized_rmse",
                "query_region_normalized_rmse",
            ]
        ]

        merged = new_rows.merge(
            old_rows,
            on=[
                "replicate",
                "label_seed",
            ],
            how="inner",
        )

        if merged.empty:
            continue

        continuity_records.extend(
            [
                {
                    "audit": "legacy_512_fixed_v4",
                    "model": current_model,
                    "quantity": "parent_nrmse_max_abs_difference",
                    "value": float(
                        np.max(
                            np.abs(
                                merged[
                                    "observed_primary_nrmse"
                                ]
                                - merged[
                                    "observed_primary_normalized_rmse"
                                ]
                            )
                        )
                    ),
                },
                {
                    "audit": "legacy_512_fixed_v4",
                    "model": current_model,
                    "quantity": "query_nrmse_max_abs_difference",
                    "value": float(
                        np.max(
                            np.abs(
                                merged[
                                    "query_region_nrmse"
                                ]
                                - merged[
                                    "query_region_normalized_rmse"
                                ]
                            )
                        )
                    ),
                },
            ]
        )

continuity_audit_df = pd.DataFrame(
    continuity_records
)

continuity_audit_df.to_csv(
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_continuity_audit.csv",
    index=False,
)


# ============================================================
# 14. Figures
# ============================================================

model_labels = {
    model_name: MODEL_CONFIGS[
        model_name
    ][
        "label"
    ]
    for model_name in MODEL_ORDER
}

# ------------------------------------------------------------
# 14.1 Lambda and final-risk composite
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        12.0,
        8.7,
    ),
    constrained_layout=True,
)

# (a) Lambda curves at practical basis.
ax = axes[
    0,
    0,
]

for model_name in MODEL_ORDER:
    selected = validation_aggregate_df[
        (
            validation_aggregate_df[
                "model"
            ]
            == model_name
        )
        & (
            validation_aggregate_df[
                "n_basis"
            ]
            == practical_basis
        )
    ].sort_values(
        "regularization_strength"
    )

    ax.errorbar(
        selected[
            "regularization_strength"
        ],
        selected[
            "parent_validation_nrmse_mean"
        ],
        yerr=selected[
            "parent_validation_nrmse_std"
        ],
        marker="o",
        capsize=3,
        label=model_labels[
            model_name
        ],
    )

ax.set_xscale(
    "log"
)

ax.set_xlabel(
    r"Regularization strength $\lambda$"
)

ax.set_ylabel(
    r"Parent-weighted validation $\mathrm{NRMSE}_{3\mathrm{D}}$"
)

ax.set_title(
    "(a) Lambda sensitivity at practical basis"
)

ax.legend()

# (b) Retuned observed parent risk.
ax = axes[
    0,
    1,
]

retuned_summary = protocol_summary_df[
    protocol_summary_df[
        "protocol"
    ]
    == "retuned"
].set_index(
    "model"
).reindex(
    MODEL_ORDER
)

y_positions = np.arange(
    len(
        MODEL_ORDER
    )
)

for index, model_name in enumerate(
    MODEL_ORDER
):
    mean_value = float(
        retuned_summary.loc[
            model_name,
            "observed_primary_nrmse_mean",
        ]
    )

    std_value = float(
        retuned_summary.loc[
            model_name,
            "observed_primary_nrmse_std",
        ]
    )

    ax.errorbar(
        mean_value,
        index,
        xerr=std_value,
        marker="o",
        capsize=3,
        linestyle="none",
    )

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    [
        model_labels[
            model_name
        ]
        for model_name in MODEL_ORDER
    ]
)

ax.invert_yaxis()

ax.set_xlabel(
    r"Observed parent-weighted $\mathrm{NRMSE}_{3\mathrm{D}}$"
)

ax.set_title(
    "(b) Retuned observed parent risk"
)

# (c) Retuned query risk.
ax = axes[
    1,
    0,
]

for index, model_name in enumerate(
    MODEL_ORDER
):
    mean_value = float(
        retuned_summary.loc[
            model_name,
            "query_region_nrmse_mean",
        ]
    )

    std_value = float(
        retuned_summary.loc[
            model_name,
            "query_region_nrmse_std",
        ]
    )

    ax.errorbar(
        mean_value,
        index,
        xerr=std_value,
        marker="o",
        capsize=3,
        linestyle="none",
    )

ax.set_yticks(
    y_positions
)

ax.set_yticklabels(
    [
        model_labels[
            model_name
        ]
        for model_name in MODEL_ORDER
    ]
)

ax.invert_yaxis()

ax.set_xlabel(
    r"Region-primary query $\mathrm{NRMSE}_{3\mathrm{D}}$"
)

ax.set_title(
    "(c) Retuned complete-query risk"
)

# (d) Matched-primary forest plot.
ax = axes[
    1,
    1,
]

forest_specifications = [
    (
        "global_minus_unweighted",
        "observed_primary_nrmse",
        "Global minus unweighted\nobserved parent risk",
    ),
    (
        "octant_minus_unweighted",
        "observed_primary_nrmse",
        "Octant-aware minus unweighted\nobserved parent risk",
    ),
    (
        "global_minus_unweighted",
        "query_region_nrmse",
        "Global minus unweighted\ncomplete-query risk",
    ),
    (
        "octant_minus_unweighted",
        "query_region_nrmse",
        "Octant-aware minus unweighted\ncomplete-query risk",
    ),
]

for index, (
    comparison,
    metric,
    label,
) in enumerate(
    forest_specifications
):
    row = paired_summary_df[
        (
            paired_summary_df[
                "protocol"
            ]
            == "matched_primary"
        )
        & (
            paired_summary_df[
                "comparison"
            ]
            == comparison
        )
        & (
            paired_summary_df[
                "metric"
            ]
            == metric
        )
    ]

    if row.shape[
        0
    ] != 1:
        continue

    row = row.iloc[
        0
    ]

    mean_value = float(
        row[
            "mean_difference"
        ]
    )

    ci_low = float(
        row[
            "bootstrap_ci_low"
        ]
    )

    ci_high = float(
        row[
            "bootstrap_ci_high"
        ]
    )

    ax.errorbar(
        mean_value,
        index,
        xerr=np.array(
            [
                [
                    mean_value
                    - ci_low
                ],
                [
                    ci_high
                    - mean_value
                ],
            ]
        ),
        marker="o",
        capsize=3,
        linestyle="none",
    )

ax.axvline(
    0.0,
    linestyle="--",
    linewidth=1.1,
)

ax.set_yticks(
    np.arange(
        len(
            forest_specifications
        )
    )
)

ax.set_yticklabels(
    [
        specification[
            2
        ]
        for specification in forest_specifications
    ]
)

ax.invert_yaxis()

ax.set_xlabel(
    r"Mean paired difference in $\mathrm{NRMSE}_{3\mathrm{D}}$"
)

ax.set_title(
    "(d) Matched-primary loss contrasts"
)

lambda_composite_pdf, (
    lambda_composite_png
) = save_figure(
    fig,
    "lambda_and_final_composite",
)


# ------------------------------------------------------------
# 14.2 Matched weighting effect versus basis
# ------------------------------------------------------------

fig, axes = plt.subplots(
    2,
    2,
    figsize=(
        11.8,
        8.5,
    ),
    constrained_layout=True,
)

matched_plot_specifications = [
    (
        "observed_primary_nrmse",
        r"Observed parent-risk difference",
        "(a) Parent-risk weighting effect",
    ),
    (
        "query_region_nrmse",
        r"Complete-query risk difference",
        "(b) Complete-query weighting effect",
    ),
    (
        "observed_octant_nrmse_std",
        r"Observed octant-dispersion difference",
        "(c) Observed directional heterogeneity",
    ),
    (
        "query_octant_nrmse_std",
        r"Query octant-dispersion difference",
        "(d) Query directional heterogeneity",
    ),
]

comparison_labels = {
    "global_minus_unweighted": (
        "Global minus unweighted"
    ),
    "octant_minus_unweighted": (
        "Octant-aware minus unweighted"
    ),
}

for axis, (
    metric,
    y_label,
    title,
) in zip(
    axes.ravel(),
    matched_plot_specifications,
):
    for comparison in [
        "global_minus_unweighted",
        "octant_minus_unweighted",
    ]:
        selected = (
            matched_basis_paired_summary_df[
                (
                    matched_basis_paired_summary_df[
                        "comparison"
                    ]
                    == comparison
                )
                & (
                    matched_basis_paired_summary_df[
                        "metric"
                    ]
                    == metric
                )
            ]
            .sort_values(
                "n_basis"
            )
        )

        means = selected[
            "mean_difference"
        ].to_numpy(
            dtype=float
        )

        ci_low = selected[
            "bootstrap_ci_low"
        ].to_numpy(
            dtype=float
        )

        ci_high = selected[
            "bootstrap_ci_high"
        ].to_numpy(
            dtype=float
        )

        axis.errorbar(
            selected[
                "n_basis"
            ],
            means,
            yerr=np.vstack(
                [
                    means
                    - ci_low,
                    ci_high
                    - means,
                ]
            ),
            marker="o",
            capsize=3,
            label=comparison_labels[
                comparison
            ],
        )

    axis.axhline(
        0.0,
        linestyle="--",
        linewidth=1.1,
    )

    axis.axvline(
        practical_basis,
        linestyle=":",
        linewidth=1.0,
    )

    axis.set_xlabel(
        r"Retained modes $n_{\rm b}$"
    )

    axis.set_ylabel(
        y_label
    )

    axis.set_title(
        title
    )

    axis.legend()

matched_effect_pdf, (
    matched_effect_png
) = save_figure(
    fig,
    "matched_weighting_effect_vs_basis",
)


# ------------------------------------------------------------
# 14.3 Split diagnostic for matched-primary protocol
# ------------------------------------------------------------

matched_primary_runs = protocol_runs_df[
    protocol_runs_df[
        "protocol"
    ]
    == "matched_primary"
]

fig, axes = plt.subplots(
    1,
    2,
    figsize=(
        12.0,
        4.9,
    ),
    constrained_layout=True,
)

for model_name in MODEL_ORDER:
    selected = (
        matched_primary_runs[
            matched_primary_runs[
                "model"
            ]
            == model_name
        ]
        .sort_values(
            "replicate"
        )
    )

    axes[
        0
    ].plot(
        selected[
            "replicate"
        ],
        selected[
            "observed_primary_nrmse"
        ],
        marker="o",
        label=model_labels[
            model_name
        ],
    )

    axes[
        1
    ].plot(
        selected[
            "replicate"
        ],
        selected[
            "query_region_nrmse"
        ],
        marker="o",
        label=model_labels[
            model_name
        ],
    )

axes[
    0
].set_xlabel(
    "Label-split replicate"
)

axes[
    0
].set_ylabel(
    r"Observed parent-weighted $\mathrm{NRMSE}_{3\mathrm{D}}$"
)

axes[
    0
].set_title(
    "(a) Observed parent risk by split"
)

axes[
    0
].legend()

axes[
    1
].set_xlabel(
    "Label-split replicate"
)

axes[
    1
].set_ylabel(
    r"Region-primary query $\mathrm{NRMSE}_{3\mathrm{D}}$"
)

axes[
    1
].set_title(
    "(b) Complete-query risk by split"
)

axes[
    1
].legend()

split_diagnostic_pdf, (
    split_diagnostic_png
) = save_figure(
    fig,
    "matched_primary_split_diagnostic",
)


# ============================================================
# 15. Summary JSON and console output
# ============================================================

summary = {
    "scope": (
        "Mock-3 high-mode lambda retuning and loss-family comparison "
        "with fixed unweighted alpha_DM=0 geometry"
    ),
    "inputs": {
        "mock1": str(
            MOCK1_FILE
        ),
        "mock3": str(
            MOCK3_FILE
        ),
        "source_predictions": str(
            SOURCE_PREDICTIONS_FILE
        ),
        "geometry": str(
            PRIMARY_GEOMETRY_FILE
        ),
        "primary_selection": str(
            PRIMARY_SELECTION_FILE
        ),
    },
    "practical_basis": int(
        practical_basis
    ),
    "search_basis_sizes": [
        int(
            value
        )
        for value in SEARCH_BASIS_SIZES
    ],
    "matched_basis_sizes": [
        int(
            value
        )
        for value in MATCHED_BASIS_SIZES
    ],
    "regularization_strengths": [
        float(
            value
        )
        for value in REGULARIZATION_STRENGTHS
    ],
    "fixed_matched_regularization": float(
        FIXED_MATCHED_REGULARIZATION
    ),
    "label_split_seeds": [
        int(
            value
        )
        for value in LABEL_SPLIT_SEEDS
    ],
    "model_configs": (
        MODEL_CONFIGS
    ),
    "primary_metric_weight": (
        primary_metric_info
    ),
    "global_metric_weight": (
        global_metric_info
    ),
    "loss_weight_summary": (
        weight_summary_df.to_dict(
            orient="records"
        )
    ),
    "retuned_selection_frequency": (
        retuned_selection_frequency_df.to_dict(
            orient="records"
        )
    ),
    "protocol_summary": (
        protocol_summary_df.to_dict(
            orient="records"
        )
    ),
    "paired_summary": (
        paired_summary_df.to_dict(
            orient="records"
        )
    ),
    "matched_basis_paired_summary": (
        matched_basis_paired_summary_df.to_dict(
            orient="records"
        )
    ),
    "continuity_audit": (
        continuity_audit_df.to_dict(
            orient="records"
        )
    ),
    "run_token": run_token,
    "outputs": {
        "validation_grid_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_validation_grid.csv"
        ),
        "validation_aggregate_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_validation_aggregate.csv"
        ),
        "protocol_runs_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_protocol_runs.csv"
        ),
        "protocol_summary_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_protocol_summary.csv"
        ),
        "retuned_selection_frequency_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_retuned_selection_frequency.csv"
        ),
        "paired_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_paired.csv"
        ),
        "paired_summary_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_paired_summary.csv"
        ),
        "matched_basis_grid_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_matched_basis_grid.csv"
        ),
        "matched_basis_summary_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_matched_basis_summary.csv"
        ),
        "matched_basis_paired_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_matched_basis_paired.csv"
        ),
        "matched_basis_paired_summary_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_matched_basis_paired_summary.csv"
        ),
        "continuity_audit_csv": str(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_continuity_audit.csv"
        ),
        "lambda_composite_pdf": str(
            lambda_composite_pdf
        ),
        "lambda_composite_png": str(
            lambda_composite_png
        ),
        "matched_effect_pdf": str(
            matched_effect_pdf
        ),
        "matched_effect_png": str(
            matched_effect_png
        ),
        "split_diagnostic_pdf": str(
            split_diagnostic_pdf
        ),
        "split_diagnostic_png": str(
            split_diagnostic_png
        ),
    },
}

summary_json_path = (
    OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_summary.json"
)

summary_json_path.write_text(
    json.dumps(
        summary,
        ensure_ascii=False,
        indent=2,
    ),
    encoding="utf-8",
)

print(
    "="
    * 120
)

print(
    "Mock-3 high-mode lambda and loss-family validation completed"
)

print(
    "="
    * 120
)

print(
    "Retuned selection frequency"
)

print(
    retuned_selection_frequency_df.to_string(
        index=False
    )
)

print(
    "-"
    * 120
)

print(
    "Protocol summary: primary metrics"
)

summary_columns = [
    "protocol",
    "model",
    "n_basis_mean",
    "regularization_strength_median",
    "observed_primary_nrmse_mean",
    "observed_primary_nrmse_std",
    "observed_unweighted_nrmse_mean",
    "query_region_nrmse_mean",
    "query_octant_nrmse_std_mean",
]

available_summary_columns = [
    column
    for column in summary_columns
    if column
    in protocol_summary_df.columns
]

print(
    protocol_summary_df[
        available_summary_columns
    ].to_string(
        index=False
    )
)

print(
    "-"
    * 120
)

print(
    "Matched-primary paired effects"
)

matched_primary_paired = (
    paired_summary_df[
        paired_summary_df[
            "protocol"
        ]
        == "matched_primary"
    ]
)

print(
    matched_primary_paired[
        [
            "comparison",
            "metric",
            "mean_difference",
            "bootstrap_ci_low",
            "bootstrap_ci_high",
            "n_negative",
            "n_positive",
        ]
    ].to_string(
        index=False
    )
)

if not continuity_audit_df.empty:
    print(
        "-"
        * 120
    )

    print(
        "Continuity audit"
    )

    print(
        continuity_audit_df.to_string(
            index=False
        )
    )

print(
    "-"
    * 120
)

print(
    "validation_grid_csv             : "
    f"{OUTPUT_DIR / f'{OUTPUT_PREFIX}_validation_grid.csv'}"
)

print(
    "protocol_runs_csv               : "
    f"{OUTPUT_DIR / f'{OUTPUT_PREFIX}_protocol_runs.csv'}"
)

print(
    "paired_summary_csv              : "
    f"{OUTPUT_DIR / f'{OUTPUT_PREFIX}_paired_summary.csv'}"
)

print(
    "matched_basis_paired_summary_csv: "
    f"{OUTPUT_DIR / f'{OUTPUT_PREFIX}_matched_basis_paired_summary.csv'}"
)

print(
    "lambda_composite_pdf            : "
    f"{lambda_composite_pdf}"
)

print(
    "lambda_composite_png            : "
    f"{lambda_composite_png}"
)

print(
    "matched_effect_pdf              : "
    f"{matched_effect_pdf}"
)

print(
    "matched_effect_png              : "
    f"{matched_effect_png}"
)

print(
    "split_diagnostic_pdf            : "
    f"{split_diagnostic_pdf}"
)

print(
    "split_diagnostic_png            : "
    f"{split_diagnostic_png}"
)

print(
    "summary_json                    : "
    f"{summary_json_path}"
)

print(
    "="
    * 120
)
