# Mock-1: 残る最小追加検証
# 参照/model分割seed、train/validation/test分割seed、および
# 参照バルク速度場の平滑化スケールに対する頑健性を検証します。
#
# 実行条件:
#   1. mock1_bulk_velocity_reconstruction_single_cell_v2 を正常終了済み
#   2. mock1_bulk_corrected_diffusion_single_cell_v5 を正常終了済み
#
# この内容全体を、Jupyter Notebookの新しい空のコードセル1つへ
# 貼り付けて実行してください。
#
# 各反復では、test集合をhyperparameter選択から完全に隔離したまま、
#   - corrected Cartesian diffusion spectral estimator
#   - adaptive local-kernel baseline
# をvalidationで再選択し、同じ未使用test集合で評価します。

import hashlib
import json
import math
import platform
import warnings
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from IPython.display import display
from scipy import linalg
from scipy.spatial import cKDTree

# ============================================================
# 1. 固定入出力パス
# ============================================================

INPUT_DIR = Path(
    'mock_data/mock1_complete_sphere'
)
DATA_FILE = INPUT_DIR / "mock1.npz"

OUTPUT_DIR = Path(
    'mock_data'
)

GEOMETRY_CACHE_FILE = OUTPUT_DIR / "mock1_diffusion_geometry_bulk_cache.npz"
CORRECTED_SUMMARY_FILE = OUTPUT_DIR / "mock1_bulk_corrected_diffusion_summary.json"
ROBUSTNESS_CACHE_DIR = OUTPUT_DIR / "mock1_bulk_robustness_cache"

SPHERE_CENTER = np.array(
    [250.0, 250.0, 250.0],
    dtype=np.float64,
)
VELOCITY_UNIT_LABEL = "simulation velocity unit"
POSITION_UNIT_LABEL = r"$h^{-1}\,\mathrm{Mpc}$"

# ============================================================
# 2. 反復・平滑化スケール
# ============================================================

# 第1組はこれまでのfiducial seedと一致します。
SEED_PAIRS = [
    (20260805, 20260804),
    (20260815, 20260814),
    (20260825, 20260824),
    (20260905, 20260904),
    (20260915, 20260914),
]

REFERENCE_FRACTION = 0.50
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

# bandwidth rank / n_neighbors = 0.375を保ちながら、
# 参照場の空間スケールをfine, fiducial, coarseへ変更します。
SMOOTHING_CONFIGS = [
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

# corrected diffusion estimatorではeta=0を固定し、
# basis数とDirichlet正則化だけを各反復のvalidationで再選択します。
DIFFUSION_BASIS_SIZES = [256, 384, 512]
DIFFUSION_REGULARIZATION_STRENGTHS = [
    1.0e-4,
    1.0e-2,
    1.0,
]
NUMERICAL_RIDGE = 1.0e-10

# local-kernel baselineも各反復・各平滑化スケールでvalidation選択します。
LOCAL_KERNEL_NEIGHBORS = [4, 8, 16, 32, 64]
LOCAL_KERNEL_BANDWIDTH_FRACTION = 3.0 / 8.0
LOCAL_KERNEL_MULTIPLIERS = [0.50, 0.75, 1.0, 1.50]

USE_REFERENCE_FIELD_CACHE = True

SHOW_PLOTS = True
SAVE_PDF = True
SAVE_PNG = True
OUTPUT_PREFIX = "mock1_bulk_seed_scale_robustness"

# ============================================================
# 3. 基本関数
# ============================================================

def file_signature(path):
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stable_signature(payload):
    text = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(
        text.encode("utf-8")
    ).hexdigest()


def finite_median(values):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    return (
        float(np.median(values))
        if values.size
        else math.nan
    )


def finite_quantile(values, q):
    values = np.asarray(
        values,
        dtype=np.float64,
    )
    values = values[np.isfinite(values)]
    return (
        float(np.quantile(values, q))
        if values.size
        else math.nan
    )


def direction_errors(true, predicted):
    true = np.asarray(
        true,
        dtype=np.float64,
    )
    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    true_speed = np.linalg.norm(
        true,
        axis=1,
    )
    predicted_speed = np.linalg.norm(
        predicted,
        axis=1,
    )

    output = np.full(
        true.shape[0],
        np.nan,
        dtype=np.float64,
    )
    valid = (
        (true_speed > 0.0)
        & (predicted_speed > 0.0)
    )

    if np.any(valid):
        cosine = np.sum(
            true[valid] * predicted[valid],
            axis=1,
        ) / (
            true_speed[valid]
            * predicted_speed[valid]
        )
        output[valid] = np.degrees(
            np.arccos(
                np.clip(
                    cosine,
                    -1.0,
                    1.0,
                )
            )
        )

    return output


def velocity_metrics(true, predicted):
    true = np.asarray(
        true,
        dtype=np.float64,
    )
    predicted = np.asarray(
        predicted,
        dtype=np.float64,
    )

    residual = predicted - true
    true_speed = np.linalg.norm(
        true,
        axis=1,
    )
    predicted_speed = np.linalg.norm(
        predicted,
        axis=1,
    )

    rmse_3d = float(
        np.sqrt(
            np.mean(
                np.sum(
                    np.square(residual),
                    axis=1,
                )
            )
        )
    )
    reference_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    np.square(true),
                    axis=1,
                )
            )
        )
    )
    normalized_rmse = (
        rmse_3d / reference_rms
        if reference_rms > 0.0
        else math.nan
    )

    component_rmse = np.sqrt(
        np.mean(
            np.square(residual),
            axis=0,
        )
    )

    correlations = []
    for component in range(3):
        if (
            np.std(true[:, component]) == 0.0
            or np.std(predicted[:, component]) == 0.0
        ):
            correlations.append(
                math.nan
            )
        else:
            correlations.append(
                float(
                    np.corrcoef(
                        true[:, component],
                        predicted[:, component],
                    )[0, 1]
                )
            )

    angles = direction_errors(
        true,
        predicted,
    )

    relative_speed_error = np.full(
        true.shape[0],
        np.nan,
        dtype=np.float64,
    )
    positive = true_speed > 0.0
    relative_speed_error[positive] = (
        predicted_speed[positive]
        - true_speed[positive]
    ) / true_speed[positive]

    return {
        "rmse_3d": rmse_3d,
        "normalized_rmse": float(
            normalized_rmse
        ),
        "rmse_x": float(
            component_rmse[0]
        ),
        "rmse_y": float(
            component_rmse[1]
        ),
        "rmse_z": float(
            component_rmse[2]
        ),
        "corr_x": correlations[0],
        "corr_y": correlations[1],
        "corr_z": correlations[2],
        "median_direction_error_deg": (
            finite_median(angles)
        ),
        "p90_direction_error_deg": (
            finite_quantile(
                angles,
                0.90,
            )
        ),
        "median_abs_relative_speed_error": (
            finite_median(
                np.abs(
                    relative_speed_error
                )
            )
        ),
    }


def calibration_metrics(true, predicted):
    records = {}

    for component, label in enumerate(
        ["x", "y", "z"]
    ):
        reference = true[
            :,
            component,
        ]
        reconstructed = predicted[
            :,
            component,
        ]

        centered = (
            reference
            - np.mean(reference)
        )
        denominator = float(
            np.dot(
                centered,
                centered,
            )
        )

        if denominator > 0.0:
            slope = float(
                np.dot(
                    centered,
                    reconstructed
                    - np.mean(
                        reconstructed
                    ),
                ) / denominator
            )
            intercept = float(
                np.mean(
                    reconstructed
                )
                - slope
                * np.mean(reference)
            )
        else:
            slope = math.nan
            intercept = math.nan

        origin_denominator = float(
            np.dot(
                reference,
                reference,
            )
        )
        slope_origin = (
            float(
                np.dot(
                    reference,
                    reconstructed,
                ) / origin_denominator
            )
            if origin_denominator > 0.0
            else math.nan
        )

        records[
            f"calibration_slope_{label}"
        ] = slope
        records[
            f"calibration_intercept_{label}"
        ] = intercept
        records[
            f"calibration_slope_origin_{label}"
        ] = slope_origin

    vector_denominator = float(
        np.sum(
            np.square(true)
        )
    )
    records[
        "vector_gain_through_origin"
    ] = (
        float(
            np.sum(
                true * predicted
            ) / vector_denominator
        )
        if vector_denominator > 0.0
        else math.nan
    )

    return records


def make_reference_model_split(
    n,
    reference_fraction,
    seed,
):
    rng = np.random.default_rng(
        seed
    )
    order = rng.permutation(n)
    n_reference = int(
        round(
            reference_fraction * n
        )
    )
    return (
        np.sort(
            order[:n_reference]
        ),
        np.sort(
            order[n_reference:]
        ),
    )


def make_label_split(
    n,
    train_fraction,
    validation_fraction,
    seed,
):
    rng = np.random.default_rng(
        seed
    )
    order = rng.permutation(n)

    n_train = int(
        round(
            train_fraction * n
        )
    )
    n_validation = int(
        round(
            validation_fraction * n
        )
    )

    return {
        "train": np.sort(
            order[:n_train]
        ),
        "validation": np.sort(
            order[
                n_train:
                n_train + n_validation
            ]
        ),
        "test": np.sort(
            order[
                n_train
                + n_validation:
            ]
        ),
    }


def factor_dense_system(system):
    try:
        factor = linalg.cho_factor(
            system,
            lower=True,
            check_finite=False,
        )
        return (
            "cholesky",
            factor,
        )
    except linalg.LinAlgError:
        return (
            "direct",
            np.asarray(
                system,
                dtype=np.float64,
            ),
        )


def solve_factored_dense(
    factor_kind,
    factor,
    rhs,
):
    if factor_kind == "cholesky":
        return linalg.cho_solve(
            factor,
            rhs,
            check_finite=False,
        )

    return linalg.solve(
        factor,
        rhs,
        assume_a="sym",
        check_finite=False,
    )


def save_figure(fig, stem):
    if SAVE_PDF:
        fig.savefig(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_{stem}.pdf",
            bbox_inches="tight",
        )

    if SAVE_PNG:
        fig.savefig(
            OUTPUT_DIR
            / f"{OUTPUT_PREFIX}_{stem}.png",
            dpi=180,
            bbox_inches="tight",
        )

    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 4. 参照場構成
# ============================================================

def query_reference_neighbors(
    query_positions,
    reference_positions,
    max_neighbors,
):
    tree = cKDTree(
        reference_positions
    )

    try:
        distances, indices = tree.query(
            query_positions,
            k=max_neighbors,
            workers=-1,
        )
    except TypeError:
        distances, indices = tree.query(
            query_positions,
            k=max_neighbors,
        )

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if max_neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    return distances, indices


def bulk_field_from_neighbors(
    distances_max,
    indices_max,
    reference_velocities,
    n_neighbors,
    bandwidth_neighbor,
    bandwidth_multiplier,
):
    distances = distances_max[
        :,
        :n_neighbors,
    ]
    indices = indices_max[
        :,
        :n_neighbors,
    ]

    bandwidth = (
        bandwidth_multiplier
        * distances[
            :,
            bandwidth_neighbor - 1
        ]
    )
    positive = bandwidth[
        bandwidth > 0.0
    ]

    if positive.size == 0:
        raise ValueError(
            "参照場bandwidthがすべて0です。"
        )

    bandwidth = np.maximum(
        bandwidth,
        max(
            float(
                np.median(positive)
            )
            * 1.0e-8,
            np.finfo(
                np.float64
            ).eps,
        ),
    )

    weights = np.exp(
        -np.square(
            distances
            / bandwidth[:, None]
        )
    )
    weight_sum = np.sum(
        weights,
        axis=1,
    )

    neighbor_velocities = (
        reference_velocities[
            indices
        ]
    )

    bulk_velocity = np.einsum(
        "ij,ijk->ik",
        weights,
        neighbor_velocities,
        optimize=True,
    )
    bulk_velocity /= (
        weight_sum[:, None]
    )

    residual = (
        neighbor_velocities
        - bulk_velocity[:, None, :]
    )
    dispersion_squared = (
        np.einsum(
            "ij,ijk->i",
            weights,
            np.square(residual),
            optimize=True,
        ) / weight_sum
    )
    dispersion = np.sqrt(
        np.maximum(
            dispersion_squared,
            0.0,
        )
    )

    effective_neighbors = (
        np.square(weight_sum)
        / np.sum(
            np.square(weights),
            axis=1,
        )
    )

    return {
        "bulk_velocity": bulk_velocity,
        "bandwidth": bandwidth,
        "dispersion": dispersion,
        "effective_neighbors": (
            effective_neighbors
        ),
    }


def reference_cache_path(
    reference_seed,
    scale,
):
    return (
        ROBUSTNESS_CACHE_DIR
        / (
            f"reference_seed_{reference_seed}"
            f"_scale_{scale}.npz"
        )
    )


def load_or_build_reference_field(
    pos,
    vel_raw,
    reference_indices,
    model_indices,
    reference_seed,
    config,
):
    cache_path = reference_cache_path(
        reference_seed,
        config["scale"],
    )

    signature = stable_signature(
        {
            "data": file_signature(
                DATA_FILE
            ),
            "reference_seed": int(
                reference_seed
            ),
            "reference_fraction": float(
                REFERENCE_FRACTION
            ),
            "reference_indices_hash": (
                hashlib.sha256(
                    reference_indices.tobytes()
                ).hexdigest()
            ),
            "model_indices_hash": (
                hashlib.sha256(
                    model_indices.tobytes()
                ).hexdigest()
            ),
            **config,
        }
    )

    if (
        USE_REFERENCE_FIELD_CACHE
        and cache_path.is_file()
    ):
        try:
            with np.load(
                cache_path,
                allow_pickle=False,
            ) as data:
                if (
                    str(
                        data["signature"].item()
                    )
                    == signature
                ):
                    return {
                        "bulk_velocity": np.asarray(
                            data["bulk_velocity"],
                            dtype=np.float64,
                        ),
                        "bandwidth": np.asarray(
                            data["bandwidth"],
                            dtype=np.float64,
                        ),
                        "dispersion": np.asarray(
                            data["dispersion"],
                            dtype=np.float64,
                        ),
                        "effective_neighbors": (
                            np.asarray(
                                data[
                                    "effective_neighbors"
                                ],
                                dtype=np.float64,
                            )
                        ),
                    }
        except Exception as exc:
            warnings.warn(
                "参照場cacheを読めなかったため"
                f"再計算します: {exc}"
            )

    max_neighbors = max(
        item["n_neighbors"]
        for item in SMOOTHING_CONFIGS
    )
    distances_max, indices_max = (
        query_reference_neighbors(
            query_positions=pos[
                model_indices
            ],
            reference_positions=pos[
                reference_indices
            ],
            max_neighbors=max_neighbors,
        )
    )

    result = bulk_field_from_neighbors(
        distances_max=distances_max,
        indices_max=indices_max,
        reference_velocities=vel_raw[
            reference_indices
        ],
        n_neighbors=int(
            config["n_neighbors"]
        ),
        bandwidth_neighbor=int(
            config[
                "bandwidth_neighbor"
            ]
        ),
        bandwidth_multiplier=float(
            config[
                "bandwidth_multiplier"
            ]
        ),
    )

    if USE_REFERENCE_FIELD_CACHE:
        np.savez_compressed(
            cache_path,
            signature=np.array(
                signature
            ),
            bulk_velocity=(
                result["bulk_velocity"]
            ),
            bandwidth=(
                result["bandwidth"]
            ),
            dispersion=(
                result["dispersion"]
            ),
            effective_neighbors=(
                result[
                    "effective_neighbors"
                ]
            ),
        )

    return result


# ============================================================
# 5. corrected diffusion spectral estimator
# ============================================================

def regularization_diagonal(
    generator_eigenvalues,
    n_basis,
):
    return np.maximum(
        generator_eigenvalues[
            :n_basis
        ],
        NUMERICAL_RIDGE,
    )


def prepare_diffusion_validation_factors(
    phi_train_max,
    generator_eigenvalues,
):
    factors = {}

    for n_basis in (
        DIFFUSION_BASIS_SIZES
    ):
        phi = phi_train_max[
            :,
            :n_basis,
        ]
        system_base = (
            phi.T @ phi
        ) / phi.shape[0]

        regularization = (
            regularization_diagonal(
                generator_eigenvalues,
                n_basis,
            )
        )

        for regularization_strength in (
            DIFFUSION_REGULARIZATION_STRENGTHS
        ):
            system = (
                system_base.copy()
            )
            diagonal = (
                np.diag_indices_from(
                    system
                )
            )
            system[diagonal] += (
                regularization_strength
                * regularization
                + NUMERICAL_RIDGE
            )

            factors[
                (
                    n_basis,
                    regularization_strength,
                )
            ] = factor_dense_system(
                system
            )

    return factors


def select_diffusion_hyperparameters(
    phi_train_max,
    phi_validation_max,
    target_train,
    target_validation,
    generator_eigenvalues,
    validation_factors,
):
    best_candidate = None
    records = []

    for n_basis in (
        DIFFUSION_BASIS_SIZES
    ):
        phi_train = phi_train_max[
            :,
            :n_basis,
        ]
        phi_validation = (
            phi_validation_max[
                :,
                :n_basis,
            ]
        )

        rhs = (
            phi_train.T
            @ target_train
        ) / phi_train.shape[0]

        for regularization_strength in (
            DIFFUSION_REGULARIZATION_STRENGTHS
        ):
            factor_kind, factor = (
                validation_factors[
                    (
                        n_basis,
                        regularization_strength,
                    )
                ]
            )
            coefficients = (
                solve_factored_dense(
                    factor_kind,
                    factor,
                    rhs,
                )
            )
            prediction = (
                phi_validation
                @ coefficients
            )

            metrics = velocity_metrics(
                target_validation,
                prediction,
            )
            records.append(
                {
                    "n_basis": int(
                        n_basis
                    ),
                    "regularization_strength": float(
                        regularization_strength
                    ),
                    **metrics,
                }
            )

            candidate = (
                metrics[
                    "normalized_rmse"
                ],
                int(n_basis),
                float(
                    regularization_strength
                ),
            )

            if (
                best_candidate is None
                or candidate
                < best_candidate
            ):
                best_candidate = (
                    candidate
                )

    if best_candidate is None:
        raise RuntimeError(
            "diffusion hyperparameterが"
            "選択されませんでした。"
        )

    (
        _,
        best_basis,
        best_regularization,
    ) = best_candidate

    return (
        best_basis,
        best_regularization,
        records,
    )


def fit_predict_diffusion(
    phi_fit_max,
    phi_test_max,
    target_fit,
    generator_eigenvalues,
    n_basis,
    regularization_strength,
):
    phi_fit = phi_fit_max[
        :,
        :n_basis,
    ]
    phi_test = phi_test_max[
        :,
        :n_basis,
    ]

    system = (
        phi_fit.T @ phi_fit
    ) / phi_fit.shape[0]

    diagonal = np.diag_indices_from(
        system
    )
    system[diagonal] += (
        regularization_strength
        * regularization_diagonal(
            generator_eigenvalues,
            n_basis,
        )
        + NUMERICAL_RIDGE
    )

    factor_kind, factor = (
        factor_dense_system(
            system
        )
    )

    rhs = (
        phi_fit.T
        @ target_fit
    ) / phi_fit.shape[0]

    coefficients = solve_factored_dense(
        factor_kind,
        factor,
        rhs,
    )

    return (
        phi_test @ coefficients
    )


# ============================================================
# 6. adaptive local-kernel baseline
# ============================================================

def query_label_neighbors(
    query_positions,
    labelled_positions,
    max_neighbors,
):
    tree = cKDTree(
        labelled_positions
    )
    try:
        distances, indices = tree.query(
            query_positions,
            k=max_neighbors,
            workers=-1,
        )
    except TypeError:
        distances, indices = tree.query(
            query_positions,
            k=max_neighbors,
        )

    distances = np.asarray(
        distances,
        dtype=np.float64,
    )
    indices = np.asarray(
        indices,
        dtype=np.int64,
    )

    if max_neighbors == 1:
        distances = distances[:, None]
        indices = indices[:, None]

    return distances, indices


def kernel_prediction_from_neighbors(
    distances_max,
    indices_max,
    labelled_velocities,
    n_neighbors,
    bandwidth_neighbor,
    bandwidth_multiplier,
):
    distances = distances_max[
        :,
        :n_neighbors,
    ]
    indices = indices_max[
        :,
        :n_neighbors,
    ]

    bandwidth = (
        bandwidth_multiplier
        * distances[
            :,
            bandwidth_neighbor - 1
        ]
    )
    positive = bandwidth[
        bandwidth > 0.0
    ]
    bandwidth = np.maximum(
        bandwidth,
        max(
            float(
                np.median(positive)
            )
            * 1.0e-8,
            np.finfo(
                np.float64
            ).eps,
        ),
    )

    weights = np.exp(
        -np.square(
            distances
            / bandwidth[:, None]
        )
    )
    weight_sum = np.sum(
        weights,
        axis=1,
    )

    prediction = np.einsum(
        "ij,ijk->ik",
        weights,
        labelled_velocities[
            indices
        ],
        optimize=True,
    )
    prediction /= (
        weight_sum[:, None]
    )

    return prediction


def select_local_kernel(
    validation_distances_max,
    validation_indices_max,
    train_velocity,
    validation_truth,
):
    best_candidate = None
    records = []

    for n_neighbors in (
        LOCAL_KERNEL_NEIGHBORS
    ):
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

        for multiplier in (
            LOCAL_KERNEL_MULTIPLIERS
        ):
            prediction = (
                kernel_prediction_from_neighbors(
                    distances_max=(
                        validation_distances_max
                    ),
                    indices_max=(
                        validation_indices_max
                    ),
                    labelled_velocities=(
                        train_velocity
                    ),
                    n_neighbors=(
                        n_neighbors
                    ),
                    bandwidth_neighbor=(
                        bandwidth_neighbor
                    ),
                    bandwidth_multiplier=(
                        multiplier
                    ),
                )
            )
            metrics = velocity_metrics(
                validation_truth,
                prediction,
            )
            records.append(
                {
                    "n_neighbors": int(
                        n_neighbors
                    ),
                    "bandwidth_neighbor": int(
                        bandwidth_neighbor
                    ),
                    "bandwidth_multiplier": float(
                        multiplier
                    ),
                    **metrics,
                }
            )

            candidate = (
                metrics[
                    "normalized_rmse"
                ],
                int(n_neighbors),
                float(multiplier),
            )

            if (
                best_candidate is None
                or candidate
                < best_candidate
            ):
                best_candidate = (
                    candidate
                )

    if best_candidate is None:
        raise RuntimeError(
            "local-kernel hyperparameterが"
            "選択されませんでした。"
        )

    (
        _,
        best_neighbors,
        best_multiplier,
    ) = best_candidate

    best_bandwidth_neighbor = min(
        best_neighbors,
        max(
            1,
            int(
                round(
                    best_neighbors
                    * LOCAL_KERNEL_BANDWIDTH_FRACTION
                )
            ),
        ),
    )

    return (
        best_neighbors,
        best_bandwidth_neighbor,
        best_multiplier,
        records,
    )


# ============================================================
# 7. 入力の読み込み
# ============================================================

required_files = [
    DATA_FILE,
    GEOMETRY_CACHE_FILE,
]

missing_files = [
    str(path)
    for path in required_files
    if not path.is_file()
]

if missing_files:
    raise FileNotFoundError(
        "必要な入力が見つかりません。\n"
        + "\n".join(
            missing_files
        )
    )

with np.load(
    DATA_FILE
) as data:
    pos_absolute = np.asarray(
        data["pos"],
        dtype=np.float64,
    )
    vel_raw = np.asarray(
        data["vel"],
        dtype=np.float64,
    )

pos = (
    pos_absolute
    - SPHERE_CENTER[None, :]
)

with np.load(
    GEOMETRY_CACHE_FILE,
    allow_pickle=False,
) as data:
    eigenfunctions = np.asarray(
        data["eigenfunctions"],
        dtype=np.float64,
    )
    generator_eigenvalues = np.asarray(
        data["generator_eigenvalues"],
        dtype=np.float64,
    )

if max(
    DIFFUSION_BASIS_SIZES
) > eigenfunctions.shape[1]:
    raise ValueError(
        "geometry cacheの固有関数数が"
        "DIFFUSION_BASIS_SIZESより少ないです。"
    )

ROBUSTNESS_CACHE_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

print("=" * 108)
print(
    "Mock-1 seed and smoothing-scale robustness"
)
print("=" * 108)
print(
    f"Python             : "
    f"{platform.python_version()}"
)
print(
    f"NumPy / SciPy      : "
    f"{np.__version__} / {scipy.__version__}"
)
print(
    f"Objects            : "
    f"{pos.shape[0]:,}"
)
print(
    f"Seed pairs         : "
    f"{len(SEED_PAIRS)}"
)
print(
    "Smoothing scales   : "
    + ", ".join(
        item["scale"]
        for item in SMOOTHING_CONFIGS
    )
)

if CORRECTED_SUMMARY_FILE.is_file():
    try:
        with CORRECTED_SUMMARY_FILE.open(
            "r",
            encoding="utf-8",
        ) as handle:
            corrected_summary = json.load(
                handle
            )
        selected = corrected_summary[
            "selected_model"
        ]
        print(
            "Previous selected model: "
            f"eta={selected['frame_eta']}, "
            f"basis={selected['n_basis']}, "
            "lambda="
            f"{selected['regularization_strength']}"
        )
    except Exception as exc:
        warnings.warn(
            "v5 summaryを読めませんでした: "
            f"{exc}"
        )

# ============================================================
# 8. 反復
# ============================================================

run_records = []
diffusion_validation_records = []
kernel_validation_records = []

max_reference_neighbors = max(
    item["n_neighbors"]
    for item in SMOOTHING_CONFIGS
)
max_local_neighbors = max(
    LOCAL_KERNEL_NEIGHBORS
)

for replicate_index, (
    reference_seed,
    label_seed,
) in enumerate(
    SEED_PAIRS,
    start=1,
):
    print("-" * 108)
    print(
        f"Replicate {replicate_index}/"
        f"{len(SEED_PAIRS)}: "
        f"reference_seed={reference_seed}, "
        f"label_seed={label_seed}"
    )

    (
        reference_indices,
        model_indices,
    ) = make_reference_model_split(
        n=pos.shape[0],
        reference_fraction=(
            REFERENCE_FRACTION
        ),
        seed=reference_seed,
    )

    split = make_label_split(
        n=model_indices.size,
        train_fraction=(
            TRAIN_FRACTION
        ),
        validation_fraction=(
            VALIDATION_FRACTION
        ),
        seed=label_seed,
    )

    train_local = split["train"]
    validation_local = split[
        "validation"
    ]
    test_local = split["test"]

    train_global = model_indices[
        train_local
    ]
    validation_global = model_indices[
        validation_local
    ]
    test_global = model_indices[
        test_local
    ]

    fit_local = np.sort(
        np.concatenate(
            [
                train_local,
                validation_local,
            ]
        )
    )
    fit_global = model_indices[
        fit_local
    ]

    phi_train_max = eigenfunctions[
        train_global,
        :max(
            DIFFUSION_BASIS_SIZES
        ),
    ]
    phi_validation_max = eigenfunctions[
        validation_global,
        :max(
            DIFFUSION_BASIS_SIZES
        ),
    ]
    phi_fit_max = eigenfunctions[
        fit_global,
        :max(
            DIFFUSION_BASIS_SIZES
        ),
    ]
    phi_test_max = eigenfunctions[
        test_global,
        :max(
            DIFFUSION_BASIS_SIZES
        ),
    ]

    diffusion_validation_factors = (
        prepare_diffusion_validation_factors(
            phi_train_max=(
                phi_train_max
            ),
            generator_eigenvalues=(
                generator_eigenvalues
            ),
        )
    )

    # 参照近傍は最大64近傍を一度だけ探索し、
    # 各平滑化スケールで再利用します。
    reference_distances_max, (
        reference_indices_max
    ) = query_reference_neighbors(
        query_positions=pos[
            model_indices
        ],
        reference_positions=pos[
            reference_indices
        ],
        max_neighbors=(
            max_reference_neighbors
        ),
    )

    # local-kernel validation/test用の近傍探索も一度だけ行います。
    (
        validation_label_distances_max,
        validation_label_indices_max,
    ) = query_label_neighbors(
        query_positions=pos[
            validation_global
        ],
        labelled_positions=pos[
            train_global
        ],
        max_neighbors=(
            max_local_neighbors
        ),
    )

    (
        test_label_distances_max,
        test_label_indices_max,
    ) = query_label_neighbors(
        query_positions=pos[
            test_global
        ],
        labelled_positions=pos[
            fit_global
        ],
        max_neighbors=(
            max_local_neighbors
        ),
    )

    for config in SMOOTHING_CONFIGS:
        scale = config["scale"]

        # ここでは既に近傍探索済みなのでcacheを使わず、
        # 各scaleの重みだけを高速に再計算します。
        reference_result = (
            bulk_field_from_neighbors(
                distances_max=(
                    reference_distances_max
                ),
                indices_max=(
                    reference_indices_max
                ),
                reference_velocities=(
                    vel_raw[
                        reference_indices
                    ]
                ),
                n_neighbors=int(
                    config["n_neighbors"]
                ),
                bandwidth_neighbor=int(
                    config[
                        "bandwidth_neighbor"
                    ]
                ),
                bandwidth_multiplier=float(
                    config[
                        "bandwidth_multiplier"
                    ]
                ),
            )
        )

        bulk_model = (
            reference_result[
                "bulk_velocity"
            ]
        )
        bulk_train = bulk_model[
            train_local
        ]
        bulk_validation = bulk_model[
            validation_local
        ]
        bulk_test = bulk_model[
            test_local
        ]
        bulk_fit = bulk_model[
            fit_local
        ]

        # ---------- corrected diffusion ----------
        (
            best_basis,
            best_regularization,
            diffusion_grid,
        ) = select_diffusion_hyperparameters(
            phi_train_max=(
                phi_train_max
            ),
            phi_validation_max=(
                phi_validation_max
            ),
            target_train=bulk_train,
            target_validation=(
                bulk_validation
            ),
            generator_eigenvalues=(
                generator_eigenvalues
            ),
            validation_factors=(
                diffusion_validation_factors
            ),
        )

        for row in diffusion_grid:
            diffusion_validation_records.append(
                {
                    "replicate": int(
                        replicate_index
                    ),
                    "reference_seed": int(
                        reference_seed
                    ),
                    "label_seed": int(
                        label_seed
                    ),
                    "scale": scale,
                    **row,
                }
            )

        diffusion_test = fit_predict_diffusion(
            phi_fit_max=phi_fit_max,
            phi_test_max=phi_test_max,
            target_fit=bulk_fit,
            generator_eigenvalues=(
                generator_eigenvalues
            ),
            n_basis=best_basis,
            regularization_strength=(
                best_regularization
            ),
        )

        diffusion_result = {
            "model": (
                "corrected Cartesian diffusion spectral"
            ),
            **velocity_metrics(
                bulk_test,
                diffusion_test,
            ),
            **calibration_metrics(
                bulk_test,
                diffusion_test,
            ),
            "selected_basis": int(
                best_basis
            ),
            "selected_regularization": float(
                best_regularization
            ),
            "selected_kernel_neighbors": (
                math.nan
            ),
            "selected_kernel_bandwidth_neighbor": (
                math.nan
            ),
            "selected_kernel_multiplier": (
                math.nan
            ),
        }

        # ---------- local-kernel baseline ----------
        (
            best_kernel_neighbors,
            best_kernel_bandwidth_neighbor,
            best_kernel_multiplier,
            kernel_grid,
        ) = select_local_kernel(
            validation_distances_max=(
                validation_label_distances_max
            ),
            validation_indices_max=(
                validation_label_indices_max
            ),
            train_velocity=bulk_train,
            validation_truth=(
                bulk_validation
            ),
        )

        for row in kernel_grid:
            kernel_validation_records.append(
                {
                    "replicate": int(
                        replicate_index
                    ),
                    "reference_seed": int(
                        reference_seed
                    ),
                    "label_seed": int(
                        label_seed
                    ),
                    "scale": scale,
                    **row,
                }
            )

        kernel_test = (
            kernel_prediction_from_neighbors(
                distances_max=(
                    test_label_distances_max
                ),
                indices_max=(
                    test_label_indices_max
                ),
                labelled_velocities=(
                    bulk_fit
                ),
                n_neighbors=(
                    best_kernel_neighbors
                ),
                bandwidth_neighbor=(
                    best_kernel_bandwidth_neighbor
                ),
                bandwidth_multiplier=(
                    best_kernel_multiplier
                ),
            )
        )

        kernel_result = {
            "model": (
                "adaptive local-kernel regression"
            ),
            **velocity_metrics(
                bulk_test,
                kernel_test,
            ),
            **calibration_metrics(
                bulk_test,
                kernel_test,
            ),
            "selected_basis": math.nan,
            "selected_regularization": (
                math.nan
            ),
            "selected_kernel_neighbors": int(
                best_kernel_neighbors
            ),
            "selected_kernel_bandwidth_neighbor": int(
                best_kernel_bandwidth_neighbor
            ),
            "selected_kernel_multiplier": float(
                best_kernel_multiplier
            ),
        }

        # ---------- mean baseline ----------
        mean_test = np.repeat(
            np.mean(
                bulk_fit,
                axis=0,
            )[None, :],
            bulk_test.shape[0],
            axis=0,
        )
        mean_result = {
            "model": (
                "mean-bulk baseline"
            ),
            **velocity_metrics(
                bulk_test,
                mean_test,
            ),
            **calibration_metrics(
                bulk_test,
                mean_test,
            ),
            "selected_basis": math.nan,
            "selected_regularization": (
                math.nan
            ),
            "selected_kernel_neighbors": (
                math.nan
            ),
            "selected_kernel_bandwidth_neighbor": (
                math.nan
            ),
            "selected_kernel_multiplier": (
                math.nan
            ),
        }

        common = {
            "replicate": int(
                replicate_index
            ),
            "reference_seed": int(
                reference_seed
            ),
            "label_seed": int(
                label_seed
            ),
            "scale": scale,
            "reference_n_neighbors": int(
                config["n_neighbors"]
            ),
            "reference_bandwidth_neighbor": int(
                config[
                    "bandwidth_neighbor"
                ]
            ),
            "reference_bandwidth_multiplier": float(
                config[
                    "bandwidth_multiplier"
                ]
            ),
            "median_reference_bandwidth": float(
                np.median(
                    reference_result[
                        "bandwidth"
                    ]
                )
            ),
            "median_reference_dispersion": float(
                np.median(
                    reference_result[
                        "dispersion"
                    ]
                )
            ),
            "median_reference_effective_neighbors": float(
                np.median(
                    reference_result[
                        "effective_neighbors"
                    ]
                )
            ),
            "reference_bulk_rms": float(
                np.sqrt(
                    np.mean(
                        np.sum(
                            np.square(
                                bulk_model
                            ),
                            axis=1,
                        )
                    )
                )
            ),
            "n_train": int(
                train_local.size
            ),
            "n_validation": int(
                validation_local.size
            ),
            "n_test": int(
                test_local.size
            ),
        }

        for result in [
            diffusion_result,
            kernel_result,
            mean_result,
        ]:
            run_records.append(
                {
                    **common,
                    **result,
                }
            )

        print(
            "  "
            f"scale={scale:8s}, "
            "median h="
            f"{common['median_reference_bandwidth']:.3f}, "
            "diffusion NRMSE="
            f"{diffusion_result['normalized_rmse']:.5f}, "
            "kernel NRMSE="
            f"{kernel_result['normalized_rmse']:.5f}, "
            "diffusion basis/lambda="
            f"{best_basis}/{best_regularization:g}"
        )

# ============================================================
# 9. 集約
# ============================================================

runs_df = pd.DataFrame(
    run_records
)
diffusion_validation_df = pd.DataFrame(
    diffusion_validation_records
)
kernel_validation_df = pd.DataFrame(
    kernel_validation_records
)

metric_columns = [
    "normalized_rmse",
    "median_direction_error_deg",
    "p90_direction_error_deg",
    "median_abs_relative_speed_error",
    "corr_x",
    "corr_y",
    "corr_z",
    "calibration_slope_x",
    "calibration_slope_y",
    "calibration_slope_z",
    "vector_gain_through_origin",
]

summary_records = []

for (
    scale,
    model,
), group in runs_df.groupby(
    [
        "scale",
        "model",
    ],
    sort=False,
):
    record = {
        "scale": scale,
        "model": model,
        "n_replicates": int(
            group.shape[0]
        ),
        "mean_reference_bandwidth": float(
            group[
                "median_reference_bandwidth"
            ].mean()
        ),
        "std_reference_bandwidth": float(
            group[
                "median_reference_bandwidth"
            ].std(
                ddof=1
            )
        ),
        "mean_reference_bulk_rms": float(
            group[
                "reference_bulk_rms"
            ].mean()
        ),
    }

    for column in metric_columns:
        record[
            f"{column}_mean"
        ] = float(
            group[column].mean()
        )
        record[
            f"{column}_std"
        ] = float(
            group[column].std(
                ddof=1
            )
        )
        record[
            f"{column}_min"
        ] = float(
            group[column].min()
        )
        record[
            f"{column}_max"
        ] = float(
            group[column].max()
        )

    summary_records.append(
        record
    )

summary_df = pd.DataFrame(
    summary_records
)

# diffusionとkernelのpaired差
paired_records = []

for (
    replicate,
    scale,
), group in runs_df.groupby(
    [
        "replicate",
        "scale",
    ]
):
    lookup = group.set_index(
        "model"
    )

    diffusion_name = (
        "corrected Cartesian diffusion spectral"
    )
    kernel_name = (
        "adaptive local-kernel regression"
    )

    if (
        diffusion_name in lookup.index
        and kernel_name in lookup.index
    ):
        paired_records.append(
            {
                "replicate": int(
                    replicate
                ),
                "scale": scale,
                "nrmse_diffusion_minus_kernel": float(
                    lookup.loc[
                        diffusion_name,
                        "normalized_rmse",
                    ]
                    - lookup.loc[
                        kernel_name,
                        "normalized_rmse",
                    ]
                ),
                "direction_diffusion_minus_kernel_deg": float(
                    lookup.loc[
                        diffusion_name,
                        "median_direction_error_deg",
                    ]
                    - lookup.loc[
                        kernel_name,
                        "median_direction_error_deg",
                    ]
                ),
                "amplitude_diffusion_minus_kernel": float(
                    lookup.loc[
                        diffusion_name,
                        "median_abs_relative_speed_error",
                    ]
                    - lookup.loc[
                        kernel_name,
                        "median_abs_relative_speed_error",
                    ]
                ),
            }
        )

paired_df = pd.DataFrame(
    paired_records
)

paired_summary_records = []

for scale, group in paired_df.groupby(
    "scale",
    sort=False,
):
    paired_summary_records.append(
        {
            "scale": scale,
            "n_replicates": int(
                group.shape[0]
            ),
            "mean_nrmse_diffusion_minus_kernel": float(
                group[
                    "nrmse_diffusion_minus_kernel"
                ].mean()
            ),
            "std_nrmse_diffusion_minus_kernel": float(
                group[
                    "nrmse_diffusion_minus_kernel"
                ].std(
                    ddof=1
                )
            ),
            "diffusion_nrmse_wins": int(
                np.sum(
                    group[
                        "nrmse_diffusion_minus_kernel"
                    ] < 0.0
                )
            ),
            "mean_direction_diffusion_minus_kernel_deg": float(
                group[
                    "direction_diffusion_minus_kernel_deg"
                ].mean()
            ),
            "mean_amplitude_diffusion_minus_kernel": float(
                group[
                    "amplitude_diffusion_minus_kernel"
                ].mean()
            ),
        }
    )

paired_summary_df = pd.DataFrame(
    paired_summary_records
)

# fiducial seed robustnessだけを抜き出した簡潔な表
fiducial_df = runs_df[
    runs_df["scale"] == "fiducial"
].copy()

selected_diffusion_df = runs_df[
    runs_df["model"]
    == "corrected Cartesian diffusion spectral"
][
    [
        "replicate",
        "reference_seed",
        "label_seed",
        "scale",
        "selected_basis",
        "selected_regularization",
        "normalized_rmse",
        "median_direction_error_deg",
        "calibration_slope_x",
        "calibration_slope_y",
        "calibration_slope_z",
    ]
].copy()

print("=" * 108)
print(
    "Seed and scale robustness: summary"
)
print("=" * 108)

display(
    summary_df[
        [
            "scale",
            "model",
            "n_replicates",
            "mean_reference_bandwidth",
            "normalized_rmse_mean",
            "normalized_rmse_std",
            "median_direction_error_deg_mean",
            "median_direction_error_deg_std",
            "median_abs_relative_speed_error_mean",
            "median_abs_relative_speed_error_std",
            "calibration_slope_x_mean",
            "calibration_slope_y_mean",
            "calibration_slope_z_mean",
        ]
    ]
)

print(
    "Paired diffusion-minus-kernel differences "
    "(negative favors diffusion)"
)
display(
    paired_summary_df
)

print(
    "Selected diffusion hyperparameters"
)
display(
    selected_diffusion_df
)

# ============================================================
# 10. 図
# ============================================================

plot_models = [
    "corrected Cartesian diffusion spectral",
    "adaptive local-kernel regression",
]

scale_order = [
    item["scale"]
    for item in SMOOTHING_CONFIGS
]

plot_summary = summary_df[
    summary_df["model"].isin(
        plot_models
    )
].copy()

# NRMSE vs physical smoothing scale
fig, ax = plt.subplots(
    figsize=(6.8, 4.8)
)

for model in plot_models:
    selected = (
        plot_summary[
            plot_summary["model"]
            == model
        ]
        .set_index("scale")
        .loc[scale_order]
        .reset_index()
    )

    ax.errorbar(
        selected[
            "mean_reference_bandwidth"
        ],
        selected[
            "normalized_rmse_mean"
        ],
        yerr=selected[
            "normalized_rmse_std"
        ],
        marker="o",
        capsize=4,
        label=model,
    )

ax.set_xlabel(
    f"Median reference smoothing scale "
    f"[{POSITION_UNIT_LABEL}]"
)
ax.set_ylabel(
    "Test normalized RMSE"
)
ax.legend()
fig.tight_layout()
save_figure(
    fig,
    "nrmse_vs_smoothing_scale",
)

# 方向誤差 vs physical smoothing scale
fig, ax = plt.subplots(
    figsize=(6.8, 4.8)
)

for model in plot_models:
    selected = (
        plot_summary[
            plot_summary["model"]
            == model
        ]
        .set_index("scale")
        .loc[scale_order]
        .reset_index()
    )

    ax.errorbar(
        selected[
            "mean_reference_bandwidth"
        ],
        selected[
            "median_direction_error_deg_mean"
        ],
        yerr=selected[
            "median_direction_error_deg_std"
        ],
        marker="o",
        capsize=4,
        label=model,
    )

ax.set_xlabel(
    f"Median reference smoothing scale "
    f"[{POSITION_UNIT_LABEL}]"
)
ax.set_ylabel(
    "Median direction error [deg]"
)
ax.legend()
fig.tight_layout()
save_figure(
    fig,
    "direction_error_vs_smoothing_scale",
)

# corrected diffusionの較正傾き
diffusion_summary = (
    summary_df[
        summary_df["model"]
        == "corrected Cartesian diffusion spectral"
    ]
    .set_index("scale")
    .loc[scale_order]
    .reset_index()
)

fig, ax = plt.subplots(
    figsize=(6.8, 4.8)
)

for component in [
    "x",
    "y",
    "z",
]:
    ax.errorbar(
        diffusion_summary[
            "mean_reference_bandwidth"
        ],
        diffusion_summary[
            f"calibration_slope_{component}_mean"
        ],
        yerr=diffusion_summary[
            f"calibration_slope_{component}_std"
        ],
        marker="o",
        capsize=4,
        label=f"$v_{component}$",
    )

ax.axhline(
    1.0,
    linewidth=1.2,
)
ax.set_xlabel(
    f"Median reference smoothing scale "
    f"[{POSITION_UNIT_LABEL}]"
)
ax.set_ylabel(
    "Calibration slope"
)
ax.legend()
fig.tight_layout()
save_figure(
    fig,
    "calibration_vs_smoothing_scale",
)

# fiducial scaleでのseed散布
fig, ax = plt.subplots(
    figsize=(7.0, 4.8)
)

for model in plot_models:
    selected = fiducial_df[
        fiducial_df["model"]
        == model
    ].sort_values(
        "replicate"
    )

    ax.plot(
        selected["replicate"],
        selected[
            "normalized_rmse"
        ],
        marker="o",
        label=model,
    )

ax.set_xlabel(
    "Seed replicate"
)
ax.set_ylabel(
    "Fiducial-scale test NRMSE"
)
ax.set_xticks(
    sorted(
        fiducial_df[
            "replicate"
        ].unique()
    )
)
ax.legend()
fig.tight_layout()
save_figure(
    fig,
    "fiducial_seed_nrmse",
)

# ============================================================
# 11. 保存
# ============================================================

paths = {
    "runs_csv": (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_runs.csv"
    ),
    "summary_csv": (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_summary.csv"
    ),
    "paired_csv": (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_paired_differences.csv"
    ),
    "paired_summary_csv": (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_paired_summary.csv"
    ),
    "diffusion_validation_csv": (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_diffusion_validation.csv"
    ),
    "kernel_validation_csv": (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_kernel_validation.csv"
    ),
    "selected_hyperparameters_csv": (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_selected_hyperparameters.csv"
    ),
    "summary_json": (
        OUTPUT_DIR
        / f"{OUTPUT_PREFIX}_summary.json"
    ),
}

runs_df.to_csv(
    paths["runs_csv"],
    index=False,
)
summary_df.to_csv(
    paths["summary_csv"],
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
diffusion_validation_df.to_csv(
    paths[
        "diffusion_validation_csv"
    ],
    index=False,
)
kernel_validation_df.to_csv(
    paths[
        "kernel_validation_csv"
    ],
    index=False,
)
selected_diffusion_df.to_csv(
    paths[
        "selected_hyperparameters_csv"
    ],
    index=False,
)

json_summary = {
    "seed_pairs": [
        {
            "reference_seed": int(
                reference_seed
            ),
            "label_seed": int(
                label_seed
            ),
        }
        for (
            reference_seed,
            label_seed,
        ) in SEED_PAIRS
    ],
    "smoothing_configs": (
        SMOOTHING_CONFIGS
    ),
    "summary_by_scale_and_model": (
        summary_df.to_dict(
            orient="records"
        )
    ),
    "paired_diffusion_minus_kernel": (
        paired_summary_df.to_dict(
            orient="records"
        )
    ),
    "selected_diffusion_hyperparameters": (
        selected_diffusion_df.to_dict(
            orient="records"
        )
    ),
}

with paths["summary_json"].open(
    "w",
    encoding="utf-8",
) as handle:
    json.dump(
        json_summary,
        handle,
        ensure_ascii=False,
        indent=2,
    )

print("=" * 108)
print(
    "Seed and smoothing-scale robustness "
    "validation completed."
)
for name, path in paths.items():
    print(
        f"{name:35s}: {path}"
    )
print("=" * 108)
