# Mock-1: potential-flow constrained radial-velocity reconstruction — seed/scale robustness
#
# この内容全体をJupyter Notebookの新しい空のコードセルへ貼り付けて実行する。
#
# 目的:
#   1. diffusion graphからmetric-calibrated gradient operatorを構成する。
#   2. Cartesian座標およびaffine potentialの勾配を再現できるか確認する。
#   3. diffusion-potential basis内の場についてradial in-span closureを行う。
#   4. 5組のreference/label seedとfine/fiducial/coarseの3スケール、計15 runを検証する。
#   5. radial validationだけによるoperational selectorと、controlled mock上のhidden-3D診断を分離する。
#   6. 既存のunconstrained Cartesian radial-only modelおよび3D oracleとpaired比較する。
#   7. seed/scale別のhyperparameter安定性、NRMSE、方向誤差、rank、condition numberを集約する。
#
# 重要:
#   - potentialは3個のaffine modeと、constant modeを除くdiffusion固有関数で展開する。
#   - grad phiはGamma(phi,x)を局所coordinate metric Gamma(x,x)で校正して構成する。
#   - 局所metricを速度成分へ直接挿入するのではなく、covectorからCartesian gradientへの変換にのみ用いる。
#   - hyperparameter選択にはradial validation NRMSEだけを使用する。
#   - hidden 3D truthはモデル選択後の評価、および明示的にdiagnosticと記した比較にのみ使用する。

from __future__ import annotations

import hashlib
import json
import math
import os
import platform
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy
from IPython.display import display
from scipy import linalg, sparse
from scipy.spatial import cKDTree

# ============================================================
# 1. 固定入出力パス
# ============================================================

MOCK_DATA_ROOT = Path(
    os.environ.get(
        "COSMIC_DIPOLE_MOCK_DATA_ROOT",
        'mock_data',
    )
)
INPUT_DIR = Path(
    os.environ.get(
        "MOCK1_INPUT_DIR",
        str(MOCK_DATA_ROOT / "mock1_complete_sphere"),
    )
)
DATA_FILE = INPUT_DIR / "mock1.npz"
GEOMETRY_CACHE_FILE = MOCK_DATA_ROOT / "mock1_diffusion_geometry_bulk_cache.npz"
REFERENCE_CACHE_DIR = MOCK_DATA_ROOT / "mock1_bulk_robustness_cache"
GRADIENT_CACHE_FILE = MOCK_DATA_ROOT / "mock1_potential_flow_gradient_basis_cache_v4.npz"
PREVIOUS_UNCONSTRAINED_DIR = (
    MOCK_DATA_ROOT / "mock1_radial_only_unconstrained_full_robustness_v2"
)
PREVIOUS_UNCONSTRAINED_PREFIX = (
    "mock1_radial_only_unconstrained_full_robustness"
)
OUTPUT_DIR = MOCK_DATA_ROOT / "mock1_potential_flow_full_robustness_v5"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = "mock1_potential_flow_full_robustness"

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)
VELOCITY_UNIT_LABEL = "simulation velocity unit"
POSITION_UNIT_LABEL = r"$h^{-1}\,\mathrm{Mpc}$"

# ============================================================
# 2. 実験設定
# ============================================================

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

# Geometry cacheを作成した既存Mock-1と同じgraph設定。
GRAPH_NEIGHBORS = 48
GRAPH_BANDWIDTH_NEIGHBOR = 16
GRAPH_BANDWIDTH_MULTIPLIER = 1.0
MAX_DIFFUSION_MODES = 512

# Potentialは3個の厳密なaffine mode（Phi=x,y,z）と、
# constant modeを除いたdiffusion eigenfunctionの和で表す。
# これにより一定bulk flowをexactに含め、球境界の固有関数展開だけに依存させない。
N_AFFINE_POTENTIAL_MODES = 3
POTENTIAL_DIFFUSION_MODE_COUNTS = [0, 3, 8, 16, 32, 64, 128, 256, 384, 511]
POTENTIAL_REGULARIZATION_STRENGTHS = [
    0.0,
    1.0e-12,
    1.0e-10,
    1.0e-8,
    1.0e-6,
    1.0e-4,
    1.0e-2,
    1.0,
    1.0e2,
]
# p=1: potential Dirichlet penalty、p=2: 高周波をより強く抑えるspectral penalty。
POTENTIAL_REGULARIZATION_POWERS = [1.0, 2.0]

# In-span closureはpure bulk、低・中・最大diffusion mode数で確認する。
CLOSURE_DIFFUSION_MODE_COUNTS = [0, 32, 128, 256, 511]
CLOSURE_PASS_TOLERANCE = 1.0e-8

# Metric inversionは線型potentialの再現性を保つため、必要最小限の固有値floorだけを用いる。
METRIC_EIGENVALUE_RELATIVE_FLOOR = 1.0e-10
METRIC_EIGENVALUE_ABSOLUTE_FLOOR = 1.0e-14
GRADIENT_MODE_CHUNK_SIZE = 32
USE_GRADIENT_CACHE = True
USE_REFERENCE_FIELD_CACHE = True

# Operational solverの数値ridge。Closure testではlambda=0を明示的に解く。
NUMERICAL_RIDGE = 1.0e-12

SHOW_PLOTS = True
SAVE_PDF = True
SAVE_PNG = True

# 開発用smoke test。通常は環境変数を設定しない。
SMOKE_TEST = os.environ.get("MOCK1_POTENTIAL_ROBUSTNESS_SMOKE_TEST", "0") == "1"
if SMOKE_TEST:
    SEED_PAIRS = SEED_PAIRS[:2]
    SMOOTHING_CONFIGS = [
        {
            "scale": "fiducial",
            "n_neighbors": 16,
            "bandwidth_neighbor": 6,
            "bandwidth_multiplier": 1.0,
        }
    ]
    GRAPH_NEIGHBORS = 16
    GRAPH_BANDWIDTH_NEIGHBOR = 6
    MAX_DIFFUSION_MODES = 24
    POTENTIAL_DIFFUSION_MODE_COUNTS = [0, 3, 8, 16, 23]
    POTENTIAL_REGULARIZATION_STRENGTHS = [0.0, 1.0e-8, 1.0e-4, 1.0e-2, 1.0]
    POTENTIAL_REGULARIZATION_POWERS = [1.0, 2.0]
    CLOSURE_DIFFUSION_MODE_COUNTS = [0, 3, 8, 16, 23]
    GRADIENT_MODE_CHUNK_SIZE = 8
    SHOW_PLOTS = False
    SAVE_PDF = False
    SAVE_PNG = False

# ============================================================
# 3. 汎用関数
# ============================================================


def file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def stable_signature(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_ready(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return [json_ready(item) for item in value.tolist()]
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return number if np.isfinite(number) else None
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, dict):
        return {str(key): json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_ready(item) for item in value]
    return value


def finite_quantile(values, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else math.nan


def scalar_nrmse(true, predicted) -> float:
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    denominator = float(np.sqrt(np.mean(np.square(true))))
    numerator = float(np.sqrt(np.mean(np.square(predicted - true))))
    return numerator / denominator if denominator > 0.0 else math.nan


def direction_errors(true, predicted):
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    true_speed = np.linalg.norm(true, axis=1)
    predicted_speed = np.linalg.norm(predicted, axis=1)
    scale = max(float(np.sqrt(np.mean(np.square(true_speed)))), 1.0)
    floor = np.finfo(np.float64).eps * scale
    valid = (true_speed > floor) & (predicted_speed > floor)
    angle = np.full(true.shape[0], np.nan, dtype=np.float64)
    if np.any(valid):
        cosine = np.sum(true[valid] * predicted[valid], axis=1)
        cosine /= true_speed[valid] * predicted_speed[valid]
        angle[valid] = np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))
    return angle


def vector_field_metrics(true, predicted, line_of_sight):
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    line_of_sight = np.asarray(line_of_sight, dtype=np.float64)

    residual = predicted - true
    true_radial = np.einsum("ij,ij->i", line_of_sight, true)
    predicted_radial = np.einsum("ij,ij->i", line_of_sight, predicted)
    residual_radial = predicted_radial - true_radial

    true_tangential = true - true_radial[:, None] * line_of_sight
    predicted_tangential = predicted - predicted_radial[:, None] * line_of_sight
    residual_tangential = predicted_tangential - true_tangential

    rmse_radial = float(np.sqrt(np.mean(np.square(residual_radial))))
    reference_rms_radial = float(np.sqrt(np.mean(np.square(true_radial))))
    rmse_tangential = float(
        np.sqrt(np.mean(np.sum(np.square(residual_tangential), axis=1)))
    )
    reference_rms_tangential = float(
        np.sqrt(np.mean(np.sum(np.square(true_tangential), axis=1)))
    )
    rmse_3d = float(np.sqrt(np.mean(np.sum(np.square(residual), axis=1))))
    reference_rms_3d = float(np.sqrt(np.mean(np.sum(np.square(true), axis=1))))
    predicted_rms_3d = float(
        np.sqrt(np.mean(np.sum(np.square(predicted), axis=1)))
    )
    predicted_rms_tangential = float(
        np.sqrt(np.mean(np.sum(np.square(predicted_tangential), axis=1)))
    )

    angle = direction_errors(true, predicted)
    radial_correlation = (
        float(np.corrcoef(true_radial, predicted_radial)[0, 1])
        if np.std(true_radial) > 0.0 and np.std(predicted_radial) > 0.0
        else math.nan
    )

    return {
        "rmse_radial": rmse_radial,
        "reference_rms_radial": reference_rms_radial,
        "nrmse_radial": (
            rmse_radial / reference_rms_radial
            if reference_rms_radial > 0.0
            else math.nan
        ),
        "rmse_tangential": rmse_tangential,
        "reference_rms_tangential": reference_rms_tangential,
        "nrmse_tangential": (
            rmse_tangential / reference_rms_tangential
            if reference_rms_tangential > 0.0
            else math.nan
        ),
        "rmse_3d": rmse_3d,
        "reference_rms_3d": reference_rms_3d,
        "nrmse_3d": (
            rmse_3d / reference_rms_3d if reference_rms_3d > 0.0 else math.nan
        ),
        "predicted_rms_3d": predicted_rms_3d,
        "predicted_to_reference_rms_3d": (
            predicted_rms_3d / reference_rms_3d
            if reference_rms_3d > 0.0
            else math.nan
        ),
        "predicted_rms_tangential": predicted_rms_tangential,
        "predicted_to_reference_rms_tangential": (
            predicted_rms_tangential / reference_rms_tangential
            if reference_rms_tangential > 0.0
            else math.nan
        ),
        "direction_error_median_deg": finite_quantile(angle, 0.50),
        "direction_error_p90_deg": finite_quantile(angle, 0.90),
        "radial_correlation": radial_correlation,
        "radial_mean_bias": float(np.mean(residual_radial)),
    }


def component_calibration(true, predicted):
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    records = []
    for component, label in enumerate(["x", "y", "z"]):
        x = true[:, component]
        y = predicted[:, component]
        if np.std(x) > 0.0:
            slope, intercept = np.polyfit(x, y, 1)
            correlation = (
                float(np.corrcoef(x, y)[0, 1]) if np.std(y) > 0.0 else math.nan
            )
        else:
            slope = intercept = correlation = math.nan
        records.append(
            {
                "component": label,
                "slope": float(slope),
                "intercept": float(intercept),
                "correlation": float(correlation),
            }
        )
    return records


def make_reference_model_split(n, reference_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_reference = int(round(reference_fraction * n))
    reference = np.sort(order[:n_reference])
    model = np.sort(order[n_reference:])
    return reference, model


def make_label_split(n, train_fraction, validation_fraction, seed):
    rng = np.random.default_rng(seed)
    order = rng.permutation(n)
    n_train = int(round(train_fraction * n))
    n_validation = int(round(validation_fraction * n))
    return {
        "train": np.sort(order[:n_train]),
        "validation": np.sort(order[n_train : n_train + n_validation]),
        "test": np.sort(order[n_train + n_validation :]),
    }


def lambda_label(value: float) -> str:
    if value == 0.0:
        return "0"
    exponent = int(round(math.log10(value)))
    if np.isclose(value, 10.0**exponent):
        return rf"$10^{{{exponent}}}$"
    return f"{value:.3g}"


def save_figure(fig, stem: str):
    if SAVE_PDF:
        fig.savefig(OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.pdf", bbox_inches="tight")
    if SAVE_PNG:
        fig.savefig(
            OUTPUT_DIR / f"{OUTPUT_PREFIX}_{stem}.png",
            dpi=180,
            bbox_inches="tight",
        )
    if SHOW_PLOTS:
        plt.show()
    else:
        plt.close(fig)


# ============================================================
# 4. 独立reference bulk field
# ============================================================


def query_neighbors(query_positions, reference_positions, max_neighbors):
    if reference_positions.shape[0] == 0:
        raise ValueError("参照点が空である。")
    effective_k = min(int(max_neighbors), int(reference_positions.shape[0]))
    tree = cKDTree(reference_positions)
    try:
        distances, indices = tree.query(
            query_positions,
            k=effective_k,
            workers=-1,
        )
    except TypeError:
        distances, indices = tree.query(query_positions, k=effective_k)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if effective_k == 1:
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
    n_neighbors = min(int(n_neighbors), distances_max.shape[1])
    bandwidth_neighbor = min(int(bandwidth_neighbor), n_neighbors)
    distances = distances_max[:, :n_neighbors]
    indices = indices_max[:, :n_neighbors]
    bandwidth = float(bandwidth_multiplier) * distances[:, bandwidth_neighbor - 1]
    positive = bandwidth[bandwidth > 0.0]
    floor = (
        max(float(np.median(positive)) * 1.0e-8, np.finfo(float).eps)
        if positive.size
        else np.finfo(float).eps
    )
    bandwidth = np.maximum(bandwidth, floor)
    weights = np.exp(-np.square(distances / bandwidth[:, None]))
    weight_sum = np.sum(weights, axis=1)
    neighbor_velocities = reference_velocities[indices]
    bulk_velocity = np.einsum(
        "ij,ijk->ik", weights, neighbor_velocities, optimize=True
    )
    bulk_velocity /= weight_sum[:, None]
    residual = neighbor_velocities - bulk_velocity[:, None, :]
    dispersion_squared = np.einsum(
        "ij,ijk->i", weights, np.square(residual), optimize=True
    ) / weight_sum
    dispersion = np.sqrt(np.maximum(dispersion_squared, 0.0))
    effective_neighbors = np.square(weight_sum) / np.sum(
        np.square(weights), axis=1
    )
    return {
        "bulk_velocity": bulk_velocity,
        "bandwidth": bandwidth,
        "dispersion": dispersion,
        "effective_neighbors": effective_neighbors,
    }


def reference_cache_path(reference_seed, scale):
    return REFERENCE_CACHE_DIR / f"reference_seed_{reference_seed}_scale_{scale}.npz"


def reference_field_signature(reference_indices, model_indices, reference_seed, config):
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


def load_cached_reference_field(
    reference_indices,
    model_indices,
    reference_seed,
    config,
):
    cache_path = reference_cache_path(reference_seed, config["scale"])
    if not USE_REFERENCE_FIELD_CACHE or not cache_path.is_file():
        return None
    signature = reference_field_signature(
        reference_indices,
        model_indices,
        reference_seed,
        config,
    )
    try:
        with np.load(cache_path, allow_pickle=False) as data:
            if str(data["signature"].item()) != signature:
                return None
            return {
                "bulk_velocity": np.asarray(data["bulk_velocity"], dtype=np.float64),
                "bandwidth": np.asarray(data["bandwidth"], dtype=np.float64),
                "dispersion": np.asarray(data["dispersion"], dtype=np.float64),
                "effective_neighbors": np.asarray(
                    data["effective_neighbors"], dtype=np.float64
                ),
                "cache_path": str(cache_path),
                "loaded_from_cache": True,
            }
    except Exception as exc:
        warnings.warn(f"reference field cacheを読めないため再計算する: {exc}")
        return None


def prepare_reference_fields(
    positions,
    raw_velocity,
    reference_indices,
    model_indices,
    reference_seed,
    configs,
):
    fields = {}
    missing = []
    for config in configs:
        cached = load_cached_reference_field(
            reference_indices,
            model_indices,
            reference_seed,
            config,
        )
        if cached is None:
            missing.append(config)
        else:
            fields[config["scale"]] = cached
            print(f"[Reference cache] {config['scale']}: {cached['cache_path']}")

    if missing:
        max_neighbors = max(int(config["n_neighbors"]) for config in missing)
        distances_max, indices_max = query_neighbors(
            query_positions=positions[model_indices],
            reference_positions=positions[reference_indices],
            max_neighbors=max_neighbors,
        )
        for config in missing:
            result = bulk_field_from_neighbors(
                distances_max=distances_max,
                indices_max=indices_max,
                reference_velocities=raw_velocity[reference_indices],
                n_neighbors=config["n_neighbors"],
                bandwidth_neighbor=config["bandwidth_neighbor"],
                bandwidth_multiplier=config["bandwidth_multiplier"],
            )
            cache_path = reference_cache_path(reference_seed, config["scale"])
            signature = reference_field_signature(
                reference_indices,
                model_indices,
                reference_seed,
                config,
            )
            if USE_REFERENCE_FIELD_CACHE:
                np.savez_compressed(
                    cache_path,
                    signature=np.array(signature),
                    bulk_velocity=result["bulk_velocity"],
                    bandwidth=result["bandwidth"],
                    dispersion=result["dispersion"],
                    effective_neighbors=result["effective_neighbors"],
                )
            result["cache_path"] = str(cache_path)
            result["loaded_from_cache"] = False
            fields[config["scale"]] = result
            print(
                f"[Reference build] {config['scale']}: "
                f"median bandwidth={np.median(result['bandwidth']):.6g}"
            )
    return fields


# ============================================================
# 5. Sparse diffusion graphとmetric-calibrated gradient
# ============================================================


def rebuild_markov_from_cached_bandwidth(
    positions,
    cached_bandwidth,
    n_neighbors,
):
    n = positions.shape[0]
    if n_neighbors >= n:
        raise ValueError("graph neighborsは点数より小さくする必要がある。")
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

    row = np.repeat(np.arange(n, dtype=np.int64), n_neighbors)
    col = indices.reshape(-1)
    squared_distance = np.square(distances.reshape(-1))
    denominator = cached_bandwidth[row] * cached_bandwidth[col]
    weights = np.exp(-squared_distance / denominator)

    kernel = sparse.coo_matrix((weights, (row, col)), shape=(n, n)).tocsr()
    kernel = kernel.maximum(kernel.T)
    kernel.setdiag(1.0)
    kernel.eliminate_zeros()

    degree = np.asarray(kernel.sum(axis=1)).ravel()
    if np.any(degree <= 0.0):
        raise ValueError("再構成したkernelに孤立点がある。")
    markov = sparse.diags(1.0 / degree) @ kernel

    queried_bandwidth = distances[:, GRAPH_BANDWIDTH_NEIGHBOR - 1]
    queried_bandwidth *= GRAPH_BANDWIDTH_MULTIPLIER
    return markov.tocsr(), kernel, queried_bandwidth


def compute_centered_coordinate_metric(markov, positions, bandwidth):
    mean_position = markov @ positions
    metric = np.empty((positions.shape[0], 3, 3), dtype=np.float64)
    scale = 2.0 * np.square(bandwidth)
    for j in range(3):
        for c in range(j, 3):
            second = markov @ (positions[:, j] * positions[:, c])
            covariance = second - mean_position[:, j] * mean_position[:, c]
            value = covariance / scale
            metric[:, j, c] = value
            metric[:, c, j] = value
    return 0.5 * (metric + np.swapaxes(metric, 1, 2)), mean_position


def invert_local_metric(metric):
    symmetric_metric = 0.5 * (metric + np.swapaxes(metric, 1, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric_metric)
    trace_scale = np.maximum(np.trace(symmetric_metric, axis1=1, axis2=2) / 3.0, 0.0)
    floor = np.maximum(
        METRIC_EIGENVALUE_ABSOLUTE_FLOOR,
        METRIC_EIGENVALUE_RELATIVE_FLOOR * np.maximum(trace_scale, 1.0),
    )
    clipped = np.maximum(eigenvalues, floor[:, None])
    inverse_metric = np.einsum(
        "nij,nj,nkj->nik",
        eigenvectors,
        1.0 / clipped,
        eigenvectors,
        optimize=True,
    )
    condition = clipped[:, -1] / clipped[:, 0]
    clipped_count = np.sum(eigenvalues < floor[:, None], axis=1)
    return inverse_metric, eigenvalues, clipped, condition, clipped_count


def gradients_from_markov_covariance(
    markov,
    positions,
    functions,
    bandwidth,
    inverse_metric,
    mean_position=None,
    chunk_size=32,
):
    functions = np.asarray(functions, dtype=np.float64)
    if functions.ndim == 1:
        functions = functions[:, None]
    if mean_position is None:
        mean_position = markov @ positions
    n, n_functions = functions.shape
    gradients = np.empty((n, n_functions, 3), dtype=np.float64)
    denominator = 2.0 * np.square(bandwidth)

    for start in range(0, n_functions, int(chunk_size)):
        stop = min(start + int(chunk_size), n_functions)
        block = functions[:, start:stop]
        mean_block = markov @ block
        covector = np.empty((n, stop - start, 3), dtype=np.float64)
        for c in range(3):
            mean_product = markov @ (block * positions[:, c, None])
            covariance = mean_product - mean_block * mean_position[:, c, None]
            covector[:, :, c] = covariance / denominator[:, None]
        gradients[:, start:stop, :] = np.einsum(
            "nij,nmj->nmi",
            inverse_metric,
            covector,
            optimize=True,
        )
        print(f"[Gradient basis] modes {start + 1}:{stop}/{n_functions}")
    return gradients


def gradient_cache_signature(mode_indices):
    return stable_signature(
        {
            "data": file_signature(DATA_FILE),
            "geometry": file_signature(GEOMETRY_CACHE_FILE),
            "graph_neighbors": int(GRAPH_NEIGHBORS),
            "graph_bandwidth_neighbor": int(GRAPH_BANDWIDTH_NEIGHBOR),
            "graph_bandwidth_multiplier": float(GRAPH_BANDWIDTH_MULTIPLIER),
            "mode_indices": [int(value) for value in mode_indices],
            "metric_relative_floor": float(METRIC_EIGENVALUE_RELATIVE_FLOOR),
            "metric_absolute_floor": float(METRIC_EIGENVALUE_ABSOLUTE_FLOOR),
            "gradient_definition": "centered_Gamma_metric_inverse_v1",
        }
    )


def build_or_load_gradient_basis(
    positions,
    eigenfunctions,
    cached_bandwidth,
    cached_metric,
    mode_indices,
):
    signature = gradient_cache_signature(mode_indices)
    if USE_GRADIENT_CACHE and GRADIENT_CACHE_FILE.is_file():
        try:
            with np.load(GRADIENT_CACHE_FILE, allow_pickle=False) as data:
                if str(data["signature"].item()) == signature:
                    print(f"[Gradient cache] 再利用する: {GRADIENT_CACHE_FILE}")
                    diagnostics = json.loads(str(data["diagnostics_json"].item()))
                    return {
                        "gradient_basis": np.asarray(
                            data["gradient_basis"], dtype=np.float64
                        ),
                        "metric_condition": np.asarray(
                            data["metric_condition"], dtype=np.float64
                        ),
                        "metric_eigenvalues": np.asarray(
                            data["metric_eigenvalues"], dtype=np.float64
                        ),
                        "diagnostics": diagnostics,
                        "loaded_from_cache": True,
                    }
        except Exception as exc:
            warnings.warn(f"gradient cacheを読めないため再計算する: {exc}")

    print("[Gradient graph] sparse Markov operatorを再構成する。")
    markov, kernel, queried_bandwidth = rebuild_markov_from_cached_bandwidth(
        positions,
        cached_bandwidth,
        GRAPH_NEIGHBORS,
    )
    recomputed_metric, mean_position = compute_centered_coordinate_metric(
        markov,
        positions,
        cached_bandwidth,
    )
    metric_difference = recomputed_metric - cached_metric
    cached_metric_norm = max(float(np.linalg.norm(cached_metric)), np.finfo(float).eps)
    metric_relative_error = float(np.linalg.norm(metric_difference) / cached_metric_norm)
    bandwidth_relative_error = float(
        np.linalg.norm(queried_bandwidth - cached_bandwidth)
        / max(float(np.linalg.norm(cached_bandwidth)), np.finfo(float).eps)
    )

    inverse_metric, metric_eigenvalues, clipped_eigenvalues, condition, clipped_count = (
        invert_local_metric(recomputed_metric)
    )

    # Coordinate gradient calibration。
    coordinate_records = []
    identity = np.eye(3)
    for coordinate in range(3):
        covector = recomputed_metric[:, coordinate, :]
        naive_gradient = covector
        calibrated_gradient = np.einsum(
            "nij,nj->ni", inverse_metric, covector, optimize=True
        )
        target = np.repeat(identity[coordinate][None, :], positions.shape[0], axis=0)
        naive_error = naive_gradient - target
        calibrated_error = calibrated_gradient - target
        coordinate_records.append(
            {
                "coordinate": "xyz"[coordinate],
                "naive_rms_error": float(
                    np.sqrt(np.mean(np.sum(np.square(naive_error), axis=1)))
                ),
                "naive_max_error": float(
                    np.max(np.linalg.norm(naive_error, axis=1))
                ),
                "calibrated_rms_error": float(
                    np.sqrt(np.mean(np.sum(np.square(calibrated_error), axis=1)))
                ),
                "calibrated_max_error": float(
                    np.max(np.linalg.norm(calibrated_error, axis=1))
                ),
            }
        )

    # Independent affine test。
    affine_coefficient = np.array([0.37, -0.52, 0.81], dtype=np.float64)
    affine_function = positions @ affine_coefficient + 1.7
    affine_gradient = gradients_from_markov_covariance(
        markov,
        positions,
        affine_function,
        cached_bandwidth,
        inverse_metric,
        mean_position=mean_position,
        chunk_size=1,
    )[:, 0, :]
    affine_target = np.repeat(
        affine_coefficient[None, :], positions.shape[0], axis=0
    )
    affine_error = affine_gradient - affine_target

    # Quadratic potentialは有限近傍biasを測るsecondary diagnostic。
    quadratic_function = 0.5 * np.sum(np.square(positions), axis=1)
    quadratic_gradient = gradients_from_markov_covariance(
        markov,
        positions,
        quadratic_function,
        cached_bandwidth,
        inverse_metric,
        mean_position=mean_position,
        chunk_size=1,
    )[:, 0, :]
    quadratic_nrmse = float(
        np.sqrt(np.mean(np.sum(np.square(quadratic_gradient - positions), axis=1)))
        / max(
            float(np.sqrt(np.mean(np.sum(np.square(positions), axis=1)))),
            np.finfo(float).eps,
        )
    )

    print("[Gradient basis] diffusion eigenfunctionsの勾配を構成する。")
    gradient_basis = gradients_from_markov_covariance(
        markov,
        positions,
        eigenfunctions[:, mode_indices],
        cached_bandwidth,
        inverse_metric,
        mean_position=mean_position,
        chunk_size=GRADIENT_MODE_CHUNK_SIZE,
    )

    diagnostics = {
        "kernel_nnz": int(kernel.nnz),
        "mean_graph_degree": float(kernel.nnz / positions.shape[0]),
        "bandwidth_relative_error": bandwidth_relative_error,
        "metric_relative_error_vs_cache": metric_relative_error,
        "metric_condition_median": float(np.median(condition)),
        "metric_condition_p90": float(np.quantile(condition, 0.90)),
        "metric_condition_p99": float(np.quantile(condition, 0.99)),
        "metric_condition_max": float(np.max(condition)),
        "metric_clipped_point_fraction": float(np.mean(clipped_count > 0)),
        "metric_clipped_eigenvalue_fraction": float(
            np.sum(clipped_count) / clipped_eigenvalues.size
        ),
        "coordinate_records": coordinate_records,
        "affine_rms_error": float(
            np.sqrt(np.mean(np.sum(np.square(affine_error), axis=1)))
        ),
        "affine_max_error": float(np.max(np.linalg.norm(affine_error, axis=1))),
        "quadratic_gradient_nrmse": quadratic_nrmse,
    }

    if USE_GRADIENT_CACHE:
        # 非圧縮npzとし、大きなgradient arrayの保存時間を抑える。
        np.savez(
            GRADIENT_CACHE_FILE,
            signature=np.array(signature),
            mode_indices=np.asarray(mode_indices, dtype=np.int64),
            gradient_basis=gradient_basis,
            metric_condition=condition,
            metric_eigenvalues=metric_eigenvalues,
            diagnostics_json=np.array(json.dumps(json_ready(diagnostics))),
        )
        print(f"[Gradient cache] 保存した: {GRADIENT_CACHE_FILE}")

    return {
        "gradient_basis": gradient_basis,
        "metric_condition": condition,
        "metric_eigenvalues": metric_eigenvalues,
        "diagnostics": diagnostics,
        "loaded_from_cache": False,
    }


# ============================================================
# 6. Potential-flow solverと線型代数診断
# ============================================================


def regularization_diagonal(generator_values, n_basis, power):
    raw = np.asarray(generator_values[:n_basis], dtype=np.float64)
    values = np.zeros_like(raw)
    positive = raw > 0.0
    values[positive] = np.power(
        np.maximum(raw[positive], NUMERICAL_RIDGE),
        float(power),
    )
    # affine potential modesは物理的なbulk-flow成分なのでpenaltyを課さない。
    return values


def factor_regularized_system(
    gram,
    regularization,
    strength,
    numerical_ridge=NUMERICAL_RIDGE,
):
    system = np.asarray(gram, dtype=np.float64).copy()
    diagonal = np.diag_indices_from(system)
    system[diagonal] += (
        float(strength) * np.asarray(regularization, dtype=np.float64)
        + float(numerical_ridge)
    )
    try:
        factor = linalg.cho_factor(system, lower=True, check_finite=False)
        return "cholesky", factor, system
    except linalg.LinAlgError:
        return "direct", system, system


def solve_factored(factor_kind, factor, rhs):
    if factor_kind == "cholesky":
        return linalg.cho_solve(factor, rhs, check_finite=False)
    return linalg.solve(
        factor,
        rhs,
        assume_a="sym",
        check_finite=False,
    )


def potential_velocity(gradient_basis, coefficients, n_basis):
    return np.einsum(
        "imc,m->ic",
        gradient_basis[:, :n_basis, :],
        coefficients[:n_basis],
        optimize=True,
    )


def potential_radial_design(gradient_basis, line_of_sight):
    return np.einsum(
        "ic,imc->im",
        line_of_sight,
        gradient_basis,
        optimize=True,
    )


def vector_design_matrix(gradient_basis):
    # Row orderingは(point, Cartesian component)。
    return np.transpose(gradient_basis, (0, 2, 1)).reshape(
        gradient_basis.shape[0] * 3,
        gradient_basis.shape[1],
    )


def design_spectrum(design):
    design = np.asarray(design, dtype=np.float64)
    singular = linalg.svdvals(design)
    if singular.size == 0 or singular[0] <= 0.0:
        return {
            "singular_values": singular,
            "numerical_rank": 0,
            "nullity": int(design.shape[1]),
            "condition_number": math.inf,
            "stable_rank": 0.0,
            "entropy_effective_rank": 0.0,
        }
    tolerance = (
        max(design.shape) * np.finfo(np.float64).eps * singular[0]
    )
    resolved = singular > tolerance
    rank = int(np.sum(resolved))
    smallest = float(singular[resolved][-1]) if rank else math.nan
    squared = np.square(singular)
    probability = squared / np.sum(squared)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1.0e-300)))
    return {
        "singular_values": singular,
        "numerical_rank": rank,
        "nullity": int(design.shape[1] - rank),
        "condition_number": (
            float(singular[0] / smallest)
            if rank and smallest > 0.0
            else math.inf
        ),
        "stable_rank": float(np.sum(squared) / squared[0]),
        "entropy_effective_rank": float(np.exp(entropy)),
    }


def gram_spectrum(gram, n_observations):
    gram = 0.5 * (
        np.asarray(gram, dtype=np.float64)
        + np.asarray(gram, dtype=np.float64).T
    )
    eigenvalues = np.maximum(
        linalg.eigvalsh(gram, check_finite=False),
        0.0,
    )
    singular = np.sqrt(float(n_observations) * eigenvalues[::-1])
    if singular.size == 0 or singular[0] <= 0.0:
        return {
            "singular_values": singular,
            "numerical_rank": 0,
            "nullity": int(gram.shape[0]),
            "condition_number": math.inf,
            "stable_rank": 0.0,
            "entropy_effective_rank": 0.0,
        }
    tolerance = (
        max(int(n_observations), int(gram.shape[0]))
        * np.finfo(np.float64).eps
        * singular[0]
    )
    resolved = singular > tolerance
    rank = int(np.sum(resolved))
    smallest = float(singular[resolved][-1]) if rank else math.nan
    squared = np.square(singular)
    probability = squared / np.sum(squared)
    entropy = -np.sum(
        probability * np.log(np.maximum(probability, 1.0e-300))
    )
    return {
        "singular_values": singular,
        "numerical_rank": rank,
        "nullity": int(gram.shape[0] - rank),
        "condition_number": (
            float(singular[0] / smallest)
            if rank and smallest > 0.0
            else math.inf
        ),
        "stable_rank": float(np.sum(squared) / squared[0]),
        "entropy_effective_rank": float(np.exp(entropy)),
    }


def exact_effective_degrees_of_freedom(gram, regularization, strength):
    system = np.asarray(gram, dtype=np.float64).copy()
    diagonal = np.diag_indices_from(system)
    system[diagonal] += (
        float(strength) * np.asarray(regularization, dtype=np.float64)
        + NUMERICAL_RIDGE
    )
    try:
        solved = linalg.solve(
            system,
            gram,
            assume_a="sym",
            check_finite=False,
        )
        return float(np.trace(solved))
    except linalg.LinAlgError:
        return math.nan


def select_record(dataframe, score_column):
    ordered = dataframe.sort_values(
        [score_column, "n_basis", "regularization_strength", "penalty_power"],
        kind="mergesort",
    )
    return ordered.iloc[0].to_dict()


# ============================================================
# 7. 既存unconstrained結果の読込みとfallback
# ============================================================


def previous_prediction_path(reference_seed, label_seed, scale):
    run_id = f"ref{reference_seed}_label{label_seed}_{scale}"
    return PREVIOUS_UNCONSTRAINED_DIR / (
        f"{PREVIOUS_UNCONSTRAINED_PREFIX}_{run_id}_predictions.npz"
    )


def load_previous_predictions(
    reference_seed,
    label_seed,
    scale,
    expected_test_global,
):
    path = previous_prediction_path(reference_seed, label_seed, scale)
    if not path.is_file():
        return None
    try:
        with np.load(path, allow_pickle=False) as data:
            stored = np.asarray(data["test_global_indices"], dtype=np.int64)
            if not np.array_equal(stored, expected_test_global):
                return None
            return {
                "unconstrained": np.asarray(
                    data["radial_diffusion_velocity_test"], dtype=np.float64
                ),
                "cartesian_oracle": np.asarray(
                    data["oracle_optimal_velocity_test"], dtype=np.float64
                ),
                "source": str(path),
            }
    except Exception as exc:
        warnings.warn(f"既存unconstrained predictionを読めない: {exc}")
        return None


def radial_unconstrained_design(phi, line_of_sight):
    return np.hstack(
        [
            line_of_sight[:, 0, None] * phi,
            line_of_sight[:, 1, None] * phi,
            line_of_sight[:, 2, None] * phi,
        ]
    )


def unconstrained_theta_to_matrix(theta, n_basis):
    return np.column_stack(
        [
            theta[:n_basis],
            theta[n_basis : 2 * n_basis],
            theta[2 * n_basis : 3 * n_basis],
        ]
    )


def fallback_unconstrained_predictions(
    phi_fit,
    phi_test,
    los_fit,
    true_fit,
    generator_eigenvalues,
):
    n_basis = phi_fit.shape[1]
    design_fit = radial_unconstrained_design(phi_fit, los_fit)
    target_fit = np.einsum("ij,ij->i", los_fit, true_fit)
    gram = (design_fit.T @ design_fit) / design_fit.shape[0]
    rhs = (design_fit.T @ target_fit) / design_fit.shape[0]
    regularization = np.tile(
        np.maximum(generator_eigenvalues[:n_basis], NUMERICAL_RIDGE), 3
    )
    factor_kind, factor, _ = factor_regularized_system(
        gram,
        regularization,
        1.0e-4,
    )
    theta = solve_factored(factor_kind, factor, rhs)
    coefficient = unconstrained_theta_to_matrix(theta, n_basis)
    return phi_test @ coefficient


def fallback_cartesian_oracle_predictions(
    phi_fit,
    phi_test,
    true_fit,
    generator_eigenvalues,
):
    n_basis = phi_fit.shape[1]
    gram = (phi_fit.T @ phi_fit) / phi_fit.shape[0]
    rhs = (phi_fit.T @ true_fit) / phi_fit.shape[0]
    regularization = np.maximum(
        generator_eigenvalues[:n_basis], NUMERICAL_RIDGE
    )
    factor_kind, factor, _ = factor_regularized_system(
        gram,
        regularization,
        1.0e-2,
    )
    coefficient = solve_factored(factor_kind, factor, rhs)
    return phi_test @ coefficient


# ============================================================
# 8. 入力読込み
# ============================================================

import gc

required_files = [DATA_FILE, GEOMETRY_CACHE_FILE]
missing_files = [str(path) for path in required_files if not path.is_file()]
if missing_files:
    raise FileNotFoundError(
        "必要な入力ファイルが見つからない。\n" + "\n".join(missing_files)
    )

with np.load(DATA_FILE, allow_pickle=False) as data:
    if "pos" not in data.files or "vel" not in data.files:
        raise KeyError(f"mock1.npzにはposとvelが必要である。実際のkeys: {data.files}")
    pos_absolute = np.asarray(data["pos"], dtype=np.float64)
    vel_raw = np.asarray(data["vel"], dtype=np.float64)

with np.load(GEOMETRY_CACHE_FILE, allow_pickle=False) as data:
    required_geometry_keys = {
        "eigenfunctions",
        "generator_eigenvalues",
        "bandwidth",
        "local_metric",
    }
    missing_geometry_keys = required_geometry_keys.difference(data.files)
    if missing_geometry_keys:
        raise KeyError(
            "geometry cacheに必要なkeysがない: "
            + ", ".join(sorted(missing_geometry_keys))
        )
    eigenfunctions = np.asarray(data["eigenfunctions"], dtype=np.float64)
    generator_eigenvalues = np.asarray(
        data["generator_eigenvalues"], dtype=np.float64
    )
    geometry_bandwidth = np.asarray(data["bandwidth"], dtype=np.float64)
    cached_local_metric = np.asarray(data["local_metric"], dtype=np.float64)

if pos_absolute.ndim != 2 or pos_absolute.shape[1] != 3:
    raise ValueError(f"posのshapeが不正である: {pos_absolute.shape}")
if vel_raw.shape != pos_absolute.shape:
    raise ValueError(f"velのshape {vel_raw.shape}がpos {pos_absolute.shape}と一致しない。")
if eigenfunctions.shape[0] != pos_absolute.shape[0]:
    raise ValueError("eigenfunctionsの点数がmock1 catalogと一致しない。")
if geometry_bandwidth.shape != (pos_absolute.shape[0],):
    raise ValueError("geometry bandwidthのshapeが不正である。")
if cached_local_metric.shape != (pos_absolute.shape[0], 3, 3):
    raise ValueError("local_metricのshapeが不正である。")
if eigenfunctions.shape[1] < MAX_DIFFUSION_MODES:
    raise ValueError("geometry cacheの固有関数数が不足している。")
if generator_eigenvalues.size < MAX_DIFFUSION_MODES:
    raise ValueError("generator_eigenvaluesの個数が不足している。")

positions = pos_absolute - SPHERE_CENTER[None, :]
radius = np.linalg.norm(positions, axis=1)
line_of_sight_all = np.zeros_like(positions)
nonzero_radius = radius > 0.0
line_of_sight_all[nonzero_radius] = (
    positions[nonzero_radius] / radius[nonzero_radius, None]
)
if not np.all(nonzero_radius):
    warnings.warn(
        f"球中心と一致する点が{np.sum(~nonzero_radius)}個ある。"
        "その点のradial design rowは0として扱う。"
    )

# Constant diffusion modeを自動検出し、potential expansionから除外する。
mode_pool = np.arange(MAX_DIFFUSION_MODES, dtype=np.int64)
constant_mode_index = int(np.argmin(generator_eigenvalues[:MAX_DIFFUSION_MODES]))
active_mode_indices = mode_pool[mode_pool != constant_mode_index]
if constant_mode_index != 0:
    warnings.warn(
        f"最小generator eigenvalueのmode indexが0ではなく{constant_mode_index}である。"
    )
max_diffusion_potential_modes = active_mode_indices.size
POTENTIAL_DIFFUSION_MODE_COUNTS = sorted(
    {
        int(value)
        for value in POTENTIAL_DIFFUSION_MODE_COUNTS
        if 0 <= int(value) <= max_diffusion_potential_modes
    }
)
POTENTIAL_BASIS_SIZES = [
    N_AFFINE_POTENTIAL_MODES + value
    for value in POTENTIAL_DIFFUSION_MODE_COUNTS
]
max_potential_basis = N_AFFINE_POTENTIAL_MODES + max_diffusion_potential_modes
if not POTENTIAL_BASIS_SIZES:
    raise ValueError("有効なpotential basis sizeが残っていない。")

active_generator_values = generator_eigenvalues[active_mode_indices]
potential_generator_values = np.concatenate(
    [
        np.zeros(N_AFFINE_POTENTIAL_MODES, dtype=np.float64),
        active_generator_values,
    ]
)
scale_names = [config["scale"] for config in SMOOTHING_CONFIGS]
scale_order = {scale: index for index, scale in enumerate(scale_names)}

print("=" * 120)
print("Mock-1 potential-flow constrained radial reconstruction: full robustness")
print("=" * 120)
print(f"Python                  : {platform.python_version()}")
print(f"NumPy / SciPy           : {np.__version__} / {scipy.__version__}")
print(f"Objects                 : {positions.shape[0]:,}")
print(f"Seed pairs              : {len(SEED_PAIRS)}")
print(f"Reference scales        : {', '.join(scale_names)}")
print(f"Maximum diffusion modes : {MAX_DIFFUSION_MODES}")
print(f"Constant mode index     : {constant_mode_index}")
print(
    "Potential modes         : "
    f"{N_AFFINE_POTENTIAL_MODES} affine + "
    f"{max_diffusion_potential_modes} diffusion = {max_potential_basis}"
)
print(f"Input                   : {DATA_FILE}")
print(f"Geometry cache          : {GEOMETRY_CACHE_FILE}")
print(f"Gradient cache          : {GRADIENT_CACHE_FILE}")
print(f"Unconstrained results   : {PREVIOUS_UNCONSTRAINED_DIR}")
print(f"Output                  : {OUTPUT_DIR}")

start_time = time.perf_counter()

# ============================================================
# 9. Gradient operatorを一度だけ構成する
# ============================================================

gradient_result = build_or_load_gradient_basis(
    positions=positions,
    eigenfunctions=eigenfunctions,
    cached_bandwidth=geometry_bandwidth,
    cached_metric=cached_local_metric,
    mode_indices=mode_pool,
)
gradient_all_modes = gradient_result["gradient_basis"]
if constant_mode_index == 0:
    gradient_active_all = gradient_all_modes[:, 1:MAX_DIFFUSION_MODES, :]
else:
    gradient_active_all = gradient_all_modes[:, active_mode_indices, :]

constant_gradient = gradient_all_modes[:, constant_mode_index, :]
constant_gradient_rms = float(
    np.sqrt(np.mean(np.sum(np.square(constant_gradient), axis=1)))
)

# Exact affine modesとのradial-design重複を、全点配置に基づくbasis変換で除く。
line_of_sight_valid = line_of_sight_all[nonzero_radius]
affine_radial_gram = line_of_sight_valid.T @ line_of_sight_valid
diffusion_radial_all = np.einsum(
    "ic,imc->im",
    line_of_sight_valid,
    gradient_active_all[nonzero_radius],
    optimize=True,
)
affine_radial_rhs = line_of_sight_valid.T @ diffusion_radial_all
affine_projection_vectors = linalg.solve(
    affine_radial_gram,
    affine_radial_rhs,
    assume_a="sym",
    check_finite=False,
).T
diffusion_radial_residual = (
    diffusion_radial_all
    - line_of_sight_valid @ affine_projection_vectors.T
)
affine_projection_relative_orthogonality = float(
    np.linalg.norm(line_of_sight_valid.T @ diffusion_radial_residual)
    / max(np.linalg.norm(affine_radial_rhs), np.finfo(float).eps)
)
affine_projection_vector_rms = float(
    np.sqrt(np.mean(np.sum(np.square(affine_projection_vectors), axis=1)))
)
del diffusion_radial_all, diffusion_radial_residual

# Gradient diagnostics table。
gradient_diagnostic_records = []
for record in gradient_result["diagnostics"]["coordinate_records"]:
    gradient_diagnostic_records.append(
        {"diagnostic": f"coordinate_{record['coordinate']}", **record}
    )
gradient_diagnostic_records.extend(
    [
        {
            "diagnostic": "graph_and_metric",
            "kernel_nnz": gradient_result["diagnostics"]["kernel_nnz"],
            "mean_graph_degree": gradient_result["diagnostics"]["mean_graph_degree"],
            "bandwidth_relative_error": gradient_result["diagnostics"][
                "bandwidth_relative_error"
            ],
            "metric_relative_error_vs_cache": gradient_result["diagnostics"][
                "metric_relative_error_vs_cache"
            ],
        },
        {
            "diagnostic": "metric_condition",
            "metric_condition_median": gradient_result["diagnostics"][
                "metric_condition_median"
            ],
            "metric_condition_p90": gradient_result["diagnostics"][
                "metric_condition_p90"
            ],
            "metric_condition_p99": gradient_result["diagnostics"][
                "metric_condition_p99"
            ],
            "metric_condition_max": gradient_result["diagnostics"][
                "metric_condition_max"
            ],
            "metric_clipped_point_fraction": gradient_result["diagnostics"][
                "metric_clipped_point_fraction"
            ],
        },
        {
            "diagnostic": "affine_and_quadratic",
            "affine_rms_error": gradient_result["diagnostics"]["affine_rms_error"],
            "affine_max_error": gradient_result["diagnostics"]["affine_max_error"],
            "quadratic_gradient_nrmse": gradient_result["diagnostics"][
                "quadratic_gradient_nrmse"
            ],
            "constant_mode_gradient_rms": constant_gradient_rms,
            "affine_projection_vector_rms": affine_projection_vector_rms,
            "affine_projection_relative_orthogonality": (
                affine_projection_relative_orthogonality
            ),
        },
    ]
)
gradient_diagnostics_df = pd.DataFrame(gradient_diagnostic_records)
print("-" * 120)
print("Gradient-operator diagnostics")
display(gradient_diagnostics_df)

# gradient_active_allはgradient_all_modesのviewなので、元arrayへの参照は保持される。
gradient_result["gradient_basis"] = None
del gradient_all_modes, constant_gradient

# ============================================================
# 10. 5 seed pairs × 3 scales
# ============================================================

radial_validation_records = []
potential_oracle_validation_records = []
selector_records = []
comparison_records = []
calibration_records = []
linear_algebra_records = []
singular_value_records = []
reference_records = []
closure_records = []
prediction_files = []

for replicate, (reference_seed, label_seed) in enumerate(SEED_PAIRS, start=1):
    print("-" * 120)
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
    train_local = split["train"]
    validation_local = split["validation"]
    test_local = split["test"]
    fit_local = np.sort(np.concatenate([train_local, validation_local]))
    train_global = model_indices[train_local]
    validation_global = model_indices[validation_local]
    test_global = model_indices[test_local]
    fit_global = model_indices[fit_local]

    reference_fields = prepare_reference_fields(
        positions,
        vel_raw,
        reference_indices,
        model_indices,
        reference_seed,
        SMOOTHING_CONFIGS,
    )

    # Replicate固有model sample上にpotential-gradient basisを構成する。
    diffusion_gradient_model = np.asarray(
        gradient_active_all[model_indices], dtype=np.float64
    ).copy()
    diffusion_gradient_model -= affine_projection_vectors[None, :, :]
    diffusion_gradient_column_rms = np.sqrt(
        np.mean(
            np.sum(np.square(diffusion_gradient_model), axis=2),
            axis=0,
        )
    )
    positive_gradient_rms = diffusion_gradient_column_rms[
        np.isfinite(diffusion_gradient_column_rms)
        & (diffusion_gradient_column_rms > 0.0)
    ]
    affine_mode_scale = (
        float(np.median(positive_gradient_rms))
        if positive_gradient_rms.size
        else 1.0
    )
    affine_gradient_model = np.broadcast_to(
        (affine_mode_scale * np.eye(3, dtype=np.float64))[None, :, :],
        (model_indices.size, N_AFFINE_POTENTIAL_MODES, 3),
    ).copy()
    gradient_model = np.concatenate(
        [affine_gradient_model, diffusion_gradient_model],
        axis=1,
    )
    line_of_sight_model = line_of_sight_all[model_indices]
    radial_design_model = potential_radial_design(
        gradient_model,
        line_of_sight_model,
    )

    gradient_train = gradient_model[train_local]
    gradient_validation = gradient_model[validation_local]
    gradient_test = gradient_model[test_local]
    radial_design_train = radial_design_model[train_local]
    radial_design_validation = radial_design_model[validation_local]
    radial_design_test = radial_design_model[test_local]
    line_of_sight_train = line_of_sight_model[train_local]
    line_of_sight_validation = line_of_sight_model[validation_local]
    line_of_sight_test = line_of_sight_model[test_local]
    line_of_sight_fit = line_of_sight_model[fit_local]

    # 各replicateで最大basisのin-span closureを再確認する。
    closure_n_basis = max_potential_basis
    closure_rng = np.random.default_rng(
        int(reference_seed) * 1000003 + int(label_seed)
    )
    closure_column_rms = np.sqrt(
        np.mean(
            np.sum(np.square(gradient_model[:, :closure_n_basis, :]), axis=2),
            axis=0,
        )
    )
    closure_coefficient_true = closure_rng.normal(size=closure_n_basis) / np.maximum(
        closure_column_rms,
        1.0e-12,
    )
    closure_velocity_model = potential_velocity(
        gradient_model,
        closure_coefficient_true,
        closure_n_basis,
    )
    fiducial_scale = "fiducial" if "fiducial" in scale_names else scale_names[0]
    fiducial_reference_rms = float(
        np.sqrt(
            np.mean(
                np.sum(
                    np.square(reference_fields[fiducial_scale]["bulk_velocity"]),
                    axis=1,
                )
            )
        )
    )
    closure_span_rms = float(
        np.sqrt(np.mean(np.sum(np.square(closure_velocity_model), axis=1)))
    )
    if closure_span_rms > 0.0:
        closure_coefficient_true *= fiducial_reference_rms / closure_span_rms
        closure_velocity_model = potential_velocity(
            gradient_model,
            closure_coefficient_true,
            closure_n_basis,
        )
    closure_radial_model = np.einsum(
        "ij,ij->i",
        line_of_sight_model,
        closure_velocity_model,
    )
    closure_design = radial_design_model[fit_local, :closure_n_basis]
    closure_target = closure_radial_model[fit_local]
    (
        closure_coefficient_estimated,
        _,
        closure_rank,
        closure_singular,
    ) = linalg.lstsq(
        closure_design,
        closure_target,
        cond=None,
        lapack_driver="gelsd",
        check_finite=False,
    )
    closure_prediction_test = potential_velocity(
        gradient_test,
        closure_coefficient_estimated,
        closure_n_basis,
    )
    closure_metrics = vector_field_metrics(
        closure_velocity_model[test_local],
        closure_prediction_test,
        line_of_sight_test,
    )
    closure_coefficient_error = float(
        np.linalg.norm(closure_coefficient_estimated - closure_coefficient_true)
        / max(np.linalg.norm(closure_coefficient_true), np.finfo(float).eps)
    )
    closure_full_rank = int(closure_rank) == int(closure_n_basis)
    closure_status = (
        "pass"
        if closure_full_rank
        and closure_coefficient_error < CLOSURE_PASS_TOLERANCE
        and closure_metrics["nrmse_3d"] < CLOSURE_PASS_TOLERANCE
        else (
            "rank_deficient_not_identifiable"
            if not closure_full_rank
            else "fail"
        )
    )
    closure_records.append(
        {
            "replicate": replicate,
            "reference_seed": reference_seed,
            "label_seed": label_seed,
            "n_basis": closure_n_basis,
            "n_observations": int(closure_design.shape[0]),
            "numerical_rank": int(closure_rank),
            "nullity": int(closure_n_basis - closure_rank),
            "condition_number": (
                float(closure_singular[0] / closure_singular[-1])
                if closure_singular.size and closure_singular[-1] > 0.0
                else math.inf
            ),
            "coefficient_relative_error": closure_coefficient_error,
            **closure_metrics,
            "closure_status": closure_status,
        }
    )

    velocity_train_by_scale = {
        scale: reference_fields[scale]["bulk_velocity"][train_local]
        for scale in scale_names
    }
    velocity_validation_by_scale = {
        scale: reference_fields[scale]["bulk_velocity"][validation_local]
        for scale in scale_names
    }
    velocity_test_by_scale = {
        scale: reference_fields[scale]["bulk_velocity"][test_local]
        for scale in scale_names
    }
    velocity_fit_by_scale = {
        scale: reference_fields[scale]["bulk_velocity"][fit_local]
        for scale in scale_names
    }
    radial_train_by_scale = {
        scale: np.einsum(
            "ij,ij->i",
            line_of_sight_train,
            velocity_train_by_scale[scale],
        )
        for scale in scale_names
    }
    radial_validation_by_scale = {
        scale: np.einsum(
            "ij,ij->i",
            line_of_sight_validation,
            velocity_validation_by_scale[scale],
        )
        for scale in scale_names
    }
    radial_test_by_scale = {
        scale: np.einsum(
            "ij,ij->i",
            line_of_sight_test,
            velocity_test_by_scale[scale],
        )
        for scale in scale_names
    }

    max_basis = max(POTENTIAL_BASIS_SIZES)
    A_train_max = radial_design_train[:, :max_basis]
    A_validation_max = radial_design_validation[:, :max_basis]
    G3_train_max = vector_design_matrix(gradient_train[:, :max_basis, :])
    G3_validation_max = vector_design_matrix(
        gradient_validation[:, :max_basis, :]
    )

    radial_train_matrix = np.column_stack(
        [radial_train_by_scale[scale] for scale in scale_names]
    )
    radial_validation_matrix = np.column_stack(
        [radial_validation_by_scale[scale] for scale in scale_names]
    )
    vector_train_matrix = np.column_stack(
        [velocity_train_by_scale[scale].reshape(-1) for scale in scale_names]
    )
    vector_validation_matrix = np.column_stack(
        [
            velocity_validation_by_scale[scale].reshape(-1)
            for scale in scale_names
        ]
    )

    radial_gram_train_sum = A_train_max.T @ A_train_max
    radial_gram_validation_sum = A_validation_max.T @ A_validation_max
    radial_gram_train_max = radial_gram_train_sum / train_local.size
    radial_gram_fit_max = (
        radial_gram_train_sum + radial_gram_validation_sum
    ) / fit_local.size
    radial_rhs_train_sum = A_train_max.T @ radial_train_matrix
    radial_rhs_validation_sum = A_validation_max.T @ radial_validation_matrix
    radial_rhs_train_max = radial_rhs_train_sum / train_local.size
    radial_rhs_fit_max = (
        radial_rhs_train_sum + radial_rhs_validation_sum
    ) / fit_local.size

    vector_gram_train_sum = G3_train_max.T @ G3_train_max
    vector_gram_validation_sum = G3_validation_max.T @ G3_validation_max
    vector_gram_train_max = vector_gram_train_sum / train_local.size
    vector_gram_fit_max = (
        vector_gram_train_sum + vector_gram_validation_sum
    ) / fit_local.size
    vector_rhs_train_sum = G3_train_max.T @ vector_train_matrix
    vector_rhs_validation_sum = G3_validation_max.T @ vector_validation_matrix
    vector_rhs_train_max = vector_rhs_train_sum / train_local.size
    vector_rhs_fit_max = (
        vector_rhs_train_sum + vector_rhs_validation_sum
    ) / fit_local.size

    del G3_train_max, G3_validation_max
    del vector_train_matrix, vector_validation_matrix

    replicate_radial_validation_records = []
    replicate_potential_oracle_records = []

    for n_basis in POTENTIAL_BASIS_SIZES:
        radial_gram = radial_gram_train_max[:n_basis, :n_basis]
        radial_rhs = radial_rhs_train_max[:n_basis]
        vector_gram = vector_gram_train_max[:n_basis, :n_basis]
        vector_rhs = vector_rhs_train_max[:n_basis]

        for penalty_power in POTENTIAL_REGULARIZATION_POWERS:
            regularization = regularization_diagonal(
                potential_generator_values,
                n_basis,
                penalty_power,
            )
            for strength in POTENTIAL_REGULARIZATION_STRENGTHS:
                radial_factor_kind, radial_factor, _ = factor_regularized_system(
                    radial_gram,
                    regularization,
                    strength,
                )
                radial_coefficients_all = solve_factored(
                    radial_factor_kind,
                    radial_factor,
                    radial_rhs,
                )

                vector_factor_kind, vector_factor, _ = factor_regularized_system(
                    vector_gram,
                    regularization,
                    strength,
                )
                vector_coefficients_all = solve_factored(
                    vector_factor_kind,
                    vector_factor,
                    vector_rhs,
                )

                for scale_index, scale in enumerate(scale_names):
                    radial_coefficients = radial_coefficients_all[:, scale_index]
                    radial_prediction_validation = (
                        A_validation_max[:, :n_basis] @ radial_coefficients
                    )
                    vector_prediction_validation = potential_velocity(
                        gradient_validation,
                        radial_coefficients,
                        n_basis,
                    )
                    vector_metrics_validation = vector_field_metrics(
                        velocity_validation_by_scale[scale],
                        vector_prediction_validation,
                        line_of_sight_validation,
                    )
                    record = {
                        "replicate": replicate,
                        "reference_seed": reference_seed,
                        "label_seed": label_seed,
                        "run_id": (
                            f"ref{reference_seed}_label{label_seed}_{scale}"
                        ),
                        "scale": scale,
                        "n_basis": int(n_basis),
                        "n_affine_modes": N_AFFINE_POTENTIAL_MODES,
                        "n_diffusion_modes": int(
                            n_basis - N_AFFINE_POTENTIAL_MODES
                        ),
                        "regularization_strength": float(strength),
                        "penalty_power": float(penalty_power),
                        "validation_nrmse_radial": scalar_nrmse(
                            radial_validation_by_scale[scale],
                            radial_prediction_validation,
                        ),
                        "validation_nrmse_tangential": (
                            vector_metrics_validation["nrmse_tangential"]
                        ),
                        "validation_nrmse_3d": vector_metrics_validation[
                            "nrmse_3d"
                        ],
                        "validation_direction_error_median_deg": (
                            vector_metrics_validation[
                                "direction_error_median_deg"
                            ]
                        ),
                    }
                    radial_validation_records.append(record)
                    replicate_radial_validation_records.append(record)

                    vector_coefficients = vector_coefficients_all[:, scale_index]
                    vector_oracle_prediction_validation = potential_velocity(
                        gradient_validation,
                        vector_coefficients,
                        n_basis,
                    )
                    vector_oracle_metrics = vector_field_metrics(
                        velocity_validation_by_scale[scale],
                        vector_oracle_prediction_validation,
                        line_of_sight_validation,
                    )
                    oracle_record = {
                        "replicate": replicate,
                        "reference_seed": reference_seed,
                        "label_seed": label_seed,
                        "run_id": (
                            f"ref{reference_seed}_label{label_seed}_{scale}"
                        ),
                        "scale": scale,
                        "n_basis": int(n_basis),
                        "n_affine_modes": N_AFFINE_POTENTIAL_MODES,
                        "n_diffusion_modes": int(
                            n_basis - N_AFFINE_POTENTIAL_MODES
                        ),
                        "regularization_strength": float(strength),
                        "penalty_power": float(penalty_power),
                        "validation_nrmse_radial": vector_oracle_metrics[
                            "nrmse_radial"
                        ],
                        "validation_nrmse_tangential": vector_oracle_metrics[
                            "nrmse_tangential"
                        ],
                        "validation_nrmse_3d": vector_oracle_metrics["nrmse_3d"],
                        "validation_direction_error_median_deg": (
                            vector_oracle_metrics[
                                "direction_error_median_deg"
                            ]
                        ),
                    }
                    potential_oracle_validation_records.append(oracle_record)
                    replicate_potential_oracle_records.append(oracle_record)

    replicate_radial_validation_df = pd.DataFrame(
        replicate_radial_validation_records
    )
    replicate_potential_oracle_df = pd.DataFrame(
        replicate_potential_oracle_records
    )

    for scale in scale_names:
        run_id = f"ref{reference_seed}_label{label_seed}_{scale}"
        scale_radial_grid = replicate_radial_validation_df[
            replicate_radial_validation_df["scale"] == scale
        ].copy()
        scale_oracle_grid = replicate_potential_oracle_df[
            replicate_potential_oracle_df["scale"] == scale
        ].copy()

        operational = select_record(
            scale_radial_grid,
            "validation_nrmse_radial",
        )
        hidden3d = select_record(
            scale_radial_grid,
            "validation_nrmse_3d",
        )
        potential_oracle = select_record(
            scale_oracle_grid,
            "validation_nrmse_3d",
        )

        selected_specs = [
            (
                "potential_radial_validation_selected",
                operational,
                "radial",
            ),
            (
                "potential_hidden_3d_validation_selected_diagnostic",
                hidden3d,
                "radial",
            ),
            (
                "potential_3d_oracle_validation_selected_diagnostic",
                potential_oracle,
                "vector",
            ),
        ]

        test_predictions = {}
        selected_coefficients = {}
        for selector_name, specification, fit_type in selected_specs:
            n_basis = int(specification["n_basis"])
            strength = float(specification["regularization_strength"])
            penalty_power = float(specification["penalty_power"])
            regularization = regularization_diagonal(
                potential_generator_values,
                n_basis,
                penalty_power,
            )
            if fit_type == "radial":
                gram = radial_gram_fit_max[:n_basis, :n_basis]
                rhs = radial_rhs_fit_max[:n_basis, scale_order[scale]]
            else:
                gram = vector_gram_fit_max[:n_basis, :n_basis]
                rhs = vector_rhs_fit_max[:n_basis, scale_order[scale]]
            factor_kind, factor, _ = factor_regularized_system(
                gram,
                regularization,
                strength,
            )
            coefficients = solve_factored(factor_kind, factor, rhs)
            predicted_test = potential_velocity(
                gradient_test,
                coefficients,
                n_basis,
            )
            metrics = vector_field_metrics(
                velocity_test_by_scale[scale],
                predicted_test,
                line_of_sight_test,
            )
            selector_records.append(
                {
                    "replicate": replicate,
                    "reference_seed": reference_seed,
                    "label_seed": label_seed,
                    "run_id": run_id,
                    "scale": scale,
                    "selector": selector_name,
                    "fit_type": fit_type,
                    "n_basis": n_basis,
                    "n_affine_modes": N_AFFINE_POTENTIAL_MODES,
                    "n_diffusion_modes": int(
                        n_basis - N_AFFINE_POTENTIAL_MODES
                    ),
                    "regularization_strength": strength,
                    "penalty_power": penalty_power,
                    "validation_nrmse_radial": float(
                        specification["validation_nrmse_radial"]
                    ),
                    "validation_nrmse_3d": float(
                        specification["validation_nrmse_3d"]
                    ),
                    **{f"test_{key}": value for key, value in metrics.items()},
                }
            )
            comparison_records.append(
                {
                    "replicate": replicate,
                    "reference_seed": reference_seed,
                    "label_seed": label_seed,
                    "run_id": run_id,
                    "scale": scale,
                    "model": selector_name,
                    "model_class": "potential_flow",
                    "n_basis": n_basis,
                    "n_affine_modes": N_AFFINE_POTENTIAL_MODES,
                    "n_diffusion_modes": int(
                        n_basis - N_AFFINE_POTENTIAL_MODES
                    ),
                    "regularization_strength": strength,
                    "penalty_power": penalty_power,
                    **metrics,
                }
            )
            test_predictions[selector_name] = predicted_test
            selected_coefficients[selector_name] = coefficients
            for calibration in component_calibration(
                velocity_test_by_scale[scale],
                predicted_test,
            ):
                calibration_records.append(
                    {
                        "replicate": replicate,
                        "reference_seed": reference_seed,
                        "label_seed": label_seed,
                        "run_id": run_id,
                        "scale": scale,
                        "model": selector_name,
                        **calibration,
                    }
                )

        previous = load_previous_predictions(
            reference_seed,
            label_seed,
            scale,
            test_global,
        )
        if previous is None:
            phi_fit = eigenfunctions[fit_global, :MAX_DIFFUSION_MODES]
            phi_test = eigenfunctions[test_global, :MAX_DIFFUSION_MODES]
            unconstrained_prediction = fallback_unconstrained_predictions(
                phi_fit,
                phi_test,
                line_of_sight_fit,
                velocity_fit_by_scale[scale],
                generator_eigenvalues,
            )
            cartesian_oracle_prediction = fallback_cartesian_oracle_predictions(
                phi_fit,
                phi_test,
                velocity_fit_by_scale[scale],
                generator_eigenvalues,
            )
            previous_source = "recomputed_fallback"
        else:
            unconstrained_prediction = previous["unconstrained"]
            cartesian_oracle_prediction = previous["cartesian_oracle"]
            previous_source = previous["source"]

        for model_name, model_class, predicted in [
            (
                "unconstrained_radial_validation_selected",
                "unconstrained",
                unconstrained_prediction,
            ),
            (
                "cartesian_3d_oracle",
                "cartesian_oracle",
                cartesian_oracle_prediction,
            ),
            (
                "zero_vector_baseline",
                "baseline",
                np.zeros_like(velocity_test_by_scale[scale]),
            ),
        ]:
            metrics = vector_field_metrics(
                velocity_test_by_scale[scale],
                predicted,
                line_of_sight_test,
            )
            comparison_records.append(
                {
                    "replicate": replicate,
                    "reference_seed": reference_seed,
                    "label_seed": label_seed,
                    "run_id": run_id,
                    "scale": scale,
                    "model": model_name,
                    "model_class": model_class,
                    "prediction_source": previous_source,
                    **metrics,
                }
            )
            test_predictions[model_name] = predicted
            if model_name != "zero_vector_baseline":
                for calibration in component_calibration(
                    velocity_test_by_scale[scale],
                    predicted,
                ):
                    calibration_records.append(
                        {
                            "replicate": replicate,
                            "reference_seed": reference_seed,
                            "label_seed": label_seed,
                            "run_id": run_id,
                            "scale": scale,
                            "model": model_name,
                            **calibration,
                        }
                    )

        operational_n = int(operational["n_basis"])
        operational_lambda = float(operational["regularization_strength"])
        operational_power = float(operational["penalty_power"])
        spectrum = gram_spectrum(
            radial_gram_fit_max[:operational_n, :operational_n],
            fit_local.size,
        )
        operational_regularization = regularization_diagonal(
            potential_generator_values,
            operational_n,
            operational_power,
        )
        gram_fit = radial_gram_fit_max[:operational_n, :operational_n]
        effective_df = exact_effective_degrees_of_freedom(
            gram_fit,
            operational_regularization,
            operational_lambda,
        )
        linear_algebra_records.append(
            {
                "replicate": replicate,
                "reference_seed": reference_seed,
                "label_seed": label_seed,
                "run_id": run_id,
                "scale": scale,
                "selector": "potential_radial_validation_selected",
                "n_parameters": operational_n,
                "n_affine_modes": N_AFFINE_POTENTIAL_MODES,
                "n_diffusion_modes": int(
                    operational_n - N_AFFINE_POTENTIAL_MODES
                ),
                "n_observations": int(fit_local.size),
                "numerical_rank": spectrum["numerical_rank"],
                "nullity": spectrum["nullity"],
                "condition_number": spectrum["condition_number"],
                "stable_rank": spectrum["stable_rank"],
                "entropy_effective_rank": spectrum[
                    "entropy_effective_rank"
                ],
                "effective_degrees_of_freedom": effective_df,
                "effective_degrees_of_freedom_fraction": (
                    effective_df / operational_n
                    if operational_n > 0
                    else math.nan
                ),
                "affine_mode_scale": affine_mode_scale,
            }
        )
        for singular_index, value in enumerate(spectrum["singular_values"]):
            singular_value_records.append(
                {
                    "replicate": replicate,
                    "reference_seed": reference_seed,
                    "label_seed": label_seed,
                    "run_id": run_id,
                    "scale": scale,
                    "index": int(singular_index),
                    "singular_value": float(value),
                    "normalized_singular_value": float(
                        value / spectrum["singular_values"][0]
                    ),
                }
            )

        reference_result = reference_fields[scale]
        reference_records.append(
            {
                "replicate": replicate,
                "reference_seed": reference_seed,
                "label_seed": label_seed,
                "run_id": run_id,
                "scale": scale,
                "n_reference": int(reference_indices.size),
                "n_model": int(model_indices.size),
                "n_train": int(train_local.size),
                "n_validation": int(validation_local.size),
                "n_test": int(test_local.size),
                "median_reference_bandwidth": float(
                    np.median(reference_result["bandwidth"])
                ),
                "median_reference_dispersion": float(
                    np.median(reference_result["dispersion"])
                ),
                "median_reference_effective_neighbors": float(
                    np.median(reference_result["effective_neighbors"])
                ),
                "reference_field_cache": reference_result["cache_path"],
                "reference_loaded_from_cache": reference_result[
                    "loaded_from_cache"
                ],
                "affine_mode_scale": affine_mode_scale,
            }
        )

        prediction_file = OUTPUT_DIR / f"{OUTPUT_PREFIX}_{run_id}_predictions.npz"
        prediction_payload = {
            "test_global_indices": test_global,
            "test_positions_absolute": pos_absolute[test_global],
            "test_positions_centered": positions[test_global],
            "line_of_sight_test": line_of_sight_test,
            "reference_velocity_test": velocity_test_by_scale[scale],
            "reference_radial_velocity_test": radial_test_by_scale[scale],
            "potential_radial_validation_velocity_test": test_predictions[
                "potential_radial_validation_selected"
            ],
            "potential_hidden_3d_diagnostic_velocity_test": test_predictions[
                "potential_hidden_3d_validation_selected_diagnostic"
            ],
            "potential_3d_oracle_velocity_test": test_predictions[
                "potential_3d_oracle_validation_selected_diagnostic"
            ],
            "unconstrained_radial_validation_velocity_test": (
                unconstrained_prediction
            ),
            "cartesian_3d_oracle_velocity_test": cartesian_oracle_prediction,
        }
        for coefficient_name, coefficient in selected_coefficients.items():
            prediction_payload[f"{coefficient_name}_coefficients"] = coefficient
        np.savez_compressed(prediction_file, **prediction_payload)
        prediction_files.append(str(prediction_file))

        operational_metrics = next(
            record
            for record in comparison_records[::-1]
            if record["run_id"] == run_id
            and record["model"] == "potential_radial_validation_selected"
        )
        print(
            f"  {scale:9s}: n={int(operational['n_basis'])}, "
            f"lambda={float(operational['regularization_strength']):.1e}, "
            f"p={float(operational['penalty_power']):.0f}, "
            f"NRMSE_r={operational_metrics['nrmse_radial']:.6f}, "
            f"NRMSE_t={operational_metrics['nrmse_tangential']:.6f}, "
            f"NRMSE_3D={operational_metrics['nrmse_3d']:.6f}, "
            f"angle={operational_metrics['direction_error_median_deg']:.3f} deg"
        )

    # Replicate終了後に大きなarrayを解放する。
    del reference_fields
    del diffusion_gradient_model, affine_gradient_model, gradient_model
    del radial_design_model
    del gradient_train, gradient_validation, gradient_test
    del radial_design_train, radial_design_validation, radial_design_test
    del A_train_max, A_validation_max
    del radial_gram_train_sum, radial_gram_validation_sum
    del radial_gram_train_max, radial_gram_fit_max
    del radial_rhs_train_sum, radial_rhs_validation_sum
    del radial_rhs_train_max, radial_rhs_fit_max
    del vector_gram_train_sum, vector_gram_validation_sum
    del vector_gram_train_max, vector_gram_fit_max
    del vector_rhs_train_sum, vector_rhs_validation_sum
    del vector_rhs_train_max, vector_rhs_fit_max
    del closure_velocity_model, closure_prediction_test
    gc.collect()

# ============================================================
# 11. 集約
# ============================================================

radial_validation_df = pd.DataFrame(radial_validation_records)
potential_oracle_validation_df = pd.DataFrame(
    potential_oracle_validation_records
)
selector_df = pd.DataFrame(selector_records)
comparison_df = pd.DataFrame(comparison_records)
calibration_df = pd.DataFrame(calibration_records)
linear_algebra_df = pd.DataFrame(linear_algebra_records)
singular_values_df = pd.DataFrame(singular_value_records)
reference_df = pd.DataFrame(reference_records)
closure_df = pd.DataFrame(closure_records)

operational_df = selector_df[
    selector_df["selector"] == "potential_radial_validation_selected"
].copy()
hidden3d_df = selector_df[
    selector_df["selector"]
    == "potential_hidden_3d_validation_selected_diagnostic"
].copy()

model_order = [
    "potential_radial_validation_selected",
    "potential_hidden_3d_validation_selected_diagnostic",
    "potential_3d_oracle_validation_selected_diagnostic",
    "unconstrained_radial_validation_selected",
    "cartesian_3d_oracle",
    "zero_vector_baseline",
]
model_labels = {
    "potential_radial_validation_selected": "Potential–radial CV",
    "potential_hidden_3d_validation_selected_diagnostic": (
        "Potential–3D diagnostic"
    ),
    "potential_3d_oracle_validation_selected_diagnostic": (
        "Potential–3D oracle"
    ),
    "unconstrained_radial_validation_selected": "Unconstrained–radial CV",
    "cartesian_3d_oracle": "Cartesian–3D oracle",
    "zero_vector_baseline": "Zero baseline",
}

summary_metrics = [
    "nrmse_radial",
    "nrmse_tangential",
    "nrmse_3d",
    "direction_error_median_deg",
    "direction_error_p90_deg",
    "predicted_to_reference_rms_3d",
]
summary_rows = []
for (scale, model), group in comparison_df.groupby(["scale", "model"]):
    row = {
        "scale": scale,
        "model": model,
        "model_label": model_labels.get(model, model),
        "n_replicates": int(group.shape[0]),
    }
    for metric in summary_metrics:
        values = group[metric].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        if finite.size:
            row[f"{metric}_mean"] = float(np.mean(finite))
            row[f"{metric}_std"] = (
                float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
            )
            row[f"{metric}_min"] = float(np.min(finite))
            row[f"{metric}_max"] = float(np.max(finite))
        else:
            row[f"{metric}_mean"] = math.nan
            row[f"{metric}_std"] = math.nan
            row[f"{metric}_min"] = math.nan
            row[f"{metric}_max"] = math.nan
    summary_rows.append(row)
scale_model_summary_df = pd.DataFrame(summary_rows)
scale_model_summary_df["scale_order"] = scale_model_summary_df["scale"].map(
    scale_order
)
scale_model_summary_df["model_order"] = scale_model_summary_df["model"].map(
    {name: index for index, name in enumerate(model_order)}
)
scale_model_summary_df = scale_model_summary_df.sort_values(
    ["scale_order", "model_order"]
).drop(columns=["scale_order", "model_order"])

comparison_lookup = comparison_df.set_index(["run_id", "model"])
selector_lookup = selector_df.set_index(["run_id", "selector"])
paired_records = []
for run_id in sorted(operational_df["run_id"].unique()):
    pot = comparison_lookup.loc[
        (run_id, "potential_radial_validation_selected")
    ]
    hidden = comparison_lookup.loc[
        (run_id, "potential_hidden_3d_validation_selected_diagnostic")
    ]
    potential_oracle = comparison_lookup.loc[
        (run_id, "potential_3d_oracle_validation_selected_diagnostic")
    ]
    unconstrained = comparison_lookup.loc[
        (run_id, "unconstrained_radial_validation_selected")
    ]
    cartesian_oracle = comparison_lookup.loc[(run_id, "cartesian_3d_oracle")]
    pot_spec = selector_lookup.loc[
        (run_id, "potential_radial_validation_selected")
    ]
    hidden_spec = selector_lookup.loc[
        (run_id, "potential_hidden_3d_validation_selected_diagnostic")
    ]
    paired_records.append(
        {
            "run_id": run_id,
            "replicate": int(pot["replicate"]),
            "reference_seed": int(pot["reference_seed"]),
            "label_seed": int(pot["label_seed"]),
            "scale": pot["scale"],
            "delta_nrmse_radial_potential_minus_unconstrained": float(
                pot["nrmse_radial"] - unconstrained["nrmse_radial"]
            ),
            "delta_nrmse_tangential_potential_minus_unconstrained": float(
                pot["nrmse_tangential"] - unconstrained["nrmse_tangential"]
            ),
            "delta_nrmse_3d_potential_minus_unconstrained": float(
                pot["nrmse_3d"] - unconstrained["nrmse_3d"]
            ),
            "delta_direction_error_potential_minus_unconstrained_deg": float(
                pot["direction_error_median_deg"]
                - unconstrained["direction_error_median_deg"]
            ),
            "nrmse_3d_ratio_potential_over_unconstrained": float(
                pot["nrmse_3d"] / unconstrained["nrmse_3d"]
            ),
            "mse_reduction_fraction_vs_unconstrained": float(
                1.0
                - np.square(pot["nrmse_3d"] / unconstrained["nrmse_3d"])
            ),
            "mse_reduction_fraction_vs_zero_baseline": float(
                1.0 - np.square(pot["nrmse_3d"])
            ),
            "delta_nrmse_3d_operational_minus_hidden3d": float(
                pot["nrmse_3d"] - hidden["nrmse_3d"]
            ),
            "delta_nrmse_3d_operational_minus_potential_oracle": float(
                pot["nrmse_3d"] - potential_oracle["nrmse_3d"]
            ),
            "delta_nrmse_3d_operational_minus_cartesian_oracle": float(
                pot["nrmse_3d"] - cartesian_oracle["nrmse_3d"]
            ),
            "operational_nrmse_tangential_below_one": bool(
                pot["nrmse_tangential"] < 1.0
            ),
            "operational_nrmse_3d_below_one": bool(pot["nrmse_3d"] < 1.0),
            "operational_beats_unconstrained_3d": bool(
                pot["nrmse_3d"] < unconstrained["nrmse_3d"]
            ),
            "operational_beats_unconstrained_direction": bool(
                pot["direction_error_median_deg"]
                < unconstrained["direction_error_median_deg"]
            ),
            "operational_hidden3d_same_candidate": bool(
                int(pot_spec["n_basis"]) == int(hidden_spec["n_basis"])
                and np.isclose(
                    float(pot_spec["regularization_strength"]),
                    float(hidden_spec["regularization_strength"]),
                )
                and np.isclose(
                    float(pot_spec["penalty_power"]),
                    float(hidden_spec["penalty_power"]),
                )
            ),
        }
    )
paired_df = pd.DataFrame(paired_records)

success_rows = []
for scale, group in paired_df.groupby("scale"):
    pot_group = operational_df[operational_df["scale"] == scale]
    success_rows.append(
        {
            "scale": scale,
            "n_replicates": int(group.shape[0]),
            "test_nrmse_radial_mean": float(
                np.mean(pot_group["test_nrmse_radial"])
            ),
            "test_nrmse_radial_std": float(
                np.std(pot_group["test_nrmse_radial"], ddof=1)
            ),
            "test_nrmse_tangential_mean": float(
                np.mean(pot_group["test_nrmse_tangential"])
            ),
            "test_nrmse_tangential_std": float(
                np.std(pot_group["test_nrmse_tangential"], ddof=1)
            ),
            "test_nrmse_3d_mean": float(
                np.mean(pot_group["test_nrmse_3d"])
            ),
            "test_nrmse_3d_std": float(
                np.std(pot_group["test_nrmse_3d"], ddof=1)
            ),
            "test_direction_error_median_deg_mean": float(
                np.mean(pot_group["test_direction_error_median_deg"])
            ),
            "test_direction_error_median_deg_std": float(
                np.std(pot_group["test_direction_error_median_deg"], ddof=1)
            ),
            "count_tangential_nrmse_below_one": int(
                group["operational_nrmse_tangential_below_one"].sum()
            ),
            "count_3d_nrmse_below_one": int(
                group["operational_nrmse_3d_below_one"].sum()
            ),
            "count_potential_beats_unconstrained_3d": int(
                group["operational_beats_unconstrained_3d"].sum()
            ),
            "count_potential_beats_unconstrained_direction": int(
                group["operational_beats_unconstrained_direction"].sum()
            ),
            "count_operational_hidden3d_same_candidate": int(
                group["operational_hidden3d_same_candidate"].sum()
            ),
            "mean_delta_nrmse_3d_potential_minus_unconstrained": float(
                np.mean(
                    group[
                        "delta_nrmse_3d_potential_minus_unconstrained"
                    ]
                )
            ),
            "mean_mse_reduction_fraction_vs_zero_baseline": float(
                np.mean(group["mse_reduction_fraction_vs_zero_baseline"])
            ),
            "mean_mse_reduction_fraction_vs_unconstrained": float(
                np.mean(group["mse_reduction_fraction_vs_unconstrained"])
            ),
        }
    )
operational_success_df = pd.DataFrame(success_rows).sort_values(
    "scale", key=lambda s: s.map(scale_order)
)

operational_hyperparameters_df = operational_df[
    [
        "run_id",
        "replicate",
        "reference_seed",
        "label_seed",
        "scale",
        "n_basis",
        "n_diffusion_modes",
        "regularization_strength",
        "penalty_power",
        "validation_nrmse_radial",
        "test_nrmse_radial",
        "test_nrmse_tangential",
        "test_nrmse_3d",
        "test_direction_error_median_deg",
    ]
].copy()
hyperparameter_frequency_df = (
    operational_hyperparameters_df.groupby(
        [
            "scale",
            "n_basis",
            "n_diffusion_modes",
            "regularization_strength",
            "penalty_power",
        ],
        as_index=False,
    )
    .size()
    .rename(columns={"size": "count"})
    .sort_values(
        ["scale", "count"],
        ascending=[True, False],
        key=lambda col: (
            col.map(scale_order) if col.name == "scale" else col
        ),
    )
)

linear_summary_rows = []
for scale, group in linear_algebra_df.groupby("scale"):
    row = {"scale": scale, "n_replicates": int(group.shape[0])}
    for metric in [
        "n_parameters",
        "numerical_rank",
        "nullity",
        "condition_number",
        "stable_rank",
        "entropy_effective_rank",
        "effective_degrees_of_freedom",
        "effective_degrees_of_freedom_fraction",
        "affine_mode_scale",
    ]:
        values = group[metric].to_numpy(dtype=np.float64)
        row[f"{metric}_mean"] = float(np.mean(values))
        row[f"{metric}_std"] = (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        )
    linear_summary_rows.append(row)
linear_algebra_summary_df = pd.DataFrame(linear_summary_rows).sort_values(
    "scale", key=lambda s: s.map(scale_order)
)

# ============================================================
# 12. Summary figures
# ============================================================

plot_models = [
    "potential_radial_validation_selected",
    "potential_hidden_3d_validation_selected_diagnostic",
    "potential_3d_oracle_validation_selected_diagnostic",
    "unconstrained_radial_validation_selected",
    "cartesian_3d_oracle",
]


def plot_metric_robustness(metric, ylabel, stem, baseline=None):
    fig, ax = plt.subplots(figsize=(11.0, 5.6))
    x_scale = np.arange(len(scale_names), dtype=np.float64)
    offsets = np.linspace(-0.28, 0.28, len(plot_models))
    jitter = np.linspace(-0.018, 0.018, len(SEED_PAIRS))
    default_colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for model_index, model in enumerate(plot_models):
        model_table = comparison_df[comparison_df["model"] == model]
        color = default_colors[model_index % len(default_colors)]
        mean_values = []
        std_values = []
        plot_x = []
        for scale_index, scale in enumerate(scale_names):
            group = model_table[model_table["scale"] == scale].sort_values(
                "replicate"
            )
            values = group[metric].to_numpy(dtype=np.float64)
            center = x_scale[scale_index] + offsets[model_index]
            ax.scatter(
                center + jitter[: values.size],
                values,
                s=26,
                alpha=0.55,
                color=color,
            )
            plot_x.append(center)
            mean_values.append(float(np.mean(values)))
            std_values.append(float(np.std(values, ddof=1)))
        ax.errorbar(
            plot_x,
            mean_values,
            yerr=std_values,
            marker="o",
            linewidth=1.5,
            capsize=3,
            color=color,
            label=model_labels[model],
        )
    if baseline is not None:
        ax.axhline(
            baseline,
            linestyle="--",
            linewidth=1.2,
            color="0.35",
            label="zero-vector baseline",
        )
    ax.set_xticks(x_scale)
    ax.set_xticklabels(scale_names)
    ax.set_ylabel(ylabel)
    ax.legend(ncol=2, frameon=True)
    fig.tight_layout()
    save_figure(fig, stem)


plot_metric_robustness(
    "nrmse_3d",
    "Untouched-test 3D NRMSE",
    "summary_3d_nrmse_by_scale",
    baseline=1.0,
)
plot_metric_robustness(
    "nrmse_tangential",
    "Untouched-test tangential NRMSE",
    "summary_tangential_nrmse_by_scale",
    baseline=1.0,
)
plot_metric_robustness(
    "nrmse_radial",
    "Untouched-test radial NRMSE",
    "summary_radial_nrmse_by_scale",
    baseline=1.0,
)
plot_metric_robustness(
    "direction_error_median_deg",
    "Median direction error [deg]",
    "summary_direction_error_by_scale",
    baseline=None,
)

# Paired 3D improvement relative to unconstrained model。
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for scale_index, scale in enumerate(scale_names):
    values = paired_df.loc[
        paired_df["scale"] == scale,
        "delta_nrmse_3d_potential_minus_unconstrained",
    ].to_numpy(dtype=np.float64)
    x = scale_index + np.linspace(-0.06, 0.06, values.size)
    ax.scatter(x, values, s=34, alpha=0.70)
    ax.errorbar(
        scale_index,
        np.mean(values),
        yerr=np.std(values, ddof=1),
        marker="o",
        capsize=4,
        linewidth=1.5,
    )
ax.axhline(0.0, linestyle="--", linewidth=1.2, color="0.35")
ax.set_xticks(np.arange(len(scale_names)))
ax.set_xticklabels(scale_names)
ax.set_ylabel(
    r"$\Delta$3D NRMSE (potential $-$ unconstrained)"
)
fig.tight_layout()
save_figure(fig, "summary_paired_3d_improvement")

# Operational hyperparameter selections。
fig, ax = plt.subplots(figsize=(8.2, 5.2))
marker_map = {1.0: "o", 2.0: "^"}
for scale_index, scale in enumerate(scale_names):
    group = operational_hyperparameters_df[
        operational_hyperparameters_df["scale"] == scale
    ].sort_values("replicate")
    for _, row in group.iterrows():
        lam = float(row["regularization_strength"])
        y = math.log10(lam) if lam > 0.0 else -14.0
        ax.scatter(
            scale_index + 0.06 * (int(row["replicate"]) - 3),
            y,
            s=18.0 + 0.10 * float(row["n_basis"]),
            marker=marker_map.get(float(row["penalty_power"]), "s"),
            facecolors=("none" if lam == 0.0 else None),
            edgecolors="black",
            linewidths=0.8,
        )
ax.set_xticks(np.arange(len(scale_names)))
ax.set_xticklabels(scale_names)
ax.set_ylabel(r"Selected $\log_{10}\lambda$")
ax.set_title("Marker: penalty power; size: number of potential parameters")
fig.tight_layout()
save_figure(fig, "summary_operational_hyperparameters")

# Effective degrees of freedom fraction。
fig, ax = plt.subplots(figsize=(7.6, 5.2))
for scale_index, scale in enumerate(scale_names):
    values = linear_algebra_df.loc[
        linear_algebra_df["scale"] == scale,
        "effective_degrees_of_freedom_fraction",
    ].to_numpy(dtype=np.float64)
    x = scale_index + np.linspace(-0.06, 0.06, values.size)
    ax.scatter(x, values, s=34, alpha=0.70)
    ax.errorbar(
        scale_index,
        np.mean(values),
        yerr=np.std(values, ddof=1),
        marker="o",
        capsize=4,
    )
ax.set_xticks(np.arange(len(scale_names)))
ax.set_xticklabels(scale_names)
ax.set_ylabel("Effective degrees-of-freedom fraction")
ax.set_ylim(0.0, 1.05)
fig.tight_layout()
save_figure(fig, "summary_effective_df_fraction")

# ============================================================
# 13. 保存
# ============================================================

output_files = {
    "gradient_diagnostics_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_gradient_diagnostics.csv",
    "closure_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_closure.csv",
    "radial_validation_grid_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_radial_validation_grid.csv",
    "potential_oracle_validation_grid_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_potential_oracle_validation_grid.csv",
    "selector_summary_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_selector_summary.csv",
    "comparison_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_test_comparison.csv",
    "calibration_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_calibration.csv",
    "linear_algebra_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_linear_algebra.csv",
    "singular_values_csv": OUTPUT_DIR / f"{OUTPUT_PREFIX}_singular_values.csv",
    "reference_fields_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_reference_fields.csv",
    "scale_model_summary_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_scale_model_summary.csv",
    "operational_success_summary_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_operational_success_summary.csv",
    "operational_hyperparameters_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_operational_hyperparameters.csv",
    "hyperparameter_frequency_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_hyperparameter_frequency.csv",
    "paired_differences_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_paired_differences.csv",
    "linear_algebra_summary_csv": OUTPUT_DIR
    / f"{OUTPUT_PREFIX}_linear_algebra_summary.csv",
    "summary_json": OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json",
}

gradient_diagnostics_df.to_csv(
    output_files["gradient_diagnostics_csv"], index=False
)
closure_df.to_csv(output_files["closure_csv"], index=False)
radial_validation_df.to_csv(
    output_files["radial_validation_grid_csv"], index=False
)
potential_oracle_validation_df.to_csv(
    output_files["potential_oracle_validation_grid_csv"], index=False
)
selector_df.to_csv(output_files["selector_summary_csv"], index=False)
comparison_df.to_csv(output_files["comparison_csv"], index=False)
calibration_df.to_csv(output_files["calibration_csv"], index=False)
linear_algebra_df.to_csv(output_files["linear_algebra_csv"], index=False)
singular_values_df.to_csv(output_files["singular_values_csv"], index=False)
reference_df.to_csv(output_files["reference_fields_csv"], index=False)
scale_model_summary_df.to_csv(
    output_files["scale_model_summary_csv"], index=False
)
operational_success_df.to_csv(
    output_files["operational_success_summary_csv"], index=False
)
operational_hyperparameters_df.to_csv(
    output_files["operational_hyperparameters_csv"], index=False
)
hyperparameter_frequency_df.to_csv(
    output_files["hyperparameter_frequency_csv"], index=False
)
paired_df.to_csv(output_files["paired_differences_csv"], index=False)
linear_algebra_summary_df.to_csv(
    output_files["linear_algebra_summary_csv"], index=False
)

summary = {
    "environment": {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
    },
    "paths": {
        "input": str(DATA_FILE),
        "geometry_cache": str(GEOMETRY_CACHE_FILE),
        "gradient_cache": str(GRADIENT_CACHE_FILE),
        "unconstrained_results": str(PREVIOUS_UNCONSTRAINED_DIR),
        "output": str(OUTPUT_DIR),
    },
    "experiment": {
        "seed_pairs": SEED_PAIRS,
        "scales": scale_names,
        "n_runs": int(len(SEED_PAIRS) * len(scale_names)),
        "potential_basis_sizes": POTENTIAL_BASIS_SIZES,
        "regularization_strengths": POTENTIAL_REGULARIZATION_STRENGTHS,
        "regularization_powers": POTENTIAL_REGULARIZATION_POWERS,
    },
    "gradient": {
        **gradient_result["diagnostics"],
        "constant_mode_index": constant_mode_index,
        "constant_mode_gradient_rms": constant_gradient_rms,
        "affine_projection_vector_rms": affine_projection_vector_rms,
        "affine_projection_relative_orthogonality": (
            affine_projection_relative_orthogonality
        ),
        "loaded_from_cache": gradient_result["loaded_from_cache"],
    },
    "closure": {
        "all_pass": bool(np.all(closure_df["closure_status"] == "pass")),
        "tolerance": CLOSURE_PASS_TOLERANCE,
        "records": closure_df.to_dict(orient="records"),
    },
    "operational_success_summary": operational_success_df.to_dict(
        orient="records"
    ),
    "scale_model_summary": scale_model_summary_df.to_dict(orient="records"),
    "hyperparameter_frequency": hyperparameter_frequency_df.to_dict(
        orient="records"
    ),
    "linear_algebra_summary": linear_algebra_summary_df.to_dict(
        orient="records"
    ),
    "prediction_files": prediction_files,
    "output_files": {key: str(value) for key, value in output_files.items()},
    "elapsed_minutes": (time.perf_counter() - start_time) / 60.0,
}
with output_files["summary_json"].open("w", encoding="utf-8") as handle:
    json.dump(json_ready(summary), handle, indent=2, ensure_ascii=False)

elapsed = time.perf_counter() - start_time
print("=" * 120)
print("Mock-1 potential-flow full robustness completed")
print("=" * 120)
print(f"Elapsed time: {elapsed / 60.0:.2f} min")
print("Operational potential-flow selections")
display(operational_hyperparameters_df)
print("Scale/model robustness summary")
display(scale_model_summary_df)
print("Operational success summary")
display(operational_success_df)
print("Hyperparameter frequencies")
display(hyperparameter_frequency_df)
print("Potential minus unconstrained paired differences")
display(paired_df)
print("Potential radial-design linear-algebra summary")
display(linear_algebra_summary_df)
print(
    "All maximum-basis in-span closure tests pass:",
    bool(np.all(closure_df["closure_status"] == "pass")),
)
print("Saved files")
for key, value in output_files.items():
    print(f"  {key}: {value}")
