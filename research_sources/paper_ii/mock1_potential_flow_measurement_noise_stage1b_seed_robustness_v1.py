# Mock-1 potential-flow radial reconstruction: measurement-noise seed robustness Stage 1b
#
# この内容全体をJupyter Notebookの新しい空のコードセルへ貼り付けて実行する。
#
# Stage 1bの目的:
#   1. Fiducial reference scaleを固定し、五組のreference/label seed pairsを用いる。
#   2. radial labelsだけへhomoscedastic Gaussian measurement noiseを導入する。
#   3. positive noise levelごとに複数のnoise realizationsを生成し、seed-pair variationと
#      noise-realization variationを分離する。
#   4. 全seed pairsのvalidation curvesからone-standard-error basisとpractical plateauを
#      求め、combined practical basis=max(one-SE, plateau)をnoise levelごとに定める。
#   5. Raw per-run argminはstrict-boundary diagnosticとして保持するが、primary結果には
#      shared practical basisを用いる。
#   6. Model selectionにはnoisy radial validation labelsだけを用いる。
#   7. 選択完了後にlatent radial、hidden tangential、hidden 3D truthを診断する。
#   8. Conditional measurement-noise covariance、whitened residual、coverageを検証する。
#
# 重要:
#   - operator-side geometryとselection-side lossはともにunweightedである。
#   - Homoscedastic noiseなのでmean-one normalized measurement precisionは1である。
#   - 同じseed pairとnoise-realization indexでは、全positive noise levelsに同じ
#     standard-normal drawを用い、noise-level dependenceをpairedに比較する。
#   - Positive-noise aggregateでは、まずnoise realizationsをseed pair内で平均し、
#     次に五つのseed-pair meansを等重みで集約する。
#   - Reported intervalsはfixed structural family下のconditional measurement-noise
#     propagationであり、model discrepancy、hyperparameter-selection uncertainty、
#     distance uncertainty、structural completion uncertaintyを含まない。

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
try:
    from IPython.display import display
except Exception:
    def display(value):
        print(value)
from scipy import linalg, sparse
from scipy.spatial import cKDTree


def parse_float_list(text: str) -> list[float]:
    values = []
    for item in str(text).split(','):
        item = item.strip()
        if item:
            values.append(float(item))
    if not values:
        raise ValueError('少なくとも一つのnoise fractionが必要である。')
    if any(value < 0.0 for value in values):
        raise ValueError('Noise fractionは非負でなければならない。')
    return sorted(set(values))


def parse_int_list(text: str) -> list[int]:
    values = []
    for item in str(text).split(','):
        item = item.strip()
        if item:
            values.append(int(item))
    if not values:
        raise ValueError('少なくとも一つのmode countが必要である。')
    if any(value < 0 for value in values):
        raise ValueError('Mode countは非負でなければならない。')
    return sorted(set(values))



# ============================================================
# 1. 固定入出力パス
# ============================================================

MOCK_DATA_ROOT = Path(
    os.environ.get(
        'COSMIC_DIPOLE_MOCK_DATA_ROOT',
        'mock_data',
    )
)
INPUT_DIR = Path(
    os.environ.get(
        'MOCK1_INPUT_DIR',
        str(MOCK_DATA_ROOT / 'mock1_complete_sphere'),
    )
)
DATA_FILE = INPUT_DIR / 'mock1.npz'
GEOMETRY_CACHE_FILE = Path(
    os.environ.get(
        'MOCK1_STAGE1B_GEOMETRY_CACHE',
        str(MOCK_DATA_ROOT / 'mock1_diffusion_geometry_bulk_cache_m2048.npz'),
    )
)
LEGACY_GEOMETRY_CACHE_FILE = Path(
    os.environ.get(
        'MOCK1_STAGE1B_LEGACY_GEOMETRY_CACHE',
        str(MOCK_DATA_ROOT / 'mock1_diffusion_geometry_bulk_cache.npz'),
    )
)
REFERENCE_CACHE_DIR = MOCK_DATA_ROOT / 'mock1_bulk_robustness_cache'
GRADIENT_CACHE_FILE = Path(
    os.environ.get(
        'MOCK1_STAGE1B_GRADIENT_CACHE',
        str(
            MOCK_DATA_ROOT
            / 'mock1_potential_flow_gradient_basis_cache_stage1a_highmode_m2048_v1.npz'
        ),
    )
)
PREVIOUS_H2_DIR = Path(
    os.environ.get(
        'MOCK1_STAGE1B_PREVIOUS_H2_DIR',
        str(MOCK_DATA_ROOT / 'mock1_potential_flow_measurement_noise_stage1a_highmode_h2_v1'),
    )
)
PREVIOUS_H2_PREFIX = 'mock1_potential_flow_measurement_noise_stage1a_highmode_h2'
OUTPUT_DIR = Path(
    os.environ.get(
        'MOCK1_STAGE1B_OUTPUT_DIR',
        str(MOCK_DATA_ROOT / 'mock1_potential_flow_measurement_noise_stage1b_seed_robustness_v1'),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
PREDICTION_DIR = OUTPUT_DIR / 'predictions'
PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = 'mock1_potential_flow_measurement_noise_stage1b_seed_robustness'

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)
VELOCITY_UNIT_LABEL = 'simulation velocity unit'
POSITION_UNIT_LABEL = r'$h^{-1}\,\mathrm{Mpc}$'


def parse_seed_pairs(text: str) -> list[tuple[int, int]]:
    pairs = []
    for item in str(text).split(','):
        item = item.strip()
        if not item:
            continue
        if ':' not in item:
            raise ValueError(
                'Seed pairはreference:label形式で指定する。例: 20260805:20260804'
            )
        reference_text, label_text = item.split(':', 1)
        pairs.append((int(reference_text), int(label_text)))
    if not pairs:
        raise ValueError('少なくとも一組のseed pairが必要である。')
    if len(set(pairs)) != len(pairs):
        raise ValueError('Seed pairsに重複がある。')
    return pairs


# ============================================================
# 2. Stage 1b設定
# ============================================================

SEED_PAIRS = parse_seed_pairs(
    os.environ.get(
        'MOCK1_STAGE1B_SEED_PAIRS',
        '20260805:20260804,20260815:20260814,20260825:20260824,'
        '20260905:20260904,20260915:20260914',
    )
)
REFERENCE_FRACTION = 0.50
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

SMOOTHING_CONFIG = {
    'scale': 'fiducial',
    'n_neighbors': 32,
    'bandwidth_neighbor': 12,
    'bandwidth_multiplier': 1.0,
}
SMOOTHING_CONFIGS = [SMOOTHING_CONFIG]

NOISE_FRACTIONS = parse_float_list(
    os.environ.get(
        'MOCK1_STAGE1B_NOISE_FRACTIONS',
        '0,0.05,0.10,0.20,0.40,0.80',
    )
)
N_NOISE_REALIZATIONS = int(
    os.environ.get('MOCK1_STAGE1B_NOISE_REALIZATIONS', '12')
)
NOISE_SEED_BASE = int(
    os.environ.get('MOCK1_STAGE1B_NOISE_SEED_BASE', '2026082700')
)
NOISE_SEED_STRIDE = int(
    os.environ.get('MOCK1_STAGE1B_NOISE_SEED_STRIDE', '1000')
)
if N_NOISE_REALIZATIONS < 1:
    raise ValueError('N_NOISE_REALIZATIONSは1以上でなければならない。')

GRAPH_NEIGHBORS = 48
GRAPH_BANDWIDTH_NEIGHBOR = 16
GRAPH_BANDWIDTH_MULTIPLIER = 1.0

N_AFFINE_POTENTIAL_MODES = 3
POTENTIAL_DIFFUSION_MODE_COUNTS = parse_int_list(
    os.environ.get(
        'MOCK1_STAGE1B_DIFFUSION_MODE_COUNTS',
        '0,3,8,16,32,64,128,256,384,511,640,768,896,1024,1280,1536,1792,2047',
    )
)
MAX_DIFFUSION_MODES = max(POTENTIAL_DIFFUSION_MODE_COUNTS) + 1
CAPACITY_REFERENCE_PARAMETER_COUNT = int(
    os.environ.get('MOCK1_STAGE1B_REFERENCE_PARAMETER_COUNT', '1027')
)
PLATEAU_RELATIVE_THRESHOLD = float(
    os.environ.get('MOCK1_STAGE1B_PLATEAU_THRESHOLD', '0.02')
)
PLATEAU_CONSECUTIVE_STEPS = int(
    os.environ.get('MOCK1_STAGE1B_PLATEAU_STEPS', '2')
)
PLATEAU_MIN_PARAMETER_COUNT = int(
    os.environ.get('MOCK1_STAGE1B_PLATEAU_MIN_PARAMETERS', '514')
)
POTENTIAL_REGULARIZATION_STRENGTHS = parse_float_list(
    os.environ.get(
        'MOCK1_STAGE1B_REGULARIZATION_STRENGTHS',
        '0,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1,1e2',
    )
)
POTENTIAL_REGULARIZATION_POWERS = parse_float_list(
    os.environ.get('MOCK1_STAGE1B_REGULARIZATION_POWERS', '1,2')
)

METRIC_EIGENVALUE_RELATIVE_FLOOR = 1.0e-10
METRIC_EIGENVALUE_ABSOLUTE_FLOOR = 1.0e-14
GRADIENT_MODE_CHUNK_SIZE = 32
USE_GRADIENT_CACHE = True
USE_REFERENCE_FIELD_CACHE = True
NUMERICAL_RIDGE = 1.0e-12
CLOSURE_PASS_TOLERANCE = 1.0e-8
RUN_MAX_BASIS_CLOSURE = os.environ.get('MOCK1_STAGE1B_RUN_CLOSURE', '1') == '1'
RESUME_FROM_CHECKPOINT = os.environ.get('MOCK1_STAGE1B_RESUME', '1') == '1'
FORCE_RECOMPUTE = os.environ.get('MOCK1_STAGE1B_FORCE_RECOMPUTE', '0') == '1'

BOOTSTRAP_DRAWS = int(os.environ.get('MOCK1_STAGE1B_BOOTSTRAP_DRAWS', '10000'))
BOOTSTRAP_SEED = int(os.environ.get('MOCK1_STAGE1B_BOOTSTRAP_SEED', '2026082727'))
COVERAGE_Z = {
    '68': 1.0,
    '95': 1.959963984540054,
}

SHOW_PLOTS = True
SAVE_PDF = True
SAVE_PNG = True

SMOKE_TEST = os.environ.get('MOCK1_STAGE1B_SMOKE_TEST', '0') == '1'
if SMOKE_TEST:
    SEED_PAIRS = SEED_PAIRS[:2]
    GRAPH_NEIGHBORS = 16
    GRAPH_BANDWIDTH_NEIGHBOR = 6
    POTENTIAL_DIFFUSION_MODE_COUNTS = [0, 3, 8, 16, 23]
    MAX_DIFFUSION_MODES = max(POTENTIAL_DIFFUSION_MODE_COUNTS) + 1
    CAPACITY_REFERENCE_PARAMETER_COUNT = 19
    PLATEAU_MIN_PARAMETER_COUNT = 11
    POTENTIAL_REGULARIZATION_STRENGTHS = [0.0, 1.0e-6, 1.0e-2, 1.0]
    POTENTIAL_REGULARIZATION_POWERS = [1.0, 2.0]
    NOISE_FRACTIONS = [0.0, 0.20, 0.80]
    N_NOISE_REALIZATIONS = 2
    BOOTSTRAP_DRAWS = 500
    GRADIENT_MODE_CHUNK_SIZE = 8
    GRADIENT_CACHE_FILE = (
        MOCK_DATA_ROOT
        / 'mock1_potential_flow_gradient_basis_cache_stage1b_smoke_v1.npz'
    )
    OUTPUT_DIR = Path(
        os.environ.get(
            'MOCK1_STAGE1B_SMOKE_OUTPUT_DIR',
            str(
                MOCK_DATA_ROOT
                / 'mock1_potential_flow_measurement_noise_stage1b_seed_robustness_v1_smoke'
            ),
        )
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PREDICTION_DIR = OUTPUT_DIR / 'predictions'
    PREDICTION_DIR.mkdir(parents=True, exist_ok=True)
    SHOW_PLOTS = False
    SAVE_PDF = False
    SAVE_PNG = False

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
            "legacy_metric_geometry": file_signature(LEGACY_GEOMETRY_CACHE_FILE),
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
# 7. Measurement-noise解析用関数
# ============================================================


def normalized_precision_weights(sigma: np.ndarray) -> np.ndarray:
    """Inverse-variance weightsを平均1へ規格化する。"""
    sigma = np.asarray(sigma, dtype=np.float64)
    if sigma.ndim != 1:
        raise ValueError('sigmaは1次元arrayでなければならない。')
    if np.any(~np.isfinite(sigma)) or np.any(sigma < 0.0):
        raise ValueError('sigmaには有限かつ非負の値が必要である。')
    if np.all(sigma == 0.0):
        return np.ones_like(sigma)
    positive = sigma > 0.0
    if not np.all(positive):
        raise ValueError(
            '一つのrun内でzero-noise点とpositive-noise点を混在させない。'
        )
    precision = 1.0 / np.square(sigma)
    return precision / np.mean(precision)


def weighted_cross_products(design, target, weights):
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if design.shape[0] != target.shape[0] or target.shape[0] != weights.size:
        raise ValueError('Design, target, weightsの行数が一致しない。')
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError('Weight sumは正でなければならない。')
    weighted_design = design * weights[:, None]
    gram = design.T @ weighted_design / weight_sum
    rhs = design.T @ (weights[:, None] * target) / weight_sum
    return gram, rhs, weight_sum


def hidden_vector_nrmse_from_quadratic(
    coefficients,
    vector_gram_sum,
    vector_rhs_true,
    true_vector_sum_squares,
):
    coefficients = np.asarray(coefficients, dtype=np.float64)
    projected = vector_gram_sum @ coefficients
    residual_sum_squares = (
        np.sum(coefficients * projected, axis=0)
        - 2.0 * np.sum(coefficients * vector_rhs_true[:, None], axis=0)
        + float(true_vector_sum_squares)
    )
    residual_sum_squares = np.maximum(residual_sum_squares, 0.0)
    denominator = max(float(true_vector_sum_squares), np.finfo(float).eps)
    return np.sqrt(residual_sum_squares / denominator), residual_sum_squares


def marginal_interval_coverage(true, mean, variance, z):
    true = np.asarray(true, dtype=np.float64)
    mean = np.asarray(mean, dtype=np.float64)
    variance = np.asarray(variance, dtype=np.float64)
    standard_deviation = np.sqrt(np.maximum(variance, 0.0))
    tolerance = float(z) * standard_deviation
    return float(np.mean(np.abs(true - mean) <= tolerance))


def conditional_coefficient_covariance(
    gram,
    factor_kind,
    factor,
    sigma_noise,
    n_observations,
):
    """Mean-normalized penalized estimatorのmeasurement-noise covariance。"""
    n_parameters = int(gram.shape[0])
    if float(sigma_noise) == 0.0:
        return np.zeros((n_parameters, n_parameters), dtype=np.float64)
    inverse_system = solve_factored(
        factor_kind,
        factor,
        np.eye(n_parameters, dtype=np.float64),
    )
    covariance = (
        float(sigma_noise) ** 2
        / float(n_observations)
        * inverse_system
        @ gram
        @ inverse_system.T
    )
    return 0.5 * (covariance + covariance.T)


def radial_prediction_variance(design, coefficient_covariance):
    design = np.asarray(design, dtype=np.float64)
    covariance = np.asarray(coefficient_covariance, dtype=np.float64)
    return np.maximum(
        np.einsum('im,mn,in->i', design, covariance, design, optimize=True),
        0.0,
    )


def vector_component_prediction_variance(gradient_basis, coefficient_covariance):
    gradient_basis = np.asarray(gradient_basis, dtype=np.float64)
    covariance = np.asarray(coefficient_covariance, dtype=np.float64)
    return np.maximum(
        np.einsum(
            'imc,mn,inc->ic',
            gradient_basis,
            covariance,
            gradient_basis,
            optimize=True,
        ),
        0.0,
    )


def noise_fraction_token(value: float) -> str:
    return f'{int(round(10000.0 * float(value))):05d}'


def normal_matrix_condition(system):
    eigenvalues = linalg.eigvalsh(
        0.5 * (np.asarray(system) + np.asarray(system).T),
        check_finite=False,
    )
    positive = eigenvalues[eigenvalues > 0.0]
    if positive.size == 0:
        return math.inf, math.nan, math.nan
    return (
        float(positive[-1] / positive[0]),
        float(positive[0]),
        float(positive[-1]),
    )


def aggregate_mean_std(group, column):
    values = group[column].to_numpy(dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    return (
        float(np.mean(values)),
        float(np.std(values, ddof=1)) if values.size > 1 else 0.0,
    )


def bootstrap_mean_interval(
    values,
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    if values.size == 1:
        value = float(values[0])
        return value, value
    rng = np.random.default_rng(int(seed))
    draw_indices = rng.integers(0, values.size, size=(int(n_draws), values.size))
    draw_means = np.mean(values[draw_indices], axis=1)
    alpha = 0.5 * (1.0 - float(confidence))
    return (
        float(np.quantile(draw_means, alpha)),
        float(np.quantile(draw_means, 1.0 - alpha)),
    )


def practical_plateau_basis(
    basis_values,
    risk_values,
    minimum_basis: int,
    relative_threshold: float,
    consecutive_steps: int,
):
    basis = np.asarray(basis_values, dtype=np.int64)
    risk = np.asarray(risk_values, dtype=np.float64)
    order = np.argsort(basis)
    basis = basis[order]
    risk = risk[order]
    valid = (basis >= int(minimum_basis)) & np.isfinite(risk)
    basis = basis[valid]
    risk = risk[valid]
    if basis.size < int(consecutive_steps) + 1:
        return None, np.array([], dtype=np.float64)
    improvement = (risk[:-1] - risk[1:]) / np.maximum(
        np.abs(risk[:-1]), np.finfo(float).eps
    )
    for start in range(0, improvement.size - int(consecutive_steps) + 1):
        window = improvement[start : start + int(consecutive_steps)]
        if np.all(window < float(relative_threshold)):
            return int(basis[start]), improvement
    return None, improvement



# ============================================================
# 8. 入力読込みとglobal geometry
# ============================================================

import gc

required_files = [DATA_FILE, GEOMETRY_CACHE_FILE, LEGACY_GEOMETRY_CACHE_FILE]
missing_files = [str(path) for path in required_files if not path.is_file()]
if missing_files:
    raise FileNotFoundError(
        '必要な入力ファイルが見つからない。\n'
        + '\n'.join(missing_files)
        + '\nHigh-mode geometry cacheがない場合は、Paper-IのMock-1 mode-number '
          'convergence scriptを先に実行する。'
    )

with np.load(DATA_FILE, allow_pickle=False) as data:
    if 'pos' not in data.files or 'vel' not in data.files:
        raise KeyError(f'mock1.npzにはposとvelが必要である。実際のkeys: {data.files}')
    pos_absolute = np.asarray(data['pos'], dtype=np.float64)
    vel_raw = np.asarray(data['vel'], dtype=np.float64)

with np.load(GEOMETRY_CACHE_FILE, allow_pickle=False) as data:
    required_geometry_keys = {
        'eigenfunctions',
        'generator_eigenvalues',
        'bandwidth',
        'stationary_measure',
    }
    missing_geometry_keys = required_geometry_keys.difference(data.files)
    if missing_geometry_keys:
        raise KeyError(
            'high-mode geometry cacheに必要なkeysがない: '
            + ', '.join(sorted(missing_geometry_keys))
        )
    eigenfunctions = np.asarray(data['eigenfunctions'], dtype=np.float64)
    generator_eigenvalues = np.asarray(
        data['generator_eigenvalues'], dtype=np.float64
    )
    geometry_bandwidth = np.asarray(data['bandwidth'], dtype=np.float64)
    stationary_measure = np.asarray(data['stationary_measure'], dtype=np.float64)

with np.load(LEGACY_GEOMETRY_CACHE_FILE, allow_pickle=False) as data:
    required_legacy_keys = {
        'eigenfunctions',
        'generator_eigenvalues',
        'bandwidth',
        'local_metric',
    }
    missing_legacy_keys = required_legacy_keys.difference(data.files)
    if missing_legacy_keys:
        raise KeyError(
            'legacy geometry cacheに必要なkeysがない: '
            + ', '.join(sorted(missing_legacy_keys))
        )
    legacy_eigenfunctions = np.asarray(data['eigenfunctions'], dtype=np.float64)
    legacy_generator_eigenvalues = np.asarray(
        data['generator_eigenvalues'], dtype=np.float64
    )
    legacy_bandwidth = np.asarray(data['bandwidth'], dtype=np.float64)
    cached_local_metric = np.asarray(data['local_metric'], dtype=np.float64)

if pos_absolute.ndim != 2 or pos_absolute.shape[1] != 3:
    raise ValueError(f'posのshapeが不正である: {pos_absolute.shape}')
if vel_raw.shape != pos_absolute.shape:
    raise ValueError(f'velのshape {vel_raw.shape}がpos {pos_absolute.shape}と一致しない。')
if eigenfunctions.shape[0] != pos_absolute.shape[0]:
    raise ValueError('eigenfunctionsの点数がmock1 catalogと一致しない。')
if eigenfunctions.shape[1] < MAX_DIFFUSION_MODES:
    raise ValueError('geometry cacheの固有関数数が不足している。')
if generator_eigenvalues.size < MAX_DIFFUSION_MODES:
    raise ValueError('generator_eigenvaluesの個数が不足している。')
if geometry_bandwidth.shape != (pos_absolute.shape[0],):
    raise ValueError('high-mode geometry bandwidthのshapeが不正である。')
if stationary_measure.shape != (pos_absolute.shape[0],):
    raise ValueError('stationary_measureのshapeが不正である。')
if not np.isclose(np.sum(stationary_measure), 1.0, rtol=1.0e-10, atol=1.0e-12):
    raise ValueError('stationary_measureが1に規格化されていない。')
if cached_local_metric.shape != (pos_absolute.shape[0], 3, 3):
    raise ValueError('legacy local_metricのshapeが不正である。')

n_low_audit = min(512, legacy_eigenfunctions.shape[1], eigenfunctions.shape[1])
n_subspace_audit = min(64, n_low_audit)
weighted_cross = (
    legacy_eigenfunctions[:, :n_subspace_audit].T
    @ (stationary_measure[:, None] * eigenfunctions[:, :n_subspace_audit])
)
low_mode_correlations = np.abs(np.diag(weighted_cross))
subspace_singular_values = linalg.svdvals(weighted_cross)
geometry_continuity_df = pd.DataFrame(
    [
        {
            'highmode_cache': str(GEOMETRY_CACHE_FILE),
            'legacy_cache': str(LEGACY_GEOMETRY_CACHE_FILE),
            'audited_low_modes': int(n_low_audit),
            'audited_subspace_modes': int(n_subspace_audit),
            'bandwidth_max_abs_difference': float(
                np.max(np.abs(geometry_bandwidth - legacy_bandwidth))
            ),
            'generator_max_abs_difference': float(
                np.max(
                    np.abs(
                        generator_eigenvalues[:n_low_audit]
                        - legacy_generator_eigenvalues[:n_low_audit]
                    )
                )
            ),
            'first64_mode_correlation_median': float(
                np.median(low_mode_correlations)
            ),
            'first64_mode_correlation_minimum': float(
                np.min(low_mode_correlations)
            ),
            'first64_subspace_minimum_singular_value': float(
                np.min(subspace_singular_values)
            ),
        }
    ]
)

positions = pos_absolute - SPHERE_CENTER[None, :]
radius = np.linalg.norm(positions, axis=1)
line_of_sight_all = np.zeros_like(positions)
nonzero_radius = radius > 0.0
line_of_sight_all[nonzero_radius] = (
    positions[nonzero_radius] / radius[nonzero_radius, None]
)
if not np.all(nonzero_radius):
    warnings.warn(
        f'球中心と一致する点が{np.sum(~nonzero_radius)}個ある。'
        'その点のradial design rowは0として扱う。'
    )

mode_pool = np.arange(MAX_DIFFUSION_MODES, dtype=np.int64)
constant_mode_index = int(np.argmin(generator_eigenvalues[:MAX_DIFFUSION_MODES]))
active_mode_indices = mode_pool[mode_pool != constant_mode_index]
if constant_mode_index != 0:
    warnings.warn(
        f'最小generator eigenvalueのmode indexが0ではなく{constant_mode_index}である。'
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
if CAPACITY_REFERENCE_PARAMETER_COUNT not in POTENTIAL_BASIS_SIZES:
    raise ValueError(
        'CAPACITY_REFERENCE_PARAMETER_COUNTはcandidate basisに含める必要がある。'
    )
max_potential_basis = N_AFFINE_POTENTIAL_MODES + max_diffusion_potential_modes
active_generator_values = generator_eigenvalues[active_mode_indices]
potential_generator_values = np.concatenate(
    [
        np.zeros(N_AFFINE_POTENTIAL_MODES, dtype=np.float64),
        active_generator_values,
    ]
)

print('=' * 124)
print('Mock-1 potential-flow measurement-noise seed robustness: Stage 1b')
print('=' * 124)
print(f'Python / NumPy / SciPy : {platform.python_version()} / {np.__version__} / {scipy.__version__}')
print(f'Objects                : {positions.shape[0]:,}')
print(f'Seed pairs             : {SEED_PAIRS}')
print(f'Reference scale        : {SMOOTHING_CONFIG["scale"]}')
print(f'Noise fractions        : {NOISE_FRACTIONS}')
print(f'Noise realizations     : {N_NOISE_REALIZATIONS} per positive level and seed pair')
print(f'Diffusion mode counts  : {POTENTIAL_DIFFUSION_MODE_COUNTS}')
print(f'Potential parameters   : {POTENTIAL_BASIS_SIZES}')
print(f'Maximum basis          : {max_potential_basis}')
print(f'Resume                 : {RESUME_FROM_CHECKPOINT and not FORCE_RECOMPUTE}')
print(f'Input                  : {DATA_FILE}')
print(f'Full geometry          : {GEOMETRY_CACHE_FILE}')
print(f'Gradient cache         : {GRADIENT_CACHE_FILE}')
print(f'Previous H2            : {PREVIOUS_H2_DIR}')
print(f'Output                 : {OUTPUT_DIR}')
print('-' * 124)
print('Full-geometry versus legacy geometry continuity audit')
display(geometry_continuity_df)

start_time = time.perf_counter()

# ============================================================
# 9. Global gradient basis and affine residualization
# ============================================================

gradient_result = build_or_load_gradient_basis(
    positions=positions,
    eigenfunctions=eigenfunctions,
    cached_bandwidth=geometry_bandwidth,
    cached_metric=cached_local_metric,
    mode_indices=mode_pool,
)
gradient_all_modes = gradient_result['gradient_basis']
if constant_mode_index == 0:
    gradient_active_all = gradient_all_modes[:, 1:MAX_DIFFUSION_MODES, :]
else:
    gradient_active_all = gradient_all_modes[:, active_mode_indices, :]

constant_gradient = gradient_all_modes[:, constant_mode_index, :]
constant_gradient_rms = float(
    np.sqrt(np.mean(np.sum(np.square(constant_gradient), axis=1)))
)

line_of_sight_valid = line_of_sight_all[nonzero_radius]
affine_radial_gram = line_of_sight_valid.T @ line_of_sight_valid
diffusion_radial_all = np.einsum(
    'ic,imc->im',
    line_of_sight_valid,
    gradient_active_all[nonzero_radius],
    optimize=True,
)
affine_radial_rhs = line_of_sight_valid.T @ diffusion_radial_all
affine_projection_vectors = linalg.solve(
    affine_radial_gram,
    affine_radial_rhs,
    assume_a='sym',
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
del diffusion_radial_all, diffusion_radial_residual
# The constant-gradient array is no longer needed, but the active full-catalog
# gradient basis is retained and reused sequentially for all seed pairs.
del gradient_all_modes
gc.collect()

gradient_diagnostics_df = pd.DataFrame(
    [
        {
            'constant_mode_index': constant_mode_index,
            'constant_gradient_rms': constant_gradient_rms,
            'affine_projection_relative_orthogonality': (
                affine_projection_relative_orthogonality
            ),
            'loaded_from_cache': bool(gradient_result['loaded_from_cache']),
            **{
                key: value
                for key, value in gradient_result['diagnostics'].items()
                if not isinstance(value, (list, dict))
            },
        }
    ]
)


# ============================================================
# 10. Seed-specific preparation and checkpoint utilities
# ============================================================


def seed_tag(seed_pair_index: int, reference_seed: int, label_seed: int) -> str:
    return (
        f'seedpair{seed_pair_index:02d}_'
        f'ref{reference_seed}_label{label_seed}'
    )


def checkpoint_paths(seed_pair_index: int, reference_seed: int, label_seed: int):
    tag = seed_tag(seed_pair_index, reference_seed, label_seed)
    return {
        'candidate': CHECKPOINT_DIR / f'{tag}_candidate.csv',
        'capacity': CHECKPOINT_DIR / f'{tag}_capacity_runs.csv',
        'closure': CHECKPOINT_DIR / f'{tag}_closure.csv',
        'noise_design': CHECKPOINT_DIR / f'{tag}_noise_design.csv',
        'metadata': CHECKPOINT_DIR / f'{tag}_metadata.json',
        'final': CHECKPOINT_DIR / f'{tag}_final_runs.csv',
        'predictions': PREDICTION_DIR / f'{tag}_predictions.npz',
    }


def seed_checkpoint_signature(
    seed_pair_index: int,
    reference_seed: int,
    label_seed: int,
) -> str:
    payload = {
        'data': file_signature(DATA_FILE),
        'geometry': file_signature(GEOMETRY_CACHE_FILE),
        'legacy_geometry': file_signature(LEGACY_GEOMETRY_CACHE_FILE),
        'seed_pair_index': int(seed_pair_index),
        'reference_seed': int(reference_seed),
        'label_seed': int(label_seed),
        'smoothing_config': SMOOTHING_CONFIG,
        'noise_fractions': NOISE_FRACTIONS,
        'n_noise_realizations': N_NOISE_REALIZATIONS,
        'noise_seed_base': NOISE_SEED_BASE,
        'noise_seed_stride': NOISE_SEED_STRIDE,
        'potential_basis_sizes': POTENTIAL_BASIS_SIZES,
        'regularization_strengths': POTENTIAL_REGULARIZATION_STRENGTHS,
        'regularization_powers': POTENTIAL_REGULARIZATION_POWERS,
        'max_diffusion_modes': MAX_DIFFUSION_MODES,
        'metric_relative_floor': METRIC_EIGENVALUE_RELATIVE_FLOOR,
        'metric_absolute_floor': METRIC_EIGENVALUE_ABSOLUTE_FLOOR,
    }
    return stable_signature(payload)


def checkpoint_is_current(paths: dict, signature: str) -> bool:
    required = [
        paths['candidate'],
        paths['capacity'],
        paths['closure'],
        paths['noise_design'],
        paths['metadata'],
    ]
    if not all(path.is_file() for path in required):
        return False
    try:
        metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    except Exception:
        return False
    return str(metadata.get('signature', '')) == str(signature)


def prepare_seed_context(seed_pair_index: int, reference_seed: int, label_seed: int):
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
    train_local = split['train']
    validation_local = split['validation']
    test_local = split['test']
    fit_local = np.sort(np.concatenate([train_local, validation_local]))

    reference_fields = prepare_reference_fields(
        positions,
        vel_raw,
        reference_indices,
        model_indices,
        reference_seed,
        SMOOTHING_CONFIGS,
    )
    scale = SMOOTHING_CONFIG['scale']
    reference_result = reference_fields[scale]
    velocity_model_true = np.asarray(
        reference_result['bulk_velocity'], dtype=np.float64
    )

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
    del affine_gradient_model, diffusion_gradient_model

    line_of_sight_model = line_of_sight_all[model_indices]
    radial_design_model = potential_radial_design(
        gradient_model,
        line_of_sight_model,
    )
    radial_true_model = np.einsum(
        'ij,ij->i', line_of_sight_model, velocity_model_true
    )

    train_radial_rms = float(
        np.sqrt(np.mean(np.square(radial_true_model[train_local])))
    )
    if train_radial_rms <= 0.0:
        raise RuntimeError('Training radial RMSが正でない。')

    run_metadata = []
    observed_model_columns = []
    seed_offset = int(seed_pair_index) * NOISE_SEED_STRIDE
    for noise_fraction in NOISE_FRACTIONS:
        n_realizations = (
            1 if np.isclose(noise_fraction, 0.0) else N_NOISE_REALIZATIONS
        )
        sigma_noise = float(noise_fraction) * train_radial_rms
        for realization in range(n_realizations):
            noise_seed = NOISE_SEED_BASE + seed_offset + realization
            if sigma_noise == 0.0:
                noise_model = np.zeros(model_indices.size, dtype=np.float64)
            else:
                rng = np.random.default_rng(noise_seed)
                noise_model = sigma_noise * rng.normal(size=model_indices.size)
            observed_model = radial_true_model + noise_model
            seed_run_index = len(run_metadata)
            run_key = (
                f'{seed_tag(seed_pair_index, reference_seed, label_seed)}_'
                f'noise{noise_fraction_token(noise_fraction)}_'
                f'rep{realization:03d}_seed{noise_seed}'
            )
            run_metadata.append(
                {
                    'seed_pair_index': int(seed_pair_index),
                    'seed_run_index': int(seed_run_index),
                    'run_key': run_key,
                    'reference_seed': int(reference_seed),
                    'label_seed': int(label_seed),
                    'scale': scale,
                    'noise_fraction': float(noise_fraction),
                    'noise_sigma': sigma_noise,
                    'noise_realization': int(realization),
                    'noise_seed': int(noise_seed),
                    'train_radial_rms': train_radial_rms,
                }
            )
            observed_model_columns.append(observed_model)

    run_metadata_df = pd.DataFrame(run_metadata)
    observed_model_matrix = np.column_stack(observed_model_columns)

    return {
        'seed_pair_index': int(seed_pair_index),
        'reference_seed': int(reference_seed),
        'label_seed': int(label_seed),
        'scale': scale,
        'reference_indices': reference_indices,
        'model_indices': model_indices,
        'train_local': train_local,
        'validation_local': validation_local,
        'test_local': test_local,
        'fit_local': fit_local,
        'velocity_model_true': velocity_model_true,
        'velocity_train_true': velocity_model_true[train_local],
        'velocity_validation_true': velocity_model_true[validation_local],
        'velocity_test_true': velocity_model_true[test_local],
        'line_of_sight_model': line_of_sight_model,
        'line_of_sight_train': line_of_sight_model[train_local],
        'line_of_sight_validation': line_of_sight_model[validation_local],
        'line_of_sight_test': line_of_sight_model[test_local],
        'gradient_model': gradient_model,
        'gradient_train': gradient_model[train_local],
        'gradient_validation': gradient_model[validation_local],
        'gradient_test': gradient_model[test_local],
        'radial_design_model': radial_design_model,
        'A_train_max': radial_design_model[train_local, :max_potential_basis],
        'A_validation_max': radial_design_model[validation_local, :max_potential_basis],
        'A_test_max': radial_design_model[test_local, :max_potential_basis],
        'A_fit_max': radial_design_model[fit_local, :max_potential_basis],
        'radial_true_model': radial_true_model,
        'radial_train_true': radial_true_model[train_local],
        'radial_validation_true': radial_true_model[validation_local],
        'radial_test_true': radial_true_model[test_local],
        'run_metadata_df': run_metadata_df,
        'observed_model_matrix': observed_model_matrix,
        'observed_train_matrix': observed_model_matrix[train_local],
        'observed_validation_matrix': observed_model_matrix[validation_local],
        'observed_test_matrix': observed_model_matrix[test_local],
        'observed_fit_matrix': observed_model_matrix[fit_local],
        'train_radial_rms': train_radial_rms,
        'affine_mode_scale': affine_mode_scale,
        'reference_bandwidth_median': float(
            reference_result.get('bandwidth_median', math.nan)
        ),
    }


def release_seed_context(context: dict):
    context.clear()
    gc.collect()


def maximum_basis_closure(context: dict) -> pd.DataFrame:
    if not RUN_MAX_BASIS_CLOSURE:
        return pd.DataFrame(
            [
                {
                    'seed_pair_index': context['seed_pair_index'],
                    'reference_seed': context['reference_seed'],
                    'label_seed': context['label_seed'],
                    'n_basis': max_potential_basis,
                    'closure_status': 'skipped',
                }
            ]
        )
    gradient_model = context['gradient_model']
    fit_local = context['fit_local']
    test_local = context['test_local']
    A_fit_max = context['A_fit_max']
    rng = np.random.default_rng(
        int(context['reference_seed']) * 1000003
        + int(context['label_seed'])
    )
    column_rms = np.sqrt(
        np.mean(
            np.sum(
                np.square(gradient_model[:, :max_potential_basis, :]),
                axis=2,
            ),
            axis=0,
        )
    )
    coefficient_true = rng.normal(size=max_potential_basis) / np.maximum(
        column_rms, 1.0e-12
    )
    velocity_closure = potential_velocity(
        gradient_model,
        coefficient_true,
        max_potential_basis,
    )
    target_rms = float(
        np.sqrt(
            np.mean(
                np.sum(np.square(context['velocity_model_true']), axis=1)
            )
        )
    )
    closure_rms = float(
        np.sqrt(np.mean(np.sum(np.square(velocity_closure), axis=1)))
    )
    if closure_rms > 0.0:
        coefficient_true *= target_rms / closure_rms
        velocity_closure = potential_velocity(
            gradient_model,
            coefficient_true,
            max_potential_basis,
        )
    radial_closure = np.einsum(
        'ij,ij->i', context['line_of_sight_model'], velocity_closure
    )
    coefficient_estimated, _, rank, singular_values = linalg.lstsq(
        A_fit_max,
        radial_closure[fit_local],
        cond=None,
        lapack_driver='gelsd',
        check_finite=False,
    )
    prediction_test = potential_velocity(
        context['gradient_test'],
        coefficient_estimated,
        max_potential_basis,
    )
    metrics = vector_field_metrics(
        velocity_closure[test_local],
        prediction_test,
        context['line_of_sight_test'],
    )
    coefficient_error = float(
        np.linalg.norm(coefficient_estimated - coefficient_true)
        / max(np.linalg.norm(coefficient_true), np.finfo(float).eps)
    )
    condition_number = (
        float(singular_values[0] / singular_values[-1])
        if singular_values.size and singular_values[-1] > 0.0
        else math.inf
    )
    status = (
        'pass'
        if int(rank) == max_potential_basis
        and coefficient_error < CLOSURE_PASS_TOLERANCE
        and metrics['nrmse_3d'] < CLOSURE_PASS_TOLERANCE
        else 'fail'
    )
    return pd.DataFrame(
        [
            {
                'seed_pair_index': context['seed_pair_index'],
                'reference_seed': context['reference_seed'],
                'label_seed': context['label_seed'],
                'n_basis': max_potential_basis,
                'n_observations': int(A_fit_max.shape[0]),
                'numerical_rank': int(rank),
                'nullity': int(max_potential_basis - rank),
                'condition_number': condition_number,
                'coefficient_relative_error': coefficient_error,
                **metrics,
                'closure_status': status,
            }
        ]
    )


def scan_seed_pair(context: dict):
    run_metadata_df = context['run_metadata_df']
    observed_train_matrix = context['observed_train_matrix']
    observed_validation_matrix = context['observed_validation_matrix']
    observed_fit_matrix = context['observed_fit_matrix']
    A_train_max = context['A_train_max']
    A_validation_max = context['A_validation_max']
    A_fit_max = context['A_fit_max']
    gradient_validation = context['gradient_validation']
    radial_validation_true = context['radial_validation_true']
    velocity_validation_true = context['velocity_validation_true']
    train_local = context['train_local']
    validation_local = context['validation_local']
    fit_local = context['fit_local']

    n_runs = int(run_metadata_df.shape[0])
    train_gram_max = A_train_max.T @ A_train_max / float(train_local.size)
    train_rhs_all = A_train_max.T @ observed_train_matrix / float(train_local.size)
    fit_gram_max = A_fit_max.T @ A_fit_max / float(fit_local.size)
    fit_rhs_all = A_fit_max.T @ observed_fit_matrix / float(fit_local.size)

    G3_validation_max = vector_design_matrix(
        gradient_validation[:, :max_potential_basis, :]
    )
    validation_vector_true_flat = velocity_validation_true.reshape(-1)
    validation_vector_gram_sum = G3_validation_max.T @ G3_validation_max
    validation_vector_rhs_true = (
        G3_validation_max.T @ validation_vector_true_flat
    )
    validation_vector_true_sum_squares = float(
        validation_vector_true_flat @ validation_vector_true_flat
    )
    validation_radial_true_sum_squares = float(
        radial_validation_true @ radial_validation_true
    )
    validation_tangential_true_sum_squares = max(
        validation_vector_true_sum_squares
        - validation_radial_true_sum_squares,
        np.finfo(float).eps,
    )

    candidate_records = []
    for n_basis in POTENTIAL_BASIS_SIZES:
        gram = train_gram_max[:n_basis, :n_basis]
        rhs_all = train_rhs_all[:n_basis]
        A_validation = A_validation_max[:, :n_basis]
        vector_gram_sum = validation_vector_gram_sum[:n_basis, :n_basis]
        vector_rhs_true = validation_vector_rhs_true[:n_basis]
        for penalty_power in POTENTIAL_REGULARIZATION_POWERS:
            regularization = regularization_diagonal(
                potential_generator_values,
                n_basis,
                penalty_power,
            )
            for strength in POTENTIAL_REGULARIZATION_STRENGTHS:
                factor_kind, factor, _ = factor_regularized_system(
                    gram,
                    regularization,
                    strength,
                )
                coefficients_all = solve_factored(
                    factor_kind,
                    factor,
                    rhs_all,
                )
                validation_prediction = A_validation @ coefficients_all
                observed_residual = (
                    validation_prediction - observed_validation_matrix
                )
                observed_sse = np.sum(np.square(observed_residual), axis=0)
                observed_denominator = np.sum(
                    np.square(observed_validation_matrix), axis=0
                )
                validation_observed_nrmse = np.sqrt(
                    observed_sse
                    / np.maximum(
                        observed_denominator,
                        np.finfo(float).eps,
                    )
                )
                latent_radial_residual = (
                    validation_prediction
                    - radial_validation_true[:, None]
                )
                latent_radial_sse = np.sum(
                    np.square(latent_radial_residual), axis=0
                )
                validation_latent_radial_nrmse = np.sqrt(
                    latent_radial_sse
                    / max(
                        validation_radial_true_sum_squares,
                        np.finfo(float).eps,
                    )
                )
                (
                    validation_hidden_3d_nrmse,
                    validation_hidden_3d_sse,
                ) = hidden_vector_nrmse_from_quadratic(
                    coefficients_all,
                    vector_gram_sum,
                    vector_rhs_true,
                    validation_vector_true_sum_squares,
                )
                validation_hidden_tangential_sse = np.maximum(
                    validation_hidden_3d_sse - latent_radial_sse,
                    0.0,
                )
                validation_hidden_tangential_nrmse = np.sqrt(
                    validation_hidden_tangential_sse
                    / validation_tangential_true_sum_squares
                )

                for seed_run_index, metadata in run_metadata_df.iterrows():
                    sigma_noise = float(metadata['noise_sigma'])
                    if sigma_noise > 0.0:
                        reduced_chi2 = float(
                            observed_sse[seed_run_index]
                            / (validation_local.size * sigma_noise**2)
                        )
                        selection_score = reduced_chi2
                    else:
                        reduced_chi2 = math.nan
                        selection_score = float(
                            validation_observed_nrmse[seed_run_index]
                        )
                    candidate_records.append(
                        {
                            **metadata.to_dict(),
                            'n_basis': int(n_basis),
                            'n_affine_modes': N_AFFINE_POTENTIAL_MODES,
                            'n_diffusion_modes': int(
                                n_basis - N_AFFINE_POTENTIAL_MODES
                            ),
                            'regularization_strength': float(strength),
                            'penalty_power': float(penalty_power),
                            'validation_selection_score': selection_score,
                            'validation_observed_nrmse': float(
                                validation_observed_nrmse[seed_run_index]
                            ),
                            'validation_reduced_chi2': reduced_chi2,
                            'validation_latent_nrmse_radial': float(
                                validation_latent_radial_nrmse[seed_run_index]
                            ),
                            'validation_hidden_nrmse_tangential': float(
                                validation_hidden_tangential_nrmse[seed_run_index]
                            ),
                            'validation_hidden_nrmse_3d': float(
                                validation_hidden_3d_nrmse[seed_run_index]
                            ),
                        }
                    )

    candidate_df = pd.DataFrame(candidate_records)
    within_basis_operational_df = (
        candidate_df.sort_values(
            [
                'seed_run_index',
                'n_basis',
                'validation_selection_score',
                'regularization_strength',
                'penalty_power',
            ],
            kind='mergesort',
        )
        .groupby(['seed_run_index', 'n_basis'], as_index=False)
        .first()
    )
    within_basis_hidden_df = (
        candidate_df.sort_values(
            [
                'seed_run_index',
                'n_basis',
                'validation_hidden_nrmse_3d',
                'regularization_strength',
                'penalty_power',
            ],
            kind='mergesort',
        )
        .groupby(['seed_run_index', 'n_basis'], as_index=False)
        .first()
    )
    hidden_lookup = within_basis_hidden_df.set_index(
        ['seed_run_index', 'n_basis']
    )

    factor_cache = {}
    capacity_records = []
    for _, selected in within_basis_operational_df.iterrows():
        seed_run_index = int(selected['seed_run_index'])
        n_basis = int(selected['n_basis'])
        strength = float(selected['regularization_strength'])
        penalty_power = float(selected['penalty_power'])
        key = (n_basis, strength, penalty_power)
        regularization = regularization_diagonal(
            potential_generator_values,
            n_basis,
            penalty_power,
        )
        if key not in factor_cache:
            factor_cache[key] = factor_regularized_system(
                fit_gram_max[:n_basis, :n_basis],
                regularization,
                strength,
            )
        factor_kind, factor, _ = factor_cache[key]
        coefficients = solve_factored(
            factor_kind,
            factor,
            fit_rhs_all[:n_basis, seed_run_index],
        )
        prediction_test = potential_velocity(
            context['gradient_test'],
            coefficients,
            n_basis,
        )
        metrics = vector_field_metrics(
            context['velocity_test_true'],
            prediction_test,
            context['line_of_sight_test'],
        )

        hidden_selected = hidden_lookup.loc[(seed_run_index, n_basis)]
        hidden_strength = float(
            hidden_selected['regularization_strength']
        )
        hidden_power = float(hidden_selected['penalty_power'])
        hidden_key = (n_basis, hidden_strength, hidden_power)
        hidden_regularization = regularization_diagonal(
            potential_generator_values,
            n_basis,
            hidden_power,
        )
        if hidden_key not in factor_cache:
            factor_cache[hidden_key] = factor_regularized_system(
                fit_gram_max[:n_basis, :n_basis],
                hidden_regularization,
                hidden_strength,
            )
        hidden_kind, hidden_factor, _ = factor_cache[hidden_key]
        hidden_coefficients = solve_factored(
            hidden_kind,
            hidden_factor,
            fit_rhs_all[:n_basis, seed_run_index],
        )
        hidden_prediction_test = potential_velocity(
            context['gradient_test'],
            hidden_coefficients,
            n_basis,
        )
        hidden_metrics = vector_field_metrics(
            context['velocity_test_true'],
            hidden_prediction_test,
            context['line_of_sight_test'],
        )
        capacity_records.append(
            {
                **selected.to_dict(),
                **{f'test_{key_name}': value for key_name, value in metrics.items()},
                'within_basis_hidden_regularization_strength': hidden_strength,
                'within_basis_hidden_penalty_power': hidden_power,
                'within_basis_selector_same_lambda_power': bool(
                    np.isclose(strength, hidden_strength)
                    and np.isclose(penalty_power, hidden_power)
                ),
                'test_hidden_selector_nrmse_3d': hidden_metrics['nrmse_3d'],
                'delta_test_nrmse_3d_operational_minus_hidden_selector': (
                    metrics['nrmse_3d'] - hidden_metrics['nrmse_3d']
                ),
                'effective_degrees_of_freedom': (
                    exact_effective_degrees_of_freedom(
                        fit_gram_max[:n_basis, :n_basis],
                        regularization,
                        strength,
                    )
                ),
            }
        )

    capacity_df = pd.DataFrame(capacity_records)
    capacity_df['effective_degrees_of_freedom_fraction'] = (
        capacity_df['effective_degrees_of_freedom']
        / capacity_df['n_basis']
    )
    return candidate_df, capacity_df


# ============================================================
# 11. Candidate scan across seed pairs
# ============================================================

all_candidate_frames = []
all_capacity_frames = []
all_closure_frames = []
all_noise_design_frames = []
seed_scan_records = []

for seed_pair_index, (reference_seed, label_seed) in enumerate(SEED_PAIRS):
    tag = seed_tag(seed_pair_index, reference_seed, label_seed)
    paths = checkpoint_paths(seed_pair_index, reference_seed, label_seed)
    signature = seed_checkpoint_signature(
        seed_pair_index,
        reference_seed,
        label_seed,
    )
    use_checkpoint = (
        RESUME_FROM_CHECKPOINT
        and not FORCE_RECOMPUTE
        and checkpoint_is_current(paths, signature)
    )
    print('-' * 124)
    print(
        f'Seed pair {seed_pair_index + 1}/{len(SEED_PAIRS)}: '
        f'reference={reference_seed}, label={label_seed}'
    )
    if use_checkpoint:
        print(f'[Checkpoint] 再利用する: {tag}')
        candidate_seed = pd.read_csv(paths['candidate'])
        capacity_seed = pd.read_csv(paths['capacity'])
        closure_seed = pd.read_csv(paths['closure'])
        noise_design_seed = pd.read_csv(paths['noise_design'])
    else:
        context = prepare_seed_context(
            seed_pair_index,
            reference_seed,
            label_seed,
        )
        print(
            f"[Seed context] train/validation/test = "
            f"{context['train_local'].size}/"
            f"{context['validation_local'].size}/"
            f"{context['test_local'].size}; "
            f"radial RMS={context['train_radial_rms']:.6g}"
        )
        closure_seed = maximum_basis_closure(context)
        candidate_seed, capacity_seed = scan_seed_pair(context)
        noise_design_seed = (
            context['run_metadata_df']
            .groupby(
                [
                    'seed_pair_index',
                    'reference_seed',
                    'label_seed',
                    'noise_fraction',
                ],
                as_index=False,
            )
            .agg(
                n_runs=('seed_run_index', 'size'),
                noise_sigma=('noise_sigma', 'first'),
                train_radial_rms=('train_radial_rms', 'first'),
            )
        )
        candidate_seed.to_csv(paths['candidate'], index=False)
        capacity_seed.to_csv(paths['capacity'], index=False)
        closure_seed.to_csv(paths['closure'], index=False)
        noise_design_seed.to_csv(paths['noise_design'], index=False)
        paths['metadata'].write_text(
            json.dumps(
                json_ready(
                    {
                        'signature': signature,
                        'seed_pair_index': seed_pair_index,
                        'reference_seed': reference_seed,
                        'label_seed': label_seed,
                        'created_at_unix': time.time(),
                    }
                ),
                indent=2,
                ensure_ascii=False,
            ),
            encoding='utf-8',
        )
        release_seed_context(context)

    all_candidate_frames.append(candidate_seed)
    all_capacity_frames.append(capacity_seed)
    all_closure_frames.append(closure_seed)
    all_noise_design_frames.append(noise_design_seed)
    seed_scan_records.append(
        {
            'seed_pair_index': seed_pair_index,
            'reference_seed': reference_seed,
            'label_seed': label_seed,
            'checkpoint_reused': use_checkpoint,
            'candidate_rows': int(candidate_seed.shape[0]),
            'capacity_rows': int(capacity_seed.shape[0]),
            'closure_status': str(closure_seed['closure_status'].iloc[0]),
        }
    )

candidate_df = pd.concat(all_candidate_frames, ignore_index=True)
capacity_run_df = pd.concat(all_capacity_frames, ignore_index=True)
closure_df = pd.concat(all_closure_frames, ignore_index=True)
noise_design_df = pd.concat(all_noise_design_frames, ignore_index=True)
seed_scan_df = pd.DataFrame(seed_scan_records)

if np.any(closure_df['closure_status'].astype(str) == 'fail'):
    warnings.warn('少なくとも一つのseed pairでmaximum-basis closureがfailした。')

raw_operational_selected_df = (
    candidate_df.sort_values(
        [
            'run_key',
            'validation_selection_score',
            'n_basis',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    )
    .groupby('run_key', as_index=False)
    .first()
)
hidden_selected_df = (
    candidate_df.sort_values(
        [
            'run_key',
            'validation_hidden_nrmse_3d',
            'n_basis',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    )
    .groupby('run_key', as_index=False)
    .first()
)
within_basis_selected_df = (
    candidate_df.sort_values(
        [
            'run_key',
            'n_basis',
            'validation_selection_score',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    )
    .groupby(['run_key', 'n_basis'], as_index=False)
    .first()
)

# ============================================================
# 12. Nested capacity aggregation and combined practical rule
# ============================================================

seed_capacity_mean_df = (
    capacity_run_df.groupby(
        ['noise_fraction', 'n_basis', 'seed_pair_index'],
        as_index=False,
    )
    .agg(
        validation_observed_nrmse_seed_mean=(
            'validation_observed_nrmse', 'mean'
        ),
        test_hidden_nrmse_3d_seed_mean=('test_nrmse_3d', 'mean'),
        test_latent_nrmse_radial_seed_mean=('test_nrmse_radial', 'mean'),
        test_hidden_nrmse_tangential_seed_mean=(
            'test_nrmse_tangential', 'mean'
        ),
        effective_degrees_of_freedom_seed_mean=(
            'effective_degrees_of_freedom', 'mean'
        ),
        selector_agreement_seed_mean=(
            'within_basis_selector_same_lambda_power', 'mean'
        ),
    )
)

capacity_records = []
for (noise_fraction, n_basis), group in seed_capacity_mean_df.groupby(
    ['noise_fraction', 'n_basis'], sort=True
):
    validation_values = group[
        'validation_observed_nrmse_seed_mean'
    ].to_numpy(dtype=np.float64)
    hidden_values = group[
        'test_hidden_nrmse_3d_seed_mean'
    ].to_numpy(dtype=np.float64)
    radial_values = group[
        'test_latent_nrmse_radial_seed_mean'
    ].to_numpy(dtype=np.float64)
    tangential_values = group[
        'test_hidden_nrmse_tangential_seed_mean'
    ].to_numpy(dtype=np.float64)
    edf_values = group[
        'effective_degrees_of_freedom_seed_mean'
    ].to_numpy(dtype=np.float64)
    agreement_values = group[
        'selector_agreement_seed_mean'
    ].to_numpy(dtype=np.float64)
    n_seeds = int(group.shape[0])
    capacity_records.append(
        {
            'noise_fraction': float(noise_fraction),
            'n_basis': int(n_basis),
            'n_seed_pairs': n_seeds,
            'validation_observed_nrmse_mean': float(
                np.mean(validation_values)
            ),
            'validation_observed_nrmse_std_between_seeds': float(
                np.std(validation_values, ddof=1)
                if n_seeds > 1 else 0.0
            ),
            'validation_observed_nrmse_sem_between_seeds': float(
                np.std(validation_values, ddof=1) / np.sqrt(n_seeds)
                if n_seeds > 1 else 0.0
            ),
            'test_hidden_nrmse_3d_mean': float(np.mean(hidden_values)),
            'test_hidden_nrmse_3d_std_between_seeds': float(
                np.std(hidden_values, ddof=1)
                if n_seeds > 1 else 0.0
            ),
            'test_latent_nrmse_radial_mean': float(np.mean(radial_values)),
            'test_hidden_nrmse_tangential_mean': float(
                np.mean(tangential_values)
            ),
            'effective_degrees_of_freedom_mean': float(np.mean(edf_values)),
            'within_basis_selector_agreement_mean': float(
                np.mean(agreement_values)
            ),
        }
    )
capacity_aggregate_df = pd.DataFrame(capacity_records).sort_values(
    ['noise_fraction', 'n_basis']
)

maximum_parameter_count = int(max(POTENTIAL_BASIS_SIZES))
capacity_selection_records = []
for noise_fraction, curve in capacity_aggregate_df.groupby(
    'noise_fraction', sort=True
):
    curve = curve.sort_values('n_basis').reset_index(drop=True)
    basis = curve['n_basis'].to_numpy(dtype=np.int64)
    risk = curve['validation_observed_nrmse_mean'].to_numpy(dtype=np.float64)
    sem = curve[
        'validation_observed_nrmse_sem_between_seeds'
    ].to_numpy(dtype=np.float64)
    minimum_index = int(np.nanargmin(risk))
    minimum_basis = int(basis[minimum_index])
    minimum_mean = float(risk[minimum_index])
    minimum_sem = float(sem[minimum_index])
    one_se_threshold = minimum_mean + minimum_sem
    eligible = basis[risk <= one_se_threshold + 1.0e-15]
    one_se_basis = int(np.min(eligible))
    plateau_basis, adjacent_improvement = practical_plateau_basis(
        basis,
        risk,
        PLATEAU_MIN_PARAMETER_COUNT,
        PLATEAU_RELATIVE_THRESHOLD,
        PLATEAU_CONSECUTIVE_STEPS,
    )
    practical_basis = int(
        max(
            one_se_basis,
            plateau_basis if plateau_basis is not None else one_se_basis,
        )
    )
    last_improvement = (
        float(adjacent_improvement[-1])
        if adjacent_improvement.size
        else math.nan
    )
    raw_group = raw_operational_selected_df[
        np.isclose(
            raw_operational_selected_df['noise_fraction'],
            float(noise_fraction),
        )
    ]
    seed_raw_fraction = (
        raw_group.assign(
            selected_max=(
                raw_group['n_basis'].to_numpy(dtype=np.int64)
                == maximum_parameter_count
            ).astype(float)
        )
        .groupby('seed_pair_index', as_index=False)['selected_max']
        .mean()
    )
    maximum_selected_fraction = float(
        seed_raw_fraction['selected_max'].mean()
    )
    strict_argmin_boundary = bool(minimum_basis == maximum_parameter_count)
    one_se_boundary = bool(one_se_basis == maximum_parameter_count)
    practical_converged = bool(
        not one_se_boundary
        and (
            plateau_basis is not None
            or (
                np.isfinite(last_improvement)
                and last_improvement <= PLATEAU_RELATIVE_THRESHOLD
            )
        )
    )
    if practical_converged:
        practical_status = 'converged'
    elif one_se_boundary:
        practical_status = 'one_se_at_geometry_limit'
    elif (
        np.isfinite(last_improvement)
        and last_improvement > PLATEAU_RELATIVE_THRESHOLD
    ):
        practical_status = 'improvement_above_threshold_at_geometry_limit'
    else:
        practical_status = 'unresolved_without_plateau'
    strict_status = (
        'boundary_argmin' if strict_argmin_boundary else 'interior_argmin'
    )
    capacity_selection_records.append(
        {
            'noise_fraction': float(noise_fraction),
            'n_seed_pairs': int(len(SEED_PAIRS)),
            'validation_minimum_basis': minimum_basis,
            'validation_minimum_mean': minimum_mean,
            'validation_minimum_sem_between_seeds': minimum_sem,
            'one_standard_error_threshold': one_se_threshold,
            'one_standard_error_basis': one_se_basis,
            'practical_plateau_basis': (
                int(plateau_basis) if plateau_basis is not None else math.nan
            ),
            'plateau_found': plateau_basis is not None,
            'combined_practical_basis': practical_basis,
            'strict_argmin_boundary': strict_argmin_boundary,
            'one_se_boundary': one_se_boundary,
            'maximum_basis_selected_fraction_equal_seed_weight': (
                maximum_selected_fraction
            ),
            'last_adjacent_validation_improvement': last_improvement,
            'strict_argmin_status': strict_status,
            'practical_convergence_status': practical_status,
            'practical_converged': practical_converged,
            'needs_larger_eigensystem': not practical_converged,
        }
    )
capacity_selection_df = pd.DataFrame(capacity_selection_records).sort_values(
    'noise_fraction'
)

practical_basis_lookup = {
    float(row['noise_fraction']): int(row['combined_practical_basis'])
    for _, row in capacity_selection_df.iterrows()
}

print('-' * 124)
print('Nested capacity selection across seed pairs')
display(capacity_selection_df)
if np.any(~capacity_selection_df['practical_converged'].astype(bool)):
    warnings.warn(
        '少なくとも一つのnoise levelでcombined practical convergenceが確定しなかった。'
    )

# ============================================================
# 13. Representative H2 continuity audit
# ============================================================

previous_candidate_path = (
    PREVIOUS_H2_DIR / f'{PREVIOUS_H2_PREFIX}_candidate_grid.csv'
)
h2_continuity_records = []
if previous_candidate_path.is_file():
    previous_candidate_df = pd.read_csv(previous_candidate_path)
    first_reference, first_label = SEED_PAIRS[0]
    current_first = candidate_df[
        (candidate_df['reference_seed'] == first_reference)
        & (candidate_df['label_seed'] == first_label)
        & candidate_df['noise_fraction'].isin([0.0, 0.2, 0.8])
    ].copy()
    previous_subset = previous_candidate_df[
        previous_candidate_df['noise_fraction'].isin([0.0, 0.2, 0.8])
    ].copy()
    merge_keys = [
        'noise_fraction',
        'noise_realization',
        'n_basis',
        'regularization_strength',
        'penalty_power',
    ]
    common = current_first.merge(
        previous_subset,
        on=merge_keys,
        how='inner',
        suffixes=('_stage1b', '_h2'),
    )
    record = {
        'previous_candidate_available': True,
        'n_common_rows': int(common.shape[0]),
    }
    for column in [
        'validation_selection_score',
        'validation_observed_nrmse',
        'validation_latent_nrmse_radial',
        'validation_hidden_nrmse_3d',
    ]:
        record[f'{column}_max_abs_difference'] = (
            float(
                np.max(
                    np.abs(
                        common[f'{column}_stage1b'].to_numpy(dtype=np.float64)
                        - common[f'{column}_h2'].to_numpy(dtype=np.float64)
                    )
                )
            )
            if common.shape[0]
            else math.nan
        )
    h2_continuity_records.append(record)
else:
    h2_continuity_records.append(
        {
            'previous_candidate_available': False,
            'n_common_rows': 0,
        }
    )
h2_continuity_df = pd.DataFrame(h2_continuity_records)

# ============================================================
# 14. Final evaluation at shared practical bases
# ============================================================


def selected_row_for_basis(seed_candidates, run_key: str, n_basis: int):
    subset = seed_candidates[
        (seed_candidates['run_key'] == run_key)
        & (seed_candidates['n_basis'] == int(n_basis))
    ]
    if subset.empty:
        raise RuntimeError(
            f'run={run_key}, basis={n_basis}のcandidateがない。'
        )
    return subset.sort_values(
        [
            'validation_selection_score',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    ).iloc[0]


def selected_hidden_row(seed_candidates, run_key: str):
    subset = seed_candidates[seed_candidates['run_key'] == run_key]
    return subset.sort_values(
        [
            'validation_hidden_nrmse_3d',
            'n_basis',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    ).iloc[0]


def fit_selected_candidate(
    context,
    selected,
    seed_run_index,
    fit_gram_max,
    fit_rhs_all,
    factor_cache,
):
    n_basis = int(selected['n_basis'])
    strength = float(selected['regularization_strength'])
    power = float(selected['penalty_power'])
    key = (n_basis, strength, power)
    regularization = regularization_diagonal(
        potential_generator_values,
        n_basis,
        power,
    )
    if key not in factor_cache:
        factor_cache[key] = factor_regularized_system(
            fit_gram_max[:n_basis, :n_basis],
            regularization,
            strength,
        )
    factor_kind, factor, system = factor_cache[key]
    coefficients = solve_factored(
        factor_kind,
        factor,
        fit_rhs_all[:n_basis, seed_run_index],
    )
    prediction_test = potential_velocity(
        context['gradient_test'],
        coefficients,
        n_basis,
    )
    radial_prediction = (
        context['A_test_max'][:, :n_basis] @ coefficients
    )
    return {
        'n_basis': n_basis,
        'strength': strength,
        'power': power,
        'regularization': regularization,
        'factor_kind': factor_kind,
        'factor': factor,
        'system': system,
        'coefficients': coefficients,
        'prediction_test': prediction_test,
        'radial_prediction_test': radial_prediction,
    }


all_final_frames = []
for seed_pair_index, (reference_seed, label_seed) in enumerate(SEED_PAIRS):
    paths = checkpoint_paths(seed_pair_index, reference_seed, label_seed)
    signature = seed_checkpoint_signature(
        seed_pair_index,
        reference_seed,
        label_seed,
    )
    final_signature = stable_signature(
        {
            'seed_signature': signature,
            'practical_basis_lookup': practical_basis_lookup,
        }
    )
    use_final_checkpoint = False
    if (
        RESUME_FROM_CHECKPOINT
        and not FORCE_RECOMPUTE
        and paths['final'].is_file()
        and paths['metadata'].is_file()
    ):
        try:
            metadata = json.loads(paths['metadata'].read_text(encoding='utf-8'))
            use_final_checkpoint = (
                str(metadata.get('final_signature', '')) == final_signature
            )
        except Exception:
            use_final_checkpoint = False
    if use_final_checkpoint:
        print(
            f'[Final checkpoint] 再利用する: '
            f'{seed_tag(seed_pair_index, reference_seed, label_seed)}'
        )
        final_seed_df = pd.read_csv(paths['final'])
        all_final_frames.append(final_seed_df)
        continue

    context = prepare_seed_context(
        seed_pair_index,
        reference_seed,
        label_seed,
    )
    seed_candidates = candidate_df[
        candidate_df['seed_pair_index'] == seed_pair_index
    ].copy()
    run_metadata_df = context['run_metadata_df']
    fit_local = context['fit_local']
    fit_gram_max = (
        context['A_fit_max'].T @ context['A_fit_max']
        / float(fit_local.size)
    )
    fit_rhs_all = (
        context['A_fit_max'].T @ context['observed_fit_matrix']
        / float(fit_local.size)
    )
    factor_cache = {}
    covariance_cache = {}
    final_records = []
    practical_predictions = np.full(
        (run_metadata_df.shape[0], context['test_local'].size, 3),
        np.nan,
        dtype=np.float64,
    )
    raw_predictions = np.full_like(practical_predictions, np.nan)
    hidden_predictions = np.full_like(practical_predictions, np.nan)

    raw_lookup = raw_operational_selected_df.set_index('run_key')
    hidden_lookup = hidden_selected_df.set_index('run_key')

    for seed_run_index, metadata in run_metadata_df.iterrows():
        run_key = str(metadata['run_key'])
        noise_fraction = float(metadata['noise_fraction'])
        practical_basis = practical_basis_lookup[noise_fraction]
        practical_selected = selected_row_for_basis(
            seed_candidates,
            run_key,
            practical_basis,
        )
        raw_selected = raw_lookup.loc[run_key]
        hidden_selected = hidden_lookup.loc[run_key]

        practical_fit = fit_selected_candidate(
            context,
            practical_selected,
            seed_run_index,
            fit_gram_max,
            fit_rhs_all,
            factor_cache,
        )
        raw_fit = fit_selected_candidate(
            context,
            raw_selected,
            seed_run_index,
            fit_gram_max,
            fit_rhs_all,
            factor_cache,
        )
        hidden_fit = fit_selected_candidate(
            context,
            hidden_selected,
            seed_run_index,
            fit_gram_max,
            fit_rhs_all,
            factor_cache,
        )

        practical_predictions[seed_run_index] = practical_fit['prediction_test']
        raw_predictions[seed_run_index] = raw_fit['prediction_test']
        hidden_predictions[seed_run_index] = hidden_fit['prediction_test']

        practical_metrics = vector_field_metrics(
            context['velocity_test_true'],
            practical_fit['prediction_test'],
            context['line_of_sight_test'],
        )
        raw_metrics = vector_field_metrics(
            context['velocity_test_true'],
            raw_fit['prediction_test'],
            context['line_of_sight_test'],
        )
        hidden_metrics = vector_field_metrics(
            context['velocity_test_true'],
            hidden_fit['prediction_test'],
            context['line_of_sight_test'],
        )

        observed_test = context['observed_test_matrix'][:, seed_run_index]
        sigma_noise = float(metadata['noise_sigma'])
        covariance_key = (
            practical_fit['n_basis'],
            practical_fit['strength'],
            practical_fit['power'],
            sigma_noise,
        )
        if covariance_key not in covariance_cache:
            covariance_cache[covariance_key] = conditional_coefficient_covariance(
                fit_gram_max[
                    :practical_fit['n_basis'],
                    :practical_fit['n_basis'],
                ],
                practical_fit['factor_kind'],
                practical_fit['factor'],
                sigma_noise,
                fit_local.size,
            )
        coefficient_covariance = covariance_cache[covariance_key]
        radial_variance = radial_prediction_variance(
            context['A_test_max'][:, :practical_fit['n_basis']],
            coefficient_covariance,
        )
        component_variance = vector_component_prediction_variance(
            context['gradient_test'][:, :practical_fit['n_basis'], :],
            coefficient_covariance,
        )
        observed_predictive_variance = radial_variance + sigma_noise**2
        coverage = {}
        for label, z_value in COVERAGE_Z.items():
            if sigma_noise == 0.0:
                coverage[f'coverage_latent_radial_{label}'] = math.nan
                coverage[f'coverage_observed_radial_{label}'] = math.nan
                coverage[f'coverage_latent_component_{label}'] = math.nan
            else:
                coverage[f'coverage_latent_radial_{label}'] = (
                    marginal_interval_coverage(
                        context['radial_test_true'],
                        practical_fit['radial_prediction_test'],
                        radial_variance,
                        z_value,
                    )
                )
                coverage[f'coverage_observed_radial_{label}'] = (
                    marginal_interval_coverage(
                        observed_test,
                        practical_fit['radial_prediction_test'],
                        observed_predictive_variance,
                        z_value,
                    )
                )
                coverage[f'coverage_latent_component_{label}'] = (
                    marginal_interval_coverage(
                        context['velocity_test_true'],
                        practical_fit['prediction_test'],
                        component_variance,
                        z_value,
                    )
                )

        predictive_sd = np.sqrt(np.maximum(observed_predictive_variance, 0.0))
        standardized = np.full(context['test_local'].size, np.nan, dtype=np.float64)
        valid = predictive_sd > 0.0
        standardized[valid] = (
            observed_test[valid]
            - practical_fit['radial_prediction_test'][valid]
        ) / predictive_sd[valid]
        finite_z = standardized[np.isfinite(standardized)]

        design_diag = gram_spectrum(
            fit_gram_max[
                :practical_fit['n_basis'],
                :practical_fit['n_basis'],
            ],
            fit_local.size,
        )
        normal_condition, normal_min, normal_max = normal_matrix_condition(
            practical_fit['system']
        )
        effective_df = exact_effective_degrees_of_freedom(
            fit_gram_max[
                :practical_fit['n_basis'],
                :practical_fit['n_basis'],
            ],
            practical_fit['regularization'],
            practical_fit['strength'],
        )

        practical_same_hidden = bool(
            practical_fit['n_basis'] == hidden_fit['n_basis']
            and np.isclose(practical_fit['strength'], hidden_fit['strength'])
            and np.isclose(practical_fit['power'], hidden_fit['power'])
        )
        raw_same_hidden = bool(
            raw_fit['n_basis'] == hidden_fit['n_basis']
            and np.isclose(raw_fit['strength'], hidden_fit['strength'])
            and np.isclose(raw_fit['power'], hidden_fit['power'])
        )

        final_records.append(
            {
                **metadata.to_dict(),
                'selection_policy': 'combined_practical_basis',
                'practical_n_basis': practical_fit['n_basis'],
                'practical_regularization_strength': practical_fit['strength'],
                'practical_penalty_power': practical_fit['power'],
                'practical_validation_selection_score': float(
                    practical_selected['validation_selection_score']
                ),
                'practical_validation_observed_nrmse': float(
                    practical_selected['validation_observed_nrmse']
                ),
                'practical_validation_hidden_nrmse_3d': float(
                    practical_selected['validation_hidden_nrmse_3d']
                ),
                'raw_argmin_n_basis': raw_fit['n_basis'],
                'raw_argmin_regularization_strength': raw_fit['strength'],
                'raw_argmin_penalty_power': raw_fit['power'],
                'hidden_selector_n_basis': hidden_fit['n_basis'],
                'hidden_selector_regularization_strength': hidden_fit['strength'],
                'hidden_selector_penalty_power': hidden_fit['power'],
                'practical_same_hidden_candidate': practical_same_hidden,
                'raw_same_hidden_candidate': raw_same_hidden,
                'test_observed_nrmse_radial': scalar_nrmse(
                    observed_test,
                    practical_fit['radial_prediction_test'],
                ),
                'test_latent_nrmse_radial': scalar_nrmse(
                    context['radial_test_true'],
                    practical_fit['radial_prediction_test'],
                ),
                **{
                    f'test_hidden_{key_name}': value
                    for key_name, value in practical_metrics.items()
                },
                'test_raw_argmin_hidden_nrmse_3d': raw_metrics['nrmse_3d'],
                'test_hidden_selector_nrmse_3d': hidden_metrics['nrmse_3d'],
                'delta_test_nrmse_3d_practical_minus_hidden': (
                    practical_metrics['nrmse_3d'] - hidden_metrics['nrmse_3d']
                ),
                'delta_test_nrmse_3d_raw_minus_hidden': (
                    raw_metrics['nrmse_3d'] - hidden_metrics['nrmse_3d']
                ),
                'delta_test_nrmse_3d_practical_minus_raw': (
                    practical_metrics['nrmse_3d'] - raw_metrics['nrmse_3d']
                ),
                'standardized_residual_mean': (
                    float(np.mean(finite_z)) if finite_z.size else math.nan
                ),
                'standardized_residual_rms': (
                    float(np.sqrt(np.mean(np.square(finite_z))))
                    if finite_z.size else math.nan
                ),
                'standardized_residual_fraction_abs_gt_1p96': (
                    float(np.mean(np.abs(finite_z) > COVERAGE_Z['95']))
                    if finite_z.size else math.nan
                ),
                'standardized_residual_fraction_abs_gt_3': (
                    float(np.mean(np.abs(finite_z) > 3.0))
                    if finite_z.size else math.nan
                ),
                **coverage,
                'coefficient_norm': float(
                    np.linalg.norm(practical_fit['coefficients'])
                ),
                'coefficient_standard_error_median': float(
                    np.median(
                        np.sqrt(
                            np.maximum(
                                np.diag(coefficient_covariance),
                                0.0,
                            )
                        )
                    )
                ),
                'radial_prediction_standard_error_median': float(
                    np.median(np.sqrt(np.maximum(radial_variance, 0.0)))
                ),
                'component_prediction_standard_error_median': float(
                    np.median(np.sqrt(np.maximum(component_variance, 0.0)))
                ),
                'n_parameters': practical_fit['n_basis'],
                'n_observations_fit': int(fit_local.size),
                'numerical_rank': design_diag['numerical_rank'],
                'nullity': design_diag['nullity'],
                'design_condition_number': design_diag['condition_number'],
                'normal_condition_number': normal_condition,
                'normal_min_eigenvalue': normal_min,
                'normal_max_eigenvalue': normal_max,
                'effective_degrees_of_freedom': effective_df,
                'effective_degrees_of_freedom_fraction': (
                    effective_df / practical_fit['n_basis']
                ),
            }
        )

    final_seed_df = pd.DataFrame(final_records)
    final_seed_df.to_csv(paths['final'], index=False)
    np.savez_compressed(
        paths['predictions'],
        run_key=run_metadata_df['run_key'].to_numpy(dtype=str),
        noise_fraction=run_metadata_df['noise_fraction'].to_numpy(dtype=np.float64),
        test_local=np.asarray(context['test_local'], dtype=np.int64),
        test_global=np.asarray(context['model_indices'][context['test_local']], dtype=np.int64),
        velocity_test_true=np.asarray(context['velocity_test_true'], dtype=np.float64),
        practical_prediction=practical_predictions,
        raw_argmin_prediction=raw_predictions,
        hidden_selector_prediction=hidden_predictions,
    )
    try:
        metadata_payload = json.loads(paths['metadata'].read_text(encoding='utf-8'))
    except Exception:
        metadata_payload = {'signature': signature}
    metadata_payload['final_signature'] = final_signature
    paths['metadata'].write_text(
        json.dumps(json_ready(metadata_payload), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    all_final_frames.append(final_seed_df)
    release_seed_context(context)

run_df = pd.concat(all_final_frames, ignore_index=True).sort_values(
    ['noise_fraction', 'seed_pair_index', 'noise_realization']
)

# ============================================================
# 15. Nested aggregate summaries and hierarchical bootstrap
# ============================================================


def hierarchical_bootstrap_interval(
    dataframe,
    column,
    n_draws=BOOTSTRAP_DRAWS,
    seed=BOOTSTRAP_SEED,
    confidence=0.95,
):
    working = dataframe[['seed_pair_index', column]].dropna()
    seed_values = sorted(working['seed_pair_index'].unique())
    if not seed_values:
        return math.nan, math.nan
    grouped = {
        int(seed_value): working[
            working['seed_pair_index'] == seed_value
        ][column].to_numpy(dtype=np.float64)
        for seed_value in seed_values
    }
    rng = np.random.default_rng(int(seed))
    draw_means = np.empty(int(n_draws), dtype=np.float64)
    for draw in range(int(n_draws)):
        sampled_seed_values = rng.choice(
            seed_values,
            size=len(seed_values),
            replace=True,
        )
        seed_means = []
        for seed_value in sampled_seed_values:
            values = grouped[int(seed_value)]
            sampled_values = rng.choice(
                values,
                size=values.size,
                replace=True,
            )
            seed_means.append(float(np.mean(sampled_values)))
        draw_means[draw] = float(np.mean(seed_means))
    alpha = 0.5 * (1.0 - float(confidence))
    return (
        float(np.quantile(draw_means, alpha)),
        float(np.quantile(draw_means, 1.0 - alpha)),
    )


metrics_for_aggregate = [
    'test_observed_nrmse_radial',
    'test_latent_nrmse_radial',
    'test_hidden_nrmse_tangential',
    'test_hidden_nrmse_3d',
    'test_hidden_direction_error_median_deg',
    'test_hidden_predicted_to_reference_rms_3d',
    'test_raw_argmin_hidden_nrmse_3d',
    'test_hidden_selector_nrmse_3d',
    'delta_test_nrmse_3d_practical_minus_hidden',
    'delta_test_nrmse_3d_raw_minus_hidden',
    'delta_test_nrmse_3d_practical_minus_raw',
    'standardized_residual_mean',
    'standardized_residual_rms',
    'coverage_observed_radial_68',
    'coverage_observed_radial_95',
    'coverage_latent_radial_95',
    'coverage_latent_component_95',
    'effective_degrees_of_freedom',
    'effective_degrees_of_freedom_fraction',
]

seed_aggregate_records = []
for (noise_fraction, seed_pair_index), group in run_df.groupby(
    ['noise_fraction', 'seed_pair_index'], sort=True
):
    row = {
        'noise_fraction': float(noise_fraction),
        'seed_pair_index': int(seed_pair_index),
        'reference_seed': int(group['reference_seed'].iloc[0]),
        'label_seed': int(group['label_seed'].iloc[0]),
        'n_noise_runs': int(group.shape[0]),
        'practical_n_basis': int(group['practical_n_basis'].iloc[0]),
        'practical_same_hidden_fraction': float(
            group['practical_same_hidden_candidate'].astype(float).mean()
        ),
        'raw_same_hidden_fraction': float(
            group['raw_same_hidden_candidate'].astype(float).mean()
        ),
    }
    for column in metrics_for_aggregate:
        values = group[column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        row[f'{column}_seed_mean'] = (
            float(np.mean(finite)) if finite.size else math.nan
        )
        row[f'{column}_within_seed_std'] = (
            float(np.std(finite, ddof=1)) if finite.size > 1 else 0.0
        )
    seed_aggregate_records.append(row)
seed_aggregate_df = pd.DataFrame(seed_aggregate_records)

aggregate_records = []
for noise_fraction, group in run_df.groupby('noise_fraction', sort=True):
    seed_group = seed_aggregate_df[
        np.isclose(seed_aggregate_df['noise_fraction'], noise_fraction)
    ]
    row = {
        'noise_fraction': float(noise_fraction),
        'noise_sigma_mean_across_seeds': float(
            group.groupby('seed_pair_index')['noise_sigma'].first().mean()
        ),
        'n_seed_pairs': int(seed_group.shape[0]),
        'n_total_runs': int(group.shape[0]),
        'practical_n_basis': int(group['practical_n_basis'].iloc[0]),
        'raw_argmin_n_basis_median': float(
            np.median(group['raw_argmin_n_basis'])
        ),
        'raw_argmin_n_basis_min': int(group['raw_argmin_n_basis'].min()),
        'raw_argmin_n_basis_max': int(group['raw_argmin_n_basis'].max()),
        'practical_same_hidden_fraction_equal_seed_weight': float(
            seed_group['practical_same_hidden_fraction'].mean()
        ),
        'raw_same_hidden_fraction_equal_seed_weight': float(
            seed_group['raw_same_hidden_fraction'].mean()
        ),
    }
    for metric_index, column in enumerate(metrics_for_aggregate):
        seed_column = f'{column}_seed_mean'
        seed_values = seed_group[seed_column].to_numpy(dtype=np.float64)
        finite_seed = seed_values[np.isfinite(seed_values)]
        row[f'{column}_mean_equal_seed_weight'] = (
            float(np.mean(finite_seed)) if finite_seed.size else math.nan
        )
        row[f'{column}_std_between_seeds'] = (
            float(np.std(finite_seed, ddof=1))
            if finite_seed.size > 1 else 0.0
        )
        row[f'{column}_sem_between_seeds'] = (
            float(np.std(finite_seed, ddof=1) / np.sqrt(finite_seed.size))
            if finite_seed.size > 1 else 0.0
        )
        lower, upper = hierarchical_bootstrap_interval(
            group,
            column,
            seed=BOOTSTRAP_SEED
            + 1000 * int(round(100.0 * float(noise_fraction)))
            + metric_index,
        )
        row[f'{column}_bootstrap_lower_95'] = lower
        row[f'{column}_bootstrap_upper_95'] = upper
    aggregate_records.append(row)
aggregate_df = pd.DataFrame(aggregate_records).sort_values('noise_fraction')

hyperparameter_frequency_df = (
    run_df.groupby(
        [
            'noise_fraction',
            'practical_n_basis',
            'practical_regularization_strength',
            'practical_penalty_power',
        ],
        as_index=False,
    )
    .size()
    .rename(columns={'size': 'count'})
    .sort_values(
        ['noise_fraction', 'count'],
        ascending=[True, False],
    )
)

# ============================================================
# 16. Figures
# ============================================================

noise_values = aggregate_df['noise_fraction'].to_numpy(dtype=np.float64)

fig, axes = plt.subplots(2, 2, figsize=(13.4, 9.6))
ax = axes[0, 0]
for noise_fraction, curve in capacity_aggregate_df.groupby(
    'noise_fraction', sort=True
):
    ax.errorbar(
        curve['n_basis'],
        curve['validation_observed_nrmse_mean'],
        yerr=curve['validation_observed_nrmse_sem_between_seeds'],
        marker='o',
        capsize=3,
        label=rf'$\sigma_u/\mathrm{{RMS}}(u_{{\rm train}}^{{\rm true}})={noise_fraction:g}$',
    )
ax.set_xlabel('Potential parameter count')
ax.set_ylabel('Observed validation radial NRMSE')
ax.set_title('(a) Nested observable validation risk')
ax.legend(fontsize=8)

ax = axes[0, 1]
for noise_fraction, curve in capacity_aggregate_df.groupby(
    'noise_fraction', sort=True
):
    ax.errorbar(
        curve['n_basis'],
        curve['test_hidden_nrmse_3d_mean'],
        yerr=curve['test_hidden_nrmse_3d_std_between_seeds'],
        marker='o',
        capsize=3,
        label=rf'$\sigma_u/\mathrm{{RMS}}(u_{{\rm train}}^{{\rm true}})={noise_fraction:g}$',
    )
ax.axhline(1.0, linestyle='--', linewidth=1.0)
ax.set_xlabel('Potential parameter count')
ax.set_ylabel('Untouched-test hidden 3D NRMSE')
ax.set_title('(b) Hidden field recovery')
ax.legend(fontsize=8)

ax = axes[1, 0]
ax.plot(
    capacity_selection_df['noise_fraction'],
    capacity_selection_df['validation_minimum_basis'],
    marker='o',
    label='Validation minimum',
)
ax.plot(
    capacity_selection_df['noise_fraction'],
    capacity_selection_df['one_standard_error_basis'],
    marker='s',
    label='One-SE basis',
)
ax.plot(
    capacity_selection_df['noise_fraction'],
    capacity_selection_df['practical_plateau_basis'],
    marker='^',
    label='Practical plateau',
)
ax.plot(
    capacity_selection_df['noise_fraction'],
    capacity_selection_df['combined_practical_basis'],
    marker='D',
    label='Combined practical basis',
)
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel('Potential parameter count')
ax.set_title('(c) Capacity-selection rules')
ax.legend(fontsize=8)

ax = axes[1, 1]
ax.plot(
    capacity_selection_df['noise_fraction'],
    100.0 * capacity_selection_df['maximum_basis_selected_fraction_equal_seed_weight'],
    marker='o',
    label='Raw maximum-basis selection [%]',
)
ax.plot(
    capacity_selection_df['noise_fraction'],
    100.0 * capacity_selection_df['last_adjacent_validation_improvement'],
    marker='s',
    label='Final adjacent improvement [%]',
)
ax.axhline(100.0 * PLATEAU_RELATIVE_THRESHOLD, linestyle='--', linewidth=1.0)
ax.axhline(0.0, linestyle=':', linewidth=1.0)
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel('Percent')
ax.set_title('(d) Strict-boundary diagnostics')
ax.legend(fontsize=8)
fig.tight_layout()
save_figure(fig, 'capacity_seed_robustness_composite')

fig, axes = plt.subplots(2, 2, figsize=(13.2, 9.4))
ax = axes[0, 0]
for column, label, marker in [
    ('test_latent_nrmse_radial', 'Latent radial', 'o'),
    ('test_hidden_nrmse_tangential', 'Hidden tangential', 's'),
    ('test_hidden_nrmse_3d', 'Hidden 3D', '^'),
]:
    ax.errorbar(
        noise_values,
        aggregate_df[f'{column}_mean_equal_seed_weight'],
        yerr=aggregate_df[f'{column}_std_between_seeds'],
        marker=marker,
        capsize=3,
        label=label,
    )
ax.axhline(1.0, linestyle='--', linewidth=1.0)
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel('Untouched-test NRMSE')
ax.set_title('(a) Practical-basis field recovery')
ax.legend()

ax = axes[0, 1]
ax.plot(
    noise_values,
    aggregate_df['practical_n_basis'],
    marker='o',
    label='Combined practical basis',
)
ax.plot(
    noise_values,
    aggregate_df['raw_argmin_n_basis_median'],
    marker='s',
    label='Raw argmin median',
)
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel('Selected potential parameters')
ax.set_title('(b) Practical versus raw capacity')
ax.legend()

ax = axes[1, 0]
ax.errorbar(
    noise_values,
    aggregate_df['standardized_residual_rms_mean_equal_seed_weight'],
    yerr=aggregate_df['standardized_residual_rms_std_between_seeds'],
    marker='o',
    capsize=3,
)
ax.axhline(1.0, linestyle='--', linewidth=1.0)
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel('Standardized-residual RMS')
ax.set_title('(c) Held-out noisy radial residuals')

ax = axes[1, 1]
for column, label, marker in [
    ('coverage_observed_radial_68', 'Observed radial 68%', 'o'),
    ('coverage_observed_radial_95', 'Observed radial 95%', 's'),
    ('coverage_latent_radial_95', 'Latent radial 95%', '^'),
    ('coverage_latent_component_95', 'Latent components 95%', 'D'),
]:
    ax.plot(
        noise_values,
        aggregate_df[f'{column}_mean_equal_seed_weight'],
        marker=marker,
        label=label,
    )
ax.axhline(0.68, linestyle=':', linewidth=1.0)
ax.axhline(0.95, linestyle='--', linewidth=1.0)
ax.set_ylim(0.0, 1.03)
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel('Empirical coverage')
ax.set_title('(d) Conditional interval coverage')
ax.legend(fontsize=8)
fig.tight_layout()
save_figure(fig, 'noise_response_practical_composite')

fig, axes = plt.subplots(1, 2, figsize=(13.0, 4.8))
ax = axes[0]
for column, label, marker in [
    ('test_hidden_nrmse_3d', 'Combined practical', 'o'),
    ('test_raw_argmin_hidden_nrmse_3d', 'Raw validation argmin', 's'),
    ('test_hidden_selector_nrmse_3d', 'Hidden-3D diagnostic', '^'),
]:
    ax.errorbar(
        noise_values,
        aggregate_df[f'{column}_mean_equal_seed_weight'],
        yerr=aggregate_df[f'{column}_std_between_seeds'],
        marker=marker,
        capsize=3,
        label=label,
    )
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel('Untouched-test hidden 3D NRMSE')
ax.set_title('(a) Selection-policy comparison')
ax.legend(fontsize=8)

ax = axes[1]
ax.plot(
    noise_values,
    aggregate_df['practical_same_hidden_fraction_equal_seed_weight'],
    marker='o',
    label='Practical-hidden agreement',
)
ax.plot(
    noise_values,
    aggregate_df['raw_same_hidden_fraction_equal_seed_weight'],
    marker='s',
    label='Raw-hidden agreement',
)
ax.axhline(1.0, linestyle='--', linewidth=1.0)
ax.set_ylim(-0.03, 1.03)
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel('Agreement fraction')
ax.set_title('(b) Observable versus hidden selector')
ax.legend(fontsize=8)
fig.tight_layout()
save_figure(fig, 'selector_policy_comparison')

fig, ax = plt.subplots(figsize=(10.0, 6.2))
marker_map = {1.0: 'o', 2.0: '^'}
for power, marker in marker_map.items():
    subset = run_df[np.isclose(run_df['practical_penalty_power'], power)]
    if subset.empty:
        continue
    sizes = 20.0 + 100.0 * (
        subset['practical_n_basis'].to_numpy(dtype=np.float64)
        / max(POTENTIAL_BASIS_SIZES)
    )
    strength_values = subset[
        'practical_regularization_strength'
    ].to_numpy(dtype=np.float64)
    y = np.full(strength_values.shape, -14.0, dtype=np.float64)
    positive_strength = strength_values > 0.0
    y[positive_strength] = np.log10(strength_values[positive_strength])
    ax.scatter(
        subset['noise_fraction'],
        y,
        s=sizes,
        marker=marker,
        alpha=0.55,
        label=f'$p={power:g}$',
    )
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel(r'Selected $\log_{10}\lambda$; zero shown at $-14$')
ax.set_title('Marker: penalty power; size: combined practical parameter count')
ax.legend()
fig.tight_layout()
save_figure(fig, 'selected_hyperparameters')

fig, ax = plt.subplots(figsize=(10.0, 6.0))
for seed_pair_index, seed_group in seed_aggregate_df.groupby(
    'seed_pair_index', sort=True
):
    ax.plot(
        seed_group['noise_fraction'],
        seed_group['test_hidden_nrmse_3d_seed_mean'],
        marker='o',
        label=(
            f"ref{int(seed_group['reference_seed'].iloc[0])}/"
            f"label{int(seed_group['label_seed'].iloc[0])}"
        ),
    )
ax.axhline(1.0, linestyle='--', linewidth=1.0)
ax.set_xlabel(r'Noise fraction $\sigma_u/\mathrm{RMS}(u_{\rm train}^{\rm true})$')
ax.set_ylabel('Seed-mean hidden 3D NRMSE')
ax.set_title('Seed-pair robustness at combined practical capacity')
ax.legend(fontsize=8)
fig.tight_layout()
save_figure(fig, 'seed_pair_noise_response')

# ============================================================
# 17. 保存
# ============================================================

output_files = {
    'geometry_continuity_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_geometry_continuity.csv',
    'gradient_diagnostics_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_gradient_diagnostics.csv',
    'seed_scan_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_seed_scan.csv',
    'closure_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_closure.csv',
    'noise_design_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_noise_design.csv',
    'candidate_grid_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_candidate_grid.csv',
    'within_basis_selected_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_within_basis_selected.csv',
    'capacity_runs_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_capacity_runs.csv',
    'seed_capacity_mean_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_seed_capacity_mean.csv',
    'capacity_aggregate_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_capacity_aggregate.csv',
    'capacity_selection_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_capacity_selection.csv',
    'raw_argmin_selected_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_raw_argmin_selected.csv',
    'hidden_selected_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_hidden_selected_diagnostic.csv',
    'h2_continuity_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_h2_continuity.csv',
    'runs_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_runs.csv',
    'seed_aggregate_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_seed_aggregate.csv',
    'aggregate_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_aggregate.csv',
    'hyperparameter_frequency_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_hyperparameter_frequency.csv',
    'summary_json': OUTPUT_DIR / f'{OUTPUT_PREFIX}_summary.json',
}

geometry_continuity_df.to_csv(output_files['geometry_continuity_csv'], index=False)
gradient_diagnostics_df.to_csv(output_files['gradient_diagnostics_csv'], index=False)
seed_scan_df.to_csv(output_files['seed_scan_csv'], index=False)
closure_df.to_csv(output_files['closure_csv'], index=False)
noise_design_df.to_csv(output_files['noise_design_csv'], index=False)
candidate_df.to_csv(output_files['candidate_grid_csv'], index=False)
within_basis_selected_df.to_csv(output_files['within_basis_selected_csv'], index=False)
capacity_run_df.to_csv(output_files['capacity_runs_csv'], index=False)
seed_capacity_mean_df.to_csv(output_files['seed_capacity_mean_csv'], index=False)
capacity_aggregate_df.to_csv(output_files['capacity_aggregate_csv'], index=False)
capacity_selection_df.to_csv(output_files['capacity_selection_csv'], index=False)
raw_operational_selected_df.to_csv(output_files['raw_argmin_selected_csv'], index=False)
hidden_selected_df.to_csv(output_files['hidden_selected_csv'], index=False)
h2_continuity_df.to_csv(output_files['h2_continuity_csv'], index=False)
run_df.to_csv(output_files['runs_csv'], index=False)
seed_aggregate_df.to_csv(output_files['seed_aggregate_csv'], index=False)
aggregate_df.to_csv(output_files['aggregate_csv'], index=False)
hyperparameter_frequency_df.to_csv(
    output_files['hyperparameter_frequency_csv'], index=False
)

elapsed = time.perf_counter() - start_time
summary = {
    'analysis': 'Mock-1 potential-flow measurement-noise seed robustness Stage 1b',
    'software': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'pandas': pd.__version__,
        'matplotlib': plt.matplotlib.__version__,
    },
    'design': {
        'seed_pairs': SEED_PAIRS,
        'reference_scale': SMOOTHING_CONFIG['scale'],
        'noise_fractions': NOISE_FRACTIONS,
        'positive_noise_realizations_per_seed': N_NOISE_REALIZATIONS,
        'potential_basis_sizes': POTENTIAL_BASIS_SIZES,
        'regularization_strengths': POTENTIAL_REGULARIZATION_STRENGTHS,
        'regularization_powers': POTENTIAL_REGULARIZATION_POWERS,
        'combined_practical_rule': 'max(one_standard_error_basis, practical_plateau_basis)',
        'plateau_relative_threshold': PLATEAU_RELATIVE_THRESHOLD,
        'plateau_consecutive_steps': PLATEAU_CONSECUTIVE_STEPS,
    },
    'continuity': geometry_continuity_df.iloc[0].to_dict(),
    'gradient_diagnostics': gradient_diagnostics_df.iloc[0].to_dict(),
    'capacity_selection': capacity_selection_df.to_dict(orient='records'),
    'closure': closure_df.to_dict(orient='records'),
    'h2_continuity': h2_continuity_df.to_dict(orient='records'),
    'elapsed_seconds': elapsed,
    'output_files': {key: str(value) for key, value in output_files.items()},
}
output_files['summary_json'].write_text(
    json.dumps(json_ready(summary), indent=2, ensure_ascii=False),
    encoding='utf-8',
)

print('=' * 124)
print('Mock-1 potential-flow measurement-noise Stage 1b completed')
print('=' * 124)
print(f'Elapsed time: {elapsed / 60.0:.2f} min')
print('Capacity selection')
display(capacity_selection_df)
print('-' * 124)
print('Noise-response aggregate')
display(aggregate_df)
print('-' * 124)
print('Selected hyperparameter frequencies')
display(hyperparameter_frequency_df)
print('-' * 124)
for key, value in output_files.items():
    print(f'{key:32s}: {value}')
print('=' * 124)
