# Mock-1 potential-flow radial reconstruction:
# heteroscedastic measurement-noise validation Stage 2
#
# このscriptは、Stage 1bで確定したoperator-side unweighted geometry、
# selection-side unweighted loss、potential-gradient family、および
# validation-based practical-capacity ruleを保持し、天体ごとに異なる
# radial-velocity measurement varianceを導入する。
#
# 主要比較:
#   1. gls_correct: 真の相対precision profileを用いるGLS
#   2. gls_flattened: precision contrastを過小評価したGLS working model
#   3. ols: object-dependent precisionを無視するOLS working model
#
# 重要:
#   - 本scriptのdistance-dependent heteroscedastic profileはcontrolled designであり、
#     特定の実surveyの誤差則を主張するものではない。
#   - Working precision weightsはtraining set内で平均1へ規格化する。
#     したがって、全sigmaを同じ倍率で誤指定してもpoint estimatorとcapacity selectionは
#     変わらず、whitened residualとconditional coverageだけが変わる。
#   - Relative profileの誤指定はpoint estimator自体を変える。
#   - Model selectionには各working modelのobservable radial validationだけを用いる。
#   - Latent radial、hidden tangential、hidden 3D truthはpost-selection diagnosticsである。

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

SCRIPT_VERSION = 'mock1_heteroscedastic_stage2_v1_20260901'

# ============================================================
# 1. Paths
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
        'MOCK1_HETERO_GEOMETRY_CACHE',
        str(MOCK_DATA_ROOT / 'mock1_diffusion_geometry_bulk_cache_m2048.npz'),
    )
)
LEGACY_GEOMETRY_CACHE_FILE = Path(
    os.environ.get(
        'MOCK1_HETERO_LEGACY_GEOMETRY_CACHE',
        str(MOCK_DATA_ROOT / 'mock1_diffusion_geometry_bulk_cache.npz'),
    )
)
REFERENCE_CACHE_DIR = MOCK_DATA_ROOT / 'mock1_bulk_robustness_cache'
GRADIENT_CACHE_FILE = Path(
    os.environ.get(
        'MOCK1_HETERO_GRADIENT_CACHE',
        str(
            MOCK_DATA_ROOT
            / 'mock1_potential_flow_gradient_basis_cache_stage1a_highmode_m2048_v1.npz'
        ),
    )
)
STAGE1B_OUTPUT_DIR = Path(
    os.environ.get(
        'MOCK1_HETERO_STAGE1B_DIR',
        str(MOCK_DATA_ROOT / 'mock1_potential_flow_measurement_noise_stage1b_seed_robustness_v1'),
    )
)
OUTPUT_DIR = Path(
    os.environ.get(
        'MOCK1_HETERO_OUTPUT_DIR',
        str(MOCK_DATA_ROOT / 'mock1_potential_flow_heteroscedastic_noise_stage2_v1'),
    )
)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
REFERENCE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_PREFIX = 'mock1_potential_flow_heteroscedastic_noise_stage2'
PROGRESS_FILE = OUTPUT_DIR / f'{OUTPUT_PREFIX}_progress.json'

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)
VELOCITY_UNIT_LABEL = 'simulation velocity unit'
POSITION_UNIT_LABEL = r'$h^{-1}\,\mathrm{Mpc}$'

# ============================================================
# 2. Controlled design
# ============================================================

SEED_PAIRS_TEXT = os.environ.get(
    'MOCK1_HETERO_SEED_PAIRS',
    '20260805:20260804,20260815:20260814,20260825:20260824,'
    '20260905:20260904,20260915:20260914',
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

NOISE_FRACTIONS_TEXT = os.environ.get(
    'MOCK1_HETERO_NOISE_FRACTIONS',
    '0.05,0.10,0.20,0.40,0.80',
)
N_NOISE_REALIZATIONS = int(
    os.environ.get('MOCK1_HETERO_NOISE_REALIZATIONS', '12')
)
NOISE_SEED_BASE = int(
    os.environ.get('MOCK1_HETERO_NOISE_SEED_BASE', '2026090100')
)
NOISE_SEED_STRIDE = int(
    os.environ.get('MOCK1_HETERO_NOISE_SEED_STRIDE', '1000')
)

# True controlled relative error profile before training-set RMS normalization:
# q_raw(r) = 1 + (Q_MAX - 1) (r / R)^PROFILE_POWER.
HETERO_SIGMA_RATIO = float(
    os.environ.get('MOCK1_HETERO_SIGMA_RATIO', '4.0')
)
HETERO_PROFILE_POWER = float(
    os.environ.get('MOCK1_HETERO_PROFILE_POWER', '1.0')
)
FLATTENED_PROFILE_EXPONENT = float(
    os.environ.get('MOCK1_HETERO_FLATTENED_EXPONENT', '0.5')
)
GLOBAL_SCALE_FACTORS_TEXT = os.environ.get(
    'MOCK1_HETERO_GLOBAL_SCALE_FACTORS', '0.5,1.0,2.0'
)

GRAPH_NEIGHBORS = 48
GRAPH_BANDWIDTH_NEIGHBOR = 16
GRAPH_BANDWIDTH_MULTIPLIER = 1.0

N_AFFINE_POTENTIAL_MODES = 3
POTENTIAL_DIFFUSION_MODE_COUNTS_TEXT = os.environ.get(
    'MOCK1_HETERO_DIFFUSION_MODE_COUNTS',
    '0,3,8,16,32,64,128,256,384,511,640,768,896,1024,1280,1536,1792,2047',
)
CAPACITY_REFERENCE_PARAMETER_COUNT = int(
    os.environ.get('MOCK1_HETERO_REFERENCE_PARAMETER_COUNT', '1027')
)
PLATEAU_RELATIVE_THRESHOLD = float(
    os.environ.get('MOCK1_HETERO_PLATEAU_THRESHOLD', '0.02')
)
PLATEAU_CONSECUTIVE_STEPS = int(
    os.environ.get('MOCK1_HETERO_PLATEAU_STEPS', '2')
)
PLATEAU_MIN_PARAMETER_COUNT = int(
    os.environ.get('MOCK1_HETERO_PLATEAU_MIN_PARAMETERS', '514')
)
POTENTIAL_REGULARIZATION_STRENGTHS_TEXT = os.environ.get(
    'MOCK1_HETERO_REGULARIZATION_STRENGTHS',
    '0,1e-12,1e-10,1e-8,1e-6,1e-4,1e-2,1,1e2',
)
POTENTIAL_REGULARIZATION_POWERS_TEXT = os.environ.get(
    'MOCK1_HETERO_REGULARIZATION_POWERS', '1,2'
)

METRIC_EIGENVALUE_RELATIVE_FLOOR = 1.0e-10
METRIC_EIGENVALUE_ABSOLUTE_FLOOR = 1.0e-14
GRADIENT_MODE_CHUNK_SIZE = 32
USE_GRADIENT_CACHE = True
USE_REFERENCE_FIELD_CACHE = True
NUMERICAL_RIDGE = 1.0e-12
CLOSURE_PASS_TOLERANCE = 1.0e-8
RUN_MAX_BASIS_CLOSURE = os.environ.get('MOCK1_HETERO_RUN_CLOSURE', '1') == '1'
RESUME_FROM_CHECKPOINT = os.environ.get('MOCK1_HETERO_RESUME', '1') == '1'
FORCE_RECOMPUTE = os.environ.get('MOCK1_HETERO_FORCE_RECOMPUTE', '0') == '1'
BOOTSTRAP_DRAWS = int(os.environ.get('MOCK1_HETERO_BOOTSTRAP_DRAWS', '10000'))
BOOTSTRAP_SEED = int(os.environ.get('MOCK1_HETERO_BOOTSTRAP_SEED', '2026090191'))
COVERAGE_Z = {'68': 1.0, '95': 1.959963984540054}
SHOW_PLOTS = os.environ.get('MOCK1_HETERO_SHOW_PLOTS', '1') == '1'
SAVE_PDF = os.environ.get('MOCK1_HETERO_SAVE_PDF', '1') == '1'
SAVE_PNG = os.environ.get('MOCK1_HETERO_SAVE_PNG', '1') == '1'
SMOKE_TEST = os.environ.get('MOCK1_HETERO_SMOKE_TEST', '0') == '1'

# Parsed after helper definitions are available.
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
# 3. Stage-2-specific utilities and parsed configuration
# ============================================================

SEED_PAIRS = parse_seed_pairs(SEED_PAIRS_TEXT)
NOISE_FRACTIONS = parse_float_list(NOISE_FRACTIONS_TEXT)
GLOBAL_SCALE_FACTORS = parse_float_list(GLOBAL_SCALE_FACTORS_TEXT)
POTENTIAL_DIFFUSION_MODE_COUNTS = parse_int_list(
    POTENTIAL_DIFFUSION_MODE_COUNTS_TEXT
)
POTENTIAL_REGULARIZATION_STRENGTHS = parse_float_list(
    POTENTIAL_REGULARIZATION_STRENGTHS_TEXT
)
POTENTIAL_REGULARIZATION_POWERS = parse_float_list(
    POTENTIAL_REGULARIZATION_POWERS_TEXT
)

if N_NOISE_REALIZATIONS < 1:
    raise ValueError('N_NOISE_REALIZATIONSは1以上でなければならない。')
if HETERO_SIGMA_RATIO < 1.0:
    raise ValueError('HETERO_SIGMA_RATIOは1以上でなければならない。')
if HETERO_PROFILE_POWER <= 0.0:
    raise ValueError('HETERO_PROFILE_POWERは正でなければならない。')
if not 0.0 <= FLATTENED_PROFILE_EXPONENT <= 1.0:
    raise ValueError('FLATTENED_PROFILE_EXPONENTは0以上1以下とする。')
if any(value <= 0.0 for value in GLOBAL_SCALE_FACTORS):
    raise ValueError('GLOBAL_SCALE_FACTORSはすべて正でなければならない。')

if SMOKE_TEST:
    SEED_PAIRS = SEED_PAIRS[:2]
    GRAPH_NEIGHBORS = 16
    GRAPH_BANDWIDTH_NEIGHBOR = 6
    POTENTIAL_DIFFUSION_MODE_COUNTS = [0, 3, 8, 16, 23]
    CAPACITY_REFERENCE_PARAMETER_COUNT = 19
    PLATEAU_MIN_PARAMETER_COUNT = 11
    POTENTIAL_REGULARIZATION_STRENGTHS = [1.0e-6, 1.0e-2]
    POTENTIAL_REGULARIZATION_POWERS = [1.0, 2.0]
    NOISE_FRACTIONS = [0.20, 0.80]
    N_NOISE_REALIZATIONS = 2
    BOOTSTRAP_DRAWS = 400
    GRADIENT_MODE_CHUNK_SIZE = 8
    GRADIENT_CACHE_FILE = (
        MOCK_DATA_ROOT
        / 'mock1_potential_flow_gradient_basis_heteroscedastic_smoke_v1.npz'
    )
    OUTPUT_DIR = Path(
        os.environ.get(
            'MOCK1_HETERO_SMOKE_OUTPUT_DIR',
            str(MOCK_DATA_ROOT / 'mock1_potential_flow_heteroscedastic_noise_stage2_v1_smoke'),
        )
    )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    CHECKPOINT_DIR = OUTPUT_DIR / 'checkpoints'
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)
    PROGRESS_FILE = OUTPUT_DIR / f'{OUTPUT_PREFIX}_progress.json'
    SHOW_PLOTS = False
    SAVE_PDF = False
    SAVE_PNG = False

MAX_DIFFUSION_MODES = max(POTENTIAL_DIFFUSION_MODE_COUNTS) + 1

WORKING_MODELS = {
    'gls_correct': {
        'label': 'Correct GLS',
        'profile_kind': 'correct',
    },
    'gls_flattened': {
        'label': 'Flattened-profile GLS',
        'profile_kind': 'flattened',
    },
    'ols': {
        'label': 'OLS',
        'profile_kind': 'constant',
    },
}
WORKING_MODEL_ORDER = list(WORKING_MODELS)
PRIMARY_MODEL = 'gls_correct'


def atomic_write_json(path: Path, payload: dict) -> None:
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    temp.write_text(
        json.dumps(json_ready(payload), indent=2, ensure_ascii=False),
        encoding='utf-8',
    )
    os.replace(temp, path)


def atomic_to_csv(dataframe: pd.DataFrame, path: Path) -> None:
    path = Path(path)
    temp = path.with_suffix(path.suffix + '.tmp')
    dataframe.to_csv(temp, index=False)
    os.replace(temp, path)


def weighted_cross_products(design, target, weights):
    design = np.asarray(design, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    if design.ndim != 2 or weights.ndim != 1:
        raise ValueError('designは2次元、weightsは1次元でなければならない。')
    if design.shape[0] != weights.size or target.shape[0] != weights.size:
        raise ValueError('Design, target, weightsの行数が一致しない。')
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError('Weight sumは正でなければならない。')
    gram = design.T @ (weights[:, None] * design) / weight_sum
    if target.ndim == 1:
        rhs = design.T @ (weights * target) / weight_sum
    elif target.ndim == 2:
        rhs = design.T @ (weights[:, None] * target) / weight_sum
    else:
        raise ValueError('targetは1次元または2次元でなければならない。')
    return gram, rhs, weight_sum


def relative_profile_raw(radius_values: np.ndarray, radius_max: float) -> np.ndarray:
    radius_values = np.asarray(radius_values, dtype=np.float64)
    if radius_max <= 0.0:
        raise ValueError('radius_maxは正でなければならない。')
    scaled = np.clip(radius_values / float(radius_max), 0.0, 1.0)
    return 1.0 + (HETERO_SIGMA_RATIO - 1.0) * np.power(
        scaled, HETERO_PROFILE_POWER
    )


def rms_normalize_profile(profile: np.ndarray, training_indices: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    training_indices = np.asarray(training_indices, dtype=np.int64)
    scale = float(np.sqrt(np.mean(np.square(profile[training_indices]))))
    if not np.isfinite(scale) or scale <= 0.0:
        raise ValueError('Profile RMS normalizationに失敗した。')
    return profile / scale


def make_working_profiles(
    true_profile: np.ndarray,
    training_indices: np.ndarray,
) -> dict[str, np.ndarray]:
    true_profile = np.asarray(true_profile, dtype=np.float64)
    raw = {
        'gls_correct': true_profile.copy(),
        'gls_flattened': np.power(true_profile, FLATTENED_PROFILE_EXPONENT),
        'ols': np.ones_like(true_profile),
    }
    return {
        name: rms_normalize_profile(value, training_indices)
        for name, value in raw.items()
    }


def normalized_profile_precision(profile: np.ndarray, indices: np.ndarray) -> np.ndarray:
    profile = np.asarray(profile, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    selected = profile[indices]
    if np.any(selected <= 0.0) or np.any(~np.isfinite(selected)):
        raise ValueError('Working profileは有限かつ正でなければならない。')
    precision = 1.0 / np.square(selected)
    return precision / np.mean(precision)


def precision_effective_sample_size(weights: np.ndarray) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(np.square(weights)))
    if denominator <= 0.0:
        return math.nan
    return float(np.square(np.sum(weights)) / denominator)


def profile_weighted_nrmse(
    true_values: np.ndarray,
    predicted_values: np.ndarray,
    relative_scale: np.ndarray,
) -> float:
    true_values = np.asarray(true_values, dtype=np.float64)
    predicted_values = np.asarray(predicted_values, dtype=np.float64)
    relative_scale = np.asarray(relative_scale, dtype=np.float64)
    residual = (predicted_values - true_values) / relative_scale
    reference = true_values / relative_scale
    denominator = float(np.sqrt(np.mean(np.square(reference))))
    numerator = float(np.sqrt(np.mean(np.square(residual))))
    return numerator / denominator if denominator > 0.0 else math.nan


def standardized_residual_rms(
    observed: np.ndarray,
    predicted: np.ndarray,
    sigma: np.ndarray,
) -> float:
    sigma = np.asarray(sigma, dtype=np.float64)
    if np.any(sigma <= 0.0):
        return math.nan
    z = (np.asarray(observed) - np.asarray(predicted)) / sigma
    return float(np.sqrt(np.mean(np.square(z))))


def regularized_covariance(
    design: np.ndarray,
    fit_weights: np.ndarray,
    true_sigma: np.ndarray,
    factor_kind: str,
    factor,
) -> np.ndarray:
    design = np.asarray(design, dtype=np.float64)
    fit_weights = np.asarray(fit_weights, dtype=np.float64)
    true_sigma = np.asarray(true_sigma, dtype=np.float64)
    n = float(design.shape[0])
    middle = design.T @ (
        np.square(fit_weights * true_sigma)[:, None] * design
    ) / (n * n)
    inverse_system = solve_factored(
        factor_kind,
        factor,
        np.eye(design.shape[1], dtype=np.float64),
    )
    covariance = inverse_system @ middle @ inverse_system.T
    return 0.5 * (covariance + covariance.T)


def exact_sign_test_two_sided(differences: np.ndarray) -> float:
    differences = np.asarray(differences, dtype=np.float64)
    differences = differences[np.isfinite(differences) & (differences != 0.0)]
    n = int(differences.size)
    if n == 0:
        return math.nan
    k = int(min(np.sum(differences > 0.0), np.sum(differences < 0.0)))
    probability = sum(math.comb(n, j) for j in range(k + 1)) / (2.0**n)
    return float(min(1.0, 2.0 * probability))


def hierarchical_bootstrap_interval(
    dataframe: pd.DataFrame,
    value_column: str,
    seed_column: str = 'seed_pair_index',
    realization_column: str = 'noise_realization',
    n_draws: int = BOOTSTRAP_DRAWS,
    seed: int = BOOTSTRAP_SEED,
    confidence: float = 0.95,
) -> tuple[float, float]:
    data = dataframe[[seed_column, realization_column, value_column]].dropna()
    if data.empty:
        return math.nan, math.nan
    groups = {
        int(key): group[value_column].to_numpy(dtype=np.float64)
        for key, group in data.groupby(seed_column)
    }
    seed_ids = np.array(sorted(groups), dtype=np.int64)
    if seed_ids.size == 1 and groups[int(seed_ids[0])].size == 1:
        value = float(groups[int(seed_ids[0])][0])
        return value, value
    rng = np.random.default_rng(int(seed))
    draws = np.empty(int(n_draws), dtype=np.float64)
    for draw in range(int(n_draws)):
        sampled_seed_ids = rng.choice(seed_ids, size=seed_ids.size, replace=True)
        seed_means = []
        for seed_id in sampled_seed_ids:
            values = groups[int(seed_id)]
            sampled_values = rng.choice(values, size=values.size, replace=True)
            seed_means.append(float(np.mean(sampled_values)))
        draws[draw] = float(np.mean(seed_means))
    alpha = 0.5 * (1.0 - confidence)
    return (
        float(np.quantile(draws, alpha)),
        float(np.quantile(draws, 1.0 - alpha)),
    )


def update_progress(stage: str, **kwargs) -> None:
    payload = {
        'script_version': SCRIPT_VERSION,
        'stage': stage,
        'updated_unix_time': time.time(),
        **kwargs,
    }
    atomic_write_json(PROGRESS_FILE, payload)


def file_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def model_block_paths(
    seed_pair_index: int,
    reference_seed: int,
    label_seed: int,
    model_name: str,
    power: float,
    strength: float,
) -> tuple[Path, Path]:
    token = (
        f'seed{seed_pair_index:02d}_ref{reference_seed}_label{label_seed}_'
        f'{model_name}_p{power:g}_lam{strength:.3e}'
    ).replace('+', '').replace('-', 'm').replace('.', 'p')
    return (
        CHECKPOINT_DIR / f'candidate_{token}.csv',
        CHECKPOINT_DIR / f'candidate_{token}.json',
    )


def seed_final_paths(
    seed_pair_index: int,
    reference_seed: int,
    label_seed: int,
) -> tuple[Path, Path]:
    token = f'seed{seed_pair_index:02d}_ref{reference_seed}_label{label_seed}'
    return (
        CHECKPOINT_DIR / f'final_{token}.csv',
        CHECKPOINT_DIR / f'final_{token}.json',
    )

# ============================================================
# 4. Load data, geometry, and global potential-gradient basis
# ============================================================

required_files = [DATA_FILE, GEOMETRY_CACHE_FILE, LEGACY_GEOMETRY_CACHE_FILE]
missing_files = [str(path) for path in required_files if not path.is_file()]
if missing_files:
    raise FileNotFoundError(
        '必要な入力ファイルが見つからない。\n'
        + '\n'.join(missing_files)
        + '\nStage 1bまでのMock-1 geometry/cacheを先に用意する。'
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
    generator_eigenvalues = np.asarray(data['generator_eigenvalues'], dtype=np.float64)
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
    legacy_generator_eigenvalues = np.asarray(data['generator_eigenvalues'], dtype=np.float64)
    legacy_bandwidth = np.asarray(data['bandwidth'], dtype=np.float64)
    cached_local_metric = np.asarray(data['local_metric'], dtype=np.float64)

if pos_absolute.ndim != 2 or pos_absolute.shape[1] != 3:
    raise ValueError(f'posのshapeが不正である: {pos_absolute.shape}')
if vel_raw.shape != pos_absolute.shape:
    raise ValueError(f'velのshape {vel_raw.shape}がpos {pos_absolute.shape}と一致しない。')
if eigenfunctions.shape[0] != pos_absolute.shape[0]:
    raise ValueError('eigenfunctionsの点数がmock1 catalogと一致しない。')
if eigenfunctions.shape[1] < MAX_DIFFUSION_MODES:
    raise ValueError(
        f'geometry cacheの固有関数数が不足している: '
        f'{eigenfunctions.shape[1]} < {MAX_DIFFUSION_MODES}'
    )
if generator_eigenvalues.size < MAX_DIFFUSION_MODES:
    raise ValueError('generator_eigenvaluesの個数が不足している。')
if geometry_bandwidth.shape != (pos_absolute.shape[0],):
    raise ValueError('high-mode geometry bandwidthのshapeが不正である。')
if stationary_measure.shape != (pos_absolute.shape[0],):
    raise ValueError('stationary_measureのshapeが不正である。')
if cached_local_metric.shape != (pos_absolute.shape[0], 3, 3):
    raise ValueError('legacy local_metricのshapeが不正である。')
if not np.isclose(np.sum(stationary_measure), 1.0, rtol=1.0e-10, atol=1.0e-12):
    raise ValueError('stationary_measureが1に規格化されていない。')

positions = pos_absolute - SPHERE_CENTER[None, :]
radius = np.linalg.norm(positions, axis=1)
radius_max = float(np.max(radius))
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
potential_generator_values = np.concatenate(
    [
        np.zeros(N_AFFINE_POTENTIAL_MODES, dtype=np.float64),
        generator_eigenvalues[active_mode_indices],
    ]
)

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
            'first64_mode_correlation_median': float(np.median(low_mode_correlations)),
            'first64_mode_correlation_minimum': float(np.min(low_mode_correlations)),
            'first64_subspace_minimum_singular_value': float(
                np.min(subspace_singular_values)
            ),
        }
    ]
)

print('=' * 132)
print('Mock-1 potential-flow heteroscedastic measurement-noise validation: Stage 2')
print('=' * 132)
print(f'Python / NumPy / SciPy : {platform.python_version()} / {np.__version__} / {scipy.__version__}')
print(f'Objects                : {positions.shape[0]:,}')
print(f'Seed pairs             : {SEED_PAIRS}')
print(f'Noise fractions        : {NOISE_FRACTIONS}')
print(f'Noise realizations     : {N_NOISE_REALIZATIONS} per positive level and seed pair')
print(f'True relative profile  : q(r)=1+({HETERO_SIGMA_RATIO:g}-1)(r/R)^{HETERO_PROFILE_POWER:g}, training-RMS normalized')
print(f'Flattened profile      : q_work proportional to q_true^{FLATTENED_PROFILE_EXPONENT:g}')
print(f'Working models         : {WORKING_MODEL_ORDER}')
print(f'Global scale audit     : {GLOBAL_SCALE_FACTORS}')
print(f'Potential parameters   : {POTENTIAL_BASIS_SIZES}')
print(f'Lambda grid            : {POTENTIAL_REGULARIZATION_STRENGTHS}')
print(f'Penalty powers         : {POTENTIAL_REGULARIZATION_POWERS}')
print(f'Resume                 : {RESUME_FROM_CHECKPOINT and not FORCE_RECOMPUTE}')
print(f'Input                  : {DATA_FILE}')
print(f'Full geometry          : {GEOMETRY_CACHE_FILE}')
print(f'Gradient cache         : {GRADIENT_CACHE_FILE}')
print(f'Output                 : {OUTPUT_DIR}')
print('-' * 132)
print('Full-geometry versus legacy geometry continuity audit')
display(geometry_continuity_df)

start_time = time.perf_counter()
update_progress(
    'loading_geometry',
    completed_candidate_blocks=0,
    expected_candidate_blocks=(
        len(SEED_PAIRS)
        * len(WORKING_MODELS)
        * len(POTENTIAL_REGULARIZATION_POWERS)
        * len(POTENTIAL_REGULARIZATION_STRENGTHS)
    ),
    completed_final_seeds=0,
    expected_final_seeds=len(SEED_PAIRS),
)

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
del diffusion_radial_all, diffusion_radial_residual, gradient_all_modes
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
# 5. Seed-specific context, closure, and candidate checkpoints
# ============================================================


def seed_tag(seed_pair_index: int, reference_seed: int, label_seed: int) -> str:
    return f'seed{seed_pair_index:02d}_ref{reference_seed}_label{label_seed}'


def seed_signature(
    seed_pair_index: int,
    reference_seed: int,
    label_seed: int,
) -> str:
    payload = {
        'script_version': SCRIPT_VERSION,
        'data': file_signature(DATA_FILE),
        'geometry': file_signature(GEOMETRY_CACHE_FILE),
        'legacy_geometry': file_signature(LEGACY_GEOMETRY_CACHE_FILE),
        'gradient_cache': file_signature(GRADIENT_CACHE_FILE) if GRADIENT_CACHE_FILE.is_file() else None,
        'seed_pair_index': seed_pair_index,
        'reference_seed': reference_seed,
        'label_seed': label_seed,
        'seed_pairs': SEED_PAIRS,
        'noise_fractions': NOISE_FRACTIONS,
        'n_noise_realizations': N_NOISE_REALIZATIONS,
        'noise_seed_base': NOISE_SEED_BASE,
        'noise_seed_stride': NOISE_SEED_STRIDE,
        'hetero_sigma_ratio': HETERO_SIGMA_RATIO,
        'hetero_profile_power': HETERO_PROFILE_POWER,
        'flattened_profile_exponent': FLATTENED_PROFILE_EXPONENT,
        'potential_basis_sizes': POTENTIAL_BASIS_SIZES,
        'regularization_strengths': POTENTIAL_REGULARIZATION_STRENGTHS,
        'regularization_powers': POTENTIAL_REGULARIZATION_POWERS,
        'plateau_threshold': PLATEAU_RELATIVE_THRESHOLD,
        'plateau_steps': PLATEAU_CONSECUTIVE_STEPS,
        'smoothing_config': SMOOTHING_CONFIG,
    }
    return stable_signature(payload)


def prepare_seed_context(
    seed_pair_index: int,
    reference_seed: int,
    label_seed: int,
) -> dict:
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
    reference_result = reference_fields[SMOOTHING_CONFIG['scale']]
    velocity_model_true = np.asarray(
        reference_result['bulk_velocity'], dtype=np.float64
    )

    diffusion_gradient_model = np.asarray(
        gradient_active_all[model_indices], dtype=np.float64
    ).copy()
    diffusion_gradient_model -= affine_projection_vectors[None, :, :]
    diffusion_gradient_column_rms = np.sqrt(
        np.mean(np.sum(np.square(diffusion_gradient_model), axis=2), axis=0)
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
        [affine_gradient_model, diffusion_gradient_model], axis=1
    )
    del affine_gradient_model, diffusion_gradient_model

    line_of_sight_model = line_of_sight_all[model_indices]
    radial_design_model = potential_radial_design(
        gradient_model, line_of_sight_model
    )
    radial_true_model = np.einsum(
        'ij,ij->i', line_of_sight_model, velocity_model_true
    )
    train_radial_rms = float(
        np.sqrt(np.mean(np.square(radial_true_model[train_local])))
    )
    if train_radial_rms <= 0.0:
        raise RuntimeError('Training radial RMSが正でない。')

    raw_true_profile = relative_profile_raw(radius[model_indices], radius_max)
    true_profile = rms_normalize_profile(raw_true_profile, train_local)
    working_profiles = make_working_profiles(true_profile, train_local)

    run_metadata = []
    observed_columns = []
    true_sigma_columns = []
    seed_offset = seed_pair_index * NOISE_SEED_STRIDE
    for noise_fraction in NOISE_FRACTIONS:
        n_realizations = (
            1 if np.isclose(noise_fraction, 0.0) else N_NOISE_REALIZATIONS
        )
        sigma_base = float(noise_fraction) * train_radial_rms
        for realization in range(n_realizations):
            noise_seed = NOISE_SEED_BASE + seed_offset + realization
            if sigma_base == 0.0:
                standard_normal = np.zeros(model_indices.size, dtype=np.float64)
            else:
                rng = np.random.default_rng(noise_seed)
                standard_normal = rng.normal(size=model_indices.size)
            true_sigma = sigma_base * true_profile
            observed = radial_true_model + true_sigma * standard_normal
            run_index = len(run_metadata)
            run_key = (
                f'{seed_tag(seed_pair_index, reference_seed, label_seed)}_'
                f'noise{noise_fraction_token(noise_fraction)}_'
                f'rep{realization:03d}_seed{noise_seed}'
            )
            run_metadata.append(
                {
                    'seed_pair_index': seed_pair_index,
                    'seed_run_index': run_index,
                    'run_key': run_key,
                    'reference_seed': reference_seed,
                    'label_seed': label_seed,
                    'noise_fraction': float(noise_fraction),
                    'noise_realization': realization,
                    'noise_seed': noise_seed,
                    'noise_sigma_base': sigma_base,
                    'train_radial_rms': train_radial_rms,
                }
            )
            observed_columns.append(observed)
            true_sigma_columns.append(true_sigma)

    run_metadata_df = pd.DataFrame(run_metadata)
    observed_model_matrix = np.column_stack(observed_columns)
    true_sigma_model_matrix = np.column_stack(true_sigma_columns)

    return {
        'seed_pair_index': seed_pair_index,
        'reference_seed': reference_seed,
        'label_seed': label_seed,
        'reference_indices': reference_indices,
        'model_indices': model_indices,
        'train_local': train_local,
        'validation_local': validation_local,
        'test_local': test_local,
        'fit_local': fit_local,
        'velocity_model_true': velocity_model_true,
        'velocity_validation_true': velocity_model_true[validation_local],
        'velocity_test_true': velocity_model_true[test_local],
        'line_of_sight_model': line_of_sight_model,
        'line_of_sight_validation': line_of_sight_model[validation_local],
        'line_of_sight_test': line_of_sight_model[test_local],
        'gradient_model': gradient_model,
        'gradient_validation': gradient_model[validation_local],
        'gradient_test': gradient_model[test_local],
        'A_train_max': radial_design_model[train_local, :max_potential_basis],
        'A_validation_max': radial_design_model[validation_local, :max_potential_basis],
        'A_test_max': radial_design_model[test_local, :max_potential_basis],
        'A_fit_max': radial_design_model[fit_local, :max_potential_basis],
        'radial_true_model': radial_true_model,
        'radial_validation_true': radial_true_model[validation_local],
        'radial_test_true': radial_true_model[test_local],
        'true_profile': true_profile,
        'working_profiles': working_profiles,
        'run_metadata_df': run_metadata_df,
        'observed_model_matrix': observed_model_matrix,
        'observed_train_matrix': observed_model_matrix[train_local],
        'observed_validation_matrix': observed_model_matrix[validation_local],
        'observed_test_matrix': observed_model_matrix[test_local],
        'observed_fit_matrix': observed_model_matrix[fit_local],
        'true_sigma_model_matrix': true_sigma_model_matrix,
        'true_sigma_validation_matrix': true_sigma_model_matrix[validation_local],
        'true_sigma_test_matrix': true_sigma_model_matrix[test_local],
        'train_radial_rms': train_radial_rms,
        'affine_mode_scale': affine_mode_scale,
    }


def release_seed_context(context: dict) -> None:
    context.clear()
    gc.collect()


def maximum_basis_closure(context: dict) -> pd.DataFrame:
    if not RUN_MAX_BASIS_CLOSURE:
        return pd.DataFrame()
    fit_local = context['fit_local']
    rng = np.random.default_rng(910000 + context['seed_pair_index'])
    coefficients_true = rng.normal(size=max_potential_basis)
    coefficients_true /= max(
        float(np.linalg.norm(coefficients_true)), np.finfo(float).eps
    )
    radial_generated = context['A_fit_max'] @ coefficients_true
    field_generated = potential_velocity(
        context['gradient_test'], coefficients_true, max_potential_basis
    )
    records = []
    for model_name in WORKING_MODEL_ORDER:
        weights = normalized_profile_precision(
            context['working_profiles'][model_name], fit_local
        )
        gram, rhs, _ = weighted_cross_products(
            context['A_fit_max'], radial_generated, weights
        )
        regularization = regularization_diagonal(
            potential_generator_values, max_potential_basis, 1.0
        )
        factor_kind, factor, system = factor_regularized_system(
            gram, regularization, 0.0
        )
        coefficients = solve_factored(factor_kind, factor, rhs)
        radial_recovered = context['A_fit_max'] @ coefficients
        field_recovered = potential_velocity(
            context['gradient_test'], coefficients, max_potential_basis
        )
        spectrum = gram_spectrum(gram, context['A_fit_max'].shape[0])
        records.append(
            {
                'seed_pair_index': context['seed_pair_index'],
                'reference_seed': context['reference_seed'],
                'label_seed': context['label_seed'],
                'working_model': model_name,
                'n_basis': max_potential_basis,
                'n_fit': int(context['A_fit_max'].shape[0]),
                'numerical_rank': int(spectrum['numerical_rank']),
                'coefficient_relative_error': float(
                    np.linalg.norm(coefficients - coefficients_true)
                    / np.linalg.norm(coefficients_true)
                ),
                'radial_fit_relative_error': float(
                    np.linalg.norm(radial_recovered - radial_generated)
                    / max(np.linalg.norm(radial_generated), np.finfo(float).eps)
                ),
                'test_field_relative_error': float(
                    np.linalg.norm(field_recovered - field_generated)
                    / max(np.linalg.norm(field_generated), np.finfo(float).eps)
                ),
                'normal_matrix_condition': normal_matrix_condition(system)[0],
                'closure_status': 'pass',
            }
        )
    dataframe = pd.DataFrame(records)
    if np.any(
        dataframe[
            [
                'coefficient_relative_error',
                'radial_fit_relative_error',
                'test_field_relative_error',
            ]
        ].to_numpy(dtype=np.float64)
        > CLOSURE_PASS_TOLERANCE
    ):
        dataframe['closure_status'] = 'review'
    return dataframe


def candidate_block_signature(
    base_signature: str,
    model_name: str,
    power: float,
    strength: float,
) -> str:
    return stable_signature(
        {
            'base_signature': base_signature,
            'working_model': model_name,
            'penalty_power': power,
            'regularization_strength': strength,
        }
    )


def load_candidate_block(
    csv_path: Path,
    metadata_path: Path,
    signature: str,
) -> pd.DataFrame | None:
    if not (
        RESUME_FROM_CHECKPOINT
        and not FORCE_RECOMPUTE
        and csv_path.is_file()
        and metadata_path.is_file()
    ):
        return None
    try:
        metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
        if metadata.get('signature') != signature:
            return None
        dataframe = pd.read_csv(csv_path)
        expected_runs = sum(
            1 if np.isclose(value, 0.0) else N_NOISE_REALIZATIONS
            for value in NOISE_FRACTIONS
        )
        expected_rows = len(POTENTIAL_BASIS_SIZES) * expected_runs
        if dataframe.shape[0] != expected_rows:
            return None
        return dataframe
    except Exception:
        return None


def scan_seed_candidates(context: dict, base_signature: str) -> pd.DataFrame:
    validation_local = context['validation_local']
    run_metadata_df = context['run_metadata_df']
    n_runs = run_metadata_df.shape[0]
    vector_design_validation = vector_design_matrix(context['gradient_validation'])
    vector_gram_sum_max = vector_design_validation.T @ vector_design_validation
    vector_truth_flat = context['velocity_validation_true'].reshape(-1)
    vector_rhs_true_max = vector_design_validation.T @ vector_truth_flat
    true_vector_sum_squares = float(np.sum(np.square(vector_truth_flat)))
    true_radial_sum_squares = float(
        np.sum(np.square(context['radial_validation_true']))
    )
    true_tangential_sum_squares = max(
        true_vector_sum_squares - true_radial_sum_squares,
        np.finfo(float).eps,
    )

    frames = []
    for model_name in WORKING_MODEL_ORDER:
        profile = context['working_profiles'][model_name]
        train_weights = normalized_profile_precision(
            profile, context['train_local']
        )
        gram_max, rhs_all, _ = weighted_cross_products(
            context['A_train_max'],
            context['observed_train_matrix'],
            train_weights,
        )
        q_validation = profile[validation_local]
        q_true_validation = context['true_profile'][validation_local]

        for power in POTENTIAL_REGULARIZATION_POWERS:
            regularization_max = regularization_diagonal(
                potential_generator_values,
                max_potential_basis,
                power,
            )
            for strength in POTENTIAL_REGULARIZATION_STRENGTHS:
                csv_path, metadata_path = model_block_paths(
                    context['seed_pair_index'],
                    context['reference_seed'],
                    context['label_seed'],
                    model_name,
                    power,
                    strength,
                )
                signature = candidate_block_signature(
                    base_signature, model_name, power, strength
                )
                cached = load_candidate_block(
                    csv_path, metadata_path, signature
                )
                if cached is not None:
                    frames.append(cached)
                    continue

                records = []
                print(
                    f"  [{context['reference_seed']}:{context['label_seed']} | "
                    f'{model_name}] p={power:g}, lambda={strength:.1e}'
                )
                for n_basis in POTENTIAL_BASIS_SIZES:
                    factor_kind, factor, system = factor_regularized_system(
                        gram_max[:n_basis, :n_basis],
                        regularization_max[:n_basis],
                        strength,
                    )
                    coefficients = solve_factored(
                        factor_kind,
                        factor,
                        rhs_all[:n_basis],
                    )
                    prediction_validation = (
                        context['A_validation_max'][:, :n_basis] @ coefficients
                    )
                    vector_nrmse, vector_sse = hidden_vector_nrmse_from_quadratic(
                        coefficients,
                        vector_gram_sum_max[:n_basis, :n_basis],
                        vector_rhs_true_max[:n_basis],
                        true_vector_sum_squares,
                    )
                    latent_radial_residual = (
                        prediction_validation
                        - context['radial_validation_true'][:, None]
                    )
                    latent_radial_sse = np.sum(
                        np.square(latent_radial_residual), axis=0
                    )
                    tangential_sse = np.maximum(
                        vector_sse - latent_radial_sse, 0.0
                    )
                    tangential_nrmse = np.sqrt(
                        tangential_sse / true_tangential_sum_squares
                    )
                    condition = normal_matrix_condition(system)[0]

                    for run_index, run_row in run_metadata_df.iterrows():
                        observed_validation = context['observed_validation_matrix'][
                            :, run_index
                        ]
                        predicted = prediction_validation[:, run_index]
                        noise_fraction = float(run_row['noise_fraction'])
                        sigma_base = float(run_row['noise_sigma_base'])
                        observed_nrmse = scalar_nrmse(
                            observed_validation, predicted
                        )
                        latent_nrmse = scalar_nrmse(
                            context['radial_validation_true'], predicted
                        )
                        if sigma_base > 0.0:
                            working_whitened_rms = standardized_residual_rms(
                                observed_validation,
                                predicted,
                                sigma_base * q_validation,
                            )
                            true_whitened_rms = standardized_residual_rms(
                                observed_validation,
                                predicted,
                                sigma_base * q_true_validation,
                            )
                            selection_score = working_whitened_rms
                        else:
                            working_whitened_rms = math.nan
                            true_whitened_rms = math.nan
                            selection_score = observed_nrmse
                        records.append(
                            {
                                **run_row.to_dict(),
                                'working_model': model_name,
                                'working_model_label': WORKING_MODELS[model_name]['label'],
                                'n_basis': int(n_basis),
                                'regularization_strength': float(strength),
                                'penalty_power': float(power),
                                'validation_selection_score': float(selection_score),
                                'validation_working_whitened_rms': float(working_whitened_rms),
                                'validation_true_whitened_rms': float(true_whitened_rms),
                                'validation_observed_nrmse': float(observed_nrmse),
                                'validation_latent_nrmse_radial': float(latent_nrmse),
                                'validation_hidden_nrmse_tangential': float(
                                    tangential_nrmse[run_index]
                                ),
                                'validation_hidden_nrmse_3d': float(
                                    vector_nrmse[run_index]
                                ),
                                'normal_matrix_condition': float(condition),
                                'precision_effective_sample_size_train': (
                                    precision_effective_sample_size(train_weights)
                                ),
                            }
                        )
                block_df = pd.DataFrame(records)
                atomic_to_csv(block_df, csv_path)
                atomic_write_json(
                    metadata_path,
                    {
                        'signature': signature,
                        'rows': int(block_df.shape[0]),
                    },
                )
                frames.append(block_df)
                update_progress(
                    'candidate_scan',
                    completed_candidate_blocks=len(
                        list(CHECKPOINT_DIR.glob('candidate_*.json'))
                    ),
                    expected_candidate_blocks=(
                        len(SEED_PAIRS)
                        * len(WORKING_MODELS)
                        * len(POTENTIAL_REGULARIZATION_POWERS)
                        * len(POTENTIAL_REGULARIZATION_STRENGTHS)
                    ),
                    completed_final_seeds=len(
                        list(CHECKPOINT_DIR.glob('final_*.json'))
                    ),
                    expected_final_seeds=len(SEED_PAIRS),
                )
    return pd.concat(frames, ignore_index=True)

# ============================================================
# 6. Candidate scan across seed pairs and nested capacity rule
# ============================================================

all_candidate_frames = []
all_closure_frames = []
profile_diagnostic_records = []
for seed_pair_index, (reference_seed, label_seed) in enumerate(SEED_PAIRS):
    print('-' * 132)
    print(
        f'Candidate scan seed pair {seed_pair_index + 1}/{len(SEED_PAIRS)}: '
        f'reference={reference_seed}, label={label_seed}'
    )
    context = prepare_seed_context(
        seed_pair_index, reference_seed, label_seed
    )
    base_signature = seed_signature(
        seed_pair_index, reference_seed, label_seed
    )
    candidate_seed_df = scan_seed_candidates(context, base_signature)
    all_candidate_frames.append(candidate_seed_df)
    closure_seed_df = maximum_basis_closure(context)
    if not closure_seed_df.empty:
        all_closure_frames.append(closure_seed_df)

    model_radius = radius[context['model_indices']]
    for model_name in WORKING_MODEL_ORDER:
        profile = context['working_profiles'][model_name]
        weights_train = normalized_profile_precision(
            profile, context['train_local']
        )
        profile_diagnostic_records.append(
            {
                'seed_pair_index': seed_pair_index,
                'reference_seed': reference_seed,
                'label_seed': label_seed,
                'working_model': model_name,
                'working_model_label': WORKING_MODELS[model_name]['label'],
                'profile_min_model': float(np.min(profile)),
                'profile_median_model': float(np.median(profile)),
                'profile_max_model': float(np.max(profile)),
                'profile_rms_train': float(
                    np.sqrt(np.mean(np.square(profile[context['train_local']])))
                ),
                'precision_weight_min_train': float(np.min(weights_train)),
                'precision_weight_max_train': float(np.max(weights_train)),
                'precision_effective_sample_size_train': (
                    precision_effective_sample_size(weights_train)
                ),
                'radius_profile_correlation': (
                    float(np.corrcoef(model_radius, profile)[0, 1])
                    if np.std(profile) > 0.0
                    else math.nan
                ),
            }
        )
    release_seed_context(context)

candidate_df = pd.concat(all_candidate_frames, ignore_index=True)
closure_df = (
    pd.concat(all_closure_frames, ignore_index=True)
    if all_closure_frames
    else pd.DataFrame()
)
profile_diagnostics_df = pd.DataFrame(profile_diagnostic_records)

candidate_path = OUTPUT_DIR / f'{OUTPUT_PREFIX}_candidate_grid.csv'
closure_path = OUTPUT_DIR / f'{OUTPUT_PREFIX}_closure.csv'
profile_path = OUTPUT_DIR / f'{OUTPUT_PREFIX}_profile_diagnostics.csv'
atomic_to_csv(candidate_df, candidate_path)
if not closure_df.empty:
    atomic_to_csv(closure_df, closure_path)
atomic_to_csv(profile_diagnostics_df, profile_path)

within_basis_selected_df = (
    candidate_df.sort_values(
        [
            'seed_pair_index',
            'working_model',
            'run_key',
            'n_basis',
            'validation_selection_score',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    )
    .groupby(
        ['seed_pair_index', 'working_model', 'run_key', 'n_basis'],
        as_index=False,
    )
    .first()
)

raw_operational_selectors_df = (
    candidate_df.sort_values(
        [
            'seed_pair_index',
            'working_model',
            'run_key',
            'validation_selection_score',
            'n_basis',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    )
    .groupby(['seed_pair_index', 'working_model', 'run_key'], as_index=False)
    .first()
)

true_selector_source = candidate_df.copy()
true_selector_source['true_selector_score'] = np.where(
    true_selector_source['noise_fraction'].to_numpy(dtype=float) > 0.0,
    true_selector_source['validation_true_whitened_rms'].to_numpy(dtype=float),
    true_selector_source['validation_latent_nrmse_radial'].to_numpy(dtype=float),
)
true_whitened_selectors_df = (
    true_selector_source.sort_values(
        [
            'seed_pair_index',
            'working_model',
            'run_key',
            'true_selector_score',
            'n_basis',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    )
    .groupby(['seed_pair_index', 'working_model', 'run_key'], as_index=False)
    .first()
)
hidden_3d_selectors_df = (
    candidate_df.sort_values(
        [
            'seed_pair_index',
            'working_model',
            'run_key',
            'validation_hidden_nrmse_3d',
            'n_basis',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    )
    .groupby(['seed_pair_index', 'working_model', 'run_key'], as_index=False)
    .first()
)

seed_capacity_mean_df = (
    within_basis_selected_df.groupby(
        ['seed_pair_index', 'working_model', 'noise_fraction', 'n_basis'],
        as_index=False,
    )
    .agg(
        validation_selection_score=(
            'validation_selection_score', 'mean'
        ),
        validation_observed_nrmse=('validation_observed_nrmse', 'mean'),
        validation_latent_nrmse_radial=(
            'validation_latent_nrmse_radial', 'mean'
        ),
        validation_hidden_nrmse_3d=(
            'validation_hidden_nrmse_3d', 'mean'
        ),
        validation_hidden_nrmse_tangential=(
            'validation_hidden_nrmse_tangential', 'mean'
        ),
    )
)
capacity_aggregate_df = (
    seed_capacity_mean_df.groupby(
        ['working_model', 'noise_fraction', 'n_basis'],
        as_index=False,
    )
    .agg(
        validation_mean=('validation_selection_score', 'mean'),
        validation_std_between_seeds=('validation_selection_score', 'std'),
        validation_observed_nrmse_mean=(
            'validation_observed_nrmse', 'mean'
        ),
        validation_latent_nrmse_radial_mean=(
            'validation_latent_nrmse_radial', 'mean'
        ),
        validation_hidden_nrmse_3d_mean=(
            'validation_hidden_nrmse_3d', 'mean'
        ),
        validation_hidden_nrmse_tangential_mean=(
            'validation_hidden_nrmse_tangential', 'mean'
        ),
        n_seed_pairs=('seed_pair_index', 'nunique'),
    )
)
capacity_aggregate_df['validation_sem_between_seeds'] = (
    capacity_aggregate_df['validation_std_between_seeds']
    / np.sqrt(capacity_aggregate_df['n_seed_pairs'])
)

capacity_selection_records = []
maximum_parameter_count = int(max(POTENTIAL_BASIS_SIZES))
for model_name in WORKING_MODEL_ORDER:
    for noise_fraction in NOISE_FRACTIONS:
        curve = capacity_aggregate_df[
            (capacity_aggregate_df['working_model'] == model_name)
            & np.isclose(
                capacity_aggregate_df['noise_fraction'], noise_fraction
            )
        ].sort_values('n_basis')
        if curve.empty:
            continue
        minimum_row = curve.sort_values(
            ['validation_mean', 'n_basis'], kind='mergesort'
        ).iloc[0]
        minimum_basis = int(minimum_row['n_basis'])
        minimum_mean = float(minimum_row['validation_mean'])
        minimum_sem = float(minimum_row['validation_sem_between_seeds'])
        one_se_threshold = minimum_mean + minimum_sem
        one_se_basis = int(
            curve[
                curve['validation_mean'] <= one_se_threshold
            ]['n_basis'].min()
        )
        plateau_basis, adjacent_improvement = practical_plateau_basis(
            curve['n_basis'].to_numpy(dtype=np.int64),
            curve['validation_mean'].to_numpy(dtype=np.float64),
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
        raw_group = raw_operational_selectors_df[
            (raw_operational_selectors_df['working_model'] == model_name)
            & np.isclose(
                raw_operational_selectors_df['noise_fraction'],
                noise_fraction,
            )
        ]
        per_seed_max_fraction = (
            raw_group.assign(
                selected_max=(
                    raw_group['n_basis'].to_numpy(dtype=np.int64)
                    == maximum_parameter_count
                ).astype(float)
            )
            .groupby('seed_pair_index', as_index=False)['selected_max']
            .mean()
        )
        maximum_basis_selected_fraction = float(
            per_seed_max_fraction['selected_max'].mean()
        )
        one_se_boundary = one_se_basis == maximum_parameter_count
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
        practical_risk_row = curve[curve['n_basis'] == practical_basis].iloc[0]
        capacity_selection_records.append(
            {
                'working_model': model_name,
                'working_model_label': WORKING_MODELS[model_name]['label'],
                'noise_fraction': float(noise_fraction),
                'n_seed_pairs': int(len(SEED_PAIRS)),
                'validation_minimum_basis': minimum_basis,
                'validation_minimum_mean': minimum_mean,
                'validation_minimum_sem_between_seeds': minimum_sem,
                'one_standard_error_threshold': one_se_threshold,
                'one_standard_error_basis': one_se_basis,
                'practical_plateau_basis': (
                    int(plateau_basis)
                    if plateau_basis is not None
                    else math.nan
                ),
                'plateau_found': plateau_basis is not None,
                'combined_practical_basis': practical_basis,
                'practical_validation_mean': float(
                    practical_risk_row['validation_mean']
                ),
                'practical_validation_sem_between_seeds': float(
                    practical_risk_row['validation_sem_between_seeds']
                ),
                'strict_argmin_boundary': minimum_basis == maximum_parameter_count,
                'one_se_boundary': one_se_boundary,
                'maximum_basis_selected_fraction_equal_seed_weight': (
                    maximum_basis_selected_fraction
                ),
                'last_adjacent_validation_improvement': last_improvement,
                'practical_converged': practical_converged,
                'needs_larger_eigensystem': not practical_converged,
            }
        )
capacity_selection_df = pd.DataFrame(capacity_selection_records).sort_values(
    ['working_model', 'noise_fraction']
)
practical_basis_lookup = {
    (row['working_model'], float(row['noise_fraction'])): int(
        row['combined_practical_basis']
    )
    for _, row in capacity_selection_df.iterrows()
}

atomic_to_csv(
    within_basis_selected_df,
    OUTPUT_DIR / f'{OUTPUT_PREFIX}_within_basis_selected.csv',
)
atomic_to_csv(
    raw_operational_selectors_df,
    OUTPUT_DIR / f'{OUTPUT_PREFIX}_raw_operational_selectors.csv',
)
atomic_to_csv(
    true_whitened_selectors_df,
    OUTPUT_DIR / f'{OUTPUT_PREFIX}_true_whitened_selectors.csv',
)
atomic_to_csv(
    hidden_3d_selectors_df,
    OUTPUT_DIR / f'{OUTPUT_PREFIX}_hidden_3d_selectors.csv',
)
atomic_to_csv(
    seed_capacity_mean_df,
    OUTPUT_DIR / f'{OUTPUT_PREFIX}_seed_capacity_mean.csv',
)
atomic_to_csv(
    capacity_aggregate_df,
    OUTPUT_DIR / f'{OUTPUT_PREFIX}_capacity_aggregate.csv',
)
atomic_to_csv(
    capacity_selection_df,
    OUTPUT_DIR / f'{OUTPUT_PREFIX}_capacity_selection.csv',
)

print('-' * 132)
print('Nested heteroscedastic capacity selection')
display(capacity_selection_df)
if np.any(~capacity_selection_df['practical_converged'].astype(bool)):
    warnings.warn(
        '少なくとも一つのworking model/noise levelでpractical convergenceが確定しなかった。'
    )

# ============================================================
# 7. Final post-selection fits and covariance diagnostics
# ============================================================


def selected_candidate_at_basis(
    seed_candidates: pd.DataFrame,
    run_key: str,
    model_name: str,
    n_basis: int,
) -> pd.Series:
    subset = seed_candidates[
        (seed_candidates['run_key'] == run_key)
        & (seed_candidates['working_model'] == model_name)
        & (seed_candidates['n_basis'] == int(n_basis))
    ]
    if subset.empty:
        raise RuntimeError(
            f'run={run_key}, model={model_name}, basis={n_basis}のcandidateがない。'
        )
    return subset.sort_values(
        [
            'validation_selection_score',
            'regularization_strength',
            'penalty_power',
        ],
        kind='mergesort',
    ).iloc[0]


def selector_row(
    selector_df: pd.DataFrame,
    run_key: str,
    model_name: str,
) -> pd.Series:
    subset = selector_df[
        (selector_df['run_key'] == run_key)
        & (selector_df['working_model'] == model_name)
    ]
    if subset.shape[0] != 1:
        raise RuntimeError(
            f'run={run_key}, model={model_name}のselector rowが一意でない。'
        )
    return subset.iloc[0]


def dataframe_digest(dataframe: pd.DataFrame) -> str:
    ordered_columns = sorted(dataframe.columns)
    ordered = dataframe[ordered_columns].sort_values(
        ordered_columns, kind='mergesort'
    ).reset_index(drop=True)
    csv_bytes = ordered.to_csv(
        index=False, float_format='%.17g', lineterminator='\n'
    ).encode('utf-8')
    return hashlib.sha256(csv_bytes).hexdigest()


def seed_candidate_checkpoint_digest(
    seed_pair_index: int, reference_seed: int, label_seed: int
) -> str:
    prefix = f'candidate_seed{seed_pair_index:02d}_ref{reference_seed}_label{label_seed}_'
    files = sorted(CHECKPOINT_DIR.glob(prefix + '*.csv'))
    if not files:
        raise RuntimeError(f'Candidate checkpoint filesが見つからない: {prefix}')
    digest = hashlib.sha256()
    for path in files:
        digest.update(path.name.encode('utf-8'))
        digest.update(file_digest(path).encode('ascii'))
    return digest.hexdigest()


def fit_from_configuration(
    context: dict,
    model_name: str,
    run_index: int,
    n_basis: int,
    strength: float,
    power: float,
    model_fit_cache: dict,
) -> dict:
    model_cache = model_fit_cache[model_name]
    key = (int(n_basis), float(strength), float(power))
    if key not in model_cache['factor_cache']:
        regularization = regularization_diagonal(
            potential_generator_values, int(n_basis), float(power)
        )
        factor_kind, factor, system = factor_regularized_system(
            model_cache['gram_max'][:n_basis, :n_basis],
            regularization,
            float(strength),
        )
        model_cache['factor_cache'][key] = {
            'regularization': regularization,
            'factor_kind': factor_kind,
            'factor': factor,
            'system': system,
        }
    factor_record = model_cache['factor_cache'][key]
    coefficients = solve_factored(
        factor_record['factor_kind'],
        factor_record['factor'],
        model_cache['rhs_all'][:n_basis, run_index],
    )
    radial_prediction = context['A_test_max'][:, :n_basis] @ coefficients
    vector_prediction = potential_velocity(
        context['gradient_test'], coefficients, n_basis
    )
    return {
        'model_name': model_name,
        'n_basis': int(n_basis),
        'strength': float(strength),
        'power': float(power),
        'regularization': factor_record['regularization'],
        'factor_kind': factor_record['factor_kind'],
        'factor': factor_record['factor'],
        'system': factor_record['system'],
        'coefficients': coefficients,
        'radial_prediction': radial_prediction,
        'vector_prediction': vector_prediction,
    }


def covariance_for_fit(
    context: dict,
    fit_record: dict,
    model_fit_cache: dict,
    noise_fraction: float,
    covariance_kind: str,
    global_scale_factor: float = 1.0,
) -> np.ndarray:
    model_name = fit_record['model_name']
    n_basis = fit_record['n_basis']
    key = (
        covariance_kind,
        n_basis,
        fit_record['strength'],
        fit_record['power'],
        float(noise_fraction),
        float(global_scale_factor),
    )
    covariance_cache = model_fit_cache[model_name]['covariance_cache']
    if key in covariance_cache:
        return covariance_cache[key]
    sigma_base = float(noise_fraction) * context['train_radial_rms']
    if sigma_base == 0.0:
        covariance = np.zeros((n_basis, n_basis), dtype=np.float64)
    else:
        fit_local = context['fit_local']
        if covariance_kind == 'model':
            profile = context['working_profiles'][model_name][fit_local]
            sigma = global_scale_factor * sigma_base * profile
        elif covariance_kind == 'oracle':
            sigma = sigma_base * context['true_profile'][fit_local]
        else:
            raise ValueError(f'未知のcovariance_kind: {covariance_kind}')
        covariance = regularized_covariance(
            context['A_fit_max'][:, :n_basis],
            model_fit_cache[model_name]['fit_weights'],
            sigma,
            fit_record['factor_kind'],
            fit_record['factor'],
        )
    covariance_cache[key] = covariance
    return covariance


all_final_frames = []
all_global_scale_frames = []
for seed_pair_index, (reference_seed, label_seed) in enumerate(SEED_PAIRS):
    final_csv_path, final_metadata_path = seed_final_paths(
        seed_pair_index, reference_seed, label_seed
    )
    seed_candidates = candidate_df[
        candidate_df['seed_pair_index'] == seed_pair_index
    ].copy()
    base_signature = seed_signature(
        seed_pair_index, reference_seed, label_seed
    )
    final_signature = stable_signature(
        {
            'script_version': SCRIPT_VERSION,
            'base_signature': base_signature,
            'candidate_digest': seed_candidate_checkpoint_digest(
                seed_pair_index, reference_seed, label_seed
            ),
            'practical_basis_lookup': {
                f'{model}:{noise:g}': basis
                for (model, noise), basis in practical_basis_lookup.items()
            },
            'global_scale_factors': GLOBAL_SCALE_FACTORS,
        }
    )
    use_checkpoint = False
    if (
        RESUME_FROM_CHECKPOINT
        and not FORCE_RECOMPUTE
        and final_csv_path.is_file()
        and final_metadata_path.is_file()
    ):
        try:
            metadata = json.loads(
                final_metadata_path.read_text(encoding='utf-8')
            )
            use_checkpoint = metadata.get('signature') == final_signature
        except Exception:
            use_checkpoint = False
    global_scale_csv = final_csv_path.with_name(
        final_csv_path.stem + '_global_scale.csv'
    )
    if use_checkpoint and global_scale_csv.is_file():
        print(
            f'[Final checkpoint] reuse: '
            f'{seed_tag(seed_pair_index, reference_seed, label_seed)}'
        )
        all_final_frames.append(pd.read_csv(final_csv_path))
        all_global_scale_frames.append(pd.read_csv(global_scale_csv))
        continue

    print('-' * 132)
    print(
        f'Final evaluation seed pair {seed_pair_index + 1}/{len(SEED_PAIRS)}: '
        f'reference={reference_seed}, label={label_seed}'
    )
    context = prepare_seed_context(
        seed_pair_index, reference_seed, label_seed
    )
    run_metadata_df = context['run_metadata_df']
    fit_local = context['fit_local']
    model_fit_cache = {}
    for model_name in WORKING_MODEL_ORDER:
        fit_weights = normalized_profile_precision(
            context['working_profiles'][model_name], fit_local
        )
        gram_max, rhs_all, _ = weighted_cross_products(
            context['A_fit_max'],
            context['observed_fit_matrix'],
            fit_weights,
        )
        model_fit_cache[model_name] = {
            'fit_weights': fit_weights,
            'gram_max': gram_max,
            'rhs_all': rhs_all,
            'factor_cache': {},
            'covariance_cache': {},
        }

    final_records = []
    global_scale_records = []
    for run_index, metadata in run_metadata_df.iterrows():
        run_key = str(metadata['run_key'])
        noise_fraction = float(metadata['noise_fraction'])
        sigma_base = float(metadata['noise_sigma_base'])
        correct_practical_basis = practical_basis_lookup[
            (PRIMARY_MODEL, noise_fraction)
        ]
        correct_practical_candidate = selected_candidate_at_basis(
            seed_candidates,
            run_key,
            PRIMARY_MODEL,
            correct_practical_basis,
        )

        protocol_configurations = []
        for model_name in WORKING_MODEL_ORDER:
            model_practical_basis = practical_basis_lookup[
                (model_name, noise_fraction)
            ]
            model_practical_candidate = selected_candidate_at_basis(
                seed_candidates,
                run_key,
                model_name,
                model_practical_basis,
            )
            model_raw_candidate = selector_row(
                raw_operational_selectors_df,
                run_key,
                model_name,
            )
            protocol_configurations.extend(
                [
                    (
                        'retuned_practical',
                        model_name,
                        model_practical_candidate,
                    ),
                    (
                        'matched_correct_gls_practical',
                        model_name,
                        correct_practical_candidate,
                    ),
                    (
                        'retuned_raw',
                        model_name,
                        model_raw_candidate,
                    ),
                ]
            )

        for protocol, model_name, selected in protocol_configurations:
            fit_record = fit_from_configuration(
                context,
                model_name,
                run_index,
                int(selected['n_basis']),
                float(selected['regularization_strength']),
                float(selected['penalty_power']),
                model_fit_cache,
            )
            observed_test = context['observed_test_matrix'][:, run_index]
            true_sigma_test = context['true_sigma_test_matrix'][:, run_index]
            q_work_test = context['working_profiles'][model_name][
                context['test_local']
            ]
            assumed_sigma_test = sigma_base * q_work_test
            metrics = vector_field_metrics(
                context['velocity_test_true'],
                fit_record['vector_prediction'],
                context['line_of_sight_test'],
            )
            base_record = {
                **metadata.to_dict(),
                'protocol': protocol,
                'working_model': model_name,
                'working_model_label': WORKING_MODELS[model_name]['label'],
                'n_basis': fit_record['n_basis'],
                'regularization_strength': fit_record['strength'],
                'penalty_power': fit_record['power'],
                'validation_selection_score': float(
                    selected['validation_selection_score']
                ),
                'test_observed_nrmse_radial': scalar_nrmse(
                    observed_test, fit_record['radial_prediction']
                ),
                'test_latent_nrmse_radial': scalar_nrmse(
                    context['radial_test_true'],
                    fit_record['radial_prediction'],
                ),
                'test_working_measurement_whitened_residual_rms': (
                    standardized_residual_rms(
                        observed_test,
                        fit_record['radial_prediction'],
                        assumed_sigma_test,
                    )
                    if sigma_base > 0.0
                    else math.nan
                ),
                'test_true_measurement_whitened_residual_rms': (
                    standardized_residual_rms(
                        observed_test,
                        fit_record['radial_prediction'],
                        true_sigma_test,
                    )
                    if sigma_base > 0.0
                    else math.nan
                ),
                **{f'test_hidden_{key}': value for key, value in metrics.items()},
                'precision_effective_sample_size_fit': (
                    precision_effective_sample_size(
                        model_fit_cache[model_name]['fit_weights']
                    )
                ),
                'normal_matrix_condition': normal_matrix_condition(
                    fit_record['system']
                )[0],
                'effective_degrees_of_freedom': exact_effective_degrees_of_freedom(
                    model_fit_cache[model_name]['gram_max'][
                        :fit_record['n_basis'], :fit_record['n_basis']
                    ],
                    fit_record['regularization'],
                    fit_record['strength'],
                ),
            }

            if protocol == 'retuned_practical' and sigma_base > 0.0:
                covariance_model = covariance_for_fit(
                    context,
                    fit_record,
                    model_fit_cache,
                    noise_fraction,
                    'model',
                    1.0,
                )
                covariance_oracle = covariance_for_fit(
                    context,
                    fit_record,
                    model_fit_cache,
                    noise_fraction,
                    'oracle',
                    1.0,
                )
                radial_variance_model = radial_prediction_variance(
                    context['A_test_max'][:, :fit_record['n_basis']],
                    covariance_model,
                )
                radial_variance_oracle = radial_prediction_variance(
                    context['A_test_max'][:, :fit_record['n_basis']],
                    covariance_oracle,
                )
                component_variance_model = vector_component_prediction_variance(
                    context['gradient_test'][:, :fit_record['n_basis'], :],
                    covariance_model,
                )
                component_variance_oracle = vector_component_prediction_variance(
                    context['gradient_test'][:, :fit_record['n_basis'], :],
                    covariance_oracle,
                )
                for label, z_value in COVERAGE_Z.items():
                    base_record[f'coverage_model_observed_radial_{label}'] = (
                        marginal_interval_coverage(
                            observed_test,
                            fit_record['radial_prediction'],
                            radial_variance_model + np.square(assumed_sigma_test),
                            z_value,
                        )
                    )
                    base_record[f'coverage_model_latent_radial_{label}'] = (
                        marginal_interval_coverage(
                            context['radial_test_true'],
                            fit_record['radial_prediction'],
                            radial_variance_model,
                            z_value,
                        )
                    )
                    base_record[f'coverage_model_latent_component_{label}'] = (
                        marginal_interval_coverage(
                            context['velocity_test_true'],
                            fit_record['vector_prediction'],
                            component_variance_model,
                            z_value,
                        )
                    )
                    base_record[f'coverage_oracle_observed_radial_{label}'] = (
                        marginal_interval_coverage(
                            observed_test,
                            fit_record['radial_prediction'],
                            radial_variance_oracle + np.square(true_sigma_test),
                            z_value,
                        )
                    )
                    base_record[f'coverage_oracle_latent_radial_{label}'] = (
                        marginal_interval_coverage(
                            context['radial_test_true'],
                            fit_record['radial_prediction'],
                            radial_variance_oracle,
                            z_value,
                        )
                    )
                    base_record[f'coverage_oracle_latent_component_{label}'] = (
                        marginal_interval_coverage(
                            context['velocity_test_true'],
                            fit_record['vector_prediction'],
                            component_variance_oracle,
                            z_value,
                        )
                    )
                base_record['model_coefficient_se_median'] = float(
                    np.median(np.sqrt(np.maximum(np.diag(covariance_model), 0.0)))
                )
                base_record['oracle_coefficient_se_median'] = float(
                    np.median(np.sqrt(np.maximum(np.diag(covariance_oracle), 0.0)))
                )

                for scale_factor in GLOBAL_SCALE_FACTORS:
                    covariance_scaled = covariance_for_fit(
                        context,
                        fit_record,
                        model_fit_cache,
                        noise_fraction,
                        'model',
                        scale_factor,
                    )
                    radial_variance_scaled = radial_prediction_variance(
                        context['A_test_max'][:, :fit_record['n_basis']],
                        covariance_scaled,
                    )
                    assumed_sigma_scaled = scale_factor * assumed_sigma_test
                    global_scale_record = {
                        **metadata.to_dict(),
                        'working_model': model_name,
                        'global_scale_factor': float(scale_factor),
                        'n_basis': fit_record['n_basis'],
                        'regularization_strength': fit_record['strength'],
                        'penalty_power': fit_record['power'],
                        'coefficient_point_estimate_invariant': True,
                        'measurement_whitened_residual_rms': (
                            standardized_residual_rms(
                                observed_test,
                                fit_record['radial_prediction'],
                                assumed_sigma_scaled,
                            )
                        ),
                    }
                    for label, z_value in COVERAGE_Z.items():
                        global_scale_record[
                            f'coverage_observed_radial_{label}'
                        ] = marginal_interval_coverage(
                            observed_test,
                            fit_record['radial_prediction'],
                            radial_variance_scaled
                            + np.square(assumed_sigma_scaled),
                            z_value,
                        )
                        global_scale_record[
                            f'coverage_latent_radial_{label}'
                        ] = marginal_interval_coverage(
                            context['radial_test_true'],
                            fit_record['radial_prediction'],
                            radial_variance_scaled,
                            z_value,
                        )
                    global_scale_records.append(global_scale_record)
            final_records.append(base_record)

    final_seed_df = pd.DataFrame(final_records)
    global_scale_seed_df = pd.DataFrame(global_scale_records)
    atomic_to_csv(final_seed_df, final_csv_path)
    atomic_to_csv(global_scale_seed_df, global_scale_csv)
    atomic_write_json(
        final_metadata_path,
        {
            'signature': final_signature,
            'rows': int(final_seed_df.shape[0]),
            'global_scale_rows': int(global_scale_seed_df.shape[0]),
        },
    )
    all_final_frames.append(final_seed_df)
    all_global_scale_frames.append(global_scale_seed_df)
    release_seed_context(context)
    update_progress(
        'final_evaluation',
        completed_candidate_blocks=len(
            list(CHECKPOINT_DIR.glob('candidate_*.json'))
        ),
        expected_candidate_blocks=(
            len(SEED_PAIRS)
            * len(WORKING_MODELS)
            * len(POTENTIAL_REGULARIZATION_POWERS)
            * len(POTENTIAL_REGULARIZATION_STRENGTHS)
        ),
        completed_final_seeds=len(
            list(CHECKPOINT_DIR.glob('final_*.json'))
        ),
        expected_final_seeds=len(SEED_PAIRS),
    )

final_df = pd.concat(all_final_frames, ignore_index=True)
global_scale_df = pd.concat(all_global_scale_frames, ignore_index=True)
atomic_to_csv(final_df, OUTPUT_DIR / f'{OUTPUT_PREFIX}_final_evaluation.csv')
atomic_to_csv(
    global_scale_df,
    OUTPUT_DIR / f'{OUTPUT_PREFIX}_global_scale_audit.csv',
)

# ============================================================
# 8. Aggregation, selector diagnostics, and paired comparisons
# ============================================================

summary_metric_columns = [
    'test_observed_nrmse_radial',
    'test_latent_nrmse_radial',
    'test_hidden_nrmse_tangential',
    'test_hidden_nrmse_3d',
    'test_hidden_direction_error_median_deg',
    'test_working_measurement_whitened_residual_rms',
    'test_true_measurement_whitened_residual_rms',
    'precision_effective_sample_size_fit',
    'normal_matrix_condition',
    'effective_degrees_of_freedom',
    'coverage_model_observed_radial_68',
    'coverage_model_observed_radial_95',
    'coverage_model_latent_radial_68',
    'coverage_model_latent_radial_95',
    'coverage_model_latent_component_68',
    'coverage_model_latent_component_95',
    'coverage_oracle_observed_radial_68',
    'coverage_oracle_observed_radial_95',
    'coverage_oracle_latent_radial_68',
    'coverage_oracle_latent_radial_95',
    'coverage_oracle_latent_component_68',
    'coverage_oracle_latent_component_95',
]
summary_metric_columns = [
    column for column in summary_metric_columns if column in final_df.columns
]

seed_aggregate_df = (
    final_df.groupby(
        ['seed_pair_index', 'protocol', 'working_model', 'noise_fraction'],
        as_index=False,
    )[summary_metric_columns]
    .mean()
)
aggregate_records = []
for keys, group in seed_aggregate_df.groupby(
    ['protocol', 'working_model', 'noise_fraction']
):
    protocol, model_name, noise_fraction = keys
    record = {
        'protocol': protocol,
        'working_model': model_name,
        'working_model_label': WORKING_MODELS[model_name]['label'],
        'noise_fraction': float(noise_fraction),
        'n_seed_pairs': int(group['seed_pair_index'].nunique()),
    }
    for column in summary_metric_columns:
        values = group[column].to_numpy(dtype=np.float64)
        values = values[np.isfinite(values)]
        record[f'{column}_mean_equal_seed_weight'] = (
            float(np.mean(values)) if values.size else math.nan
        )
        record[f'{column}_std_between_seeds'] = (
            float(np.std(values, ddof=1)) if values.size > 1 else 0.0
        )
        record[f'{column}_sem_between_seeds'] = (
            float(np.std(values, ddof=1) / np.sqrt(values.size))
            if values.size > 1
            else 0.0
        )
    aggregate_records.append(record)
aggregate_df = pd.DataFrame(aggregate_records).sort_values(
    ['protocol', 'working_model', 'noise_fraction']
)

practical_configurations_df = final_df[
    final_df['protocol'] == 'retuned_practical'
][
    [
        'seed_pair_index',
        'reference_seed',
        'label_seed',
        'noise_fraction',
        'noise_realization',
        'working_model',
        'n_basis',
        'regularization_strength',
        'penalty_power',
    ]
].copy()
hyperparameter_frequency_df = (
    practical_configurations_df.groupby(
        [
            'working_model',
            'noise_fraction',
            'n_basis',
            'regularization_strength',
            'penalty_power',
        ],
        as_index=False,
    )
    .size()
    .rename(columns={'size': 'count'})
    .sort_values(
        ['working_model', 'noise_fraction', 'count'],
        ascending=[True, True, False],
    )
)

# Selector comparison.
practical_selector_rows = []
for _, operational in raw_operational_selectors_df.iterrows():
    model_name = str(operational['working_model'])
    noise_fraction = float(operational['noise_fraction'])
    practical_selector_rows.append(
        selected_candidate_at_basis(
            candidate_df[
                candidate_df['seed_pair_index']
                == int(operational['seed_pair_index'])
            ],
            str(operational['run_key']),
            model_name,
            practical_basis_lookup[(model_name, noise_fraction)],
        ).to_dict()
    )
practical_selectors_df = pd.DataFrame(practical_selector_rows)

selector_records = []
selector_pairs = [
    (
        'raw_operational_vs_true_whitened',
        raw_operational_selectors_df,
        true_whitened_selectors_df,
    ),
    (
        'raw_operational_vs_hidden_3d',
        raw_operational_selectors_df,
        hidden_3d_selectors_df,
    ),
    (
        'practical_operational_vs_hidden_3d',
        practical_selectors_df,
        hidden_3d_selectors_df,
    ),
]
for comparison_name, left_df, right_df in selector_pairs:
    merged = left_df.merge(
        right_df,
        on=['seed_pair_index', 'working_model', 'run_key'],
        suffixes=('_left', '_right'),
    )
    for _, row in merged.iterrows():
        selector_records.append(
            {
                'comparison': comparison_name,
                'seed_pair_index': int(row['seed_pair_index']),
                'working_model': row['working_model'],
                'run_key': row['run_key'],
                'noise_fraction': float(row['noise_fraction_left']),
                'noise_realization': int(row['noise_realization_left']),
                'left_n_basis': int(row['n_basis_left']),
                'right_n_basis': int(row['n_basis_right']),
                'basis_difference_left_minus_right': int(
                    row['n_basis_left'] - row['n_basis_right']
                ),
                'same_candidate': bool(
                    int(row['n_basis_left']) == int(row['n_basis_right'])
                    and np.isclose(
                        row['regularization_strength_left'],
                        row['regularization_strength_right'],
                    )
                    and np.isclose(
                        row['penalty_power_left'], row['penalty_power_right']
                    )
                ),
            }
        )
selector_differences_df = pd.DataFrame(selector_records)
selector_seed_mean_df = (
    selector_differences_df.groupby(
        ['comparison', 'working_model', 'noise_fraction', 'seed_pair_index'],
        as_index=False,
    )
    .agg(
        same_candidate_fraction=('same_candidate', 'mean'),
        mean_basis_difference=(
            'basis_difference_left_minus_right', 'mean'
        ),
    )
)
selector_aggregate_df = (
    selector_seed_mean_df.groupby(
        ['comparison', 'working_model', 'noise_fraction'],
        as_index=False,
    )
    .agg(
        same_candidate_fraction_equal_seed_weight=(
            'same_candidate_fraction', 'mean'
        ),
        mean_basis_difference_equal_seed_weight=(
            'mean_basis_difference', 'mean'
        ),
        n_seed_pairs=('seed_pair_index', 'nunique'),
    )
)

# Paired estimator differences.
comparison_specs = [
    ('ols_minus_correct', 'ols', 'gls_correct'),
    ('flattened_minus_correct', 'gls_flattened', 'gls_correct'),
    ('flattened_minus_ols', 'gls_flattened', 'ols'),
]
paired_metrics = [
    'test_observed_nrmse_radial',
    'test_latent_nrmse_radial',
    'test_hidden_nrmse_tangential',
    'test_hidden_nrmse_3d',
    'test_working_measurement_whitened_residual_rms',
    'test_true_measurement_whitened_residual_rms',
]
paired_records = []
for protocol in ['retuned_practical', 'matched_correct_gls_practical']:
    protocol_df = final_df[final_df['protocol'] == protocol]
    for comparison_name, left_model, right_model in comparison_specs:
        left = protocol_df[protocol_df['working_model'] == left_model]
        right = protocol_df[protocol_df['working_model'] == right_model]
        merged = left.merge(
            right,
            on=[
                'seed_pair_index',
                'reference_seed',
                'label_seed',
                'run_key',
                'noise_fraction',
                'noise_realization',
                'noise_seed',
            ],
            suffixes=('_left', '_right'),
        )
        for _, row in merged.iterrows():
            record = {
                'protocol': protocol,
                'comparison': comparison_name,
                'left_model': left_model,
                'right_model': right_model,
                'seed_pair_index': int(row['seed_pair_index']),
                'reference_seed': int(row['reference_seed']),
                'label_seed': int(row['label_seed']),
                'run_key': row['run_key'],
                'noise_fraction': float(row['noise_fraction']),
                'noise_realization': int(row['noise_realization']),
            }
            for metric in paired_metrics:
                record[f'delta_{metric}'] = float(
                    row[f'{metric}_left'] - row[f'{metric}_right']
                )
            paired_records.append(record)
paired_differences_df = pd.DataFrame(paired_records)

paired_summary_records = []
for keys, group in paired_differences_df.groupby(
    ['protocol', 'comparison', 'noise_fraction']
):
    protocol, comparison_name, noise_fraction = keys
    for metric in paired_metrics:
        delta_column = f'delta_{metric}'
        values = group[delta_column].to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        lower, upper = hierarchical_bootstrap_interval(
            group,
            delta_column,
            seed=BOOTSTRAP_SEED
            + int(round(1000 * float(noise_fraction)))
            + paired_metrics.index(metric) * 37,
        )
        seed_means = (
            group.groupby('seed_pair_index')[delta_column]
            .mean()
            .to_numpy(dtype=np.float64)
        )
        paired_summary_records.append(
            {
                'protocol': protocol,
                'comparison': comparison_name,
                'noise_fraction': float(noise_fraction),
                'metric': metric,
                'n_seed_pairs': int(group['seed_pair_index'].nunique()),
                'n_runs': int(finite.size),
                'delta_mean': float(np.mean(finite)) if finite.size else math.nan,
                'delta_median': float(np.median(finite)) if finite.size else math.nan,
                'delta_ci_low': lower,
                'delta_ci_high': upper,
                'seed_mean_wins_left_lower': int(np.sum(seed_means < 0.0)),
                'seed_mean_losses_left_higher': int(np.sum(seed_means > 0.0)),
                'seed_mean_ties': int(np.sum(seed_means == 0.0)),
                'seed_mean_sign_test_pvalue_two_sided': (
                    exact_sign_test_two_sided(seed_means)
                ),
            }
        )
paired_summary_df = pd.DataFrame(paired_summary_records)

# Global scale audit aggregation.
global_scale_seed_mean_df = (
    global_scale_df.groupby(
        [
            'seed_pair_index',
            'working_model',
            'noise_fraction',
            'global_scale_factor',
        ],
        as_index=False,
    )
    .agg(
        measurement_whitened_residual_rms=(
            'measurement_whitened_residual_rms', 'mean'
        ),
        coverage_observed_radial_68=(
            'coverage_observed_radial_68', 'mean'
        ),
        coverage_observed_radial_95=(
            'coverage_observed_radial_95', 'mean'
        ),
        coverage_latent_radial_68=(
            'coverage_latent_radial_68', 'mean'
        ),
        coverage_latent_radial_95=(
            'coverage_latent_radial_95', 'mean'
        ),
    )
)
global_scale_aggregate_df = (
    global_scale_seed_mean_df.groupby(
        ['working_model', 'noise_fraction', 'global_scale_factor'],
        as_index=False,
    )
    .agg(
        measurement_whitened_residual_rms_mean=(
            'measurement_whitened_residual_rms', 'mean'
        ),
        coverage_observed_radial_68_mean=(
            'coverage_observed_radial_68', 'mean'
        ),
        coverage_observed_radial_95_mean=(
            'coverage_observed_radial_95', 'mean'
        ),
        coverage_latent_radial_68_mean=(
            'coverage_latent_radial_68', 'mean'
        ),
        coverage_latent_radial_95_mean=(
            'coverage_latent_radial_95', 'mean'
        ),
        n_seed_pairs=('seed_pair_index', 'nunique'),
    )
)

# Save tabular results.
for dataframe, suffix in [
    (seed_aggregate_df, 'seed_aggregate'),
    (aggregate_df, 'aggregate'),
    (practical_configurations_df, 'practical_configurations'),
    (hyperparameter_frequency_df, 'hyperparameter_frequency'),
    (practical_selectors_df, 'practical_selectors'),
    (selector_differences_df, 'selector_differences'),
    (selector_aggregate_df, 'selector_aggregate'),
    (paired_differences_df, 'paired_differences'),
    (paired_summary_df, 'paired_summary'),
    (global_scale_seed_mean_df, 'global_scale_seed_mean'),
    (global_scale_aggregate_df, 'global_scale_aggregate'),
    (geometry_continuity_df, 'geometry_continuity'),
    (gradient_diagnostics_df, 'gradient_diagnostics'),
]:
    atomic_to_csv(dataframe, OUTPUT_DIR / f'{OUTPUT_PREFIX}_{suffix}.csv')

# ============================================================
# 9. Optional Stage-1b comparison, figures, and decision
# ============================================================

stage1b_capacity_path = (
    STAGE1B_OUTPUT_DIR
    / 'mock1_potential_flow_measurement_noise_stage1b_seed_robustness_capacity_selection.csv'
)
if stage1b_capacity_path.is_file():
    stage1b_capacity_df = pd.read_csv(stage1b_capacity_path)
    stage1b_columns = [
        column
        for column in [
            'noise_fraction',
            'validation_minimum_basis',
            'one_standard_error_basis',
            'practical_plateau_basis',
            'combined_practical_basis',
            'practical_converged',
        ]
        if column in stage1b_capacity_df.columns
    ]
    stage1b_comparison_df = capacity_selection_df[
        capacity_selection_df['working_model'] == PRIMARY_MODEL
    ].merge(
        stage1b_capacity_df[stage1b_columns],
        on='noise_fraction',
        how='left',
        suffixes=('_heteroscedastic', '_homoscedastic_stage1b'),
    )
else:
    stage1b_comparison_df = pd.DataFrame(
        [
            {
                'stage1b_capacity_available': False,
                'path': str(stage1b_capacity_path),
            }
        ]
    )
atomic_to_csv(
    stage1b_comparison_df,
    OUTPUT_DIR / f'{OUTPUT_PREFIX}_stage1b_capacity_comparison.csv',
)

# Representative controlled profile without rebuilding the reference field.
first_reference_seed, first_label_seed = SEED_PAIRS[0]
_, first_model_indices = make_reference_model_split(
    positions.shape[0], REFERENCE_FRACTION, first_reference_seed
)
first_split = make_label_split(
    first_model_indices.size,
    TRAIN_FRACTION,
    VALIDATION_FRACTION,
    first_label_seed,
)
first_raw_profile = relative_profile_raw(radius[first_model_indices], radius_max)
first_true_profile = rms_normalize_profile(
    first_raw_profile, first_split['train']
)
first_working_profiles = make_working_profiles(
    first_true_profile, first_split['train']
)
first_model_radius = radius[first_model_indices]

if SAVE_PDF or SAVE_PNG or SHOW_PLOTS:
    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    order = np.argsort(first_model_radius)
    axes[0, 0].plot(
        first_model_radius[order],
        first_true_profile[order],
        label='True relative scale',
    )
    axes[0, 0].plot(
        first_model_radius[order],
        first_working_profiles['gls_flattened'][order],
        label='Flattened working scale',
    )
    axes[0, 0].plot(
        first_model_radius[order],
        first_working_profiles['ols'][order],
        label='OLS working scale',
    )
    axes[0, 0].set_xlabel(r'Observer-centered radius $r$')
    axes[0, 0].set_ylabel(r'Relative uncertainty $q_i$')
    axes[0, 0].set_title('(a) Controlled heteroscedastic profile')
    axes[0, 0].legend()

    axes[0, 1].hist(first_true_profile, bins=35, alpha=0.65, label='True')
    axes[0, 1].hist(
        first_working_profiles['gls_flattened'],
        bins=35,
        alpha=0.65,
        label='Flattened',
    )
    axes[0, 1].set_xlabel(r'Relative uncertainty $q_i$')
    axes[0, 1].set_ylabel('Count')
    axes[0, 1].set_title('(b) Profile distribution')
    axes[0, 1].legend()

    for model_name in WORKING_MODEL_ORDER:
        precision = normalized_profile_precision(
            first_working_profiles[model_name], first_split['train']
        )
        axes[1, 0].scatter(
            first_model_radius[first_split['train']],
            precision,
            s=5,
            alpha=0.35,
            label=WORKING_MODELS[model_name]['label'],
        )
    axes[1, 0].set_xlabel(r'Observer-centered radius $r$')
    axes[1, 0].set_ylabel('Mean-normalized precision weight')
    axes[1, 0].set_title('(c) Fitting precision weights')
    axes[1, 0].legend()

    ess_values = []
    for model_name in WORKING_MODEL_ORDER:
        precision = normalized_profile_precision(
            first_working_profiles[model_name], first_split['train']
        )
        ess_values.append(precision_effective_sample_size(precision))
    axes[1, 1].bar(
        np.arange(len(WORKING_MODEL_ORDER)), ess_values
    )
    axes[1, 1].set_xticks(
        np.arange(len(WORKING_MODEL_ORDER)),
        ['Correct GLS', 'Flattened GLS', 'OLS'],
        rotation=15,
    )
    axes[1, 1].set_ylabel('Precision effective sample size')
    axes[1, 1].set_title('(d) Weight concentration')
    fig.tight_layout()
    save_figure(fig, 'heteroscedastic_design_composite')

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    for model_name in WORKING_MODEL_ORDER:
        subset = capacity_selection_df[
            capacity_selection_df['working_model'] == model_name
        ].sort_values('noise_fraction')
        axes[0, 0].plot(
            subset['noise_fraction'],
            subset['combined_practical_basis'],
            marker='o',
            label=WORKING_MODELS[model_name]['label'],
        )
        axes[0, 1].errorbar(
            subset['noise_fraction'],
            subset['practical_validation_mean'],
            yerr=subset['practical_validation_sem_between_seeds'],
            marker='o',
            capsize=3,
            label=WORKING_MODELS[model_name]['label'],
        )
    axes[0, 0].set_xlabel(r'Noise fraction $\nu_u$')
    axes[0, 0].set_ylabel('Combined practical parameters')
    axes[0, 0].set_title('(a) Validation-selected capacity')
    axes[0, 0].legend()
    axes[0, 1].set_xlabel(r'Noise fraction $\nu_u$')
    axes[0, 1].set_ylabel('Practical validation score')
    axes[0, 1].set_title('(b) Observable validation risk')
    axes[0, 1].legend()

    practical_aggregate = aggregate_df[
        aggregate_df['protocol'] == 'retuned_practical'
    ]
    for model_name in WORKING_MODEL_ORDER:
        subset = practical_aggregate[
            practical_aggregate['working_model'] == model_name
        ].sort_values('noise_fraction')
        axes[1, 0].plot(
            subset['noise_fraction'],
            subset['effective_degrees_of_freedom_mean_equal_seed_weight'],
            marker='o',
            label=WORKING_MODELS[model_name]['label'],
        )
        profile_subset = profile_diagnostics_df[
            profile_diagnostics_df['working_model'] == model_name
        ]
        axes[1, 1].errorbar(
            [WORKING_MODEL_ORDER.index(model_name)],
            [profile_subset['precision_effective_sample_size_train'].mean()],
            yerr=[profile_subset['precision_effective_sample_size_train'].std(ddof=1)],
            marker='o',
            capsize=3,
            linestyle='none',
            label=WORKING_MODELS[model_name]['label'],
        )
    axes[1, 0].set_xlabel(r'Noise fraction $\nu_u$')
    axes[1, 0].set_ylabel('Effective degrees of freedom')
    axes[1, 0].set_title('(c) Regularized effective capacity')
    axes[1, 0].legend()
    axes[1, 1].set_xticks(
        np.arange(len(WORKING_MODEL_ORDER)),
        ['Correct GLS', 'Flattened GLS', 'OLS'],
        rotation=15,
    )
    axes[1, 1].set_ylabel('Training precision ESS')
    axes[1, 1].set_title('(d) Precision-weight concentration')
    fig.tight_layout()
    save_figure(fig, 'capacity_precision_composite')

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    contrast_panels = [
        ('ols_minus_correct', 'test_latent_nrmse_radial', '(a) OLS minus correct GLS: latent radial'),
        ('flattened_minus_correct', 'test_latent_nrmse_radial', '(b) Flattened minus correct GLS: latent radial'),
        ('ols_minus_correct', 'test_hidden_nrmse_3d', '(c) OLS minus correct GLS: hidden 3D'),
        ('flattened_minus_correct', 'test_hidden_nrmse_3d', '(d) Flattened minus correct GLS: hidden 3D'),
    ]
    for axis, (comparison_name, metric, title) in zip(axes.ravel(), contrast_panels):
        subset = paired_summary_df[
            (paired_summary_df['protocol'] == 'matched_correct_gls_practical')
            & (paired_summary_df['comparison'] == comparison_name)
            & (paired_summary_df['metric'] == metric)
        ].sort_values('noise_fraction')
        lower = subset['delta_mean'] - subset['delta_ci_low']
        upper = subset['delta_ci_high'] - subset['delta_mean']
        axis.errorbar(
            subset['noise_fraction'],
            subset['delta_mean'],
            yerr=np.vstack([lower, upper]),
            marker='o',
            capsize=3,
        )
        axis.axhline(0.0, linestyle='--', linewidth=1)
        axis.set_xlabel(r'Noise fraction $\nu_u$')
        axis.set_ylabel('NRMSE difference')
        axis.set_title(title)
    fig.tight_layout()
    save_figure(fig, 'paired_estimator_contrasts')

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9.5))
    for model_name in WORKING_MODEL_ORDER:
        subset = practical_aggregate[
            practical_aggregate['working_model'] == model_name
        ].sort_values('noise_fraction')
        axes[0, 0].plot(
            subset['noise_fraction'],
            subset['test_working_measurement_whitened_residual_rms_mean_equal_seed_weight'],
            marker='o',
            label=WORKING_MODELS[model_name]['label'],
        )
        axes[0, 1].plot(
            subset['noise_fraction'],
            subset['coverage_model_observed_radial_68_mean_equal_seed_weight'],
            marker='o',
            label=WORKING_MODELS[model_name]['label'],
        )
        axes[1, 0].plot(
            subset['noise_fraction'],
            subset['coverage_model_observed_radial_95_mean_equal_seed_weight'],
            marker='o',
            label=WORKING_MODELS[model_name]['label'],
        )
    axes[0, 0].axhline(1.0, linestyle='--', linewidth=1)
    axes[0, 0].set_xlabel(r'Noise fraction $\nu_u$')
    axes[0, 0].set_ylabel('Whitened-residual RMS')
    axes[0, 0].set_title('(a) Working-model residual calibration')
    axes[0, 0].legend()
    axes[0, 1].axhline(0.6827, linestyle='--', linewidth=1)
    axes[0, 1].set_xlabel(r'Noise fraction $\nu_u$')
    axes[0, 1].set_ylabel('Observed radial coverage')
    axes[0, 1].set_title('(b) Model-based 68% coverage')
    axes[0, 1].legend()
    axes[1, 0].axhline(0.95, linestyle='--', linewidth=1)
    axes[1, 0].set_xlabel(r'Noise fraction $\nu_u$')
    axes[1, 0].set_ylabel('Observed radial coverage')
    axes[1, 0].set_title('(c) Model-based 95% coverage')
    axes[1, 0].legend()

    correct_scale = global_scale_aggregate_df[
        global_scale_aggregate_df['working_model'] == PRIMARY_MODEL
    ]
    for noise_fraction in NOISE_FRACTIONS:
        subset = correct_scale[np.isclose(
            correct_scale['noise_fraction'], noise_fraction
        )].sort_values('global_scale_factor')
        axes[1, 1].plot(
            subset['global_scale_factor'],
            subset['coverage_observed_radial_95_mean'],
            marker='o',
            label=rf'$\nu_u={noise_fraction:g}$',
        )
    axes[1, 1].axhline(0.95, linestyle='--', linewidth=1)
    axes[1, 1].set_xlabel('Assumed global noise-scale factor')
    axes[1, 1].set_ylabel('Observed radial 95% coverage')
    axes[1, 1].set_title('(d) Global-scale misspecification audit')
    axes[1, 1].legend()
    fig.tight_layout()
    save_figure(fig, 'residual_coverage_composite')

# Decision summary.
all_practical_converged = bool(
    capacity_selection_df['practical_converged'].astype(bool).all()
)
ols_hidden_rows = paired_summary_df[
    (paired_summary_df['protocol'] == 'matched_correct_gls_practical')
    & (paired_summary_df['comparison'] == 'ols_minus_correct')
    & (paired_summary_df['metric'] == 'test_hidden_nrmse_3d')
]
ols_latent_rows = paired_summary_df[
    (paired_summary_df['protocol'] == 'matched_correct_gls_practical')
    & (paired_summary_df['comparison'] == 'ols_minus_correct')
    & (paired_summary_df['metric'] == 'test_latent_nrmse_radial')
]
correct_gls_hidden_advantage_levels = int(
    np.sum(ols_hidden_rows['delta_ci_low'].to_numpy(dtype=float) > 0.0)
)
correct_gls_latent_advantage_levels = int(
    np.sum(ols_latent_rows['delta_ci_low'].to_numpy(dtype=float) > 0.0)
)
any_correct_gls_advantage = bool(
    correct_gls_hidden_advantage_levels > 0
    or correct_gls_latent_advantage_levels > 0
)
needs_larger_eigensystem = bool(
    capacity_selection_df['needs_larger_eigensystem'].astype(bool).any()
)
if needs_larger_eigensystem:
    recommended_next_step = 'extend_capacity_before_text_update'
elif all_practical_converged and any_correct_gls_advantage:
    recommended_next_step = 'finalize_heteroscedastic_text_and_then_correlated_noise'
elif all_practical_converged:
    recommended_next_step = 'review_precision_profile_effect_before_text_update'
else:
    recommended_next_step = 'review_capacity_and_heteroscedastic_design'

decision_df = pd.DataFrame(
    [
        {
            'all_working_models_practical_converged': all_practical_converged,
            'needs_larger_eigensystem': needs_larger_eigensystem,
            'primary_working_model': PRIMARY_MODEL,
            'true_profile_sigma_ratio': HETERO_SIGMA_RATIO,
            'true_profile_power': HETERO_PROFILE_POWER,
            'flattened_profile_exponent': FLATTENED_PROFILE_EXPONENT,
            'correct_gls_hidden_3d_advantage_noise_levels_vs_ols': (
                correct_gls_hidden_advantage_levels
            ),
            'correct_gls_latent_radial_advantage_noise_levels_vs_ols': (
                correct_gls_latent_advantage_levels
            ),
            'global_scale_point_estimate_invariant_by_design': True,
            'recommended_next_step': recommended_next_step,
        }
    ]
)
atomic_to_csv(decision_df, OUTPUT_DIR / f'{OUTPUT_PREFIX}_decision.csv')

elapsed = time.perf_counter() - start_time
output_files = {
    'candidate_grid_csv': candidate_path,
    'capacity_selection_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_capacity_selection.csv',
    'aggregate_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_aggregate.csv',
    'paired_summary_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_paired_summary.csv',
    'selector_aggregate_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_selector_aggregate.csv',
    'global_scale_aggregate_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_global_scale_aggregate.csv',
    'profile_diagnostics_csv': profile_path,
    'closure_csv': closure_path,
    'decision_csv': OUTPUT_DIR / f'{OUTPUT_PREFIX}_decision.csv',
    'summary_json': OUTPUT_DIR / f'{OUTPUT_PREFIX}_summary.json',
}
summary_payload = {
    'script_version': SCRIPT_VERSION,
    'software': {
        'python': platform.python_version(),
        'numpy': np.__version__,
        'scipy': scipy.__version__,
        'pandas': pd.__version__,
        'matplotlib': plt.matplotlib.__version__,
    },
    'controlled_heteroscedastic_design': {
        'profile': 'q_raw(r)=1+(q_max-1)(r/R)^power, training-RMS normalized',
        'q_max': HETERO_SIGMA_RATIO,
        'power': HETERO_PROFILE_POWER,
        'flattened_exponent': FLATTENED_PROFILE_EXPONENT,
        'global_scale_factors': GLOBAL_SCALE_FACTORS,
        'note': 'Controlled monotone distance profile; not a survey-specific empirical error law.',
    },
    'seed_pairs': SEED_PAIRS,
    'noise_fractions': NOISE_FRACTIONS,
    'noise_realizations_per_positive_level': N_NOISE_REALIZATIONS,
    'working_models': WORKING_MODELS,
    'potential_basis_sizes': POTENTIAL_BASIS_SIZES,
    'regularization_strengths': POTENTIAL_REGULARIZATION_STRENGTHS,
    'regularization_powers': POTENTIAL_REGULARIZATION_POWERS,
    'capacity_selection': capacity_selection_df.to_dict(orient='records'),
    'decision': decision_df.iloc[0].to_dict(),
    'geometry_continuity': geometry_continuity_df.iloc[0].to_dict(),
    'gradient_diagnostics': gradient_diagnostics_df.iloc[0].to_dict(),
    'elapsed_seconds': elapsed,
    'output_files': {key: str(value) for key, value in output_files.items()},
}
atomic_write_json(output_files['summary_json'], summary_payload)
update_progress(
    'completed',
    completed_candidate_blocks=(
        len(SEED_PAIRS)
        * len(WORKING_MODELS)
        * len(POTENTIAL_REGULARIZATION_POWERS)
        * len(POTENTIAL_REGULARIZATION_STRENGTHS)
    ),
    expected_candidate_blocks=(
        len(SEED_PAIRS)
        * len(WORKING_MODELS)
        * len(POTENTIAL_REGULARIZATION_POWERS)
        * len(POTENTIAL_REGULARIZATION_STRENGTHS)
    ),
    completed_final_seeds=len(SEED_PAIRS),
    expected_final_seeds=len(SEED_PAIRS),
)

print('\n' + '=' * 132)
print('Mock-1 heteroscedastic measurement-noise Stage 2 completed')
print('=' * 132)
print(f'Elapsed time: {elapsed / 60.0:.2f} min')
print('Capacity selection')
display(capacity_selection_df)
print('-' * 132)
print('Primary practical aggregate')
primary_columns = [
    'protocol',
    'working_model',
    'noise_fraction',
    'test_latent_nrmse_radial_mean_equal_seed_weight',
    'test_hidden_nrmse_3d_mean_equal_seed_weight',
    'test_working_measurement_whitened_residual_rms_mean_equal_seed_weight',
    'coverage_model_observed_radial_95_mean_equal_seed_weight',
]
primary_columns = [column for column in primary_columns if column in aggregate_df.columns]
display(
    aggregate_df[aggregate_df['protocol'] == 'retuned_practical'][primary_columns]
)
print('-' * 132)
print('Paired correct-GLS comparisons')
display(
    paired_summary_df[
        (paired_summary_df['protocol'] == 'matched_correct_gls_practical')
        & paired_summary_df['comparison'].isin(
            ['ols_minus_correct', 'flattened_minus_correct']
        )
        & paired_summary_df['metric'].isin(
            ['test_latent_nrmse_radial', 'test_hidden_nrmse_3d']
        )
    ]
)
print('-' * 132)
print('Selector aggregate')
display(selector_aggregate_df)
print('-' * 132)
print('Global noise-scale audit')
display(global_scale_aggregate_df)
print('-' * 132)
print('Decision')
display(decision_df)
print('-' * 132)
print('Saved principal files')
for key, value in output_files.items():
    print(f'  {key:32s}: {value}')
print('=' * 132)
