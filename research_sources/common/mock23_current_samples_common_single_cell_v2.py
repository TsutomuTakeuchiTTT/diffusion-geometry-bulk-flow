# Mock-2 / Mock-3: current-sample selection-aware diffusion reconstruction
# このセルは共通関数を定義します。Jupyter Notebookの新しい空のコードセルへ
# 内容全体を貼り付け、最初に一度だけ実行してください。
#
# 続いて、mock2_current_samples_run_single_cell_v2.txt または
# mock3_current_samples_run_single_cell_v2.txt を別セルで実行します。
#
# 設計:
#   - Mock-1: complete control sample（独立reference + complete query）
#   - Mock-2/3: flux-selected survey graph
#   - Schechter shape: Mock-2/3の実サンプルからjoint STY likelihoodで推定
#   - survey graph: sparse variable-bandwidth kernel（dense N x N行列は作らない）
#   - corrected Cartesian diffusion-spectral estimator
#   - complete-query positions: Nyström extension
#   - geometry variantsを逐次処理し、ピークメモリを抑制

from __future__ import annotations

import gc
import hashlib
import json
import math
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
from scipy.integrate import cumulative_trapezoid
from scipy.optimize import minimize
from scipy.sparse.linalg import ArpackNoConvergence, eigsh
from scipy.spatial import cKDTree

# ============================================================
# A. General utilities
# ============================================================

def _file_signature(path: Path) -> dict:
    stat = path.stat()
    return {
        "path": str(path.resolve()),
        "size": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def _stable_signature(payload: dict) -> str:
    text = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _finite_median(values) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.median(values)) if values.size else math.nan


def _finite_quantile(values, q: float) -> float:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.quantile(values, q)) if values.size else math.nan


def _effective_sample_size(weights) -> float:
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(np.square(weights)))
    return (
        float(np.square(np.sum(weights)) / denominator)
        if denominator > 0.0
        else math.nan
    )


def _weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    total = float(np.sum(weights))
    if total <= 0.0:
        return np.full(values.shape[1:], np.nan)
    return np.sum(values * weights.reshape((-1,) + (1,) * (values.ndim - 1)), axis=0) / total


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


def velocity_metrics(true, predicted, weights=None) -> dict:
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if true.shape != predicted.shape or true.ndim != 2 or true.shape[1] != 3:
        raise ValueError("true and predicted must have shape (n, 3).")

    if weights is None:
        weights = np.ones(true.shape[0], dtype=np.float64)
    else:
        weights = np.asarray(weights, dtype=np.float64)
        if weights.shape != (true.shape[0],):
            raise ValueError("weights must have shape (n,).")
        weights = np.maximum(weights, 0.0)

    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("Metric weights have zero total.")
    normalized_weights = weights / weight_sum

    residual = predicted - true
    residual_squared = np.sum(np.square(residual), axis=1)
    true_squared = np.sum(np.square(true), axis=1)
    rmse_3d = float(np.sqrt(np.sum(normalized_weights * residual_squared)))
    true_rms = float(np.sqrt(np.sum(normalized_weights * true_squared)))
    normalized_rmse = rmse_3d / true_rms if true_rms > 0.0 else math.nan

    component_rmse = np.sqrt(
        np.sum(normalized_weights[:, None] * np.square(residual), axis=0)
    )

    correlations = []
    for component in range(3):
        x = true[:, component]
        y = predicted[:, component]
        mean_x = float(np.sum(normalized_weights * x))
        mean_y = float(np.sum(normalized_weights * y))
        covariance = float(
            np.sum(normalized_weights * (x - mean_x) * (y - mean_y))
        )
        variance_x = float(np.sum(normalized_weights * np.square(x - mean_x)))
        variance_y = float(np.sum(normalized_weights * np.square(y - mean_y)))
        correlations.append(
            covariance / math.sqrt(variance_x * variance_y)
            if variance_x > 0.0 and variance_y > 0.0
            else math.nan
        )

    angles = _direction_errors(true, predicted)
    finite_angle = np.isfinite(angles)
    angle_weights = normalized_weights[finite_angle]
    angle_values = angles[finite_angle]
    if angle_values.size:
        order = np.argsort(angle_values)
        sorted_angle = angle_values[order]
        sorted_weight = angle_weights[order]
        cumulative = np.cumsum(sorted_weight) / np.sum(sorted_weight)
        median_angle = float(np.interp(0.5, cumulative, sorted_angle))
        p90_angle = float(np.interp(0.9, cumulative, sorted_angle))
    else:
        median_angle = math.nan
        p90_angle = math.nan

    true_speed = np.linalg.norm(true, axis=1)
    predicted_speed = np.linalg.norm(predicted, axis=1)
    valid_speed = true_speed > 0.0
    relative_speed_error = np.full(true.shape[0], np.nan)
    relative_speed_error[valid_speed] = (
        predicted_speed[valid_speed] - true_speed[valid_speed]
    ) / true_speed[valid_speed]
    valid_relative = np.isfinite(relative_speed_error)
    if np.any(valid_relative):
        values = np.abs(relative_speed_error[valid_relative])
        local_weights = normalized_weights[valid_relative]
        order = np.argsort(values)
        cumulative = np.cumsum(local_weights[order]) / np.sum(local_weights)
        median_abs_relative = float(np.interp(0.5, cumulative, values[order]))
    else:
        median_abs_relative = math.nan

    mean_residual = np.sum(normalized_weights[:, None] * residual, axis=0)

    output = {
        "n": int(true.shape[0]),
        "rmse_3d": rmse_3d,
        "normalized_rmse": float(normalized_rmse),
        "rmse_x": float(component_rmse[0]),
        "rmse_y": float(component_rmse[1]),
        "rmse_z": float(component_rmse[2]),
        "corr_x": float(correlations[0]),
        "corr_y": float(correlations[1]),
        "corr_z": float(correlations[2]),
        "median_direction_error_deg": median_angle,
        "p90_direction_error_deg": p90_angle,
        "median_abs_relative_speed_error": median_abs_relative,
        "mean_residual_x": float(mean_residual[0]),
        "mean_residual_y": float(mean_residual[1]),
        "mean_residual_z": float(mean_residual[2]),
        "mean_residual_magnitude": float(np.linalg.norm(mean_residual)),
    }
    output.update(calibration_metrics(true, predicted, weights=weights))
    return output


def calibration_metrics(true, predicted, weights=None) -> dict:
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if weights is None:
        weights = np.ones(true.shape[0], dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = weights / np.sum(weights)

    output = {}
    for component, label in enumerate(("x", "y", "z")):
        x = true[:, component]
        y = predicted[:, component]
        mean_x = float(np.sum(weights * x))
        mean_y = float(np.sum(weights * y))
        denominator = float(np.sum(weights * np.square(x - mean_x)))
        slope = (
            float(np.sum(weights * (x - mean_x) * (y - mean_y)) / denominator)
            if denominator > 0.0
            else math.nan
        )
        intercept = mean_y - slope * mean_x if np.isfinite(slope) else math.nan
        origin_denominator = float(np.sum(weights * np.square(x)))
        slope_origin = (
            float(np.sum(weights * x * y) / origin_denominator)
            if origin_denominator > 0.0
            else math.nan
        )
        output[f"calibration_slope_{label}"] = slope
        output[f"calibration_intercept_{label}"] = intercept
        output[f"calibration_slope_origin_{label}"] = slope_origin

    vector_denominator = float(np.sum(weights[:, None] * np.square(true)))
    output["vector_gain_through_origin"] = (
        float(np.sum(weights[:, None] * true * predicted) / vector_denominator)
        if vector_denominator > 0.0
        else math.nan
    )
    return output


def _save_figure(fig, output_dir: Path, prefix: str, stem: str, show_plots: bool):
    fig.savefig(output_dir / f"{prefix}_{stem}.pdf", bbox_inches="tight")
    fig.savefig(output_dir / f"{prefix}_{stem}.png", dpi=180, bbox_inches="tight")
    if show_plots:
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
        "validation": np.sort(order[n_train:n_train + n_validation]),
        "test": np.sort(order[n_train + n_validation:]),
    }


# ============================================================
# B. Actual catalog loading and validation
# ============================================================

def _load_mock1(path: Path) -> dict:
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


def _load_survey(path: Path, mock_name: str) -> dict:
    with np.load(path, allow_pickle=False) as data:
        required = {"pos", "vel", "ids", "dist", "m_app", "M_abs"}
        if mock_name == "mock2":
            required.add("m_lim")
        elif mock_name == "mock3":
            required.update({"octant", "m_lim", "mlim_per_octant", "delta_m"})
        else:
            raise ValueError("mock_name must be 'mock2' or 'mock3'.")
        missing = required.difference(data.files)
        if missing:
            raise KeyError(f"{mock_name} is missing keys: {sorted(missing)}")

        pos = np.asarray(data["pos"], dtype=np.float64)
        n = pos.shape[0]
        m_lim = np.asarray(data["m_lim"], dtype=np.float64)
        if m_lim.ndim == 0:
            m_lim = np.full(n, float(m_lim), dtype=np.float64)

        output = {
            "pos": pos,
            "vel": np.asarray(data["vel"], dtype=np.float64),
            "ids": np.asarray(data["ids"]),
            "dist": np.asarray(data["dist"], dtype=np.float64),
            "m_app": np.asarray(data["m_app"], dtype=np.float64),
            "M_abs": np.asarray(data["M_abs"], dtype=np.float64),
            "m_lim": m_lim,
        }
        if mock_name == "mock3":
            output["region_id"] = np.asarray(data["octant"], dtype=np.int64)
            output["mlim_per_region"] = np.asarray(
                data["mlim_per_octant"], dtype=np.float64
            )
            output["delta_m"] = np.asarray(data["delta_m"], dtype=np.float64)
        else:
            output["region_id"] = np.zeros(n, dtype=np.int64)
            output["mlim_per_region"] = np.array([float(m_lim[0])])
            output["delta_m"] = np.array([0.0])

    if output["vel"].shape != pos.shape:
        raise ValueError(f"{mock_name}: vel must have shape {pos.shape}.")
    for key in ("ids", "dist", "m_app", "M_abs", "m_lim", "region_id"):
        if output[key].shape != (n,):
            raise ValueError(f"{mock_name}: {key} must have shape ({n},).")

    h_values = output["dist"] / np.power(
        10.0,
        (output["m_app"] - output["M_abs"] - 25.0) / 5.0,
    )
    output["h"] = float(np.median(h_values))
    output["h_scatter"] = float(np.std(h_values))
    output["magnitude_relation_max_abs_error"] = float(
        np.max(np.abs(h_values - output["h"]))
    )

    if np.any(output["m_app"] > output["m_lim"] + 1.0e-8):
        raise ValueError(f"{mock_name}: some main-catalog objects violate m_app <= m_lim.")

    if mock_name == "mock3":
        centered = output["pos"] - np.array([250.0, 250.0, 250.0])[None, :]
        expected = (
            (centered[:, 0] >= 0.0).astype(np.int64)
            + 2 * (centered[:, 1] >= 0.0).astype(np.int64)
            + 4 * (centered[:, 2] >= 0.0).astype(np.int64)
        )
        # The actual files use x-bit=1, y-bit=2, z-bit=4.
        if not np.array_equal(expected, output["region_id"]):
            raise ValueError("Mock-3 octant labels do not match Cartesian signs.")
        expected_limit = output["mlim_per_region"][output["region_id"]]
        if not np.allclose(expected_limit, output["m_lim"]):
            raise ValueError("Mock-3 object-wise m_lim does not match mlim_per_octant.")

    return output


# ============================================================
# C. Schechter STY fit and selection function
# ============================================================

def _infer_parent_magnitude_bounds(mock2: dict, mock3: dict):
    # The actual generated samples show hard endpoints near -24 and -14.
    # Mock-2 reaches the faint endpoint at the nearest distances.
    bright = float(np.floor(min(np.min(mock2["M_abs"]), np.min(mock3["M_abs"]))))
    faint = float(np.ceil(np.max(mock2["M_abs"])))
    if not (bright < faint):
        raise ValueError("Could not infer parent magnitude bounds.")
    return bright, faint


def _schechter_log_shape(M, M_star, alpha):
    x = np.power(10.0, 0.4 * (M_star - M))
    return np.log(0.4 * np.log(10.0)) + (alpha + 1.0) * np.log(x) - x


def _fit_joint_schechter(mock2: dict, mock3: dict, M_bright: float, M_faint: float):
    datasets = []
    for survey in (mock2, mock3):
        M_limit = survey["m_lim"] - 5.0 * np.log10(survey["dist"] / survey["h"]) - 25.0
        M_observable_max = np.minimum(M_faint, M_limit)
        if np.any(survey["M_abs"] < M_bright - 1.0e-8):
            raise ValueError("Absolute magnitudes are brighter than M_bright.")
        if np.any(survey["M_abs"] > M_observable_max + 1.0e-7):
            raise ValueError("Catalog violates the inferred magnitude truncation.")
        datasets.append((survey["M_abs"], M_observable_max))

    grid = np.linspace(M_bright, M_faint, 4096)

    def negative_log_likelihood(parameters):
        M_star, alpha = parameters
        log_phi_grid = _schechter_log_shape(grid, M_star, alpha)
        offset = float(np.max(log_phi_grid))
        phi_grid = np.exp(log_phi_grid - offset)
        cumulative = cumulative_trapezoid(phi_grid, grid, initial=0.0)

        total = 0.0
        for M, M_observable_max in datasets:
            denominator = np.interp(M_observable_max, grid, cumulative)
            if np.any(denominator <= 0.0) or not np.all(np.isfinite(denominator)):
                return 1.0e100
            log_numerator = _schechter_log_shape(M, M_star, alpha)
            total -= float(np.sum(log_numerator - (np.log(denominator) + offset)))
        return total

    starts = [
        np.array([-22.1, -1.1]),
        np.array([-21.5, -0.8]),
        np.array([-22.7, -1.4]),
    ]
    solutions = []
    for start in starts:
        result = minimize(
            negative_log_likelihood,
            start,
            method="L-BFGS-B",
            bounds=[(-24.0, -18.0), (-2.5, 0.5)],
            options={"maxiter": 1000, "ftol": 1.0e-12},
        )
        solutions.append(result)
    best = min(solutions, key=lambda result: result.fun)
    if not best.success:
        warnings.warn(f"Schechter optimization did not report success: {best.message}")

    M_star, alpha = map(float, best.x)
    log_phi_grid = _schechter_log_shape(grid, M_star, alpha)
    offset = float(np.max(log_phi_grid))
    phi_grid = np.exp(log_phi_grid - offset)
    cumulative = cumulative_trapezoid(phi_grid, grid, initial=0.0)
    cumulative /= cumulative[-1]

    return {
        "M_star": M_star,
        "alpha": alpha,
        "M_bright": float(M_bright),
        "M_faint": float(M_faint),
        "negative_log_likelihood": float(best.fun),
        "grid_M": grid,
        "grid_cdf": cumulative,
    }


def _selection_probability_from_mlim(dist, m_lim, h, lf_model):
    dist = np.asarray(dist, dtype=np.float64)
    m_lim = np.asarray(m_lim, dtype=np.float64)
    if m_lim.ndim == 0:
        m_lim = np.full(dist.shape, float(m_lim))
    M_limit = m_lim - 5.0 * np.log10(dist / h) - 25.0
    M_clipped = np.clip(M_limit, lf_model["M_bright"], lf_model["M_faint"])
    probability = np.interp(M_clipped, lf_model["grid_M"], lf_model["grid_cdf"])
    probability = np.where(M_limit < lf_model["M_bright"], 0.0, probability)
    probability = np.where(M_limit >= lf_model["M_faint"], 1.0, probability)
    return np.clip(probability, 0.0, 1.0)


def _global_mock3_probability(dist, h, lf_model, mlim_per_region):
    curves = [
        _selection_probability_from_mlim(
            dist,
            np.full(np.asarray(dist).shape, float(m_lim)),
            h,
            lf_model,
        )
        for m_lim in np.asarray(mlim_per_region, dtype=np.float64)
    ]
    return np.mean(np.vstack(curves), axis=0)


def _fit_or_load_lf(root_dir: Path, mock2_file: Path, mock3_file: Path, use_cache=True):
    cache_file = root_dir / "mock23_joint_schechter_fit_v2.npz"
    signature = _stable_signature(
        {
            "mock2": _file_signature(mock2_file),
            "mock3": _file_signature(mock3_file),
            "version": 2,
        }
    )
    if use_cache and cache_file.is_file():
        try:
            with np.load(cache_file, allow_pickle=False) as data:
                if str(data["signature"].item()) == signature:
                    return {
                        "M_star": float(data["M_star"].item()),
                        "alpha": float(data["alpha"].item()),
                        "M_bright": float(data["M_bright"].item()),
                        "M_faint": float(data["M_faint"].item()),
                        "negative_log_likelihood": float(data["negative_log_likelihood"].item()),
                        "grid_M": np.asarray(data["grid_M"], dtype=np.float64),
                        "grid_cdf": np.asarray(data["grid_cdf"], dtype=np.float64),
                    }
        except Exception as exc:
            warnings.warn(f"LF cache could not be read; refitting: {exc}")

    mock2 = _load_survey(mock2_file, "mock2")
    mock3 = _load_survey(mock3_file, "mock3")
    M_bright, M_faint = _infer_parent_magnitude_bounds(mock2, mock3)
    model = _fit_joint_schechter(mock2, mock3, M_bright, M_faint)
    if use_cache:
        np.savez_compressed(
            cache_file,
            signature=np.array(signature),
            M_star=np.array(model["M_star"]),
            alpha=np.array(model["alpha"]),
            M_bright=np.array(model["M_bright"]),
            M_faint=np.array(model["M_faint"]),
            negative_log_likelihood=np.array(model["negative_log_likelihood"]),
            grid_M=model["grid_M"],
            grid_cdf=model["grid_cdf"],
        )
    return model


# ============================================================
# D. Independent complete reference field
# ============================================================

def _estimate_bulk_field_fixed_bandwidth(
    query_positions,
    reference_positions,
    reference_velocities,
    bandwidth,
    n_neighbors,
):
    tree = cKDTree(reference_positions)
    try:
        distances, indices = tree.query(
            query_positions,
            k=min(n_neighbors, reference_positions.shape[0]),
            workers=-1,
        )
    except TypeError:
        distances, indices = tree.query(
            query_positions,
            k=min(n_neighbors, reference_positions.shape[0]),
        )
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    weights = np.exp(-np.square(distances / bandwidth))
    weight_sum = np.sum(weights, axis=1)
    velocity = np.einsum(
        "ij,ijk->ik",
        weights,
        reference_velocities[indices],
        optimize=True,
    ) / weight_sum[:, None]
    residual = reference_velocities[indices] - velocity[:, None, :]
    dispersion = np.sqrt(
        np.maximum(
            np.einsum("ij,ijk->i", weights, np.square(residual), optimize=True)
            / weight_sum,
            0.0,
        )
    )
    effective_neighbors = np.square(weight_sum) / np.sum(np.square(weights), axis=1)
    return {
        "velocity": velocity,
        "dispersion": dispersion,
        "effective_neighbors": effective_neighbors,
        "nearest_distance": distances[:, 0],
        "kth_distance": distances[:, -1],
    }


# ============================================================
# E. Sparse kernel and weighted diffusion geometry
# ============================================================

def _build_base_kernel(positions, n_neighbors, bandwidth_neighbor, multiplier):
    n = positions.shape[0]
    n_neighbors = min(n_neighbors, n - 1)
    bandwidth_neighbor = min(bandwidth_neighbor, n_neighbors)
    tree = cKDTree(positions)
    try:
        distances, indices = tree.query(positions, k=n_neighbors + 1, workers=-1)
    except TypeError:
        distances, indices = tree.query(positions, k=n_neighbors + 1)
    distances = np.asarray(distances[:, 1:], dtype=np.float64)
    indices = np.asarray(indices[:, 1:], dtype=np.int64)
    rho = multiplier * distances[:, bandwidth_neighbor - 1]
    positive = rho[rho > 0.0]
    rho = np.maximum(
        rho,
        max(float(np.median(positive)) * 1.0e-8, np.finfo(np.float64).eps),
    )
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
        "multiplier": multiplier,
    }


def _clip_inverse_probability(probability, floor, max_weight, quantile):
    probability = np.maximum(np.asarray(probability, dtype=np.float64), floor)
    raw = 1.0 / probability
    cap = min(float(max_weight), float(np.quantile(raw, quantile)))
    clipped = np.minimum(raw, cap)
    clipped /= np.mean(clipped)
    return clipped, {
        "probability_min": float(np.min(probability)),
        "probability_median": float(np.median(probability)),
        "raw_weight_max": float(np.max(raw)),
        "clip_cap_before_normalization": float(cap),
        "normalized_weight_max": float(np.max(clipped)),
        "effective_sample_size": _effective_sample_size(clipped),
    }


def _build_weighted_geometry(base_kernel, quadrature_weights, alpha, max_basis):
    kernel = base_kernel["kernel"]
    n = kernel.shape[0]
    quadrature_weights = np.asarray(quadrature_weights, dtype=np.float64)
    q = np.asarray(kernel @ quadrature_weights).ravel()
    if np.any(q <= 0.0):
        raise ValueError("Non-positive weighted kernel density q.")
    q_factor = np.power(q, -alpha)
    kernel_alpha = (
        sparse.diags(q_factor) @ kernel @ sparse.diags(q_factor)
    ).tocsr()
    d = np.asarray(kernel_alpha @ quadrature_weights).ravel()
    if np.any(d <= 0.0):
        raise ValueError("Non-positive row normalization d.")
    stationary = quadrature_weights * d
    stationary /= np.sum(stationary)
    symmetric_factor = np.sqrt(quadrature_weights / d)
    symmetric_operator = (
        sparse.diags(symmetric_factor)
        @ kernel_alpha
        @ sparse.diags(symmetric_factor)
    )
    symmetric_operator = (0.5 * (symmetric_operator + symmetric_operator.T)).tocsr()
    k = min(max_basis, n - 2)
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
        raise RuntimeError(f"ARPACK did not converge: {n_converged}/{k} eigenpairs.") from exc
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.asarray(eigenvalues[order], dtype=np.float64)
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    eigenfunctions = eigenvectors / np.sqrt(stationary)[:, None]
    for column in range(eigenfunctions.shape[1]):
        pivot = int(np.argmax(np.abs(eigenfunctions[:, column])))
        if eigenfunctions[pivot, column] < 0.0:
            eigenfunctions[:, column] *= -1.0
    generator = np.maximum(1.0 - eigenvalues, 0.0)
    positive = generator[1:][generator[1:] > 0.0]
    if positive.size:
        generator /= np.median(positive)
    return {
        "eigenvalues": eigenvalues,
        "generator_eigenvalues": generator,
        "eigenfunctions": eigenfunctions,
        "quadrature_weights": quadrature_weights,
        "q": q,
        "d": d,
        "stationary": stationary,
        "alpha": float(alpha),
    }


def _nystrom_extend(query_positions, observed_positions, base_kernel, geometry):
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
    rho_query = np.maximum(
        rho_query,
        max(float(np.median(positive)) * 1.0e-8, np.finfo(np.float64).eps),
    )
    kernel_query = np.exp(
        -np.square(distances) / (rho_query[:, None] * rho_observed[indices])
    )
    quadrature = geometry["quadrature_weights"]
    q_observed = geometry["q"]
    alpha = geometry["alpha"]
    q_query = np.sum(kernel_query * quadrature[indices], axis=1)
    q_query = np.maximum(q_query, np.finfo(np.float64).eps)
    kernel_alpha_query = kernel_query / (
        np.power(q_query[:, None], alpha)
        * np.power(q_observed[indices], alpha)
    )
    row_weight = kernel_alpha_query * quadrature[indices]
    row_sum = np.sum(row_weight, axis=1)
    transition = row_weight / row_sum[:, None]
    eigenvalues = geometry["eigenvalues"]
    safe = np.where(np.abs(eigenvalues) > 1.0e-10, eigenvalues, np.nan)
    extended = np.einsum(
        "ij,ijm->im",
        transition,
        geometry["eigenfunctions"][indices, :],
        optimize=True,
    ) / safe[None, :]
    return {
        "eigenfunctions": extended,
        "rho_query": rho_query,
        "nearest_distance": distances[:, 0],
        "row_normalization": row_sum,
    }


# ============================================================
# F. Weighted Cartesian diffusion-spectral regression
# ============================================================

def _factor_dense_system(system):
    try:
        return "cholesky", linalg.cho_factor(
            system, lower=True, check_finite=False
        )
    except linalg.LinAlgError:
        return "direct", np.asarray(system, dtype=np.float64)


def _solve_factored_dense(kind, factor, rhs):
    if kind == "cholesky":
        return linalg.cho_solve(factor, rhs, check_finite=False)
    return linalg.solve(factor, rhs, assume_a="sym", check_finite=False)


def _fit_weighted_spectral(
    phi,
    velocity,
    loss_weights,
    generator_eigenvalues,
    regularization_strength,
    numerical_ridge,
):
    loss_weights = np.asarray(loss_weights, dtype=np.float64)
    loss_weights = loss_weights / np.mean(loss_weights)
    weighted_phi = phi * np.sqrt(loss_weights)[:, None]
    weighted_velocity = velocity * np.sqrt(loss_weights)[:, None]
    system = (weighted_phi.T @ weighted_phi) / phi.shape[0]
    rhs = (weighted_phi.T @ weighted_velocity) / phi.shape[0]
    diagonal = np.diag_indices_from(system)
    system[diagonal] += (
        regularization_strength
        * np.maximum(generator_eigenvalues[: phi.shape[1]], numerical_ridge)
        + numerical_ridge
    )
    kind, factor = _factor_dense_system(system)
    return _solve_factored_dense(kind, factor, rhs)


def _select_spectral(
    eigenfunctions,
    generator_eigenvalues,
    train_indices,
    validation_indices,
    target,
    loss_weights,
    metric_weights,
    basis_sizes,
    regularization_strengths,
    numerical_ridge,
):
    records = []
    best = None
    for n_basis in basis_sizes:
        if n_basis > eigenfunctions.shape[1]:
            continue
        phi_train = eigenfunctions[train_indices, :n_basis]
        phi_validation = eigenfunctions[validation_indices, :n_basis]
        for regularization in regularization_strengths:
            coefficients = _fit_weighted_spectral(
                phi_train,
                target[train_indices],
                loss_weights[train_indices],
                generator_eigenvalues,
                regularization,
                numerical_ridge,
            )
            prediction = phi_validation @ coefficients
            metrics = velocity_metrics(
                target[validation_indices],
                prediction,
                weights=metric_weights[validation_indices],
            )
            record = {
                "n_basis": int(n_basis),
                "regularization_strength": float(regularization),
                **metrics,
            }
            records.append(record)
            candidate = (metrics["normalized_rmse"], n_basis, regularization)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("No spectral model was evaluated.")
    return int(best[1]), float(best[2]), pd.DataFrame(records)


# ============================================================
# G. Local-kernel baselines
# ============================================================

def _query_neighbors(query_positions, labelled_positions, max_neighbors):
    tree = cKDTree(labelled_positions)
    try:
        distances, indices = tree.query(query_positions, k=max_neighbors, workers=-1)
    except TypeError:
        distances, indices = tree.query(query_positions, k=max_neighbors)
    distances = np.asarray(distances, dtype=np.float64)
    indices = np.asarray(indices, dtype=np.int64)
    if distances.ndim == 1:
        distances = distances[:, None]
        indices = indices[:, None]
    return distances, indices


def _local_kernel_from_neighbors(
    distances_max,
    indices_max,
    labelled_velocity,
    source_weights,
    n_neighbors,
    bandwidth_neighbor,
    multiplier,
):
    distances = distances_max[:, :n_neighbors]
    indices = indices_max[:, :n_neighbors]
    bandwidth = multiplier * distances[:, bandwidth_neighbor - 1]
    positive = bandwidth[bandwidth > 0.0]
    bandwidth = np.maximum(
        bandwidth,
        max(float(np.median(positive)) * 1.0e-8, np.finfo(np.float64).eps),
    )
    weights = np.exp(-np.square(distances / bandwidth[:, None]))
    weights *= source_weights[indices]
    weight_sum = np.sum(weights, axis=1)
    prediction = np.einsum(
        "ij,ijk->ik",
        weights,
        labelled_velocity[indices],
        optimize=True,
    ) / weight_sum[:, None]
    return prediction


def _select_local_kernel(
    validation_distances,
    validation_indices,
    train_velocity,
    train_source_weights,
    validation_truth,
    validation_metric_weights,
    neighbor_grid,
    multiplier_grid,
    bandwidth_fraction,
):
    best = None
    records = []
    for n_neighbors in neighbor_grid:
        bandwidth_neighbor = min(
            n_neighbors,
            max(1, int(round(n_neighbors * bandwidth_fraction))),
        )
        for multiplier in multiplier_grid:
            prediction = _local_kernel_from_neighbors(
                validation_distances,
                validation_indices,
                train_velocity,
                train_source_weights,
                n_neighbors,
                bandwidth_neighbor,
                multiplier,
            )
            metrics = velocity_metrics(
                validation_truth,
                prediction,
                weights=validation_metric_weights,
            )
            record = {
                "n_neighbors": int(n_neighbors),
                "bandwidth_neighbor": int(bandwidth_neighbor),
                "bandwidth_multiplier": float(multiplier),
                **metrics,
            }
            records.append(record)
            candidate = (metrics["normalized_rmse"], n_neighbors, multiplier)
            if best is None or candidate < best:
                best = candidate
    if best is None:
        raise RuntimeError("No local-kernel model was evaluated.")
    best_neighbors = int(best[1])
    best_multiplier = float(best[2])
    best_bandwidth_neighbor = min(
        best_neighbors,
        max(1, int(round(best_neighbors * bandwidth_fraction))),
    )
    return best_neighbors, best_bandwidth_neighbor, best_multiplier, pd.DataFrame(records)


# ============================================================
# H. Evaluation helpers
# ============================================================

def _append_evaluation(records, model, subset, true, predicted, mask=None, weights=None, extra=None):
    if mask is None:
        mask = np.ones(true.shape[0], dtype=bool)
    mask = np.asarray(mask, dtype=bool)
    if np.sum(mask) < 5:
        return
    local_weights = None if weights is None else np.asarray(weights)[mask]
    record = {
        "model": model,
        "subset": subset,
        **velocity_metrics(true[mask], predicted[mask], weights=local_weights),
    }
    if extra:
        record.update(extra)
    records.append(record)


def _radial_records(model, true, predicted, radius, mask, n_bins, weights=None):
    mask = np.asarray(mask, dtype=bool)
    selected_radius = radius[mask]
    if selected_radius.size < n_bins * 5:
        return []
    edges = np.unique(np.quantile(selected_radius, np.linspace(0.0, 1.0, n_bins + 1)))
    records = []
    for index in range(edges.size - 1):
        lower = edges[index]
        upper = edges[index + 1]
        bin_mask = mask & (radius >= lower) & (
            radius <= upper if index == edges.size - 2 else radius < upper
        )
        if np.sum(bin_mask) < 5:
            continue
        local_weights = None if weights is None else weights[bin_mask]
        records.append(
            {
                "model": model,
                "bin": int(index),
                "radius_min": float(lower),
                "radius_max": float(upper),
                "radius_median": float(np.median(radius[bin_mask])),
                **velocity_metrics(
                    true[bin_mask],
                    predicted[bin_mask],
                    weights=local_weights,
                ),
            }
        )
    return records


# ============================================================
# I. Main pipeline
# ============================================================

def run_current_selection_mock(config: dict):
    """Run Mock-2 or Mock-3 using Mock-1 as independent complete control."""

    required_config = {
        "mock_name",
        "root_dir",
        "mock1_file",
        "survey_file",
        "mock2_file",
        "mock3_file",
        "output_dir",
        "geometry_variants",
        "primary_variant",
    }
    missing = required_config.difference(config)
    if missing:
        raise KeyError(f"Missing configuration entries: {sorted(missing)}")

    mock_name = str(config["mock_name"])
    root_dir = Path(config["root_dir"])
    mock1_file = Path(config["mock1_file"])
    survey_file = Path(config["survey_file"])
    mock2_file = Path(config["mock2_file"])
    mock3_file = Path(config["mock3_file"])
    output_dir = Path(config["output_dir"])
    cache_dir = output_dir / "cache"
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    center = np.asarray(config.get("sphere_center", [250.0, 250.0, 250.0]), dtype=np.float64)
    sphere_radius = float(config.get("sphere_radius", 169.35))
    complete_reference_fraction = float(config.get("complete_reference_fraction", 0.70))
    complete_split_seed = int(config.get("complete_split_seed", 20261201))
    label_split_seed = int(config.get("label_split_seed", 20261202 if mock_name == "mock2" else 20261203))
    train_fraction = float(config.get("train_fraction", 0.70))
    validation_fraction = float(config.get("validation_fraction", 0.15))

    reference_bandwidth = float(config.get("reference_bandwidth", 22.0))
    reference_neighbors = int(config.get("reference_neighbors", 128))

    graph_neighbors = int(config.get("graph_neighbors", 48))
    graph_bandwidth_neighbor = int(config.get("graph_bandwidth_neighbor", 16))
    graph_bandwidth_multiplier = float(config.get("graph_bandwidth_multiplier", 1.0))

    basis_sizes = list(config.get("basis_sizes", [128, 256, 384, 512]))
    regularization_strengths = list(config.get("regularization_strengths", [1.0e-4, 1.0e-2, 1.0]))
    numerical_ridge = float(config.get("numerical_ridge", 1.0e-10))

    min_selection_probability = float(config.get("min_selection_probability", 1.0e-4))
    max_inverse_weight = float(config.get("max_inverse_weight", 50.0))
    weight_clip_quantile = float(config.get("weight_clip_quantile", 0.995))

    local_neighbors = list(config.get("local_kernel_neighbors", [8, 16, 32, 64]))
    local_multipliers = list(config.get("local_kernel_multipliers", [0.50, 0.75, 1.0, 1.50]))
    local_bandwidth_fraction = float(config.get("local_kernel_bandwidth_fraction", 3.0 / 8.0))

    selection_support_min = float(config.get("selection_support_min", 0.001))
    strict_selection_support_min = float(config.get("strict_selection_support_min", 0.01))
    support_bandwidth_factor = float(config.get("support_bandwidth_factor", 1.5))
    radial_bins = int(config.get("radial_bins", 8))
    use_cache = bool(config.get("use_cache", True))
    show_plots = bool(config.get("show_plots", True))

    print("=" * 112)
    print(f"{mock_name}: current-sample selection-aware diffusion reconstruction")
    print("=" * 112)
    print(f"Python             : {platform.python_version()}")
    print(f"NumPy / SciPy      : {np.__version__} / {scipy.__version__}")
    print(f"Mock-1             : {mock1_file}")
    print(f"Survey             : {survey_file}")
    print(f"Output             : {output_dir}")

    for path in (mock1_file, survey_file, mock2_file, mock3_file):
        if not path.is_file():
            raise FileNotFoundError(path)

    complete = _load_mock1(mock1_file)
    survey = _load_survey(survey_file, mock_name)
    mock2 = _load_survey(mock2_file, "mock2")
    mock3 = _load_survey(mock3_file, "mock3")
    lf_model = _fit_or_load_lf(root_dir, mock2_file, mock3_file, use_cache=use_cache)

    print(
        "Joint STY Schechter shape: "
        f"M*={lf_model['M_star']:.6f}, alpha={lf_model['alpha']:.6f}, "
        f"parent range=[{lf_model['M_bright']:.1f},{lf_model['M_faint']:.1f}]"
    )
    print(
        f"Magnitude-distance h={survey['h']:.7f} "
        f"(scatter={survey['h_scatter']:.3e})"
    )

    centered_survey = survey["pos"] - center[None, :]
    centered_complete = complete["pos"] - center[None, :]

    # Exclude any complete-control IDs appearing in either selected survey.
    excluded_ids = np.union1d(mock2["ids"], mock3["ids"])
    complete_available_mask = ~np.isin(complete["ids"], excluded_ids)
    complete_available = np.flatnonzero(complete_available_mask)
    rng = np.random.default_rng(complete_split_seed)
    order = rng.permutation(complete_available)
    n_reference = int(round(complete_reference_fraction * order.size))
    reference_global = np.sort(order[:n_reference])
    query_global = np.sort(order[n_reference:])

    print(
        f"Survey objects     : {survey['pos'].shape[0]:,}\n"
        f"Complete available : {complete_available.size:,}\n"
        f"Reference/query    : {reference_global.size:,}/{query_global.size:,}\n"
        f"Survey overlaps removed from Mock-1: "
        f"{complete['pos'].shape[0] - complete_available.size:,}"
    )

    # Selection probabilities.
    probability_region_survey = _selection_probability_from_mlim(
        survey["dist"], survey["m_lim"], survey["h"], lf_model
    )
    if mock_name == "mock3":
        probability_global_survey = _global_mock3_probability(
            survey["dist"], survey["h"], lf_model, survey["mlim_per_region"]
        )
    else:
        probability_global_survey = probability_region_survey.copy()

    query_dist = complete["dist"][query_global]
    if mock_name == "mock3":
        query_centered = centered_complete[query_global]
        query_region = (
            (query_centered[:, 0] >= 0.0).astype(np.int64)
            + 2 * (query_centered[:, 1] >= 0.0).astype(np.int64)
            + 4 * (query_centered[:, 2] >= 0.0).astype(np.int64)
        )
        query_m_lim = survey["mlim_per_region"][query_region]
        probability_region_query = _selection_probability_from_mlim(
            query_dist, query_m_lim, survey["h"], lf_model
        )
        probability_global_query = _global_mock3_probability(
            query_dist, survey["h"], lf_model, survey["mlim_per_region"]
        )
    else:
        query_region = np.zeros(query_global.size, dtype=np.int64)
        query_m_lim = np.full(query_global.size, survey["m_lim"][0])
        probability_region_query = _selection_probability_from_mlim(
            query_dist, query_m_lim, survey["h"], lf_model
        )
        probability_global_query = probability_region_query.copy()

    survey_region = survey["region_id"]

    # Selection audit.
    audit_records = []
    for region in np.unique(survey_region):
        mask = survey_region == region
        audit_records.append(
            {
                "region_id": int(region),
                "n_survey": int(np.sum(mask)),
                "m_lim": float(np.unique(survey["m_lim"][mask])[0]),
                "distance_min": float(np.min(survey["dist"][mask])),
                "distance_median": float(np.median(survey["dist"][mask])),
                "distance_max": float(np.max(survey["dist"][mask])),
                "selection_probability_min": float(np.min(probability_region_survey[mask])),
                "selection_probability_median": float(np.median(probability_region_survey[mask])),
                "selection_probability_max": float(np.max(probability_region_survey[mask])),
            }
        )
    audit_df = pd.DataFrame(audit_records)
    display(audit_df)

    # Independent fixed-scale reference truth.
    truth_cache = cache_dir / "independent_reference_truth_v2.npz"
    truth_signature = _stable_signature(
        {
            "mock1": _file_signature(mock1_file),
            "survey": _file_signature(survey_file),
            "complete_split_seed": complete_split_seed,
            "reference_fraction": complete_reference_fraction,
            "reference_bandwidth": reference_bandwidth,
            "reference_neighbors": reference_neighbors,
            "reference_hash": hashlib.sha256(reference_global.tobytes()).hexdigest(),
            "query_hash": hashlib.sha256(query_global.tobytes()).hexdigest(),
        }
    )
    truth_loaded = False
    if use_cache and truth_cache.is_file():
        try:
            with np.load(truth_cache, allow_pickle=False) as data:
                if str(data["signature"].item()) == truth_signature:
                    truth_survey = np.asarray(data["truth_survey"], dtype=np.float64)
                    truth_query = np.asarray(data["truth_query"], dtype=np.float64)
                    truth_survey_dispersion = np.asarray(data["truth_survey_dispersion"], dtype=np.float64)
                    truth_query_dispersion = np.asarray(data["truth_query_dispersion"], dtype=np.float64)
                    truth_loaded = True
        except Exception as exc:
            warnings.warn(f"Truth cache could not be read; rebuilding: {exc}")
    if not truth_loaded:
        reference_positions = centered_complete[reference_global]
        reference_velocities = complete["vel"][reference_global]
        survey_truth_result = _estimate_bulk_field_fixed_bandwidth(
            centered_survey,
            reference_positions,
            reference_velocities,
            reference_bandwidth,
            reference_neighbors,
        )
        query_truth_result = _estimate_bulk_field_fixed_bandwidth(
            centered_complete[query_global],
            reference_positions,
            reference_velocities,
            reference_bandwidth,
            reference_neighbors,
        )
        truth_survey = survey_truth_result["velocity"]
        truth_query = query_truth_result["velocity"]
        truth_survey_dispersion = survey_truth_result["dispersion"]
        truth_query_dispersion = query_truth_result["dispersion"]
        if use_cache:
            np.savez_compressed(
                truth_cache,
                signature=np.array(truth_signature),
                truth_survey=truth_survey,
                truth_query=truth_query,
                truth_survey_dispersion=truth_survey_dispersion,
                truth_query_dispersion=truth_query_dispersion,
            )

    label_split = _make_split(
        survey["pos"].shape[0], train_fraction, validation_fraction, label_split_seed
    )
    train_indices = label_split["train"]
    validation_indices = label_split["validation"]
    test_indices = label_split["test"]
    fit_indices = np.sort(np.concatenate([train_indices, validation_indices]))

    print(
        f"Survey label split : {train_indices.size:,}/"
        f"{validation_indices.size:,}/{test_indices.size:,}"
    )

    base_kernel = _build_base_kernel(
        centered_survey,
        graph_neighbors,
        graph_bandwidth_neighbor,
        graph_bandwidth_multiplier,
    )
    max_basis = min(max(basis_sizes), survey["pos"].shape[0] - 2)

    # Parent-balanced metric weights for comparing models on the observed validation/test sample.
    observed_metric_weights, observed_metric_info = _clip_inverse_probability(
        probability_region_survey,
        min_selection_probability,
        max_inverse_weight,
        weight_clip_quantile,
    )

    # Common support diagnostics from the selected-point graph.
    query_knn = _nystrom_extend(
        centered_complete[query_global],
        centered_survey,
        base_kernel,
        {
            "quadrature_weights": np.ones(survey["pos"].shape[0]),
            "q": np.asarray(base_kernel["kernel"] @ np.ones(survey["pos"].shape[0])).ravel(),
            "alpha": 0.0,
            "eigenvalues": np.ones(1),
            "eigenfunctions": np.ones((survey["pos"].shape[0], 1)),
        },
    )
    query_rho = query_knn["rho_query"]
    support_rho_threshold = support_bandwidth_factor * float(
        np.quantile(base_kernel["rho"], 0.95)
    )

    region_radial_limit = {}
    for region in np.unique(survey_region):
        region_radial_limit[int(region)] = float(
            np.quantile(survey["dist"][survey_region == region], 0.995)
        )
    radial_support = np.array(
        [query_dist[index] <= region_radial_limit[int(query_region[index])] for index in range(query_global.size)],
        dtype=bool,
    )
    extended_probability_support = probability_region_query >= selection_support_min
    strict_probability_support = probability_region_query >= strict_selection_support_min
    neighbor_support = query_rho <= support_rho_threshold
    boundary_safe = query_dist <= (sphere_radius - reference_bandwidth)
    in_support_extended = extended_probability_support & neighbor_support & radial_support
    in_support_strict = strict_probability_support & neighbor_support & radial_support
    primary_query_mask = in_support_strict & boundary_safe

    support_df = pd.DataFrame(
        [
            {"subset": "query_all", "n": int(query_global.size)},
            {"subset": "query_extended_support", "n": int(np.sum(in_support_extended))},
            {"subset": "query_strict_support", "n": int(np.sum(in_support_strict))},
            {"subset": "query_primary_boundary_safe", "n": int(np.sum(primary_query_mask))},
            {"subset": "query_high_completeness", "n": int(np.sum(probability_region_query >= 0.20))},
            {"subset": "query_medium_completeness", "n": int(np.sum((probability_region_query >= 0.05) & (probability_region_query < 0.20)))},
            {"subset": "query_low_completeness", "n": int(np.sum((probability_region_query >= strict_selection_support_min) & (probability_region_query < 0.05)))},
            {"subset": "query_very_low_completeness", "n": int(np.sum((probability_region_query >= selection_support_min) & (probability_region_query < strict_selection_support_min)))},
            {"subset": "query_out_of_extended_support", "n": int(np.sum(~in_support_extended))},
        ]
    )
    display(support_df)

    # Selection curves.
    radius_grid = np.linspace(max(0.1, float(np.min(survey["dist"]))), sphere_radius, 300)
    fig, ax = plt.subplots(figsize=(6.8, 4.8))
    if mock_name == "mock3":
        for region, m_lim in enumerate(survey["mlim_per_region"]):
            curve = _selection_probability_from_mlim(
                radius_grid,
                np.full(radius_grid.shape, m_lim),
                survey["h"],
                lf_model,
            )
            ax.plot(radius_grid, curve, label=f"octant {region}: m_lim={m_lim:.3f}")
        ax.legend(fontsize="small", ncol=2)
    else:
        curve = _selection_probability_from_mlim(
            radius_grid,
            np.full(radius_grid.shape, survey["m_lim"][0]),
            survey["h"],
            lf_model,
        )
        ax.plot(radius_grid, curve)
    ax.set_xlabel(r"Radius [$h^{-1}\,\mathrm{Mpc}$]")
    ax.set_ylabel("Schechter selection probability")
    ax.set_ylim(-0.02, 1.02)
    fig.tight_layout()
    _save_figure(fig, output_dir, mock_name, "selection_function", show_plots)

    # Process diffusion variants sequentially.
    results = []
    validation_frames = []
    radial_frames = []
    predictions = {}
    variant_diagnostics = []

    for variant in config["geometry_variants"]:
        variant_name = str(variant["name"])
        probability_mode = str(variant.get("probability_mode", "none"))
        loss_weight_mode = str(variant.get("loss_weight_mode", probability_mode))
        alpha = float(variant.get("alpha", 0.0))

        def probability_for_mode(mode, region_probability, global_probability):
            if mode == "none":
                return np.ones_like(region_probability)
            if mode == "region":
                return region_probability
            if mode == "global":
                return global_probability
            raise ValueError(f"Unknown probability mode: {mode}")

        geometry_probability = probability_for_mode(
            probability_mode,
            probability_region_survey,
            probability_global_survey,
        )
        loss_probability = probability_for_mode(
            loss_weight_mode,
            probability_region_survey,
            probability_global_survey,
        )

        if probability_mode == "none":
            quadrature = np.ones(survey["pos"].shape[0])
            quadrature_info = {
                "probability_min": 1.0,
                "probability_median": 1.0,
                "raw_weight_max": 1.0,
                "clip_cap_before_normalization": 1.0,
                "normalized_weight_max": 1.0,
                "effective_sample_size": float(survey["pos"].shape[0]),
            }
        else:
            quadrature, quadrature_info = _clip_inverse_probability(
                geometry_probability,
                min_selection_probability,
                max_inverse_weight,
                weight_clip_quantile,
            )
        if loss_weight_mode == "none":
            loss_weights = np.ones(survey["pos"].shape[0])
        else:
            loss_weights, _ = _clip_inverse_probability(
                loss_probability,
                min_selection_probability,
                max_inverse_weight,
                weight_clip_quantile,
            )

        geometry_cache = cache_dir / f"geometry_{variant_name}_v2.npz"
        geometry_signature = _stable_signature(
            {
                "survey": _file_signature(survey_file),
                "variant": variant,
                "graph_neighbors": graph_neighbors,
                "bandwidth_neighbor": graph_bandwidth_neighbor,
                "multiplier": graph_bandwidth_multiplier,
                "max_basis": max_basis,
                "quadrature_cap": quadrature_info["clip_cap_before_normalization"],
                "lf_M_star": lf_model["M_star"],
                "lf_alpha": lf_model["alpha"],
            }
        )
        geometry = None
        if use_cache and geometry_cache.is_file():
            try:
                with np.load(geometry_cache, allow_pickle=False) as data:
                    if str(data["signature"].item()) == geometry_signature:
                        geometry = {
                            "eigenvalues": np.asarray(data["eigenvalues"], dtype=np.float64),
                            "generator_eigenvalues": np.asarray(data["generator_eigenvalues"], dtype=np.float64),
                            "eigenfunctions": np.asarray(data["eigenfunctions"], dtype=np.float64),
                            "quadrature_weights": np.asarray(data["quadrature_weights"], dtype=np.float64),
                            "q": np.asarray(data["q"], dtype=np.float64),
                            "d": np.asarray(data["d"], dtype=np.float64),
                            "stationary": np.asarray(data["stationary"], dtype=np.float64),
                            "alpha": float(data["alpha"].item()),
                        }
            except Exception as exc:
                warnings.warn(f"{variant_name} cache could not be read; rebuilding: {exc}")
        if geometry is None:
            start = time.perf_counter()
            geometry = _build_weighted_geometry(
                base_kernel,
                quadrature,
                alpha,
                max_basis,
            )
            print(f"{variant_name}: eigenbasis completed in {time.perf_counter() - start:.1f} s")
            if use_cache:
                np.savez_compressed(
                    geometry_cache,
                    signature=np.array(geometry_signature),
                    eigenvalues=geometry["eigenvalues"],
                    generator_eigenvalues=geometry["generator_eigenvalues"],
                    eigenfunctions=geometry["eigenfunctions"],
                    quadrature_weights=geometry["quadrature_weights"],
                    q=geometry["q"],
                    d=geometry["d"],
                    stationary=geometry["stationary"],
                    alpha=np.array(geometry["alpha"]),
                )

        best_basis, best_regularization, validation_df = _select_spectral(
            geometry["eigenfunctions"],
            geometry["generator_eigenvalues"],
            train_indices,
            validation_indices,
            truth_survey,
            loss_weights,
            observed_metric_weights,
            basis_sizes,
            regularization_strengths,
            numerical_ridge,
        )
        validation_df["model"] = variant_name
        validation_frames.append(validation_df)

        coefficients = _fit_weighted_spectral(
            geometry["eigenfunctions"][fit_indices, :best_basis],
            truth_survey[fit_indices],
            loss_weights[fit_indices],
            geometry["generator_eigenvalues"],
            best_regularization,
            numerical_ridge,
        )
        observed_prediction = geometry["eigenfunctions"][:, :best_basis] @ coefficients
        query_extension = _nystrom_extend(
            centered_complete[query_global],
            centered_survey,
            base_kernel,
            geometry,
        )
        query_prediction = query_extension["eigenfunctions"][:, :best_basis] @ coefficients

        predictions[f"{variant_name}_observed"] = observed_prediction
        predictions[f"{variant_name}_query"] = query_prediction

        _append_evaluation(
            results,
            variant_name,
            "observed_test_unweighted",
            truth_survey[test_indices],
            observed_prediction[test_indices],
        )
        _append_evaluation(
            results,
            variant_name,
            "observed_test_parent_weighted",
            truth_survey[test_indices],
            observed_prediction[test_indices],
            weights=observed_metric_weights[test_indices],
        )
        _append_evaluation(results, variant_name, "query_all", truth_query, query_prediction)
        _append_evaluation(
            results,
            variant_name,
            "query_extended_support",
            truth_query,
            query_prediction,
            mask=in_support_extended,
        )
        _append_evaluation(
            results,
            variant_name,
            "query_strict_support",
            truth_query,
            query_prediction,
            mask=in_support_strict,
        )
        _append_evaluation(
            results,
            variant_name,
            "query_primary_boundary_safe",
            truth_query,
            query_prediction,
            mask=primary_query_mask,
        )
        _append_evaluation(
            results,
            variant_name,
            "query_high_completeness",
            truth_query,
            query_prediction,
            mask=probability_region_query >= 0.20,
        )
        _append_evaluation(
            results,
            variant_name,
            "query_medium_completeness",
            truth_query,
            query_prediction,
            mask=(probability_region_query >= 0.05) & (probability_region_query < 0.20),
        )
        _append_evaluation(
            results,
            variant_name,
            "query_low_completeness",
            truth_query,
            query_prediction,
            mask=(probability_region_query >= strict_selection_support_min) & (probability_region_query < 0.05),
        )
        _append_evaluation(
            results,
            variant_name,
            "query_very_low_completeness",
            truth_query,
            query_prediction,
            mask=(probability_region_query >= selection_support_min) & (probability_region_query < strict_selection_support_min),
        )

        radial = _radial_records(
            variant_name,
            truth_query,
            query_prediction,
            query_dist,
            primary_query_mask,
            radial_bins,
        )
        if radial:
            radial_frames.append(pd.DataFrame(radial))

        if mock_name == "mock3":
            observed_test_region = survey_region[test_indices]
            for region in range(8):
                observed_region_mask = observed_test_region == region
                _append_evaluation(
                    results,
                    variant_name,
                    f"observed_test_parent_weighted_octant_{region}",
                    truth_survey[test_indices],
                    observed_prediction[test_indices],
                    mask=observed_region_mask,
                    weights=observed_metric_weights[test_indices],
                    extra={"region_id": region, "m_lim": float(survey["mlim_per_region"][region])},
                )
                region_mask = primary_query_mask & (query_region == region)
                _append_evaluation(
                    results,
                    variant_name,
                    f"query_primary_octant_{region}",
                    truth_query,
                    query_prediction,
                    mask=region_mask,
                    extra={"region_id": region, "m_lim": float(survey["mlim_per_region"][region])},
                )

        variant_diagnostics.append(
            {
                "model": variant_name,
                "probability_mode": probability_mode,
                "loss_weight_mode": loss_weight_mode,
                "alpha": alpha,
                "selected_basis": best_basis,
                "selected_regularization": best_regularization,
                **quadrature_info,
            }
        )

        print(
            f"{variant_name}: basis={best_basis}, lambda={best_regularization:g}, "
            f"ESS={quadrature_info['effective_sample_size']:.1f}"
        )
        del geometry, query_extension, observed_prediction, query_prediction
        gc.collect()

    # Local-kernel baselines.
    max_local_neighbors = max(local_neighbors)
    validation_distances, validation_neighbor_indices = _query_neighbors(
        centered_survey[validation_indices],
        centered_survey[train_indices],
        max_local_neighbors,
    )
    test_distances, test_neighbor_indices = _query_neighbors(
        centered_survey[test_indices],
        centered_survey[fit_indices],
        max_local_neighbors,
    )
    query_distances, query_neighbor_indices = _query_neighbors(
        centered_complete[query_global],
        centered_survey[fit_indices],
        max_local_neighbors,
    )

    local_variants = [
        ("local_kernel_uniform", np.ones(survey["pos"].shape[0])),
        ("local_kernel_selection_weighted", observed_metric_weights),
    ]
    for local_name, source_weights_all in local_variants:
        best_neighbors, best_bandwidth_neighbor, best_multiplier, local_validation_df = _select_local_kernel(
            validation_distances,
            validation_neighbor_indices,
            truth_survey[train_indices],
            source_weights_all[train_indices],
            truth_survey[validation_indices],
            observed_metric_weights[validation_indices],
            local_neighbors,
            local_multipliers,
            local_bandwidth_fraction,
        )
        local_validation_df["model"] = local_name
        validation_frames.append(local_validation_df)
        observed_test_prediction = _local_kernel_from_neighbors(
            test_distances,
            test_neighbor_indices,
            truth_survey[fit_indices],
            source_weights_all[fit_indices],
            best_neighbors,
            best_bandwidth_neighbor,
            best_multiplier,
        )
        query_prediction = _local_kernel_from_neighbors(
            query_distances,
            query_neighbor_indices,
            truth_survey[fit_indices],
            source_weights_all[fit_indices],
            best_neighbors,
            best_bandwidth_neighbor,
            best_multiplier,
        )
        predictions[f"{local_name}_observed_test"] = observed_test_prediction
        predictions[f"{local_name}_query"] = query_prediction
        _append_evaluation(
            results,
            local_name,
            "observed_test_unweighted",
            truth_survey[test_indices],
            observed_test_prediction,
        )
        _append_evaluation(
            results,
            local_name,
            "observed_test_parent_weighted",
            truth_survey[test_indices],
            observed_test_prediction,
            weights=observed_metric_weights[test_indices],
        )
        _append_evaluation(results, local_name, "query_all", truth_query, query_prediction)
        _append_evaluation(
            results,
            local_name,
            "query_extended_support",
            truth_query,
            query_prediction,
            mask=in_support_extended,
        )
        _append_evaluation(
            results,
            local_name,
            "query_strict_support",
            truth_query,
            query_prediction,
            mask=in_support_strict,
        )
        _append_evaluation(
            results,
            local_name,
            "query_primary_boundary_safe",
            truth_query,
            query_prediction,
            mask=primary_query_mask,
        )
        radial = _radial_records(
            local_name,
            truth_query,
            query_prediction,
            query_dist,
            primary_query_mask,
            radial_bins,
        )
        if radial:
            radial_frames.append(pd.DataFrame(radial))
        if mock_name == "mock3":
            observed_test_region = survey_region[test_indices]
            for region in range(8):
                _append_evaluation(
                    results,
                    local_name,
                    f"observed_test_parent_weighted_octant_{region}",
                    truth_survey[test_indices],
                    observed_test_prediction,
                    mask=observed_test_region == region,
                    weights=observed_metric_weights[test_indices],
                    extra={"region_id": region, "m_lim": float(survey["mlim_per_region"][region])},
                )
                _append_evaluation(
                    results,
                    local_name,
                    f"query_primary_octant_{region}",
                    truth_query,
                    query_prediction,
                    mask=primary_query_mask & (query_region == region),
                    extra={"region_id": region, "m_lim": float(survey["mlim_per_region"][region])},
                )
        variant_diagnostics.append(
            {
                "model": local_name,
                "selected_neighbors": best_neighbors,
                "selected_bandwidth_neighbor": best_bandwidth_neighbor,
                "selected_multiplier": best_multiplier,
            }
        )

    # Mean baseline.
    mean_prediction_observed = np.repeat(
        np.mean(truth_survey[fit_indices], axis=0)[None, :], test_indices.size, axis=0
    )
    mean_prediction_query = np.repeat(
        np.mean(truth_survey[fit_indices], axis=0)[None, :], query_global.size, axis=0
    )
    _append_evaluation(
        results,
        "mean_bulk_baseline",
        "observed_test_unweighted",
        truth_survey[test_indices],
        mean_prediction_observed,
    )
    _append_evaluation(
        results,
        "mean_bulk_baseline",
        "query_primary_boundary_safe",
        truth_query,
        mean_prediction_query,
        mask=primary_query_mask,
    )

    results_df = pd.DataFrame(results)
    validation_df = pd.concat(validation_frames, ignore_index=True)
    radial_df = pd.concat(radial_frames, ignore_index=True) if radial_frames else pd.DataFrame()
    diagnostics_df = pd.DataFrame(variant_diagnostics)

    print("=" * 112)
    print("Primary complete-query comparison")
    print("=" * 112)
    primary_table = results_df[
        results_df["subset"] == "query_primary_boundary_safe"
    ].sort_values("normalized_rmse")
    display(primary_table)

    # Model-comparison figure.
    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    ax.bar(primary_table["model"], primary_table["normalized_rmse"])
    ax.set_ylabel("Complete-query in-support NRMSE")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    _save_figure(fig, output_dir, mock_name, "primary_query_nrmse", show_plots)

    # Primary corrected model speed plot.
    primary_name = str(config["primary_variant"])
    primary_prediction = predictions[f"{primary_name}_query"]
    primary_mask = primary_query_mask
    if np.sum(primary_mask) >= 5:
        true_speed = np.linalg.norm(truth_query[primary_mask], axis=1)
        predicted_speed = np.linalg.norm(primary_prediction[primary_mask], axis=1)
        fig, ax = plt.subplots(figsize=(6.0, 5.2))
        ax.scatter(true_speed, predicted_speed, s=8, alpha=0.35)
        lower = min(true_speed.min(), predicted_speed.min())
        upper = max(true_speed.max(), predicted_speed.max())
        ax.plot([lower, upper], [lower, upper], linewidth=1.2)
        ax.set_xlabel("Reference bulk speed [simulation velocity unit]")
        ax.set_ylabel("Reconstructed bulk speed [simulation velocity unit]")
        fig.tight_layout()
        _save_figure(fig, output_dir, mock_name, "primary_true_vs_reconstructed_speed", show_plots)

    # Radial comparison for selected models.
    if not radial_df.empty:
        plot_models = [
            config["primary_variant"],
            "naive_alpha0",
            "local_kernel_uniform",
        ]
        if mock_name == "mock3":
            plot_models.insert(1, "global_radial_corrected_alpha0")
        fig, ax = plt.subplots(figsize=(7.0, 4.8))
        for model in plot_models:
            selected = radial_df[radial_df["model"] == model].sort_values("radius_median")
            if not selected.empty:
                ax.plot(
                    selected["radius_median"],
                    selected["normalized_rmse"],
                    marker="o",
                    label=model,
                )
        ax.set_xlabel(r"Radius [$h^{-1}\,\mathrm{Mpc}$]")
        ax.set_ylabel("Complete-query radial NRMSE")
        ax.legend(fontsize="small")
        fig.tight_layout()
        _save_figure(fig, output_dir, mock_name, "radial_nrmse", show_plots)

    if mock_name == "mock3":
        octant_rows = results_df[
            results_df["subset"].str.startswith("query_primary_octant_")
        ]
        if not octant_rows.empty:
            fig, ax = plt.subplots(figsize=(7.2, 4.8))
            for model in ["naive_alpha0", "global_radial_corrected_alpha0", "region_aware_corrected_alpha0"]:
                selected = octant_rows[octant_rows["model"] == model].sort_values("region_id")
                if not selected.empty:
                    ax.plot(selected["region_id"], selected["normalized_rmse"], marker="o", label=model)
            ax.set_xlabel("Octant")
            ax.set_ylabel("Primary-query NRMSE")
            ax.set_xticks(range(8))
            ax.legend(fontsize="small")
            fig.tight_layout()
            _save_figure(fig, output_dir, mock_name, "octant_nrmse", show_plots)

    # Save results.
    audit_df.to_csv(output_dir / f"{mock_name}_selection_audit.csv", index=False)
    support_df.to_csv(output_dir / f"{mock_name}_support_counts.csv", index=False)
    results_df.to_csv(output_dir / f"{mock_name}_reconstruction_results.csv", index=False)
    validation_df.to_csv(output_dir / f"{mock_name}_validation_grid.csv", index=False)
    radial_df.to_csv(output_dir / f"{mock_name}_radial_results.csv", index=False)
    diagnostics_df.to_csv(output_dir / f"{mock_name}_model_diagnostics.csv", index=False)

    npz_payload = {
        "query_global_indices": query_global,
        "query_region": query_region,
        "query_dist": query_dist,
        "query_selection_probability_region": probability_region_query,
        "query_selection_probability_global": probability_global_query,
        "query_extended_support": in_support_extended,
        "query_strict_support": in_support_strict,
        "query_primary_boundary_safe": primary_query_mask,
        "truth_query": truth_query,
        "truth_survey": truth_survey,
        "survey_train_indices": train_indices,
        "survey_validation_indices": validation_indices,
        "survey_test_indices": test_indices,
    }
    for key, value in predictions.items():
        npz_payload[key] = value
    np.savez_compressed(output_dir / f"{mock_name}_predictions.npz", **npz_payload)

    summary = {
        "mock_name": mock_name,
        "input_files": {
            "mock1": str(mock1_file),
            "survey": str(survey_file),
            "mock2": str(mock2_file),
            "mock3": str(mock3_file),
        },
        "lf_model": {
            "M_star": lf_model["M_star"],
            "alpha": lf_model["alpha"],
            "M_bright": lf_model["M_bright"],
            "M_faint": lf_model["M_faint"],
        },
        "catalog": {
            "n_survey": int(survey["pos"].shape[0]),
            "n_complete_available": int(complete_available.size),
            "n_reference": int(reference_global.size),
            "n_query": int(query_global.size),
            "h": survey["h"],
        },
        "truth_definition": {
            "fixed_gaussian_bandwidth_hinv_mpc": reference_bandwidth,
            "reference_neighbors": reference_neighbors,
        },
        "support": support_df.to_dict(orient="records"),
        "primary_variant": primary_name,
        "primary_results": primary_table.to_dict(orient="records"),
        "model_diagnostics": diagnostics_df.to_dict(orient="records"),
        "memory_design": {
            "dense_N_by_N_created": False,
            "geometry_variants_processed_sequentially": True,
            "max_cached_eigenfunctions_shape": [int(survey["pos"].shape[0]), int(max_basis)],
        },
    }
    with (output_dir / f"{mock_name}_summary.json").open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    print("=" * 112)
    print(f"{mock_name} completed.")
    print(f"Results: {output_dir}")
    print("=" * 112)

    return {
        "results": results_df,
        "validation": validation_df,
        "radial": radial_df,
        "diagnostics": diagnostics_df,
        "audit": audit_df,
        "support": support_df,
        "summary": summary,
    }
