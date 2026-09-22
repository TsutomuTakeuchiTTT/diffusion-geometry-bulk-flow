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

# ==================================================================================================
# Radial Mock-2: potential-flow reconstruction with distance-dependent selection
# ==================================================================================================

import os
from matplotlib.lines import Line2D

# --------------------------------------------------------------------------------------------------
# 1. Fixed paths and experiment configuration
# --------------------------------------------------------------------------------------------------

ROOT_DIR = Path(
    'mock_data'
)
MOCK1_FILE = ROOT_DIR / "mock1_complete_sphere" / "mock1.npz"
MOCK2_FILE = ROOT_DIR / "mock2_schechter_selection" / "mock2.npz"
MOCK3_FILE = ROOT_DIR / "mock3_inhomogeneous_survey" / "mock3.npz"

PRIOR_MOCK2_DIR = ROOT_DIR / "mock2_selection_correction_optimization_v4"
PRIOR_GEOMETRY_FILE = (
    PRIOR_MOCK2_DIR
    / "final_geometry_cache"
    / "geometry_a0p00_g0p00_c1p0_m512.npz"
)
PRIOR_TRUTH_FILE = PRIOR_MOCK2_DIR / "independent_reference_truth_v4.npz"

OUTPUT_DIR = ROOT_DIR / "mock2_radial_potential_selection_v1"
CACHE_DIR = OUTPUT_DIR / "cache"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)

GEOMETRY_CACHE_FILE = CACHE_DIR / "unweighted_alpha0_geometry_m512.npz"
GRADIENT_CACHE_FILE = CACHE_DIR / "potential_gradient_basis_observed_query_m512.npz"
TRUTH_CACHE_FILE = CACHE_DIR / "independent_reference_truth_fiducial.npz"

SPHERE_CENTER = np.array([250.0, 250.0, 250.0], dtype=np.float64)
SPHERE_RADIUS = 169.35

COMPLETE_REFERENCE_FRACTION = 0.50
COMPLETE_SPLIT_SEED = 20261201
LABEL_SPLIT_SEED = 20261202
TRAIN_FRACTION = 0.70
VALIDATION_FRACTION = 0.15

REFERENCE_BANDWIDTH = 22.0
REFERENCE_NEIGHBORS = 128

GRAPH_NEIGHBORS = 48
GRAPH_BANDWIDTH_NEIGHBOR = 16
GRAPH_BANDWIDTH_MULTIPLIER = 1.0
MAX_DIFFUSION_MODES = 512
N_AFFINE_MODES = 3

MIN_SELECTION_PROBABILITY = 1.0e-4
WEIGHT_CLIP_QUANTILE = 0.995
EVALUATION_WEIGHT_GAMMA = 1.0
EVALUATION_WEIGHT_CAP = 50.0

SELECTION_SUPPORT_MIN = 0.001
STRICT_SELECTION_SUPPORT_MIN = 0.01
SUPPORT_BANDWIDTH_FACTOR = 1.5
RADIAL_BINS = 8

# The non-radial Mock-2 optimum (gamma=1, cap=50) is retained as a mandatory comparison.
LOSS_CONFIGS = [
    {"name": "unweighted", "gamma": 0.0, "cap": 1.0},
    {"name": "gamma0p50_cap20", "gamma": 0.50, "cap": 20.0},
    {"name": "gamma0p50_cap50", "gamma": 0.50, "cap": 50.0},
    {"name": "gamma0p75_cap20", "gamma": 0.75, "cap": 20.0},
    {"name": "gamma0p75_cap50", "gamma": 0.75, "cap": 50.0},
    {"name": "gamma1p00_cap20", "gamma": 1.00, "cap": 20.0},
    {"name": "previous_mock2_gamma1p00_cap50", "gamma": 1.00, "cap": 50.0},
]

POTENTIAL_DIFFUSION_MODE_COUNTS = [0, 32, 64, 128, 256, 384, 511]
POTENTIAL_REGULARIZATION_STRENGTHS = [1.0e-8, 1.0e-6, 1.0e-4, 1.0e-2, 1.0]
POTENTIAL_REGULARIZATION_POWERS = [1.0, 2.0]
CLOSURE_DIFFUSION_MODE_COUNTS = [0, 128, 511]
CLOSURE_PASS_TOLERANCE = 1.0e-8

CARTESIAN_ORACLE_BASIS_SIZES = [128, 256, 384, 512]
CARTESIAN_ORACLE_REGULARIZATION = [1.0e-4, 1.0e-2, 1.0]

LOCAL_KERNEL_NEIGHBORS = [8, 16, 32, 64]
LOCAL_KERNEL_MULTIPLIERS = [0.50, 0.75, 1.0, 1.50]
LOCAL_KERNEL_BANDWIDTH_FRACTION = 3.0 / 8.0

METRIC_EIGENVALUE_RELATIVE_FLOOR = 1.0e-10
METRIC_EIGENVALUE_ABSOLUTE_FLOOR = 1.0e-14
GRADIENT_MODE_CHUNK_SIZE = 32
QUERY_CHUNK_SIZE = 256
NUMERICAL_RIDGE = 1.0e-12

BOOTSTRAP_REPLICATES = 1000
BOOTSTRAP_SEED = 20261230

SHOW_PLOTS = True
SAVE_PDF = True
SAVE_PNG = True
USE_GEOMETRY_CACHE = True
USE_GRADIENT_CACHE = True
USE_TRUTH_CACHE = True
OUTPUT_PREFIX = "mock2_radial_potential_selection"

# A reduced smoke mode is available for code-path checks, but it still expects the real input files.
SMOKE_TEST = os.environ.get("RADIAL_MOCK2_SMOKE_TEST", "0") == "1"
if SMOKE_TEST:
    MAX_DIFFUSION_MODES = 64
    POTENTIAL_DIFFUSION_MODE_COUNTS = [0, 16, 63]
    POTENTIAL_REGULARIZATION_STRENGTHS = [1.0e-6, 1.0e-4, 1.0e-2]
    LOSS_CONFIGS = [LOSS_CONFIGS[0], LOSS_CONFIGS[-1]]
    CARTESIAN_ORACLE_BASIS_SIZES = [32, 64]
    CARTESIAN_ORACLE_REGULARIZATION = [1.0e-4, 1.0e-2]
    CLOSURE_DIFFUSION_MODE_COUNTS = [0, 16, 63]
    BOOTSTRAP_REPLICATES = 100
    SHOW_PLOTS = False
    SAVE_PDF = False
    SAVE_PNG = False

# --------------------------------------------------------------------------------------------------
# 2. Small utilities
# --------------------------------------------------------------------------------------------------

def _json_ready(value):
    if isinstance(value, dict):
        return {str(k): _json_ready(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_ready(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.floating, float)):
        x = float(value)
        return x if np.isfinite(x) else None
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    return value


def _save_radial_mock2_figure(fig, stem):
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


def _weighted_quantile(values, quantile, weights=None):
    values = np.asarray(values, dtype=np.float64)
    finite = np.isfinite(values)
    values = values[finite]
    if values.size == 0:
        return math.nan
    if weights is None:
        return float(np.quantile(values, quantile))
    weights = np.asarray(weights, dtype=np.float64)[finite]
    weights = np.maximum(weights, 0.0)
    if np.sum(weights) <= 0.0:
        return math.nan
    order = np.argsort(values)
    values = values[order]
    weights = weights[order]
    cumulative = np.cumsum(weights) / np.sum(weights)
    return float(np.interp(float(quantile), cumulative, values))


def _scalar_nrmse(true, predicted, weights=None):
    true = np.asarray(true, dtype=np.float64)
    predicted = np.asarray(predicted, dtype=np.float64)
    if true.shape != predicted.shape:
        raise ValueError("scalar arrays must have identical shapes")
    if weights is None:
        weights = np.ones(true.size, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    weights = np.maximum(weights, 0.0)
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
    if true.shape != predicted.shape or true.shape != line_of_sight.shape:
        raise ValueError("true, predicted, and line_of_sight must have shape (n, 3)")
    n = true.shape[0]
    if weights is None:
        weights = np.ones(n, dtype=np.float64)
    weights = np.maximum(np.asarray(weights, dtype=np.float64), 0.0)
    weight_sum = float(np.sum(weights))
    if weight_sum <= 0.0:
        raise ValueError("metric weights have zero total")
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

    nrmse_radial = _scalar_nrmse(
        true_radial_scalar, predicted_radial_scalar, weights=weights
    )
    nrmse_tangential = ratio(residual_tangential, true_tangential)
    nrmse_3d = ratio(residual, true)

    angles = _direction_errors(true, predicted)
    median_angle = _weighted_quantile(angles, 0.5, weights=weights)
    p90_angle = _weighted_quantile(angles, 0.9, weights=weights)

    true_speed = np.linalg.norm(true, axis=1)
    predicted_speed = np.linalg.norm(predicted, axis=1)
    valid = true_speed > 0.0
    relative_speed_error = np.full(n, np.nan)
    relative_speed_error[valid] = (
        predicted_speed[valid] - true_speed[valid]
    ) / true_speed[valid]
    median_abs_relative_speed_error = _weighted_quantile(
        np.abs(relative_speed_error), 0.5, weights=weights
    )

    mean_residual = np.sum(w[:, None] * residual, axis=0)
    true_energy = float(np.sum(w[:, None] * np.square(true)))
    vector_gain = (
        float(np.sum(w[:, None] * true * predicted) / true_energy)
        if true_energy > 0.0
        else math.nan
    )
    predicted_rms = math.sqrt(float(np.sum(w * np.sum(np.square(predicted), axis=1))))
    true_rms = math.sqrt(float(np.sum(w * np.sum(np.square(true), axis=1))))

    return {
        "n": int(n),
        "nrmse_radial": float(nrmse_radial),
        "nrmse_tangential": float(nrmse_tangential),
        "nrmse_3d": float(nrmse_3d),
        "direction_error_median_deg": float(median_angle),
        "direction_error_p90_deg": float(p90_angle),
        "median_abs_relative_speed_error": float(median_abs_relative_speed_error),
        "mean_residual_x": float(mean_residual[0]),
        "mean_residual_y": float(mean_residual[1]),
        "mean_residual_z": float(mean_residual[2]),
        "mean_residual_magnitude": float(np.linalg.norm(mean_residual)),
        "vector_gain_through_origin": float(vector_gain),
        "predicted_to_reference_rms_3d": (
            float(predicted_rms / true_rms) if true_rms > 0.0 else math.nan
        ),
        "radial_calibration_slope": _weighted_origin_slope(
            true_radial_scalar, predicted_radial_scalar, weights
        ),
    }


def _weighted_origin_slope(x, y, weights):
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denominator = float(np.sum(weights * np.square(x)))
    return (
        float(np.sum(weights * x * y) / denominator)
        if denominator > 0.0
        else math.nan
    )


def _tempered_inverse_selection_weights(probability, gamma, cap):
    probability = np.maximum(
        np.asarray(probability, dtype=np.float64), MIN_SELECTION_PROBABILITY
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


def _hash_array(values):
    values = np.ascontiguousarray(values)
    return hashlib.sha256(values.view(np.uint8)).hexdigest()

# --------------------------------------------------------------------------------------------------
# 3. Geometry and metric-calibrated observed/query gradients
# --------------------------------------------------------------------------------------------------

def _load_or_build_unweighted_geometry(base_kernel, survey_n):
    candidate_files = [PRIOR_GEOMETRY_FILE, GEOMETRY_CACHE_FILE]
    if USE_GEOMETRY_CACHE:
        for path in candidate_files:
            if not path.is_file():
                continue
            try:
                with np.load(path, allow_pickle=False) as data:
                    geometry = {
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
                        "alpha": float(data["alpha"].item()),
                    }
                if (
                    geometry["eigenfunctions"].shape[0] == survey_n
                    and geometry["eigenfunctions"].shape[1] >= MAX_DIFFUSION_MODES
                    and abs(geometry["alpha"]) < 1.0e-12
                ):
                    print(f"[Geometry cache] reuse: {path}")
                    return geometry, str(path), True
            except Exception as exc:
                warnings.warn(f"geometry cache could not be loaded ({path}): {exc}")

    print("[Geometry] building unweighted alpha=0 geometry")
    geometry = _build_weighted_geometry(
        base_kernel,
        np.ones(survey_n, dtype=np.float64),
        alpha=0.0,
        max_basis=MAX_DIFFUSION_MODES,
    )
    if USE_GEOMETRY_CACHE:
        np.savez_compressed(
            GEOMETRY_CACHE_FILE,
            eigenvalues=geometry["eigenvalues"],
            generator_eigenvalues=geometry["generator_eigenvalues"],
            eigenfunctions=geometry["eigenfunctions"],
            quadrature_weights=geometry["quadrature_weights"],
            q=geometry["q"],
            d=geometry["d"],
            stationary=geometry["stationary"],
            alpha=np.array(geometry["alpha"]),
        )
    return geometry, str(GEOMETRY_CACHE_FILE), False


def _observed_markov_from_base_kernel(base_kernel):
    kernel = base_kernel["kernel"]
    degree = np.asarray(kernel.sum(axis=1)).ravel()
    if np.any(degree <= 0.0):
        raise ValueError("observed graph contains a non-positive degree")
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


def _observed_function_gradients(
    markov,
    positions,
    functions,
    bandwidth,
    inverse_metric,
    mean_position,
    chunk_size=32,
):
    functions = np.asarray(functions, dtype=np.float64)
    if functions.ndim == 1:
        functions = functions[:, None]
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
            "nij,nmj->nmi", inverse_metric, covector, optimize=True
        )
        print(f"[Observed gradient] modes {start + 1}:{stop}/{n_functions}")
    return gradients


def _query_transition(query_positions, base_kernel):
    tree = base_kernel["tree"]
    n_neighbors = base_kernel["n_neighbors"]
    bandwidth_neighbor = base_kernel["bandwidth_neighbor"]
    rho_observed = base_kernel["rho"]
    try:
        distances, indices = tree.query(
            query_positions, k=n_neighbors, workers=-1
        )
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
        "row_sum": row_sum,
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


def _query_function_gradients(
    transition_info,
    observed_positions,
    observed_functions,
    inverse_metric,
    mean_position,
    chunk_size=32,
):
    transition = transition_info["transition"]
    indices = transition_info["indices"]
    rho_query = transition_info["rho_query"]
    observed_functions = np.asarray(observed_functions, dtype=np.float64)
    if observed_functions.ndim == 1:
        observed_functions = observed_functions[:, None]
    n_query = transition.shape[0]
    n_functions = observed_functions.shape[1]
    gradients = np.empty((n_query, n_functions, 3), dtype=np.float64)
    denominator = 2.0 * np.square(rho_query)
    neighbor_positions = observed_positions[indices]

    for start in range(0, n_functions, int(chunk_size)):
        stop = min(start + int(chunk_size), n_functions)
        neighbor_functions = observed_functions[indices, start:stop]
        mean_function = np.einsum(
            "qk,qkm->qm", transition, neighbor_functions, optimize=True
        )
        covector = np.empty((n_query, stop - start, 3), dtype=np.float64)
        for c in range(3):
            mean_product = np.einsum(
                "qk,qkm,qk->qm",
                transition,
                neighbor_functions,
                neighbor_positions[:, :, c],
                optimize=True,
            )
            covariance = mean_product - mean_function * mean_position[:, c, None]
            covector[:, :, c] = covariance / denominator[:, None]
        gradients[:, start:stop, :] = np.einsum(
            "qij,qmj->qmi", inverse_metric, covector, optimize=True
        )
        print(f"[Query gradient] modes {start + 1}:{stop}/{n_functions}")
    return gradients


def _gradient_cache_signature(
    survey_file,
    query_indices,
    geometry_source,
    constant_mode_index,
):
    return _stable_signature(
        {
            "survey": _file_signature(survey_file),
            "query_hash": _hash_array(np.asarray(query_indices, dtype=np.int64)),
            "geometry_source": (
                _file_signature(Path(geometry_source))
                if Path(geometry_source).is_file()
                else str(geometry_source)
            ),
            "graph_neighbors": int(GRAPH_NEIGHBORS),
            "graph_bandwidth_neighbor": int(GRAPH_BANDWIDTH_NEIGHBOR),
            "graph_multiplier": float(GRAPH_BANDWIDTH_MULTIPLIER),
            "max_modes": int(MAX_DIFFUSION_MODES),
            "constant_mode_index": int(constant_mode_index),
            "metric_relative_floor": float(METRIC_EIGENVALUE_RELATIVE_FLOOR),
            "metric_absolute_floor": float(METRIC_EIGENVALUE_ABSOLUTE_FLOOR),
            "version": 1,
        }
    )


def _build_or_load_observed_query_gradients(
    centered_survey,
    centered_query,
    base_kernel,
    geometry,
    query_indices,
    geometry_source,
    constant_mode_index,
):
    signature = _gradient_cache_signature(
        MOCK2_FILE, query_indices, geometry_source, constant_mode_index
    )
    if USE_GRADIENT_CACHE and GRADIENT_CACHE_FILE.is_file():
        try:
            with np.load(GRADIENT_CACHE_FILE, allow_pickle=False) as data:
                if str(data["signature"].item()) == signature:
                    print(f"[Gradient cache] reuse: {GRADIENT_CACHE_FILE}")
                    return {
                        "observed_gradient_all": np.asarray(
                            data["observed_gradient_all"], dtype=np.float64
                        ),
                        "query_gradient_all": np.asarray(
                            data["query_gradient_all"], dtype=np.float64
                        ),
                        "observed_metric_condition": np.asarray(
                            data["observed_metric_condition"], dtype=np.float64
                        ),
                        "query_metric_condition": np.asarray(
                            data["query_metric_condition"], dtype=np.float64
                        ),
                        "query_rho": np.asarray(data["query_rho"], dtype=np.float64),
                        "query_nearest_distance": np.asarray(
                            data["query_nearest_distance"], dtype=np.float64
                        ),
                        "diagnostics": json.loads(str(data["diagnostics_json"].item())),
                        "loaded_from_cache": True,
                    }
        except Exception as exc:
            warnings.warn(f"gradient cache could not be loaded: {exc}")

    markov = _observed_markov_from_base_kernel(base_kernel)
    observed_metric, observed_mean = _centered_coordinate_metric_sparse(
        markov, centered_survey, base_kernel["rho"]
    )
    (
        inverse_observed_metric,
        observed_metric_eigenvalues,
        observed_condition,
        observed_clipped_count,
    ) = _invert_local_metric(observed_metric)

    query_transition = _query_transition(centered_query, base_kernel)
    query_metric, query_mean = _query_metric(query_transition, centered_survey)
    (
        inverse_query_metric,
        query_metric_eigenvalues,
        query_condition,
        query_clipped_count,
    ) = _invert_local_metric(query_metric)

    eigenfunctions = geometry["eigenfunctions"][:, :MAX_DIFFUSION_MODES]
    observed_gradient_all = _observed_function_gradients(
        markov,
        centered_survey,
        eigenfunctions,
        base_kernel["rho"],
        inverse_observed_metric,
        observed_mean,
        chunk_size=GRADIENT_MODE_CHUNK_SIZE,
    )
    query_gradient_all = _query_function_gradients(
        query_transition,
        centered_survey,
        eigenfunctions,
        inverse_query_metric,
        query_mean,
        chunk_size=GRADIENT_MODE_CHUNK_SIZE,
    )

    # Affine and quadratic diagnostics at observed and out-of-sample query positions.
    affine_coefficient = np.array([0.37, -0.52, 0.81], dtype=np.float64)
    affine_observed = centered_survey @ affine_coefficient + 1.7
    affine_observed_gradient = _observed_function_gradients(
        markov,
        centered_survey,
        affine_observed,
        base_kernel["rho"],
        inverse_observed_metric,
        observed_mean,
        chunk_size=1,
    )[:, 0, :]
    affine_query_gradient = _query_function_gradients(
        query_transition,
        centered_survey,
        affine_observed,
        inverse_query_metric,
        query_mean,
        chunk_size=1,
    )[:, 0, :]

    quadratic_observed = 0.5 * np.sum(np.square(centered_survey), axis=1)
    quadratic_observed_gradient = _observed_function_gradients(
        markov,
        centered_survey,
        quadratic_observed,
        base_kernel["rho"],
        inverse_observed_metric,
        observed_mean,
        chunk_size=1,
    )[:, 0, :]
    quadratic_query_gradient = _query_function_gradients(
        query_transition,
        centered_survey,
        quadratic_observed,
        inverse_query_metric,
        query_mean,
        chunk_size=1,
    )[:, 0, :]

    observed_affine_error = affine_observed_gradient - affine_coefficient[None, :]
    query_affine_error = affine_query_gradient - affine_coefficient[None, :]
    observed_quadratic_nrmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    np.square(quadratic_observed_gradient - centered_survey), axis=1
                )
            )
        )
        / max(
            float(np.sqrt(np.mean(np.sum(np.square(centered_survey), axis=1)))),
            np.finfo(float).eps,
        )
    )
    query_quadratic_nrmse = float(
        np.sqrt(
            np.mean(
                np.sum(
                    np.square(quadratic_query_gradient - centered_query), axis=1
                )
            )
        )
        / max(
            float(np.sqrt(np.mean(np.sum(np.square(centered_query), axis=1)))),
            np.finfo(float).eps,
        )
    )

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
            np.sqrt(np.mean(np.sum(np.square(observed_affine_error), axis=1)))
        ),
        "query_affine_rms_error": float(
            np.sqrt(np.mean(np.sum(np.square(query_affine_error), axis=1)))
        ),
        "observed_quadratic_gradient_nrmse": observed_quadratic_nrmse,
        "query_quadratic_gradient_nrmse": query_quadratic_nrmse,
        "constant_mode_gradient_rms_observed": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(observed_gradient_all[:, constant_mode_index, :]),
                        axis=1,
                    )
                )
            )
        ),
        "constant_mode_gradient_rms_query": float(
            np.sqrt(
                np.mean(
                    np.sum(
                        np.square(query_gradient_all[:, constant_mode_index, :]),
                        axis=1,
                    )
                )
            )
        ),
    }

    if USE_GRADIENT_CACHE:
        np.savez(
            GRADIENT_CACHE_FILE,
            signature=np.array(signature),
            observed_gradient_all=observed_gradient_all,
            query_gradient_all=query_gradient_all,
            observed_metric_condition=observed_condition,
            query_metric_condition=query_condition,
            query_rho=query_transition["rho_query"],
            query_nearest_distance=query_transition["nearest_distance"],
            diagnostics_json=np.array(json.dumps(_json_ready(diagnostics))),
        )
        print(f"[Gradient cache] saved: {GRADIENT_CACHE_FILE}")

    return {
        "observed_gradient_all": observed_gradient_all,
        "query_gradient_all": query_gradient_all,
        "observed_metric_condition": observed_condition,
        "query_metric_condition": query_condition,
        "query_rho": query_transition["rho_query"],
        "query_nearest_distance": query_transition["nearest_distance"],
        "diagnostics": diagnostics,
        "loaded_from_cache": False,
    }

# --------------------------------------------------------------------------------------------------
# 4. Potential solver, design diagnostics, and model selection
# --------------------------------------------------------------------------------------------------

def _regularization_diagonal(generator_values, n_basis, power):
    raw = np.asarray(generator_values[:n_basis], dtype=np.float64)
    output = np.zeros_like(raw)
    positive = raw > 0.0
    output[positive] = np.power(
        np.maximum(raw[positive], NUMERICAL_RIDGE), float(power)
    )
    return output


def _factor_regularized_system(gram, regularization, strength):
    system = np.asarray(gram, dtype=np.float64).copy()
    diagonal = np.diag_indices_from(system)
    system[diagonal] += (
        float(strength) * np.asarray(regularization, dtype=np.float64)
        + NUMERICAL_RIDGE
    )
    try:
        return "cholesky", linalg.cho_factor(
            system, lower=True, check_finite=False
        ), system
    except linalg.LinAlgError:
        return "direct", system, system


def _solve_factored(kind, factor, rhs):
    if kind == "cholesky":
        return linalg.cho_solve(factor, rhs, check_finite=False)
    return linalg.solve(factor, rhs, assume_a="sym", check_finite=False)


def _weighted_crossproducts(design, targets, weights):
    design = np.asarray(design, dtype=np.float64)
    targets = np.asarray(targets, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    normalization = float(np.sum(weights))
    weighted_design = weights[:, None] * design
    gram = design.T @ weighted_design / normalization
    rhs = design.T @ (weights[:, None] * targets) / normalization
    return gram, rhs


def _potential_velocity(gradient_basis, coefficients, n_basis):
    return np.einsum(
        "imc,m->ic",
        gradient_basis[:, :n_basis, :],
        coefficients[:n_basis],
        optimize=True,
    )


def _potential_radial_design(gradient_basis, line_of_sight):
    return np.einsum("ic,imc->im", line_of_sight, gradient_basis, optimize=True)


def _vector_design_matrix(gradient_basis):
    return np.transpose(gradient_basis, (0, 2, 1)).reshape(
        gradient_basis.shape[0] * 3,
        gradient_basis.shape[1],
    )


def _gram_spectrum(gram, n_observations):
    gram = 0.5 * (np.asarray(gram) + np.asarray(gram).T)
    eigenvalues = np.maximum(linalg.eigvalsh(gram, check_finite=False), 0.0)
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
    tolerance = max(n_observations, gram.shape[0]) * np.finfo(float).eps * singular[0]
    resolved = singular > tolerance
    rank = int(np.sum(resolved))
    smallest = float(singular[resolved][-1]) if rank else math.nan
    squared = np.square(singular)
    probability = squared / np.sum(squared)
    entropy = -np.sum(probability * np.log(np.maximum(probability, 1.0e-300)))
    return {
        "singular_values": singular,
        "numerical_rank": rank,
        "nullity": int(gram.shape[0] - rank),
        "condition_number": (
            float(singular[0] / smallest) if rank and smallest > 0.0 else math.inf
        ),
        "stable_rank": float(np.sum(squared) / squared[0]),
        "entropy_effective_rank": float(np.exp(entropy)),
    }


def _effective_degrees_of_freedom(gram, regularization, strength):
    system = np.asarray(gram, dtype=np.float64).copy()
    diagonal = np.diag_indices_from(system)
    system[diagonal] += (
        float(strength) * np.asarray(regularization, dtype=np.float64)
        + NUMERICAL_RIDGE
    )
    solved = linalg.solve(system, gram, assume_a="sym", check_finite=False)
    return float(np.trace(solved))


def _select_record(frame, columns):
    if frame.empty:
        raise RuntimeError("no candidate records are available")
    preferred_ties = [
        "n_basis",
        "regularization_strength",
        "penalty_power",
        "loss_gamma",
        "loss_cap",
    ]
    sort_columns = list(columns) + [
        name for name in preferred_ties if name in frame.columns
    ]
    return frame.sort_values(sort_columns, kind="mergesort").iloc[0].to_dict()


def _fit_final_radial_potential(
    specification,
    loss_weights,
    radial_design_observed,
    gradient_observed,
    gradient_query,
    radial_truth_observed,
    fit_indices,
    generator_values,
):
    n_basis = int(specification["n_basis"])
    strength = float(specification["regularization_strength"])
    power = float(specification["penalty_power"])
    design = radial_design_observed[fit_indices, :n_basis]
    target = radial_truth_observed[fit_indices, None]
    weights = loss_weights[fit_indices]
    gram, rhs = _weighted_crossproducts(design, target, weights)
    regularization = _regularization_diagonal(generator_values, n_basis, power)
    kind, factor, _ = _factor_regularized_system(gram, regularization, strength)
    coefficients = _solve_factored(kind, factor, rhs[:, 0])
    return {
        "coefficients": coefficients,
        "observed_prediction": _potential_velocity(
            gradient_observed, coefficients, n_basis
        ),
        "query_prediction": _potential_velocity(gradient_query, coefficients, n_basis),
        "n_basis": n_basis,
        "regularization_strength": strength,
        "penalty_power": power,
    }


def _fit_potential_3d_oracle_grid(
    gradient_observed,
    true_velocity_observed,
    train_indices,
    validation_indices,
    fit_indices,
    evaluation_weights,
    generator_values,
):
    max_basis = max(3 + n for n in POTENTIAL_DIFFUSION_MODE_COUNTS)
    train_design = _vector_design_matrix(gradient_observed[train_indices, :max_basis, :])
    validation_gradient = gradient_observed[validation_indices]
    train_targets = true_velocity_observed[train_indices].reshape(-1, 1)
    point_weights = evaluation_weights[train_indices]
    row_weights = np.repeat(point_weights, 3)
    normalization = float(np.sum(point_weights))
    gram = (
        train_design.T @ (row_weights[:, None] * train_design)
        / normalization
    )
    rhs = (
        train_design.T @ (row_weights[:, None] * train_targets)
        / normalization
    )
    records = []
    for n_diffusion in POTENTIAL_DIFFUSION_MODE_COUNTS:
        n_basis = N_AFFINE_MODES + int(n_diffusion)
        for power in POTENTIAL_REGULARIZATION_POWERS:
            regularization = _regularization_diagonal(generator_values, n_basis, power)
            for strength in POTENTIAL_REGULARIZATION_STRENGTHS:
                kind, factor, _ = _factor_regularized_system(
                    gram[:n_basis, :n_basis], regularization, strength
                )
                coefficients = _solve_factored(kind, factor, rhs[:n_basis, 0])
                prediction = _potential_velocity(
                    validation_gradient, coefficients, n_basis
                )
                metrics = _radial_vector_metrics(
                    true_velocity_observed[validation_indices],
                    prediction,
                    line_of_sight_observed[validation_indices],
                    weights=evaluation_weights[validation_indices],
                )
                records.append(
                    {
                        "n_basis": n_basis,
                        "n_diffusion_modes": int(n_diffusion),
                        "regularization_strength": float(strength),
                        "penalty_power": float(power),
                        "validation_nrmse_3d": metrics["nrmse_3d"],
                        "validation_nrmse_radial": metrics["nrmse_radial"],
                    }
                )
    grid = pd.DataFrame(records)
    best = _select_record(grid, ["validation_nrmse_3d", "validation_nrmse_radial"])

    n_basis = int(best["n_basis"])
    fit_design = _vector_design_matrix(gradient_observed[fit_indices, :n_basis, :])
    fit_targets = true_velocity_observed[fit_indices].reshape(-1, 1)
    fit_point_weights = evaluation_weights[fit_indices]
    fit_row_weights = np.repeat(fit_point_weights, 3)
    fit_normalization = float(np.sum(fit_point_weights))
    fit_gram = (
        fit_design.T @ (fit_row_weights[:, None] * fit_design)
        / fit_normalization
    )
    fit_rhs = (
        fit_design.T @ (fit_row_weights[:, None] * fit_targets)
        / fit_normalization
    )
    regularization = _regularization_diagonal(
        generator_values, n_basis, best["penalty_power"]
    )
    kind, factor, _ = _factor_regularized_system(
        fit_gram, regularization, best["regularization_strength"]
    )
    coefficients = _solve_factored(kind, factor, fit_rhs[:, 0])
    return best, grid, coefficients


def _fit_cartesian_3d_oracle(
    geometry,
    query_extension,
    true_velocity_observed,
    train_indices,
    validation_indices,
    fit_indices,
    evaluation_weights,
):
    records = []
    best = None
    for n_basis in CARTESIAN_ORACLE_BASIS_SIZES:
        for strength in CARTESIAN_ORACLE_REGULARIZATION:
            coefficients = _fit_weighted_spectral(
                geometry["eigenfunctions"][train_indices, :n_basis],
                true_velocity_observed[train_indices],
                evaluation_weights[train_indices],
                geometry["generator_eigenvalues"],
                strength,
                NUMERICAL_RIDGE,
            )
            prediction = geometry["eigenfunctions"][validation_indices, :n_basis] @ coefficients
            metrics = _radial_vector_metrics(
                true_velocity_observed[validation_indices],
                prediction,
                line_of_sight_observed[validation_indices],
                weights=evaluation_weights[validation_indices],
            )
            record = {
                "n_basis": int(n_basis),
                "regularization_strength": float(strength),
                "validation_nrmse_3d": metrics["nrmse_3d"],
                "validation_nrmse_radial": metrics["nrmse_radial"],
            }
            records.append(record)
            score = (
                record["validation_nrmse_3d"],
                record["validation_nrmse_radial"],
                int(n_basis),
                float(strength),
            )
            if best is None or score < best[0]:
                best = (score, record)
    best_record = dict(best[1])
    n_basis = int(best_record["n_basis"])
    coefficients = _fit_weighted_spectral(
        geometry["eigenfunctions"][fit_indices, :n_basis],
        true_velocity_observed[fit_indices],
        evaluation_weights[fit_indices],
        geometry["generator_eigenvalues"],
        best_record["regularization_strength"],
        NUMERICAL_RIDGE,
    )
    return {
        "best": best_record,
        "grid": pd.DataFrame(records),
        "coefficients": coefficients,
        "observed_prediction": geometry["eigenfunctions"][:, :n_basis] @ coefficients,
        "query_prediction": query_extension["eigenfunctions"][:, :n_basis] @ coefficients,
    }


def _paired_bootstrap_difference(true, pred_a, pred_b, los, weights, seed):
    rng = np.random.default_rng(seed)
    n = true.shape[0]
    differences_3d = np.empty(BOOTSTRAP_REPLICATES)
    differences_radial = np.empty(BOOTSTRAP_REPLICATES)
    for i in range(BOOTSTRAP_REPLICATES):
        selected = rng.integers(0, n, size=n)
        local_weights = None if weights is None else weights[selected]
        metric_a = _radial_vector_metrics(
            true[selected], pred_a[selected], los[selected], weights=local_weights
        )
        metric_b = _radial_vector_metrics(
            true[selected], pred_b[selected], los[selected], weights=local_weights
        )
        differences_3d[i] = metric_a["nrmse_3d"] - metric_b["nrmse_3d"]
        differences_radial[i] = metric_a["nrmse_radial"] - metric_b["nrmse_radial"]
    return {
        "delta_3d_mean": float(np.mean(differences_3d)),
        "delta_3d_ci_low": float(np.quantile(differences_3d, 0.025)),
        "delta_3d_ci_high": float(np.quantile(differences_3d, 0.975)),
        "probability_a_better_3d": float(np.mean(differences_3d < 0.0)),
        "delta_radial_mean": float(np.mean(differences_radial)),
        "delta_radial_ci_low": float(np.quantile(differences_radial, 0.025)),
        "delta_radial_ci_high": float(np.quantile(differences_radial, 0.975)),
        "probability_a_better_radial": float(np.mean(differences_radial < 0.0)),
    }

# --------------------------------------------------------------------------------------------------
# 5. Load catalogs, reproduce the independent truth, and build support masks
# --------------------------------------------------------------------------------------------------

start_time = time.perf_counter()
print("=" * 122)
print("Radial Mock-2: potential-flow reconstruction with distance-dependent selection")
print("=" * 122)
print(f"Python                  : {platform.python_version()}")
print(f"NumPy / SciPy           : {np.__version__} / {scipy.__version__}")
print(f"Mock-1                  : {MOCK1_FILE}")
print(f"Mock-2                  : {MOCK2_FILE}")
print(f"Output                  : {OUTPUT_DIR}")

for path in [MOCK1_FILE, MOCK2_FILE, MOCK3_FILE]:
    if not path.is_file():
        raise FileNotFoundError(path)

complete = _load_mock1(MOCK1_FILE)
survey = _load_survey(MOCK2_FILE, "mock2")
mock3 = _load_survey(MOCK3_FILE, "mock3")
lf_model = _fit_or_load_lf(ROOT_DIR, MOCK2_FILE, MOCK3_FILE, use_cache=True)

centered_complete = complete["pos"] - SPHERE_CENTER[None, :]
centered_survey = survey["pos"] - SPHERE_CENTER[None, :]

excluded_ids = np.union1d(survey["ids"], mock3["ids"])
complete_available = np.flatnonzero(~np.isin(complete["ids"], excluded_ids))
rng = np.random.default_rng(COMPLETE_SPLIT_SEED)
complete_order = rng.permutation(complete_available)
n_reference = int(round(COMPLETE_REFERENCE_FRACTION * complete_order.size))
reference_global = np.sort(complete_order[:n_reference])
query_global = np.sort(complete_order[n_reference:])
centered_query = centered_complete[query_global]

probability_survey = _selection_probability_from_mlim(
    survey["dist"], survey["m_lim"], survey["h"], lf_model
)
query_dist = complete["dist"][query_global]
query_m_lim = np.full(query_global.size, float(survey["m_lim"][0]))
probability_query = _selection_probability_from_mlim(
    query_dist, query_m_lim, survey["h"], lf_model
)

truth_signature = _stable_signature(
    {
        "mock1": _file_signature(MOCK1_FILE),
        "mock2": _file_signature(MOCK2_FILE),
        "complete_split_seed": int(COMPLETE_SPLIT_SEED),
        "reference_fraction": float(COMPLETE_REFERENCE_FRACTION),
        "reference_bandwidth": float(REFERENCE_BANDWIDTH),
        "reference_neighbors": int(REFERENCE_NEIGHBORS),
        "reference_hash": _hash_array(reference_global),
        "query_hash": _hash_array(query_global),
    }
)
truth_loaded = False
for candidate in [PRIOR_TRUTH_FILE, TRUTH_CACHE_FILE]:
    if not (USE_TRUTH_CACHE and candidate.is_file()):
        continue
    try:
        with np.load(candidate, allow_pickle=False) as data:
            if str(data["signature"].item()) == truth_signature:
                truth_survey = np.asarray(data["truth_survey"], dtype=np.float64)
                truth_query = np.asarray(data["truth_query"], dtype=np.float64)
                truth_loaded = True
                print(f"[Truth cache] reuse: {candidate}")
                break
    except Exception as exc:
        warnings.warn(f"truth cache could not be loaded ({candidate}): {exc}")

if not truth_loaded:
    print("[Truth] building independent fiducial bulk field")
    reference_positions = centered_complete[reference_global]
    reference_velocities = complete["vel"][reference_global]
    truth_survey = _estimate_bulk_field_fixed_bandwidth(
        centered_survey,
        reference_positions,
        reference_velocities,
        REFERENCE_BANDWIDTH,
        REFERENCE_NEIGHBORS,
    )["velocity"]
    truth_query = _estimate_bulk_field_fixed_bandwidth(
        centered_query,
        reference_positions,
        reference_velocities,
        REFERENCE_BANDWIDTH,
        REFERENCE_NEIGHBORS,
    )["velocity"]
    if USE_TRUTH_CACHE:
        np.savez_compressed(
            TRUTH_CACHE_FILE,
            signature=np.array(truth_signature),
            truth_survey=truth_survey,
            truth_query=truth_query,
        )

split = _make_split(
    survey["pos"].shape[0], TRAIN_FRACTION, VALIDATION_FRACTION, LABEL_SPLIT_SEED
)
train_indices = split["train"]
validation_indices = split["validation"]
test_indices = split["test"]
fit_indices = np.sort(np.concatenate([train_indices, validation_indices]))

line_of_sight_observed = centered_survey / np.linalg.norm(
    centered_survey, axis=1
)[:, None]
line_of_sight_query = centered_query / np.linalg.norm(centered_query, axis=1)[:, None]
radial_truth_observed = np.einsum(
    "ij,ij->i", line_of_sight_observed, truth_survey
)
radial_truth_query = np.einsum("ij,ij->i", line_of_sight_query, truth_query)

base_kernel = _build_base_kernel(
    centered_survey,
    GRAPH_NEIGHBORS,
    GRAPH_BANDWIDTH_NEIGHBOR,
    GRAPH_BANDWIDTH_MULTIPLIER,
)
geometry, geometry_source, geometry_loaded = _load_or_build_unweighted_geometry(
    base_kernel, survey["pos"].shape[0]
)
geometry["eigenvalues"] = geometry["eigenvalues"][:MAX_DIFFUSION_MODES]
geometry["generator_eigenvalues"] = geometry["generator_eigenvalues"][
    :MAX_DIFFUSION_MODES
]
geometry["eigenfunctions"] = geometry["eigenfunctions"][:, :MAX_DIFFUSION_MODES]

constant_mode_index = int(np.argmin(geometry["generator_eigenvalues"]))
mode_pool = np.arange(MAX_DIFFUSION_MODES, dtype=np.int64)
active_mode_indices = mode_pool[mode_pool != constant_mode_index]
if active_mode_indices.size < max(POTENTIAL_DIFFUSION_MODE_COUNTS):
    raise ValueError("not enough nonconstant diffusion modes")

query_extension = _nystrom_extend(
    centered_query, centered_survey, base_kernel, geometry
)

# Support is never used for hyperparameter selection.
support_rho_threshold = SUPPORT_BANDWIDTH_FACTOR * float(
    np.quantile(base_kernel["rho"], 0.95)
)
radial_limit = float(np.quantile(survey["dist"], 0.995))
neighbor_support = query_extension["rho_query"] <= support_rho_threshold
radial_support = query_dist <= radial_limit
boundary_safe = query_dist <= (SPHERE_RADIUS - REFERENCE_BANDWIDTH)
in_support_extended = (
    (probability_query >= SELECTION_SUPPORT_MIN)
    & neighbor_support
    & radial_support
)
in_support_strict = (
    (probability_query >= STRICT_SELECTION_SUPPORT_MIN)
    & neighbor_support
    & radial_support
)
primary_query_mask = in_support_strict & boundary_safe

query_subsets = {
    "query_all": np.ones(query_global.size, dtype=bool),
    "query_extended_support": in_support_extended,
    "query_strict_support": in_support_strict,
    "query_primary_boundary_safe": primary_query_mask,
    "query_high_completeness": probability_query >= 0.20,
    "query_medium_completeness": (probability_query >= 0.05)
    & (probability_query < 0.20),
    "query_low_completeness": (probability_query >= STRICT_SELECTION_SUPPORT_MIN)
    & (probability_query < 0.05),
    "query_very_low_completeness": (probability_query >= SELECTION_SUPPORT_MIN)
    & (probability_query < STRICT_SELECTION_SUPPORT_MIN),
}

support_df = pd.DataFrame(
    [{"subset": key, "n": int(np.sum(mask))} for key, mask in query_subsets.items()]
)

evaluation_weights, evaluation_weight_info = _tempered_inverse_selection_weights(
    probability_survey, EVALUATION_WEIGHT_GAMMA, EVALUATION_WEIGHT_CAP
)

print(f"Survey objects          : {survey['pos'].shape[0]:,}")
print(f"Complete available      : {complete_available.size:,}")
print(f"Reference/query         : {reference_global.size:,}/{query_global.size:,}")
print(
    "Train/validation/test : "
    f"{train_indices.size:,}/{validation_indices.size:,}/{test_indices.size:,}"
)
print(
    "Selection probability : "
    f"min={probability_survey.min():.3e}, median={np.median(probability_survey):.4f}"
)
print(
    "Evaluation weights     : "
    f"gamma={EVALUATION_WEIGHT_GAMMA:g}, cap={EVALUATION_WEIGHT_CAP:g}, "
    f"ESS={evaluation_weight_info['effective_sample_size']:.1f}"
)
display(support_df)

# --------------------------------------------------------------------------------------------------
# 6. Build observed and out-of-sample potential-gradient bases
# --------------------------------------------------------------------------------------------------

gradient_result = _build_or_load_observed_query_gradients(
    centered_survey,
    centered_query,
    base_kernel,
    geometry,
    query_global,
    geometry_source,
    constant_mode_index,
)
observed_gradient_all = gradient_result["observed_gradient_all"]
query_gradient_all = gradient_result["query_gradient_all"]

observed_active = observed_gradient_all[:, active_mode_indices, :]
query_active = query_gradient_all[:, active_mode_indices, :]
active_generator = geometry["generator_eigenvalues"][active_mode_indices]

# Remove the best affine radial component from every diffusion-gradient mode.
affine_gram = line_of_sight_observed.T @ line_of_sight_observed
observed_diffusion_radial = np.einsum(
    "ic,imc->im", line_of_sight_observed, observed_active, optimize=True
)
affine_rhs = line_of_sight_observed.T @ observed_diffusion_radial
affine_projection_vectors = linalg.solve(
    affine_gram, affine_rhs, assume_a="sym", check_finite=False
).T
observed_active = np.asarray(observed_active, dtype=np.float64).copy()
query_active = np.asarray(query_active, dtype=np.float64).copy()
observed_active -= affine_projection_vectors[None, :, :]
query_active -= affine_projection_vectors[None, :, :]

column_rms = np.sqrt(
    np.mean(np.sum(np.square(observed_active), axis=2), axis=0)
)
positive_column_rms = column_rms[np.isfinite(column_rms) & (column_rms > 0.0)]
affine_mode_scale = (
    float(np.median(positive_column_rms)) if positive_column_rms.size else 1.0
)
affine_observed = np.broadcast_to(
    (affine_mode_scale * np.eye(3))[None, :, :],
    (centered_survey.shape[0], 3, 3),
).copy()
affine_query = np.broadcast_to(
    (affine_mode_scale * np.eye(3))[None, :, :],
    (centered_query.shape[0], 3, 3),
).copy()
gradient_observed = np.concatenate([affine_observed, observed_active], axis=1)
gradient_query = np.concatenate([affine_query, query_active], axis=1)
potential_generator_values = np.concatenate(
    [np.zeros(N_AFFINE_MODES), active_generator]
)
radial_design_observed = _potential_radial_design(
    gradient_observed, line_of_sight_observed
)
radial_design_query = _potential_radial_design(gradient_query, line_of_sight_query)
max_potential_basis = gradient_observed.shape[1]

# Large all-mode arrays are no longer needed.
del observed_gradient_all, query_gradient_all, observed_active, query_active

# --------------------------------------------------------------------------------------------------
# 7. Out-of-sample in-span closure
# --------------------------------------------------------------------------------------------------

print("-" * 122)
print("Test A: observed-to-complete-query potential in-span closure")
closure_rng = np.random.default_rng(20261231)
closure_records = []
reference_rms = float(
    np.sqrt(np.mean(np.sum(np.square(truth_query[primary_query_mask]), axis=1)))
)
for n_diffusion in CLOSURE_DIFFUSION_MODE_COUNTS:
    n_basis = N_AFFINE_MODES + int(n_diffusion)
    local_column_rms = np.sqrt(
        np.mean(
            np.sum(np.square(gradient_observed[:, :n_basis, :]), axis=2), axis=0
        )
    )
    coefficient_true = closure_rng.normal(size=n_basis) / np.maximum(
        local_column_rms, 1.0e-12
    )
    query_true = _potential_velocity(gradient_query, coefficient_true, n_basis)
    query_rms = float(np.sqrt(np.mean(np.sum(np.square(query_true), axis=1))))
    if query_rms > 0.0:
        coefficient_true *= reference_rms / query_rms
    observed_true = _potential_velocity(gradient_observed, coefficient_true, n_basis)
    query_true = _potential_velocity(gradient_query, coefficient_true, n_basis)
    observed_radial = np.einsum(
        "ij,ij->i", line_of_sight_observed, observed_true
    )
    design = radial_design_observed[fit_indices, :n_basis]
    coefficient_estimated, _, rank, singular = linalg.lstsq(
        design,
        observed_radial[fit_indices],
        cond=None,
        lapack_driver="gelsd",
        check_finite=False,
    )
    query_pred = _potential_velocity(
        gradient_query, coefficient_estimated, n_basis
    )
    metrics = _radial_vector_metrics(
        query_true[primary_query_mask],
        query_pred[primary_query_mask],
        line_of_sight_query[primary_query_mask],
    )
    coefficient_error = float(
        np.linalg.norm(coefficient_estimated - coefficient_true)
        / max(np.linalg.norm(coefficient_true), np.finfo(float).eps)
    )
    status = (
        "pass"
        if int(rank) == n_basis
        and coefficient_error < CLOSURE_PASS_TOLERANCE
        and metrics["nrmse_3d"] < CLOSURE_PASS_TOLERANCE
        else "fail"
    )
    closure_records.append(
        {
            "n_basis": n_basis,
            "n_diffusion_modes": int(n_diffusion),
            "n_fit_observations": int(fit_indices.size),
            "n_query_primary": int(np.sum(primary_query_mask)),
            "numerical_rank": int(rank),
            "nullity": int(n_basis - rank),
            "condition_number": (
                float(singular[0] / singular[-1])
                if singular.size and singular[-1] > 0.0
                else math.inf
            ),
            "coefficient_relative_error": coefficient_error,
            **metrics,
            "closure_status": status,
        }
    )
closure_df = pd.DataFrame(closure_records)
display(closure_df)

# --------------------------------------------------------------------------------------------------
# 8. Joint validation over selection-loss and potential hyperparameters
# --------------------------------------------------------------------------------------------------

print("-" * 122)
print("Test B: radial validation of selection-loss and potential-flow hyperparameters")

weight_arrays = {}
weight_records = []
for config in LOSS_CONFIGS:
    weights, info = _tempered_inverse_selection_weights(
        probability_survey, config["gamma"], config["cap"]
    )
    weight_arrays[config["name"]] = weights
    weight_records.append({"loss_name": config["name"], **info})
weight_df = pd.DataFrame(weight_records)

max_basis = max(N_AFFINE_MODES + n for n in POTENTIAL_DIFFUSION_MODE_COUNTS)
A_train_max = radial_design_observed[train_indices, :max_basis]
A_validation_max = radial_design_observed[validation_indices, :max_basis]
gradient_validation = gradient_observed[validation_indices]

validation_records = []
for config in LOSS_CONFIGS:
    loss_name = config["name"]
    loss_weights = weight_arrays[loss_name]
    gram, rhs = _weighted_crossproducts(
        A_train_max,
        radial_truth_observed[train_indices, None],
        loss_weights[train_indices],
    )
    for n_diffusion in POTENTIAL_DIFFUSION_MODE_COUNTS:
        n_basis = N_AFFINE_MODES + int(n_diffusion)
        for power in POTENTIAL_REGULARIZATION_POWERS:
            regularization = _regularization_diagonal(
                potential_generator_values, n_basis, power
            )
            for strength in POTENTIAL_REGULARIZATION_STRENGTHS:
                kind, factor, _ = _factor_regularized_system(
                    gram[:n_basis, :n_basis], regularization, strength
                )
                coefficients = _solve_factored(
                    kind, factor, rhs[:n_basis, 0]
                )
                radial_prediction = A_validation_max[:, :n_basis] @ coefficients
                vector_prediction = _potential_velocity(
                    gradient_validation, coefficients, n_basis
                )
                vector_metrics_weighted = _radial_vector_metrics(
                    truth_survey[validation_indices],
                    vector_prediction,
                    line_of_sight_observed[validation_indices],
                    weights=evaluation_weights[validation_indices],
                )
                validation_records.append(
                    {
                        "loss_name": loss_name,
                        "loss_gamma": float(config["gamma"]),
                        "loss_cap": float(config["cap"]),
                        "loss_ess": float(
                            _effective_sample_size(loss_weights[train_indices])
                        ),
                        "n_basis": n_basis,
                        "n_diffusion_modes": int(n_diffusion),
                        "regularization_strength": float(strength),
                        "penalty_power": float(power),
                        "weighted_validation_nrmse_radial": _scalar_nrmse(
                            radial_truth_observed[validation_indices],
                            radial_prediction,
                            weights=evaluation_weights[validation_indices],
                        ),
                        "unweighted_validation_nrmse_radial": _scalar_nrmse(
                            radial_truth_observed[validation_indices],
                            radial_prediction,
                        ),
                        "weighted_validation_nrmse_tangential": vector_metrics_weighted[
                            "nrmse_tangential"
                        ],
                        "weighted_validation_nrmse_3d": vector_metrics_weighted[
                            "nrmse_3d"
                        ],
                        "weighted_validation_direction_error_median_deg": vector_metrics_weighted[
                            "direction_error_median_deg"
                        ],
                    }
                )
validation_df = pd.DataFrame(validation_records)

operational = _select_record(
    validation_df,
    ["weighted_validation_nrmse_radial", "unweighted_validation_nrmse_radial"],
)
hidden_3d = _select_record(
    validation_df,
    ["weighted_validation_nrmse_3d", "weighted_validation_nrmse_radial"],
)
unweighted_selected = _select_record(
    validation_df[validation_df["loss_name"] == "unweighted"],
    ["weighted_validation_nrmse_radial", "unweighted_validation_nrmse_radial"],
)
previous_selected = _select_record(
    validation_df[
        validation_df["loss_name"] == "previous_mock2_gamma1p00_cap50"
    ],
    ["weighted_validation_nrmse_radial", "unweighted_validation_nrmse_radial"],
)

selector_df = pd.DataFrame(
    [
        {"selector": "operational_radial_validation", **operational},
        {"selector": "hidden_3d_validation_diagnostic", **hidden_3d},
        {"selector": "unweighted_loss_radial_validation", **unweighted_selected},
        {"selector": "previous_mock2_loss_radial_validation", **previous_selected},
    ]
)
display(selector_df)

# --------------------------------------------------------------------------------------------------
# 9. Final model fits and oracle ceilings
# --------------------------------------------------------------------------------------------------

selected_models = {}
selected_specs = {
    "potential_optimized_loss": operational,
    "potential_unweighted_loss": unweighted_selected,
    "potential_previous_mock2_loss": previous_selected,
    "potential_hidden_3d_diagnostic": hidden_3d,
}
for model_name, specification in selected_specs.items():
    selected_models[model_name] = _fit_final_radial_potential(
        specification,
        weight_arrays[specification["loss_name"]],
        radial_design_observed,
        gradient_observed,
        gradient_query,
        radial_truth_observed,
        fit_indices,
        potential_generator_values,
    )

# Matched structural comparisons isolate the loss weighting itself.
for model_name, loss_name in [
    ("potential_matched_unweighted", "unweighted"),
    ("potential_matched_previous_mock2", "previous_mock2_gamma1p00_cap50"),
]:
    spec = dict(operational)
    spec["loss_name"] = loss_name
    config = next(item for item in LOSS_CONFIGS if item["name"] == loss_name)
    spec["loss_gamma"] = config["gamma"]
    spec["loss_cap"] = config["cap"]
    selected_models[model_name] = _fit_final_radial_potential(
        spec,
        weight_arrays[loss_name],
        radial_design_observed,
        gradient_observed,
        gradient_query,
        radial_truth_observed,
        fit_indices,
        potential_generator_values,
    )

potential_oracle_best, potential_oracle_grid, potential_oracle_coefficients = (
    _fit_potential_3d_oracle_grid(
        gradient_observed,
        truth_survey,
        train_indices,
        validation_indices,
        fit_indices,
        evaluation_weights,
        potential_generator_values,
    )
)
potential_oracle_n_basis = int(potential_oracle_best["n_basis"])
selected_models["potential_3d_oracle"] = {
    "coefficients": potential_oracle_coefficients,
    "observed_prediction": _potential_velocity(
        gradient_observed, potential_oracle_coefficients, potential_oracle_n_basis
    ),
    "query_prediction": _potential_velocity(
        gradient_query, potential_oracle_coefficients, potential_oracle_n_basis
    ),
    **potential_oracle_best,
}

cartesian_oracle = _fit_cartesian_3d_oracle(
    geometry,
    query_extension,
    truth_survey,
    train_indices,
    validation_indices,
    fit_indices,
    evaluation_weights,
)
selected_models["cartesian_3d_oracle"] = cartesian_oracle

# Weighted radial bulk-flow baseline using the operational loss.
operational_weights = weight_arrays[operational["loss_name"]]
los_fit = line_of_sight_observed[fit_indices]
w_fit = operational_weights[fit_indices]
bulk_gram = los_fit.T @ (w_fit[:, None] * los_fit)
bulk_rhs = los_fit.T @ (w_fit * radial_truth_observed[fit_indices])
bulk_vector = linalg.solve(
    bulk_gram, bulk_rhs, assume_a="sym", check_finite=False
)
selected_models["radial_bulk_flow"] = {
    "observed_prediction": np.repeat(
        bulk_vector[None, :], centered_survey.shape[0], axis=0
    ),
    "query_prediction": np.repeat(
        bulk_vector[None, :], centered_query.shape[0], axis=0
    ),
    "bulk_vector": bulk_vector,
}
selected_models["zero_vector_baseline"] = {
    "observed_prediction": np.zeros_like(truth_survey),
    "query_prediction": np.zeros_like(truth_query),
}

# Fair scalar radial local-kernel baselines.
local_records = []
local_predictions = {}
for local_name, source_loss_name in [
    ("local_radial_unweighted", "unweighted"),
    ("local_radial_optimized_loss", operational["loss_name"]),
]:
    source_weights = weight_arrays[source_loss_name]
    validation_distances, validation_neighbors = _query_neighbors(
        centered_survey[validation_indices],
        centered_survey[train_indices],
        max(LOCAL_KERNEL_NEIGHBORS),
    )
    best = None
    for neighbors in LOCAL_KERNEL_NEIGHBORS:
        bandwidth_neighbor = min(
            neighbors,
            max(1, int(round(neighbors * LOCAL_KERNEL_BANDWIDTH_FRACTION))),
        )
        for multiplier in LOCAL_KERNEL_MULTIPLIERS:
            prediction = _local_kernel_from_neighbors(
                validation_distances,
                validation_neighbors,
                radial_truth_observed[train_indices, None],
                source_weights[train_indices],
                neighbors,
                bandwidth_neighbor,
                multiplier,
            )[:, 0]
            score = _scalar_nrmse(
                radial_truth_observed[validation_indices],
                prediction,
                weights=evaluation_weights[validation_indices],
            )
            record = {
                "model": local_name,
                "source_loss_name": source_loss_name,
                "n_neighbors": int(neighbors),
                "bandwidth_neighbor": int(bandwidth_neighbor),
                "bandwidth_multiplier": float(multiplier),
                "weighted_validation_nrmse_radial": float(score),
            }
            local_records.append(record)
            candidate = (score, neighbors, multiplier)
            if best is None or candidate < best[0]:
                best = (candidate, record)
    best_record = best[1]
    test_distances, test_neighbors = _query_neighbors(
        centered_survey[test_indices],
        centered_survey[fit_indices],
        max(LOCAL_KERNEL_NEIGHBORS),
    )
    query_distances_neighbors, query_neighbors = _query_neighbors(
        centered_query,
        centered_survey[fit_indices],
        max(LOCAL_KERNEL_NEIGHBORS),
    )
    observed_test_prediction = _local_kernel_from_neighbors(
        test_distances,
        test_neighbors,
        radial_truth_observed[fit_indices, None],
        source_weights[fit_indices],
        best_record["n_neighbors"],
        best_record["bandwidth_neighbor"],
        best_record["bandwidth_multiplier"],
    )[:, 0]
    query_prediction = _local_kernel_from_neighbors(
        query_distances_neighbors,
        query_neighbors,
        radial_truth_observed[fit_indices, None],
        source_weights[fit_indices],
        best_record["n_neighbors"],
        best_record["bandwidth_neighbor"],
        best_record["bandwidth_multiplier"],
    )[:, 0]
    local_predictions[local_name] = {
        "observed_test_prediction": observed_test_prediction,
        "query_prediction": query_prediction,
        **best_record,
    }
local_validation_df = pd.DataFrame(local_records)

# --------------------------------------------------------------------------------------------------
# 10. Untouched observed-test and complete-query evaluation
# --------------------------------------------------------------------------------------------------

evaluation_records = []
radial_bin_records = []
completeness_records = []
prediction_payload = {
    "query_global_indices": query_global,
    "survey_train_indices": train_indices,
    "survey_validation_indices": validation_indices,
    "survey_test_indices": test_indices,
    "truth_survey": truth_survey,
    "truth_query": truth_query,
    "line_of_sight_survey": line_of_sight_observed,
    "line_of_sight_query": line_of_sight_query,
    "selection_probability_survey": probability_survey,
    "selection_probability_query": probability_query,
    "primary_query_mask": primary_query_mask,
}

for model_name, model in selected_models.items():
    observed_prediction = model["observed_prediction"]
    query_prediction = model["query_prediction"]
    prediction_payload[f"{model_name}_observed"] = observed_prediction
    prediction_payload[f"{model_name}_query"] = query_prediction

    for subset_name, true_values, pred_values, los_values, mask, weights in [
        (
            "observed_test_unweighted",
            truth_survey[test_indices],
            observed_prediction[test_indices],
            line_of_sight_observed[test_indices],
            np.ones(test_indices.size, dtype=bool),
            None,
        ),
        (
            "observed_test_parent_weighted",
            truth_survey[test_indices],
            observed_prediction[test_indices],
            line_of_sight_observed[test_indices],
            np.ones(test_indices.size, dtype=bool),
            evaluation_weights[test_indices],
        ),
    ]:
        metrics = _radial_vector_metrics(
            true_values[mask],
            pred_values[mask],
            los_values[mask],
            weights=None if weights is None else weights[mask],
        )
        evaluation_records.append(
            {"model": model_name, "subset": subset_name, **metrics}
        )

    for subset_name, mask in query_subsets.items():
        if np.sum(mask) < 5:
            continue
        metrics = _radial_vector_metrics(
            truth_query[mask],
            query_prediction[mask],
            line_of_sight_query[mask],
        )
        evaluation_records.append(
            {"model": model_name, "subset": subset_name, **metrics}
        )
        completeness_records.append(
            {
                "model": model_name,
                "subset": subset_name,
                "selection_probability_median": float(
                    np.median(probability_query[mask])
                ),
                **metrics,
            }
        )

    # Equal-count radial bins inside the primary query set.
    primary_radius = query_dist[primary_query_mask]
    if primary_radius.size >= RADIAL_BINS * 5:
        edges = np.unique(
            np.quantile(primary_radius, np.linspace(0.0, 1.0, RADIAL_BINS + 1))
        )
        for bin_index in range(len(edges) - 1):
            if bin_index == len(edges) - 2:
                local = primary_query_mask & (query_dist >= edges[bin_index]) & (
                    query_dist <= edges[bin_index + 1]
                )
            else:
                local = primary_query_mask & (query_dist >= edges[bin_index]) & (
                    query_dist < edges[bin_index + 1]
                )
            if np.sum(local) < 5:
                continue
            metrics = _radial_vector_metrics(
                truth_query[local],
                query_prediction[local],
                line_of_sight_query[local],
            )
            radial_bin_records.append(
                {
                    "model": model_name,
                    "radial_bin": int(bin_index),
                    "radius_min": float(edges[bin_index]),
                    "radius_max": float(edges[bin_index + 1]),
                    "radius_median": float(np.median(query_dist[local])),
                    **metrics,
                }
            )

evaluation_df = pd.DataFrame(evaluation_records)
radial_bins_df = pd.DataFrame(radial_bin_records)
completeness_df = pd.DataFrame(completeness_records)

# Scalar-only local radial records.
local_evaluation_records = []
for model_name, model in local_predictions.items():
    local_evaluation_records.extend(
        [
            {
                "model": model_name,
                "subset": "observed_test_unweighted",
                "n": int(test_indices.size),
                "nrmse_radial": _scalar_nrmse(
                    radial_truth_observed[test_indices],
                    model["observed_test_prediction"],
                ),
            },
            {
                "model": model_name,
                "subset": "observed_test_parent_weighted",
                "n": int(test_indices.size),
                "nrmse_radial": _scalar_nrmse(
                    radial_truth_observed[test_indices],
                    model["observed_test_prediction"],
                    weights=evaluation_weights[test_indices],
                ),
            },
            {
                "model": model_name,
                "subset": "query_primary_boundary_safe",
                "n": int(np.sum(primary_query_mask)),
                "nrmse_radial": _scalar_nrmse(
                    radial_truth_query[primary_query_mask],
                    model["query_prediction"][primary_query_mask],
                ),
            },
        ]
    )
local_evaluation_df = pd.DataFrame(local_evaluation_records)

primary_query_df = evaluation_df[
    evaluation_df["subset"] == "query_primary_boundary_safe"
].sort_values("nrmse_3d")
observed_parent_df = evaluation_df[
    evaluation_df["subset"] == "observed_test_parent_weighted"
].sort_values("nrmse_3d")
print("Primary complete-query comparison")
display(primary_query_df)
print("Observed held-out parent-weighted comparison")
display(observed_parent_df)
print("Scalar radial baselines")
display(local_evaluation_df)

# --------------------------------------------------------------------------------------------------
# 11. Selected-design linear algebra and paired uncertainty
# --------------------------------------------------------------------------------------------------

operational_n_basis = int(operational["n_basis"])
operational_regularization = _regularization_diagonal(
    potential_generator_values,
    operational_n_basis,
    operational["penalty_power"],
)
operational_fit_weights = weight_arrays[operational["loss_name"]][fit_indices]
operational_fit_design = radial_design_observed[
    fit_indices, :operational_n_basis
]
operational_gram, _ = _weighted_crossproducts(
    operational_fit_design,
    radial_truth_observed[fit_indices, None],
    operational_fit_weights,
)
spectrum = _gram_spectrum(operational_gram, fit_indices.size)
effective_df = _effective_degrees_of_freedom(
    operational_gram,
    operational_regularization,
    operational["regularization_strength"],
)
linear_algebra_df = pd.DataFrame(
    [
        {
            "n_parameters": operational_n_basis,
            "n_observations": int(fit_indices.size),
            "numerical_rank": spectrum["numerical_rank"],
            "nullity": spectrum["nullity"],
            "condition_number": spectrum["condition_number"],
            "stable_rank": spectrum["stable_rank"],
            "entropy_effective_rank": spectrum["entropy_effective_rank"],
            "effective_degrees_of_freedom": effective_df,
            "effective_degrees_of_freedom_fraction": effective_df
            / operational_n_basis,
            "loss_name": operational["loss_name"],
            "loss_gamma": operational["loss_gamma"],
            "loss_cap": operational["loss_cap"],
            "loss_ess_fit": _effective_sample_size(operational_fit_weights),
            "regularization_strength": operational["regularization_strength"],
            "penalty_power": operational["penalty_power"],
        }
    ]
)
singular_values_df = pd.DataFrame(
    {
        "singular_value_index": np.arange(spectrum["singular_values"].size),
        "singular_value": spectrum["singular_values"],
        "normalized_singular_value": spectrum["singular_values"]
        / spectrum["singular_values"][0],
    }
)
display(linear_algebra_df)

bootstrap_records = []
optimized_model = selected_models["potential_optimized_loss"]
unweighted_model = selected_models["potential_unweighted_loss"]
previous_model = selected_models["potential_previous_mock2_loss"]
for subset_name, true, los, mask, weights in [
    (
        "observed_test_parent_weighted",
        truth_survey[test_indices],
        line_of_sight_observed[test_indices],
        np.ones(test_indices.size, dtype=bool),
        evaluation_weights[test_indices],
    ),
    (
        "query_primary_boundary_safe",
        truth_query,
        line_of_sight_query,
        primary_query_mask,
        None,
    ),
]:
    if subset_name.startswith("observed"):
        pred_opt = optimized_model["observed_prediction"][test_indices]
        pred_unweighted = unweighted_model["observed_prediction"][test_indices]
        pred_previous = previous_model["observed_prediction"][test_indices]
    else:
        pred_opt = optimized_model["query_prediction"]
        pred_unweighted = unweighted_model["query_prediction"]
        pred_previous = previous_model["query_prediction"]
    for comparison_name, comparison_prediction, seed_offset in [
        ("optimized_minus_unweighted", pred_unweighted, 0),
        ("optimized_minus_previous_mock2", pred_previous, 1),
    ]:
        result = _paired_bootstrap_difference(
            true[mask],
            pred_opt[mask],
            comparison_prediction[mask],
            los[mask],
            None if weights is None else weights[mask],
            BOOTSTRAP_SEED + seed_offset,
        )
        bootstrap_records.append(
            {
                "subset": subset_name,
                "comparison": comparison_name,
                **result,
            }
        )
bootstrap_df = pd.DataFrame(bootstrap_records)
display(bootstrap_df)

# --------------------------------------------------------------------------------------------------
# 12. Diagnostic figures
# --------------------------------------------------------------------------------------------------

# Selection probability as a function of radius.
fig, ax = plt.subplots(figsize=(7.4, 5.0))
ax.scatter(survey["dist"], probability_survey, s=7, alpha=0.25, label="observed survey")
order = np.argsort(query_dist)
ax.plot(query_dist[order], probability_query[order], linewidth=1.4, label="complete query")
ax.set_yscale("log")
ax.set_xlabel(r"Radius [$h^{-1}\,\mathrm{Mpc}$]")
ax.set_ylabel("Selection probability")
ax.legend(frameon=True)
fig.tight_layout()
_save_radial_mock2_figure(fig, "selection_probability_vs_radius")

# Primary-query model comparison.
plot_models = [
    "potential_optimized_loss",
    "potential_unweighted_loss",
    "potential_previous_mock2_loss",
    "potential_hidden_3d_diagnostic",
    "potential_3d_oracle",
    "cartesian_3d_oracle",
    "radial_bulk_flow",
    "zero_vector_baseline",
]
labels = {
    "potential_optimized_loss": "Potential optimized loss",
    "potential_unweighted_loss": "Potential unweighted loss",
    "potential_previous_mock2_loss": "Potential previous Mock-2 loss",
    "potential_hidden_3d_diagnostic": "Potential hidden-3D diagnostic",
    "potential_3d_oracle": "Potential 3D oracle",
    "cartesian_3d_oracle": "Cartesian 3D oracle",
    "radial_bulk_flow": "Radial bulk flow",
    "zero_vector_baseline": "Zero baseline",
}
plot_df = primary_query_df.set_index("model").loc[plot_models].reset_index()
x = np.arange(len(plot_df))
width = 0.25
fig, ax = plt.subplots(figsize=(13.0, 5.6))
ax.bar(x - width, plot_df["nrmse_radial"], width, label="radial")
ax.bar(x, plot_df["nrmse_tangential"], width, label="tangential")
ax.bar(x + width, plot_df["nrmse_3d"], width, label="3D")
ax.axhline(1.0, linestyle="--", linewidth=1.2, color="0.35", label="zero-vector baseline")
ax.set_xticks(x)
ax.set_xticklabels([labels[name] for name in plot_df["model"]], rotation=24, ha="right")
ax.set_ylabel("Primary complete-query NRMSE")
ax.legend(frameon=True, ncol=2)
fig.tight_layout()
_save_radial_mock2_figure(fig, "primary_query_model_comparison")

# Observed parent-weighted model comparison.
plot_df_obs = observed_parent_df.set_index("model").loc[plot_models].reset_index()
fig, ax = plt.subplots(figsize=(13.0, 5.6))
ax.bar(x - width, plot_df_obs["nrmse_radial"], width, label="radial")
ax.bar(x, plot_df_obs["nrmse_tangential"], width, label="tangential")
ax.bar(x + width, plot_df_obs["nrmse_3d"], width, label="3D")
ax.axhline(1.0, linestyle="--", linewidth=1.2, color="0.35", label="zero-vector baseline")
ax.set_xticks(x)
ax.set_xticklabels([labels[name] for name in plot_df_obs["model"]], rotation=24, ha="right")
ax.set_ylabel("Parent-weighted observed-test NRMSE")
ax.legend(frameon=True, ncol=2)
fig.tight_layout()
_save_radial_mock2_figure(fig, "observed_parent_weighted_model_comparison")

# Completeness dependence.
fig, ax = plt.subplots(figsize=(8.0, 5.2))
subset_order = [
    "query_high_completeness",
    "query_medium_completeness",
    "query_low_completeness",
    "query_very_low_completeness",
]
for model_name in [
    "potential_optimized_loss",
    "potential_unweighted_loss",
    "potential_previous_mock2_loss",
]:
    selected = completeness_df[
        (completeness_df["model"] == model_name)
        & completeness_df["subset"].isin(subset_order)
    ].copy()
    selected["order"] = selected["subset"].map(
        {name: i for i, name in enumerate(subset_order)}
    )
    selected = selected.sort_values("order")
    ax.plot(
        selected["selection_probability_median"],
        selected["nrmse_3d"],
        marker="o",
        label=labels[model_name],
    )
ax.set_xscale("log")
ax.set_xlabel("Median selection probability")
ax.set_ylabel("Complete-query 3D NRMSE")
ax.legend(frameon=True)
fig.tight_layout()
_save_radial_mock2_figure(fig, "query_3d_nrmse_by_completeness")

# Radius dependence.
fig, ax = plt.subplots(figsize=(8.0, 5.2))
for model_name in [
    "potential_optimized_loss",
    "potential_unweighted_loss",
    "potential_previous_mock2_loss",
]:
    selected = radial_bins_df[radial_bins_df["model"] == model_name].sort_values(
        "radius_median"
    )
    ax.plot(
        selected["radius_median"],
        selected["nrmse_3d"],
        marker="o",
        label=labels[model_name],
    )
ax.set_xlabel(r"Radius [$h^{-1}\,\mathrm{Mpc}$]")
ax.set_ylabel("Primary complete-query 3D NRMSE")
ax.legend(frameon=True)
fig.tight_layout()
_save_radial_mock2_figure(fig, "query_3d_nrmse_by_radius")

# Validation weight/configuration summary.
config_best = (
    validation_df.sort_values(
        ["weighted_validation_nrmse_radial", "unweighted_validation_nrmse_radial"]
    )
    .groupby("loss_name", as_index=False)
    .first()
)
fig, ax = plt.subplots(figsize=(9.0, 5.2))
positions = np.arange(len(config_best))
ax.scatter(
    positions,
    config_best["weighted_validation_nrmse_radial"],
    s=65,
)
ax.set_xticks(positions)
ax.set_xticklabels(config_best["loss_name"], rotation=28, ha="right")
ax.set_ylabel("Parent-balanced radial validation NRMSE")
fig.tight_layout()
_save_radial_mock2_figure(fig, "loss_weight_validation_summary")

# Selected design singular spectrum.
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.plot(
    singular_values_df["singular_value_index"],
    singular_values_df["normalized_singular_value"],
)
ax.set_yscale("log")
ax.set_xlabel("Singular-value index")
ax.set_ylabel(r"$\sigma_j/\sigma_1$")
row = linear_algebra_df.iloc[0]
ax.text(
    0.04,
    0.06,
    f"rank = {int(row.numerical_rank)}/{int(row.n_parameters)}\n"
    f"condition number = {row.condition_number:.1f}\n"
    f"stable rank = {row.stable_rank:.1f}\n"
    f"entropy effective rank = {row.entropy_effective_rank:.1f}",
    transform=ax.transAxes,
    bbox=dict(boxstyle="round", facecolor="white", alpha=0.9),
)
fig.tight_layout()
_save_radial_mock2_figure(fig, "operational_design_singular_spectrum")

# Out-of-sample closure.
fig, ax = plt.subplots(figsize=(7.6, 5.2))
ax.plot(closure_df["n_basis"], closure_df["nrmse_3d"], marker="o")
ax.axhline(CLOSURE_PASS_TOLERANCE, linestyle="--", linewidth=1.2, label="pass tolerance")
ax.set_yscale("log")
ax.set_xlabel(r"Number of potential parameters $3+n_{\Phi}$")
ax.set_ylabel("Complete-query in-span 3D NRMSE")
ax.legend(frameon=True)
fig.tight_layout()
_save_radial_mock2_figure(fig, "out_of_sample_in_span_closure")

# Direction-error CDF on primary complete query.
fig, ax = plt.subplots(figsize=(8.0, 5.4))
for model_name in [
    "potential_optimized_loss",
    "potential_unweighted_loss",
    "potential_previous_mock2_loss",
    "cartesian_3d_oracle",
]:
    prediction = selected_models[model_name]["query_prediction"]
    angles = _direction_errors(
        truth_query[primary_query_mask], prediction[primary_query_mask]
    )
    angles = np.sort(angles[np.isfinite(angles)])
    cumulative = np.arange(1, angles.size + 1) / angles.size
    ax.plot(angles, cumulative, label=labels[model_name])
ax.set_xlim(0.0, 180.0)
ax.set_ylim(0.0, 1.0)
ax.set_xlabel("Direction error [deg]")
ax.set_ylabel("Cumulative fraction")
ax.legend(frameon=True)
fig.tight_layout()
_save_radial_mock2_figure(fig, "primary_query_direction_error_cdf")

# --------------------------------------------------------------------------------------------------
# 13. Save all outputs
# --------------------------------------------------------------------------------------------------

gradient_diagnostics_df = pd.DataFrame([gradient_result["diagnostics"]])
weight_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_selection_weight_diagnostics.csv", index=False)
support_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_support.csv", index=False)
gradient_diagnostics_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_gradient_diagnostics.csv", index=False
)
closure_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_out_of_sample_closure.csv", index=False)
validation_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_validation_grid.csv", index=False)
selector_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_selector_summary.csv", index=False)
potential_oracle_grid.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_potential_oracle_validation_grid.csv", index=False
)
cartesian_oracle["grid"].to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_cartesian_oracle_validation_grid.csv", index=False
)
local_validation_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_local_radial_validation.csv", index=False
)
evaluation_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_evaluation.csv", index=False)
local_evaluation_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_local_radial_evaluation.csv", index=False
)
radial_bins_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_radial_bins.csv", index=False)
completeness_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_completeness_subsets.csv", index=False
)
linear_algebra_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_linear_algebra.csv", index=False
)
singular_values_df.to_csv(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_singular_values.csv", index=False
)
bootstrap_df.to_csv(OUTPUT_DIR / f"{OUTPUT_PREFIX}_bootstrap.csv", index=False)

for model_name, model in selected_models.items():
    if "coefficients" in model:
        prediction_payload[f"{model_name}_coefficients"] = model["coefficients"]
for model_name, model in local_predictions.items():
    prediction_payload[f"{model_name}_observed_test_radial"] = model[
        "observed_test_prediction"
    ]
    prediction_payload[f"{model_name}_query_radial"] = model["query_prediction"]
np.savez_compressed(
    OUTPUT_DIR / f"{OUTPUT_PREFIX}_predictions.npz", **prediction_payload
)

summary = {
    "experiment": "Radial Mock-2 potential-flow selection-loss validation",
    "selection_geometry": "unweighted alpha=0",
    "hyperparameter_selection": "parent-balanced observed radial validation NRMSE only",
    "query_not_used_for_selection": True,
    "catalog": {
        "n_survey": int(survey["pos"].shape[0]),
        "n_complete_available": int(complete_available.size),
        "n_reference": int(reference_global.size),
        "n_query": int(query_global.size),
        "n_train": int(train_indices.size),
        "n_validation": int(validation_indices.size),
        "n_test": int(test_indices.size),
    },
    "lf_model": {
        key: lf_model[key]
        for key in ["M_star", "alpha", "M_bright", "M_faint"]
    },
    "operational_selection": operational,
    "hidden_3d_selection_diagnostic": hidden_3d,
    "unweighted_selection": unweighted_selected,
    "previous_mock2_selection": previous_selected,
    "potential_oracle_selection": potential_oracle_best,
    "cartesian_oracle_selection": cartesian_oracle["best"],
    "primary_query_results": primary_query_df.to_dict(orient="records"),
    "observed_parent_weighted_results": observed_parent_df.to_dict(orient="records"),
    "local_radial_results": local_evaluation_df.to_dict(orient="records"),
    "bootstrap": bootstrap_df.to_dict(orient="records"),
    "support": support_df.to_dict(orient="records"),
    "linear_algebra": linear_algebra_df.to_dict(orient="records"),
    "gradient_diagnostics": gradient_result["diagnostics"],
    "closure_all_pass": bool(np.all(closure_df["closure_status"] == "pass")),
    "memory_design": {
        "dense_N_by_N_created": False,
        "query_transition_shape": [int(query_global.size), int(GRAPH_NEIGHBORS)],
        "maximum_potential_parameters": int(max_potential_basis),
    },
}
with (OUTPUT_DIR / f"{OUTPUT_PREFIX}_summary.json").open("w", encoding="utf-8") as handle:
    json.dump(_json_ready(summary), handle, ensure_ascii=False, indent=2)

print("=" * 122)
print("Radial Mock-2 potential-flow selection experiment completed")
print("=" * 122)
print(f"Elapsed time             : {(time.perf_counter() - start_time) / 60.0:.2f} min")
print("Operational selector")
display(selector_df[selector_df["selector"] == "operational_radial_validation"])
print("Primary complete-query comparison")
display(primary_query_df)
print("Observed parent-weighted comparison")
display(observed_parent_df)
print("Selection-weight diagnostics")
display(weight_df)
print("Out-of-sample closure")
display(closure_df)
print("Linear-algebra diagnostics")
display(linear_algebra_df)
print("Saved files")
for path in sorted(OUTPUT_DIR.glob(f"{OUTPUT_PREFIX}_*")):
    print(f"  {path.name}")
