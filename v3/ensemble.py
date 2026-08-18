"""Structured failure probabilities, robust Brier blending, and calibration."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy.optimize import minimize


EPSILON = 1e-5
RECENCY_SCHEMES = {
    "primary": {2022: 0.15, 2023: 0.30, 2024: 0.55},
    "balanced": {2022: 0.20, 2023: 0.30, 2024: 0.50},
    "aggressive": {2022: 0.10, 2023: 0.30, 2024: 0.60},
}


def brier(y_true: np.ndarray, prediction: np.ndarray) -> float:
    y = np.asarray(y_true, dtype="float64")
    p = np.clip(np.asarray(prediction, dtype="float64"), EPSILON, 1 - EPSILON)
    return float(np.mean(np.square(y - p)))


def sigmoid(values: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(values, dtype="float64"), -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-values))


def structured_probability(
    reverse: np.ndarray,
    middle: np.ndarray,
    outside: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    reverse = np.asarray(reverse, dtype="float64")
    middle = np.asarray(middle, dtype="float64")
    outside = np.asarray(outside, dtype="float64")
    overlap = np.minimum(np.asarray(overlap, dtype="float64"), np.minimum(reverse, middle))
    failure = reverse + middle - overlap + outside
    return np.clip(1.0 - failure, EPSILON, 1 - EPSILON)


def failure_design(
    reverse: np.ndarray,
    middle: np.ndarray,
    outside: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    overlap = np.minimum(overlap, np.minimum(reverse, middle))
    return np.column_stack([-reverse, -middle, -outside, overlap]).astype("float64")


def fit_failure_combiner(
    y_true: np.ndarray,
    reverse: np.ndarray,
    middle: np.ndarray,
    outside: np.ndarray,
    overlap: np.ndarray,
) -> dict[str, Any]:
    design = failure_design(reverse, middle, outside, overlap)
    target = np.asarray(y_true, dtype="float64")

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        intercept = parameters[0]
        coefficients = parameters[1:]
        probability = sigmoid(intercept + design @ coefficients)
        residual = probability - target
        derivative = probability * (1.0 - probability)
        penalty = 1e-3 * float(np.square(coefficients).sum())
        loss = float(np.mean(np.square(residual))) + penalty
        common = 2.0 * residual * derivative / len(target)
        gradient = np.r_[common.sum(), design.T @ common + 2e-3 * coefficients]
        return loss, gradient

    initial = np.asarray([1.0, 1.0, 1.0, 1.0, 0.5], dtype="float64")
    fitted = minimize(
        lambda values: objective(values)[0],
        initial,
        jac=lambda values: objective(values)[1],
        method="L-BFGS-B",
        bounds=[(-5.0, 5.0)] + [(0.0, 10.0)] * 4,
    )
    parameters = fitted.x if fitted.success else initial
    return {
        "intercept": float(parameters[0]),
        "coefficients": [float(value) for value in parameters[1:]],
        "success": bool(fitted.success),
    }


def apply_failure_combiner(
    config: dict[str, Any],
    reverse: np.ndarray,
    middle: np.ndarray,
    outside: np.ndarray,
    overlap: np.ndarray,
) -> np.ndarray:
    design = failure_design(reverse, middle, outside, overlap)
    return np.clip(
        sigmoid(float(config["intercept"]) + design @ np.asarray(config["coefficients"])),
        EPSILON,
        1 - EPSILON,
    )


def project_simplex(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(values) + 1)
    valid = ordered - cumulative / indices > 0
    if not valid.any():
        return np.full(len(values), 1.0 / len(values))
    threshold = cumulative[valid][-1] / indices[valid][-1]
    result = np.maximum(values - threshold, 0.0)
    return result / result.sum()


def fit_simplex_brier(
    y_true: np.ndarray,
    matrix: np.ndarray,
    sample_weight: np.ndarray | None = None,
    iterations: int = 5000,
) -> np.ndarray:
    matrix = np.asarray(matrix, dtype="float64")
    target = np.asarray(y_true, dtype="float64")
    weights = np.ones(len(target), dtype="float64") if sample_weight is None else np.asarray(sample_weight, dtype="float64")
    weights /= weights.sum()
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    gram = weighted_matrix.T @ weighted_matrix
    step = 1.0 / max(2.0 * float(np.linalg.eigvalsh(gram).max()), 1e-8)
    coefficients = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    for _ in range(iterations):
        gradient = 2.0 * matrix.T @ (weights * (matrix @ coefficients - target))
        updated = project_simplex(coefficients - step * gradient)
        if np.max(np.abs(updated - coefficients)) < 1e-11:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def season_sample_weights(seasons: np.ndarray, scheme: str = "primary") -> np.ndarray:
    mapping = RECENCY_SCHEMES[scheme]
    output = np.asarray([mapping.get(int(year), min(mapping.values())) for year in seasons], dtype="float64")
    return output / output.mean()


def season_metrics(
    y_true: np.ndarray,
    prediction: np.ndarray,
    seasons: np.ndarray,
    game_types: np.ndarray,
) -> dict[str, float]:
    result: dict[str, float] = {}
    for year in [2022, 2023, 2024]:
        current = seasons == year
        if current.any():
            result[f"brier_{year}"] = brier(y_true[current], prediction[current])
            for game_type in ["R", "F"]:
                typed = current & (game_types == game_type)
                if typed.any():
                    result[f"brier_{year}_{game_type.lower()}"] = brier(y_true[typed], prediction[typed])
    return result


def robust_score(metrics: dict[str, float], scheme: str = "primary") -> float:
    weights = RECENCY_SCHEMES[scheme]
    values = np.asarray([metrics[f"brier_{year}"] for year in weights], dtype="float64")
    weighted = float(sum(weights[year] * metrics[f"brier_{year}"] for year in weights))
    dispersion = 0.10 * float(values.max() - values.min())
    regular_penalty = max(0.0, metrics.get("brier_2024_r", 0.0) - 0.24790)
    futures_penalty = max(0.0, metrics.get("brier_2024_f", 0.0) - 0.24700)
    return weighted + dispersion + regular_penalty + futures_penalty


def fit_affine(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    p = np.asarray(prediction, dtype="float64")
    y = np.asarray(y_true, dtype="float64")
    variance = float(np.mean(np.square(p - p.mean())))
    covariance = float(np.mean((p - p.mean()) * (y - y.mean())))
    slope = float(np.clip(covariance / (variance + 1e-8), 0.5, 1.5))
    intercept = float(np.clip(y.mean() - slope * p.mean(), -0.10, 0.10))
    return {"method": "affine", "slope": slope, "intercept": intercept}


def fit_beta(y_true: np.ndarray, prediction: np.ndarray) -> dict[str, Any]:
    p = np.clip(np.asarray(prediction, dtype="float64"), EPSILON, 1 - EPSILON)
    y = np.asarray(y_true, dtype="float64")
    design = np.column_stack([np.log(p), -np.log1p(-p)])

    def objective(parameters: np.ndarray) -> tuple[float, np.ndarray]:
        probability = sigmoid(parameters[2] + design @ parameters[:2])
        residual = probability - y
        derivative = probability * (1.0 - probability)
        identity = np.asarray([1.0, 1.0, 0.0])
        penalty = 5e-3 * float(np.square(parameters - identity).sum())
        loss = float(np.mean(np.square(residual))) + penalty
        common = 2.0 * residual * derivative / len(y)
        gradient = np.r_[design.T @ common, common.sum()] + 1e-2 * (parameters - identity)
        return loss, gradient

    initial = np.asarray([1.0, 1.0, 0.0])
    fitted = minimize(
        lambda values: objective(values)[0], initial,
        jac=lambda values: objective(values)[1], method="L-BFGS-B",
        bounds=[(0.05, 5.0), (0.05, 5.0), (-3.0, 3.0)],
    )
    parameters = fitted.x if fitted.success else initial
    return {
        "method": "beta",
        "a": float(parameters[0]),
        "b": float(parameters[1]),
        "c": float(parameters[2]),
        "success": bool(fitted.success),
    }


def apply_calibrator(prediction: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    p = np.clip(np.asarray(prediction, dtype="float64"), EPSILON, 1 - EPSILON)
    method = config.get("method", "identity")
    if method == "identity":
        calibrated = p
    elif method == "affine":
        calibrated = float(config["slope"]) * p + float(config["intercept"])
    elif method == "beta":
        calibrated = sigmoid(
            float(config["a"]) * np.log(p)
            + float(config["b"]) * (-np.log1p(-p))
            + float(config["c"])
        )
    else:
        raise ValueError(f"Unsupported calibrator: {method}")
    eta = float(config.get("eta", 1.0))
    return np.clip((1.0 - eta) * p + eta * calibrated, EPSILON, 1 - EPSILON)
