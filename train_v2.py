"""V2 temporal-robust CatBoost training pipeline.

Five OOF prediction sources are trained with expanding-window validation:
global long-history, global recent-history, seasonless global, routed R/F
experts, and a pitcher-only expert. Their probabilities are combined by a
non-negative Brier optimizer and calibrated strictly walk-forward.
"""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, mean_squared_error, roc_auc_score

from result import (
    ID_COL,
    TARGET_COL,
    build_features,
    initial_feature_config,
    read_compact_csv,
    run_inference,
    save_json,
)


SOURCE_ORDER = [
    "global_long",
    "global_recent",
    "global_seasonless",
    "game_type",
    "pitcher",
]
MODEL_FILES = [
    "global_long.cbm",
    "global_recent.cbm",
    "global_seasonless.cbm",
    "regular_expert.cbm",
    "futures_expert.cbm",
    "pitcher_expert.cbm",
]


def brier(y_true: np.ndarray, prediction: np.ndarray) -> float:
    return float(mean_squared_error(y_true, np.clip(prediction, 1e-5, 1 - 1e-5)))


def recency_weights(
    train_seasons: np.ndarray, prediction_year: int, half_life: float
) -> np.ndarray:
    age = np.maximum(0, (prediction_year - 1) - np.asarray(train_seasons, dtype="float64"))
    weights = np.power(0.5, age / float(half_life))
    return (weights / weights.mean()).astype("float32")


def project_simplex(values: np.ndarray) -> np.ndarray:
    """Euclidean projection onto non-negative weights summing to one."""
    values = np.asarray(values, dtype="float64")
    ordered = np.sort(values)[::-1]
    cumulative = np.cumsum(ordered) - 1.0
    indices = np.arange(1, len(values) + 1)
    valid = ordered - cumulative / indices > 0
    if not valid.any():
        return np.full(len(values), 1.0 / len(values))
    rho = indices[valid][-1]
    threshold = cumulative[valid][-1] / rho
    projected = np.maximum(values - threshold, 0.0)
    return projected / projected.sum()


def optimize_brier_weights(
    y_true: np.ndarray, prediction_matrix: np.ndarray, iterations: int = 4000
) -> np.ndarray:
    """Solve the convex non-negative linear Brier blend by projected gradient."""
    matrix = np.asarray(prediction_matrix, dtype="float64")
    target = np.asarray(y_true, dtype="float64")
    if matrix.ndim != 2 or matrix.shape[0] != len(target):
        raise ValueError("Prediction matrix and target shapes do not match")
    if not np.isfinite(matrix).all():
        raise ValueError("Prediction matrix contains non-finite values")
    gram = (matrix.T @ matrix) / len(target)
    lipschitz = max(2.0 * float(np.linalg.eigvalsh(gram).max()), 1e-8)
    step = 1.0 / lipschitz
    weights = np.full(matrix.shape[1], 1.0 / matrix.shape[1])
    for _ in range(iterations):
        gradient = 2.0 * (matrix.T @ (matrix @ weights - target)) / len(target)
        updated = project_simplex(weights - step * gradient)
        if np.max(np.abs(updated - weights)) < 1e-11:
            weights = updated
            break
        weights = updated
    return weights


def fit_affine_calibrator(
    y_true: np.ndarray, prediction: np.ndarray
) -> tuple[float, float]:
    probability = np.asarray(prediction, dtype="float64")
    target = np.asarray(y_true, dtype="float64")
    variance = float(np.mean((probability - probability.mean()) ** 2))
    covariance = float(
        np.mean((probability - probability.mean()) * (target - target.mean()))
    )
    slope = float(np.clip(covariance / (variance + 1e-8), 0.5, 1.5))
    intercept = float(np.clip(target.mean() - slope * probability.mean(), -0.10, 0.10))
    return slope, intercept


def make_feature_sets(
    columns: list[str], categorical_columns: list[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    seasonless_drop = {"season", "historical_expanding_prior"}
    game_type_drop = {"game_type"}
    pitcher_exact = {
        "pitcher_id",
        "pitcher_hand",
        "pitcher_team_id",
        "recent_global_prior",
        "historical_expanding_prior",
        "log_pitcher_n",
    }
    pitcher_prefixes = (
        "asof_pitcher_",
        "pitcher_",
        "pitchmix_",
        "fastball_minus_",
    )
    pitcher_columns = [
        column
        for column in columns
        if column in pitcher_exact or column.startswith(pitcher_prefixes)
    ]
    feature_sets = {
        "global_long": list(columns),
        "global_recent": list(columns),
        "global_seasonless": [c for c in columns if c not in seasonless_drop],
        "game_type": [c for c in columns if c not in game_type_drop],
        "pitcher": pitcher_columns,
    }
    categorical_sets = {
        name: [column for column in categorical_columns if column in selected]
        for name, selected in feature_sets.items()
    }
    return feature_sets, categorical_sets


def model_parameters(
    iterations: int,
    depth: int,
    learning_rate: float,
    seed: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": float(learning_rate),
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "random_seed": int(seed),
        "loss_function": "Logloss",
        # GPU does not implement Brier as an online eval metric. OOF model
        # selection and blending still optimize Brier after prediction.
        "eval_metric": "Logloss",
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": int(verbose),
        "task_type": task_type,
    }
    if task_type == "GPU":
        parameters["devices"] = devices
    return parameters


def fit_fold_model(
    X: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    columns: list[str],
    categorical: list[str],
    weights: np.ndarray,
    parameters: dict[str, Any],
    early_stopping: int,
) -> tuple[np.ndarray, int]:
    model = CatBoostClassifier(**parameters)
    model.fit(
        X.loc[train_mask, columns],
        y[train_mask],
        cat_features=categorical,
        sample_weight=weights,
        eval_set=(X.loc[valid_mask, columns], y[valid_mask]),
        early_stopping_rounds=early_stopping,
        use_best_model=True,
    )
    prediction = np.clip(
        model.predict_proba(X.loc[valid_mask, columns])[:, 1], 1e-5, 1 - 1e-5
    )
    best_iteration = max(1, int(model.get_best_iteration()) + 1)
    del model
    gc.collect()
    return prediction.astype("float32"), best_iteration


def _model_metrics(
    season: int,
    name: str,
    y_true: np.ndarray,
    prediction: np.ndarray,
    baseline: float,
    best_iteration: int = 0,
) -> dict[str, Any]:
    probability = np.clip(np.asarray(prediction), 1e-5, 1 - 1e-5)
    score = brier(y_true, probability)
    baseline_score = brier(y_true, np.full(len(y_true), baseline))
    return {
        "season": int(season),
        "model": name,
        "rows": int(len(y_true)),
        "brier": score,
        "brier_skill_train_prior": 1.0 - score / baseline_score,
        "log_loss": float(log_loss(y_true, probability)),
        "roc_auc": float(roc_auc_score(y_true, probability)),
        "target_mean": float(np.mean(y_true)),
        "prediction_mean": float(np.mean(probability)),
        "best_iteration": int(best_iteration),
    }


def run_training(
    root: str | Path,
    run_dir: str | Path | None = None,
    fast_mode: bool | None = None,
) -> dict[str, Any]:
    started = time.time()
    root = Path(root).resolve()
    run_path = Path(run_dir).resolve() if run_dir else root
    data_dir = root / "open" / "data"
    model_dir = run_path / "model"
    artifact_dir = run_path / "artifacts"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    seed = int(os.environ.get("BASEBALL_SEED", "42"))
    valid_years = [2022, 2023, 2024]
    iterations = int(os.environ.get("BASEBALL_ITERATIONS", "1200"))
    depth = int(os.environ.get("BASEBALL_DEPTH", "8"))
    learning_rate = float(os.environ.get("BASEBALL_LEARNING_RATE", "0.04"))
    early_stopping = int(os.environ.get("BASEBALL_EARLY_STOPPING", "120"))
    minimum_final_iterations = int(os.environ.get("BASEBALL_MIN_FINAL_ITERATIONS", "100"))
    task_type = os.environ.get("BASEBALL_TASK_TYPE", "GPU").upper()
    devices = os.environ.get("BASEBALL_GPU_DEVICES", "0")
    if task_type not in {"CPU", "GPU"}:
        raise ValueError("BASEBALL_TASK_TYPE must be CPU or GPU")
    if fast_mode is None:
        fast_mode = os.environ.get("BASEBALL_FAST_MODE", "0") == "1"
    fast_rows = int(os.environ.get("BASEBALL_FAST_ROWS_PER_SEASON", "30000"))

    train = read_compact_csv(data_dir / "train.csv", train=True)
    if train[ID_COL].duplicated().any():
        raise ValueError("train row_id is duplicated")
    if not set(train[TARGET_COL].unique()).issubset({0, 1}):
        raise ValueError("Target must contain only 0 and 1")
    if fast_mode:
        train = pd.concat(
            [
                part.sample(min(len(part), fast_rows), random_state=seed)
                for _, part in train.groupby("season", sort=True)
            ],
            ignore_index=True,
        )
        print("FAST_MODE: artifacts are diagnostic only and must not be submitted")

    feature_config = initial_feature_config(train)
    X = build_features(train, feature_config)
    feature_config["feature_columns"] = list(X.columns)
    feature_sets, categorical_sets = make_feature_sets(
        list(X.columns), list(feature_config["categorical_columns"])
    )
    feature_config["model_feature_columns"] = feature_sets
    feature_config["model_categorical_columns"] = categorical_sets
    y = train[TARGET_COL].to_numpy(dtype="int8")
    seasons = train["season"].to_numpy(dtype="int16")
    game_types = train["game_type"].astype(str).to_numpy()
    metadata_columns = [
        ID_COL,
        "season",
        "game_type",
        "pitcher_id",
        "asof_pitcher_n",
        "asof_batter_n",
    ]
    metadata = train[metadata_columns].copy()
    n_rows = len(train)
    oof = {name: np.full(n_rows, np.nan, dtype="float32") for name in SOURCE_ORDER}
    best_iterations: dict[str, list[int]] = {
        "global_long": [],
        "global_recent": [],
        "global_seasonless": [],
        "regular_expert": [],
        "futures_expert": [],
        "pitcher": [],
    }
    metric_rows: list[dict[str, Any]] = []
    fold_parameters: list[dict[str, Any]] = []
    base_parameters = model_parameters(
        iterations, depth, learning_rate, seed, task_type, devices, 100
    )

    for valid_year in valid_years:
        fold_started = time.time()
        train_mask = seasons < valid_year
        valid_mask = seasons == valid_year
        if not train_mask.any() or not valid_mask.any():
            raise ValueError(f"Missing data for fold {valid_year}")
        y_valid = y[valid_mask]
        baseline = float(y[train_mask].mean())
        metric_rows.append(
            _model_metrics(
                valid_year,
                "constant_train_prior",
                y_valid,
                np.full(len(y_valid), baseline),
                baseline,
            )
        )
        fold_specs = [
            ("global_long", "global_long", 3.0),
            ("global_recent", "global_recent", 1.0),
            ("global_seasonless", "global_seasonless", 1.0),
            ("pitcher", "pitcher", 1.5),
        ]
        for source_name, feature_name, half_life in fold_specs:
            weights = recency_weights(seasons[train_mask], valid_year, half_life)
            prediction, best = fit_fold_model(
                X,
                y,
                train_mask,
                valid_mask,
                feature_sets[feature_name],
                categorical_sets[feature_name],
                weights,
                base_parameters,
                early_stopping,
            )
            oof[source_name][valid_mask] = prediction
            best_iterations[source_name].append(best)
            metric_rows.append(
                _model_metrics(valid_year, source_name, y_valid, prediction, baseline, best)
            )

        routed_prediction = oof["global_recent"][valid_mask].copy()
        for game_type, artifact_name, half_life in [
            ("R", "regular_expert", 2.5),
            ("F", "futures_expert", 1.0),
        ]:
            typed_train = train_mask & (game_types == game_type)
            typed_valid = valid_mask & (game_types == game_type)
            if not typed_train.any() or not typed_valid.any():
                continue
            typed_weights = recency_weights(
                seasons[typed_train], valid_year, half_life
            )
            typed_prediction, best = fit_fold_model(
                X,
                y,
                typed_train,
                typed_valid,
                feature_sets["game_type"],
                categorical_sets["game_type"],
                typed_weights,
                base_parameters,
                early_stopping,
            )
            routed_prediction[game_types[valid_mask] == game_type] = typed_prediction
            best_iterations[artifact_name].append(best)
        oof["game_type"][valid_mask] = routed_prediction
        metric_rows.append(
            _model_metrics(
                valid_year, "game_type", y_valid, routed_prediction, baseline
            )
        )
        print(
            f"Fold {valid_year} complete | rows={valid_mask.sum():,} "
            f"elapsed={time.time() - fold_started:.1f}s"
        )

    oof_mask = np.logical_and.reduce(
        [np.isfinite(oof[name]) for name in SOURCE_ORDER]
    )
    matrix = np.column_stack([oof[name][oof_mask] for name in SOURCE_ORDER])
    oof_years = seasons[oof_mask]
    oof_target = y[oof_mask]
    walk_blend = np.full(n_rows, np.nan, dtype="float32")
    walk_affine = np.full(n_rows, np.nan, dtype="float32")

    for valid_year in valid_years:
        current = oof_mask & (seasons == valid_year)
        history = oof_mask & (seasons < valid_year)
        current_matrix = np.column_stack([oof[name][current] for name in SOURCE_ORDER])
        if history.any():
            history_matrix = np.column_stack([oof[name][history] for name in SOURCE_ORDER])
            weights = optimize_brier_weights(y[history], history_matrix)
            history_blend = history_matrix @ weights
            slope, intercept = fit_affine_calibrator(y[history], history_blend)
        else:
            weights = np.full(len(SOURCE_ORDER), 1.0 / len(SOURCE_ORDER))
            slope, intercept = 1.0, 0.0
        current_blend = current_matrix @ weights
        walk_blend[current] = current_blend.astype("float32")
        walk_affine[current] = np.clip(
            slope * current_blend + intercept, 1e-5, 1 - 1e-5
        ).astype("float32")
        fold_parameters.append(
            {
                "season": valid_year,
                "weights": dict(zip(SOURCE_ORDER, weights.tolist())),
                "slope": slope,
                "intercept": intercept,
            }
        )

    final_weights = optimize_brier_weights(oof_target, matrix)
    final_blend = matrix @ final_weights
    final_slope, final_intercept = fit_affine_calibrator(oof_target, final_blend)
    selection_weights = {2022: 0.2, 2023: 0.3, 2024: 0.5}
    blend_selection = sum(
        selection_weights[year]
        * brier(y[oof_mask & (seasons == year)], walk_blend[oof_mask & (seasons == year)])
        for year in valid_years
    )
    affine_selection = sum(
        selection_weights[year]
        * brier(y[oof_mask & (seasons == year)], walk_affine[oof_mask & (seasons == year)])
        for year in valid_years
    )
    calibration_method = (
        "affine" if affine_selection + 1e-7 < blend_selection else "identity"
    )
    walk_calibrated = walk_affine if calibration_method == "affine" else walk_blend
    final_calibrated = (
        np.clip(final_slope * final_blend + final_intercept, 1e-5, 1 - 1e-5)
        if calibration_method == "affine"
        else final_blend
    )

    for valid_year in valid_years:
        mask = oof_mask & (seasons == valid_year)
        baseline = float(y[seasons < valid_year].mean())
        metric_rows.extend(
            [
                _model_metrics(
                    valid_year,
                    "walk_forward_blend",
                    y[mask],
                    walk_blend[mask],
                    baseline,
                ),
                _model_metrics(
                    valid_year,
                    "walk_forward_calibrated",
                    y[mask],
                    walk_calibrated[mask],
                    baseline,
                ),
            ]
        )

    metrics = pd.DataFrame(metric_rows).drop_duplicates(["season", "model"], keep="last")
    metrics.to_csv(artifact_dir / "cv_metrics.csv", index=False, encoding="utf-8")
    correlations = pd.DataFrame(matrix, columns=SOURCE_ORDER).corr()
    correlations.to_csv(artifact_dir / "model_correlations.csv", encoding="utf-8")
    pd.DataFrame(fold_parameters).to_json(
        artifact_dir / "walk_forward_parameters.json",
        orient="records",
        indent=2,
    )

    oof_frame = metadata.loc[oof_mask].copy()
    oof_frame[TARGET_COL] = oof_target
    for index, source_name in enumerate(SOURCE_ORDER):
        oof_frame[f"pred_{source_name}"] = matrix[:, index].astype("float32")
    oof_frame["pred_blend_walk_forward"] = walk_blend[oof_mask]
    oof_frame["pred_calibrated_walk_forward"] = walk_calibrated[oof_mask]
    oof_frame["pred_blend_final_fit"] = final_blend.astype("float32")
    oof_frame["pred_calibrated_final_fit"] = final_calibrated.astype("float32")
    oof_frame.to_parquet(artifact_dir / "oof_predictions.parquet", index=False)

    final_year = int(seasons.max()) + 1
    final_iteration_map = {
        name: max(minimum_final_iterations, int(np.median(values)))
        for name, values in best_iterations.items()
        if values
    }
    importance_parts: list[pd.DataFrame] = []

    def fit_final(
        artifact_name: str,
        feature_name: str,
        train_mask: np.ndarray,
        half_life: float,
        iteration_key: str,
    ) -> None:
        parameters = model_parameters(
            final_iteration_map[iteration_key],
            depth,
            learning_rate,
            seed,
            task_type,
            devices,
            100,
        )
        model = CatBoostClassifier(**parameters)
        weights = recency_weights(seasons[train_mask], final_year, half_life)
        columns = feature_sets[feature_name]
        model.fit(
            X.loc[train_mask, columns],
            y[train_mask],
            cat_features=categorical_sets[feature_name],
            sample_weight=weights,
        )
        model.save_model(str(model_dir / artifact_name))
        importance_parts.append(
            pd.DataFrame(
                {
                    "feature": columns,
                    artifact_name.removesuffix(".cbm"): model.get_feature_importance(),
                }
            )
        )
        del model
        gc.collect()

    all_rows = np.ones(n_rows, dtype=bool)
    fit_final("global_long.cbm", "global_long", all_rows, 3.0, "global_long")
    fit_final("global_recent.cbm", "global_recent", all_rows, 1.0, "global_recent")
    fit_final(
        "global_seasonless.cbm",
        "global_seasonless",
        all_rows,
        1.0,
        "global_seasonless",
    )
    fit_final("pitcher_expert.cbm", "pitcher", all_rows, 1.5, "pitcher")
    fit_final(
        "regular_expert.cbm",
        "game_type",
        game_types == "R",
        2.5,
        "regular_expert",
    )
    fit_final(
        "futures_expert.cbm",
        "game_type",
        game_types == "F",
        1.0,
        "futures_expert",
    )

    importance = importance_parts[0]
    for part in importance_parts[1:]:
        importance = importance.merge(part, on="feature", how="outer")
    model_importance_columns = [column for column in importance if column != "feature"]
    importance[model_importance_columns] = importance[model_importance_columns].fillna(0.0)
    importance["mean_importance"] = importance[model_importance_columns].mean(axis=1)
    importance.sort_values("mean_importance", ascending=False).to_csv(
        artifact_dir / "feature_importance.csv", index=False, encoding="utf-8"
    )

    ensemble_config = {
        "version": 2,
        "source_order": SOURCE_ORDER,
        "weights": final_weights.tolist(),
        "weight_map": dict(zip(SOURCE_ORDER, final_weights.tolist())),
        "sources": [
            {
                "name": "global_long",
                "kind": "single",
                "file": "global_long.cbm",
                "feature_set": "global_long",
            },
            {
                "name": "global_recent",
                "kind": "single",
                "file": "global_recent.cbm",
                "feature_set": "global_recent",
            },
            {
                "name": "global_seasonless",
                "kind": "single",
                "file": "global_seasonless.cbm",
                "feature_set": "global_seasonless",
            },
            {
                "name": "game_type",
                "kind": "game_type_router",
                "feature_set": "game_type",
                "fallback_source": "global_recent",
                "routes": {"R": "regular_expert.cbm", "F": "futures_expert.cbm"},
            },
            {
                "name": "pitcher",
                "kind": "single",
                "file": "pitcher_expert.cbm",
                "feature_set": "pitcher",
            },
        ],
        "model_files": MODEL_FILES,
        "best_iterations": best_iterations,
        "final_iterations": final_iteration_map,
        "selection_score": float(
            affine_selection if calibration_method == "affine" else blend_selection
        ),
    }
    calibration_config = {
        "version": 2,
        "method": calibration_method,
        "slope": final_slope,
        "intercept": final_intercept,
        "trained_on_seasons": valid_years,
        "walk_forward_blend_brier": blend_selection,
        "walk_forward_affine_brier": affine_selection,
    }
    save_json(model_dir / "feature_config.json", feature_config)
    save_json(model_dir / "ensemble_config.json", ensemble_config)
    save_json(model_dir / "calibration_config.json", calibration_config)

    selected_fold_brier = {
        str(year): brier(
            y[oof_mask & (seasons == year)],
            walk_calibrated[oof_mask & (seasons == year)],
        )
        for year in valid_years
    }
    summary = {
        "version": 2,
        "fast_mode": bool(fast_mode),
        "rows": n_rows,
        "features": X.shape[1],
        "valid_years": valid_years,
        "sources": SOURCE_ORDER,
        "best_iterations": best_iterations,
        "ensemble": ensemble_config,
        "calibration": calibration_config,
        "selected_fold_brier": selected_fold_brier,
        "weighted_cv_brier": float(
            sum(selection_weights[year] * selected_fold_brier[str(year)] for year in valid_years)
        ),
        "worst_fold_brier": float(max(selected_fold_brier.values())),
        "elapsed_seconds": time.time() - started,
    }
    save_json(artifact_dir / "training_summary.json", summary)
    submission_path = run_inference(
        root,
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        output_dir=str(run_path / "output"),
    )
    print(f"V2 training complete in {time.time() - started:.1f}s")
    return {
        "summary": summary,
        "metrics": metrics,
        "correlations": correlations,
        "importance": importance.sort_values("mean_importance", ascending=False),
        "submission_path": submission_path,
    }


if __name__ == "__main__":
    run_training(Path(__file__).resolve().parent)
