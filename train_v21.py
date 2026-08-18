"""V2.1: V1 legacy global plus validated game-type and pitcher experts."""

from __future__ import annotations

import gc
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier

from result import (
    ID_COL,
    TARGET_COL,
    build_features,
    initial_feature_config,
    make_post_regime_f_prior_map,
    read_compact_csv,
    run_inference,
    save_json,
)
from train_v2 import (
    _model_metrics,
    brier,
    fit_affine_calibrator,
    recency_weights,
)


SOURCE_ORDER = ["legacy_global", "game_type", "pitcher"]
MODEL_FILES = [
    "legacy_global.cbm",
    "regular_expert.cbm",
    "futures_expert.cbm",
    "pitcher_expert.cbm",
]
SEASON_WEIGHTS = {2022: 1.0, 2023: 2.0, 2024: 4.0}
SELECTION_WEIGHTS = {2022: 0.2, 2023: 0.3, 2024: 0.5}
PRIOR_META_WEIGHTS = np.asarray([0.50, 0.30, 0.20], dtype="float64")


def project_bounded_simplex(
    values: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
) -> np.ndarray:
    values = np.asarray(values, dtype="float64")
    lower = np.zeros_like(values) if lower is None else np.asarray(lower, dtype="float64")
    upper = np.ones_like(values) if upper is None else np.asarray(upper, dtype="float64")
    if lower.sum() > 1.0 + 1e-12 or upper.sum() < 1.0 - 1e-12:
        raise ValueError("Infeasible simplex bounds")
    low = float(np.min(values - upper)) - 1.0
    high = float(np.max(values - lower)) + 1.0
    for _ in range(100):
        midpoint = (low + high) / 2.0
        projected = np.clip(values - midpoint, lower, upper)
        if projected.sum() > 1.0:
            low = midpoint
        else:
            high = midpoint
    projected = np.clip(values - (low + high) / 2.0, lower, upper)
    return projected / projected.sum()


def optimize_weighted_brier(
    y_true: np.ndarray,
    prediction_matrix: np.ndarray,
    sample_weight: np.ndarray,
    lower: np.ndarray | None = None,
    upper: np.ndarray | None = None,
    iterations: int = 4000,
) -> np.ndarray:
    matrix = np.asarray(prediction_matrix, dtype="float64")
    target = np.asarray(y_true, dtype="float64")
    weights = np.asarray(sample_weight, dtype="float64")
    weights = weights / weights.sum()
    weighted_matrix = matrix * np.sqrt(weights)[:, None]
    gram = weighted_matrix.T @ weighted_matrix
    lipschitz = max(2.0 * float(np.linalg.eigvalsh(gram).max()), 1e-8)
    step = 1.0 / lipschitz
    coefficients = project_bounded_simplex(PRIOR_META_WEIGHTS, lower, upper)
    for _ in range(iterations):
        residual = matrix @ coefficients - target
        gradient = 2.0 * matrix.T @ (weights * residual)
        updated = project_bounded_simplex(
            coefficients - step * gradient, lower, upper
        )
        if np.max(np.abs(updated - coefficients)) < 1e-11:
            coefficients = updated
            break
        coefficients = updated
    return coefficients


def fit_type_affine(
    y_true: np.ndarray,
    prediction: np.ndarray,
    game_type: np.ndarray,
) -> tuple[float, float, dict[str, float]]:
    probability = np.asarray(prediction, dtype="float64")
    target = np.asarray(y_true, dtype="float64")
    is_f = (np.asarray(game_type).astype(str) == "F").astype("float64")
    design = np.column_stack([probability, np.ones(len(probability)), is_f])
    slope, intercept, f_offset = np.linalg.lstsq(design, target, rcond=None)[0]
    return (
        float(np.clip(slope, 0.5, 1.5)),
        float(np.clip(intercept, -0.10, 0.10)),
        {"F": float(np.clip(f_offset, -0.10, 0.10)), "R": 0.0},
    )


def apply_type_affine(
    prediction: np.ndarray,
    game_type: np.ndarray,
    slope: float,
    intercept: float,
    offsets: dict[str, float],
) -> np.ndarray:
    types = np.asarray(game_type).astype(str)
    extra = np.asarray([offsets.get(value, 0.0) for value in types])
    return np.clip(slope * prediction + intercept + extra, 1e-5, 1 - 1e-5)


def make_feature_sets_v21(
    columns: list[str], categorical_columns: list[str]
) -> tuple[dict[str, list[str]], dict[str, list[str]]]:
    v21_only = {
        "historical_expanding_prior",
        "recent_global_prior",
        "recent_game_type_prior",
        "recent_f_prior",
        "pitcher_relative_historical",
        "pitcher_relative_success",
        "batter_relative_success",
        "pitcher_relative_game_type_success",
        "pitcher_relative_to_f_prior",
        "recent_success_relative",
    }
    legacy_columns = []
    for column in columns:
        if column in v21_only:
            continue
        if column.startswith("pitcher_success_shrink_"):
            continue
        if column.startswith("batter_success_shrink_"):
            continue
        legacy_columns.append(column)

    expert_columns = [
        column
        for column in columns
        if not column.startswith("legacy_") and column != "season_prior_rate"
    ]
    game_type_columns = [c for c in expert_columns if c != "game_type"]
    pitcher_exact = {
        "pitcher_id",
        "pitcher_hand",
        "pitcher_team_id",
        "asof_pitcher_n",
        "log_pitcher_n",
        "recent_global_prior",
        "recent_game_type_prior",
        "recent_f_prior",
        "recent_success_relative",
    }
    pitcher_prefixes = (
        "asof_pitcher_",
        "pitcher_",
        "pitchmix_",
        "fastball_minus_",
    )
    pitcher_columns = [
        column
        for column in expert_columns
        if column in pitcher_exact or column.startswith(pitcher_prefixes)
    ]
    feature_sets = {
        "legacy_global": legacy_columns,
        "game_type": game_type_columns,
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
    brier_early_stopping: bool,
    verbose: int = 100,
) -> dict[str, Any]:
    parameters: dict[str, Any] = {
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": float(learning_rate),
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "random_seed": int(seed),
        "loss_function": "Logloss",
        "eval_metric": "BrierScore" if brier_early_stopping else "Logloss",
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": int(verbose),
        "task_type": task_type,
    }
    if task_type == "GPU":
        parameters["devices"] = devices
    return parameters


def fit_fold(
    X: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    columns: list[str],
    categorical: list[str],
    sample_weight: np.ndarray,
    parameters: dict[str, Any],
    early_stopping: int,
) -> tuple[np.ndarray, int]:
    model = CatBoostClassifier(**parameters)
    model.fit(
        X.loc[train_mask, columns],
        y[train_mask],
        cat_features=categorical,
        sample_weight=sample_weight,
        eval_set=(X.loc[valid_mask, columns], y[valid_mask]),
        early_stopping_rounds=early_stopping,
        use_best_model=True,
    )
    prediction = np.clip(
        model.predict_proba(X.loc[valid_mask, columns])[:, 1], 1e-5, 1 - 1e-5
    ).astype("float32")
    best = max(1, int(model.get_best_iteration()) + 1)
    del model
    gc.collect()
    return prediction, best


def recent_two_iteration(values: list[int]) -> int:
    if not values:
        return 1
    return max(1, int(round(float(np.median(values[-2:])))))


def candidate_selection_score(
    prediction: np.ndarray,
    y: np.ndarray,
    seasons: np.ndarray,
    game_types: np.ndarray,
    target_type: str,
) -> float:
    score = 0.0
    for season, weight in SELECTION_WEIGHTS.items():
        mask = (seasons == season) & (game_types == target_type) & np.isfinite(prediction)
        if mask.any():
            score += weight * brier(y[mask], prediction[mask])
    return float(score)


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
    task_type = os.environ.get("BASEBALL_TASK_TYPE", "GPU").upper()
    devices = os.environ.get("BASEBALL_GPU_DEVICES", "0")
    if fast_mode is None:
        fast_mode = os.environ.get("BASEBALL_FAST_MODE", "0") == "1"
    fast_rows = int(os.environ.get("BASEBALL_FAST_ROWS_PER_SEASON", "30000"))

    train = read_compact_csv(data_dir / "train.csv", train=True)
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
    feature_config["version"] = 3
    feature_config["post_regime_f_prior_map"] = make_post_regime_f_prior_map(train)
    X = build_features(train, feature_config)
    feature_config["feature_columns"] = list(X.columns)
    feature_sets, categorical_sets = make_feature_sets_v21(
        list(X.columns), list(feature_config["categorical_columns"])
    )
    feature_config["model_feature_columns"] = feature_sets
    feature_config["model_categorical_columns"] = categorical_sets

    y = train[TARGET_COL].to_numpy(dtype="int8")
    seasons = train["season"].to_numpy(dtype="int16")
    game_types = train["game_type"].astype(str).to_numpy()
    n_rows = len(train)
    metadata = train[
        [ID_COL, "season", "game_type", "pitcher_id", "asof_pitcher_n", "asof_batter_n"]
    ].copy()
    candidate_names = [
        "legacy_global",
        "pitcher",
        "regular_hl1_5",
        "regular_hl2_5",
        "regular_uniform",
        "futures_hl1",
        "futures_post2023",
        "futures_weak_old",
    ]
    candidate_oof = {
        name: np.full(n_rows, np.nan, dtype="float32") for name in candidate_names
    }
    best_iterations = {name: [] for name in candidate_names}
    candidate_metrics: list[dict[str, Any]] = []

    legacy_parameters = model_parameters(
        iterations, depth, learning_rate, seed, task_type, devices, True
    )
    expert_parameters = model_parameters(
        iterations, depth, learning_rate, seed, task_type, devices, False
    )

    for valid_year in valid_years:
        fold_started = time.time()
        train_mask = seasons < valid_year
        valid_mask = seasons == valid_year
        baseline = float(y[train_mask].mean())
        legacy_prediction, legacy_best = fit_fold(
            X,
            y,
            train_mask,
            valid_mask,
            feature_sets["legacy_global"],
            categorical_sets["legacy_global"],
            recency_weights(seasons[train_mask], valid_year, 3.0),
            legacy_parameters,
            early_stopping,
        )
        candidate_oof["legacy_global"][valid_mask] = legacy_prediction
        best_iterations["legacy_global"].append(legacy_best)

        pitcher_prediction, pitcher_best = fit_fold(
            X,
            y,
            train_mask,
            valid_mask,
            feature_sets["pitcher"],
            categorical_sets["pitcher"],
            recency_weights(seasons[train_mask], valid_year, 1.5),
            expert_parameters,
            early_stopping,
        )
        candidate_oof["pitcher"][valid_mask] = pitcher_prediction
        best_iterations["pitcher"].append(pitcher_best)

        for game_type, candidates in {
            "R": ["regular_hl1_5", "regular_hl2_5", "regular_uniform"],
            "F": ["futures_hl1", "futures_post2023", "futures_weak_old"],
        }.items():
            typed_valid = valid_mask & (game_types == game_type)
            base_typed_train = train_mask & (game_types == game_type)
            for candidate in candidates:
                typed_train = base_typed_train.copy()
                if candidate == "futures_post2023":
                    post_mask = typed_train & (seasons >= 2023)
                    if post_mask.any():
                        typed_train = post_mask
                if candidate == "regular_hl1_5":
                    sample_weight = recency_weights(seasons[typed_train], valid_year, 1.5)
                elif candidate == "regular_hl2_5":
                    sample_weight = recency_weights(seasons[typed_train], valid_year, 2.5)
                elif candidate == "regular_uniform":
                    sample_weight = np.ones(typed_train.sum(), dtype="float32")
                elif candidate in {"futures_hl1", "futures_post2023"}:
                    sample_weight = recency_weights(seasons[typed_train], valid_year, 1.0)
                else:
                    sample_weight = np.where(
                        seasons[typed_train] >= 2023, 1.0, 0.05
                    ).astype("float32")
                    sample_weight /= sample_weight.mean()
                prediction, best = fit_fold(
                    X,
                    y,
                    typed_train,
                    typed_valid,
                    feature_sets["game_type"],
                    categorical_sets["game_type"],
                    sample_weight,
                    expert_parameters,
                    early_stopping,
                )
                candidate_oof[candidate][typed_valid] = prediction
                best_iterations[candidate].append(best)

        for candidate in candidate_names:
            mask = valid_mask & np.isfinite(candidate_oof[candidate])
            if mask.any():
                candidate_metrics.append(
                    _model_metrics(
                        valid_year,
                        candidate,
                        y[mask],
                        candidate_oof[candidate][mask],
                        baseline,
                        best_iterations[candidate][-1],
                    )
                )
        print(
            f"V2.1 fold {valid_year} complete | elapsed={time.time() - fold_started:.1f}s"
        )

    regular_candidates = ["regular_hl1_5", "regular_hl2_5", "regular_uniform"]
    futures_candidates = ["futures_hl1", "futures_post2023", "futures_weak_old"]
    regular_scores = {
        name: candidate_selection_score(candidate_oof[name], y, seasons, game_types, "R")
        for name in regular_candidates
    }
    futures_scores = {
        name: candidate_selection_score(candidate_oof[name], y, seasons, game_types, "F")
        for name in futures_candidates
    }
    selected_regular = min(regular_scores, key=regular_scores.get)
    selected_futures = min(futures_scores, key=futures_scores.get)

    source_oof = {
        "legacy_global": candidate_oof["legacy_global"],
        "pitcher": candidate_oof["pitcher"],
        "game_type": candidate_oof["legacy_global"].copy(),
    }
    source_oof["game_type"][game_types == "R"] = candidate_oof[selected_regular][
        game_types == "R"
    ]
    source_oof["game_type"][game_types == "F"] = candidate_oof[selected_futures][
        game_types == "F"
    ]
    oof_mask = np.logical_and.reduce(
        [np.isfinite(source_oof[name]) for name in SOURCE_ORDER]
    )

    walk_blend = np.full(n_rows, np.nan, dtype="float32")
    walk_affine = np.full(n_rows, np.nan, dtype="float32")
    walk_type_affine = np.full(n_rows, np.nan, dtype="float32")
    fold_parameters: list[dict[str, Any]] = []
    for valid_year in valid_years:
        current = oof_mask & (seasons == valid_year)
        history = oof_mask & (seasons < valid_year)
        current_matrix = np.column_stack(
            [source_oof[name][current] for name in SOURCE_ORDER]
        )
        current_types = game_types[current]
        current_prediction = np.empty(current.sum(), dtype="float64")
        type_weights: dict[str, list[float]] = {}
        for game_type in ["R", "F"]:
            current_type_mask = current_types == game_type
            history_type = history & (game_types == game_type)
            if history_type.any():
                history_matrix = np.column_stack(
                    [source_oof[name][history_type] for name in SOURCE_ORDER]
                )
                sample_weight = np.asarray(
                    [SEASON_WEIGHTS[int(year)] for year in seasons[history_type]],
                    dtype="float64",
                )
                lower = np.asarray([0.20, 0.0, 0.0]) if game_type == "F" else None
                upper = np.asarray([1.0, 0.70, 0.30]) if game_type == "F" else None
                coefficients = optimize_weighted_brier(
                    y[history_type], history_matrix, sample_weight, lower, upper
                )
            else:
                coefficients = PRIOR_META_WEIGHTS.copy()
            current_prediction[current_type_mask] = (
                current_matrix[current_type_mask] @ coefficients
            )
            type_weights[game_type] = coefficients.tolist()
        walk_blend[current] = current_prediction.astype("float32")

        if history.any():
            history_matrix_all = np.column_stack(
                [source_oof[name][history] for name in SOURCE_ORDER]
            )
            history_prediction = np.empty(history.sum(), dtype="float64")
            history_types = game_types[history]
            for game_type in ["R", "F"]:
                mask = history_types == game_type
                history_prediction[mask] = (
                    history_matrix_all[mask]
                    @ np.asarray(type_weights[game_type], dtype="float64")
                )
            slope, intercept = fit_affine_calibrator(y[history], history_prediction)
            type_slope, type_intercept, offsets = fit_type_affine(
                y[history], history_prediction, history_types
            )
        else:
            slope, intercept = 1.0, 0.0
            type_slope, type_intercept, offsets = 1.0, 0.0, {"R": 0.0, "F": 0.0}
        walk_affine[current] = np.clip(
            slope * current_prediction + intercept, 1e-5, 1 - 1e-5
        )
        walk_type_affine[current] = apply_type_affine(
            current_prediction, current_types, type_slope, type_intercept, offsets
        )
        fold_parameters.append(
            {
                "season": valid_year,
                "weights_by_game_type": type_weights,
                "affine": {"slope": slope, "intercept": intercept},
                "type_affine": {
                    "slope": type_slope,
                    "intercept": type_intercept,
                    "game_type_offsets": offsets,
                },
            }
        )

    final_weights_by_type: dict[str, list[float]] = {}
    final_blend = np.full(oof_mask.sum(), np.nan, dtype="float64")
    oof_indices = np.flatnonzero(oof_mask)
    oof_types = game_types[oof_mask]
    full_matrix = np.column_stack(
        [source_oof[name][oof_mask] for name in SOURCE_ORDER]
    )
    for game_type in ["R", "F"]:
        full_type = oof_mask & (game_types == game_type)
        matrix_type = np.column_stack(
            [source_oof[name][full_type] for name in SOURCE_ORDER]
        )
        sample_weight = np.asarray(
            [SEASON_WEIGHTS[int(year)] for year in seasons[full_type]], dtype="float64"
        )
        lower = np.asarray([0.20, 0.0, 0.0]) if game_type == "F" else None
        upper = np.asarray([1.0, 0.70, 0.30]) if game_type == "F" else None
        coefficients = optimize_weighted_brier(
            y[full_type], matrix_type, sample_weight, lower, upper
        )
        final_weights_by_type[game_type] = coefficients.tolist()
        final_blend[oof_types == game_type] = matrix_type @ coefficients

    final_slope, final_intercept = fit_affine_calibrator(y[oof_mask], final_blend)
    type_slope, type_intercept, type_offsets = fit_type_affine(
        y[oof_mask], final_blend, oof_types
    )
    calibration_candidates = {
        "identity": walk_blend,
        "affine": walk_affine,
        "type_affine": walk_type_affine,
    }
    calibration_scores = {
        name: float(
            sum(
                SELECTION_WEIGHTS[year]
                * brier(
                    y[oof_mask & (seasons == year)],
                    prediction[oof_mask & (seasons == year)],
                )
                for year in valid_years
            )
        )
        for name, prediction in calibration_candidates.items()
    }
    calibration_method = min(calibration_scores, key=calibration_scores.get)
    walk_calibrated = calibration_candidates[calibration_method]
    if calibration_method == "affine":
        final_calibrated = np.clip(
            final_slope * final_blend + final_intercept, 1e-5, 1 - 1e-5
        )
    elif calibration_method == "type_affine":
        final_calibrated = apply_type_affine(
            final_blend, oof_types, type_slope, type_intercept, type_offsets
        )
    else:
        final_calibrated = final_blend.copy()

    metric_rows = list(candidate_metrics)
    for valid_year in valid_years:
        mask = oof_mask & (seasons == valid_year)
        baseline = float(y[seasons < valid_year].mean())
        for source_name in SOURCE_ORDER:
            metric_rows.append(
                _model_metrics(
                    valid_year,
                    source_name,
                    y[mask],
                    source_oof[source_name][mask],
                    baseline,
                )
            )
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
    metrics = pd.DataFrame(metric_rows).drop_duplicates(
        ["season", "model"], keep="last"
    )
    metrics.to_csv(artifact_dir / "cv_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(candidate_metrics).to_csv(
        artifact_dir / "experiment_log.csv", index=False, encoding="utf-8"
    )
    pd.DataFrame(full_matrix, columns=SOURCE_ORDER).corr().to_csv(
        artifact_dir / "model_correlations.csv", encoding="utf-8"
    )
    save_json(
        artifact_dir / "expert_selection.json",
        {
            "regular_scores": regular_scores,
            "futures_scores": futures_scores,
            "selected_regular": selected_regular,
            "selected_futures": selected_futures,
        },
    )
    save_json(artifact_dir / "walk_forward_parameters.json", {"folds": fold_parameters})

    oof_frame = metadata.loc[oof_mask].copy()
    oof_frame[TARGET_COL] = y[oof_mask]
    for source_name in SOURCE_ORDER:
        oof_frame[f"pred_{source_name}"] = source_oof[source_name][oof_mask]
    oof_frame["pred_blend_walk_forward"] = walk_blend[oof_mask]
    oof_frame["pred_calibrated_walk_forward"] = walk_calibrated[oof_mask]
    oof_frame["pred_blend_final_fit"] = final_blend.astype("float32")
    oof_frame["pred_calibrated_final_fit"] = final_calibrated.astype("float32")
    oof_frame.to_parquet(artifact_dir / "oof_predictions.parquet", index=False)

    final_iterations = {
        "legacy_global": recent_two_iteration(best_iterations["legacy_global"]),
        "pitcher": recent_two_iteration(best_iterations["pitcher"]),
        "regular_expert": recent_two_iteration(best_iterations[selected_regular]),
        "futures_expert": recent_two_iteration(best_iterations[selected_futures]),
    }
    final_year = int(seasons.max()) + 1
    importance_parts: list[pd.DataFrame] = []

    def fit_final(
        filename: str,
        feature_name: str,
        mask: np.ndarray,
        sample_weight: np.ndarray,
        iteration_key: str,
        brier_early_stopping: bool,
    ) -> None:
        model = CatBoostClassifier(
            **model_parameters(
                final_iterations[iteration_key],
                depth,
                learning_rate,
                seed,
                task_type,
                devices,
                brier_early_stopping,
            )
        )
        columns = feature_sets[feature_name]
        model.fit(
            X.loc[mask, columns],
            y[mask],
            cat_features=categorical_sets[feature_name],
            sample_weight=sample_weight,
        )
        model.save_model(str(model_dir / filename))
        importance_parts.append(
            pd.DataFrame(
                {
                    "feature": columns,
                    filename.removesuffix(".cbm"): model.get_feature_importance(),
                }
            )
        )
        del model
        gc.collect()

    all_rows = np.ones(n_rows, dtype=bool)
    fit_final(
        "legacy_global.cbm",
        "legacy_global",
        all_rows,
        recency_weights(seasons, final_year, 3.0),
        "legacy_global",
        True,
    )
    fit_final(
        "pitcher_expert.cbm",
        "pitcher",
        all_rows,
        recency_weights(seasons, final_year, 1.5),
        "pitcher",
        False,
    )
    regular_mask = game_types == "R"
    if selected_regular == "regular_hl1_5":
        regular_weight = recency_weights(seasons[regular_mask], final_year, 1.5)
    elif selected_regular == "regular_hl2_5":
        regular_weight = recency_weights(seasons[regular_mask], final_year, 2.5)
    else:
        regular_weight = np.ones(regular_mask.sum(), dtype="float32")
    fit_final(
        "regular_expert.cbm",
        "game_type",
        regular_mask,
        regular_weight,
        "regular_expert",
        False,
    )
    futures_mask = game_types == "F"
    if selected_futures == "futures_post2023":
        futures_mask &= seasons >= 2023
        futures_weight = recency_weights(seasons[futures_mask], final_year, 1.0)
    elif selected_futures == "futures_weak_old":
        futures_weight = np.where(
            seasons[futures_mask] >= 2023, 1.0, 0.05
        ).astype("float32")
        futures_weight /= futures_weight.mean()
    else:
        futures_weight = recency_weights(seasons[futures_mask], final_year, 1.0)
    fit_final(
        "futures_expert.cbm",
        "game_type",
        futures_mask,
        futures_weight,
        "futures_expert",
        False,
    )

    importance = importance_parts[0]
    for part in importance_parts[1:]:
        importance = importance.merge(part, on="feature", how="outer")
    importance_columns = [column for column in importance if column != "feature"]
    importance[importance_columns] = importance[importance_columns].fillna(0.0)
    importance["mean_importance"] = importance[importance_columns].mean(axis=1)
    importance.sort_values("mean_importance", ascending=False).to_csv(
        artifact_dir / "feature_importance.csv", index=False, encoding="utf-8"
    )

    default_weights = PRIOR_META_WEIGHTS.tolist()
    ensemble_config = {
        "version": 3,
        "source_order": SOURCE_ORDER,
        "weights": default_weights,
        "weights_by_game_type": final_weights_by_type,
        "weight_map_by_game_type": {
            game_type: dict(zip(SOURCE_ORDER, values))
            for game_type, values in final_weights_by_type.items()
        },
        "sources": [
            {
                "name": "legacy_global",
                "kind": "single",
                "file": "legacy_global.cbm",
                "feature_set": "legacy_global",
            },
            {
                "name": "game_type",
                "kind": "game_type_router",
                "feature_set": "game_type",
                "fallback_source": "legacy_global",
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
        "selected_regular": selected_regular,
        "selected_futures": selected_futures,
        "best_iterations": best_iterations,
        "final_iterations": final_iterations,
    }
    calibration_config: dict[str, Any] = {
        "version": 3,
        "method": calibration_method,
        "selection_scores": calibration_scores,
        "trained_on_seasons": valid_years,
    }
    if calibration_method == "affine":
        calibration_config.update(slope=final_slope, intercept=final_intercept)
    elif calibration_method == "type_affine":
        calibration_config.update(
            slope=type_slope,
            intercept=type_intercept,
            game_type_offsets=type_offsets,
        )
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
    f_2024 = oof_mask & (seasons == 2024) & (game_types == "F")
    summary = {
        "version": "2.1",
        "fast_mode": bool(fast_mode),
        "rows": n_rows,
        "features": X.shape[1],
        "selected_regular": selected_regular,
        "selected_futures": selected_futures,
        "selected_fold_brier": selected_fold_brier,
        "weighted_cv_brier": float(
            sum(SELECTION_WEIGHTS[year] * selected_fold_brier[str(year)] for year in valid_years)
        ),
        "worst_fold_brier": float(max(selected_fold_brier.values())),
        "brier_2024_f": brier(y[f_2024], walk_calibrated[f_2024]),
        "mean_gap_2024_f": float(
            walk_calibrated[f_2024].mean() - y[f_2024].mean()
        ),
        "ensemble": ensemble_config,
        "calibration": calibration_config,
        "elapsed_seconds": time.time() - started,
    }
    save_json(artifact_dir / "training_summary.json", summary)
    submission_path = run_inference(
        root,
        data_dir=str(data_dir),
        model_dir=str(model_dir),
        output_dir=str(run_path / "output"),
    )
    print(f"V2.1 training complete in {time.time() - started:.1f}s")
    return {
        "summary": summary,
        "metrics": metrics,
        "correlations": pd.DataFrame(full_matrix, columns=SOURCE_ORDER).corr(),
        "importance": importance.sort_values("mean_importance", ascending=False),
        "submission_path": submission_path,
    }


if __name__ == "__main__":
    run_training(Path(__file__).resolve().parent)
