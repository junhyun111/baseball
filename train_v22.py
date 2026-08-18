"""V2.2 conservative three-model CatBoost training pipeline.

The production ensemble intentionally contains only the V1-compatible global
model and the fixed Regular/Futures experts.  Pitcher experiments remain in
``train_v21.py`` but are excluded from training, blending, inference, and the
submission package here.
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
from train_v2 import _model_metrics, brier, fit_affine_calibrator, recency_weights
from train_v21 import fit_fold, make_feature_sets_v21, model_parameters, recent_two_iteration


SOURCE_ORDER = ["legacy_global", "game_type"]
MODEL_FILES = ["legacy_global.cbm", "regular_expert.cbm", "futures_expert.cbm"]
VALID_YEARS = [2022, 2023, 2024]
SELECTION_WEIGHTS = {2022: 0.2, 2023: 0.3, 2024: 0.5}
AGGRESSIVE_WEIGHTS = {2022: 0.1, 2023: 0.3, 2024: 0.6}
R_WEIGHT_GRID = [0.0, 0.10, 0.20, 0.25, 0.30, 0.35, 0.40]
F_WEIGHT_GRID = [0.20, 0.25, 0.30, 0.35, 0.40]
LEGACY_ITERATION_GRID = [40, 60, 80, 100, 120]
FUTURES_ITERATION_GRID = [70, 100, 140]
ACCEPTANCE_LIMITS = {
    "weighted_cv_brier": 0.24764,
    "brier_2024": 0.247998,
    "brier_2024_f": 0.24725,
    "brier_2024_r": 0.24810,
}
V1_2024_R_REFERENCE = 0.248099


def weighted_season_brier(
    target: np.ndarray,
    prediction: np.ndarray,
    seasons: np.ndarray,
    mask: np.ndarray,
    weights: dict[int, float] = SELECTION_WEIGHTS,
) -> float:
    return float(
        sum(
            weight * brier(target[mask & (seasons == year)], prediction[mask & (seasons == year)])
            for year, weight in weights.items()
            if (mask & (seasons == year)).any()
        )
    )


def fit_fixed_iteration(
    X: pd.DataFrame,
    y: np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    columns: list[str],
    categorical: list[str],
    sample_weight: np.ndarray,
    parameters: dict[str, Any],
) -> np.ndarray:
    """Fit an exact tree count for the one-axis iteration sensitivity tests."""
    model = CatBoostClassifier(**parameters)
    model.fit(
        X.loc[train_mask, columns],
        y[train_mask],
        cat_features=categorical,
        sample_weight=sample_weight,
    )
    prediction = np.clip(
        model.predict_proba(X.loc[valid_mask, columns])[:, 1], 1e-5, 1 - 1e-5
    ).astype("float32")
    del model
    gc.collect()
    return prediction


def blend_prediction(
    legacy: np.ndarray,
    expert: np.ndarray,
    expert_weight: float,
) -> np.ndarray:
    return ((1.0 - expert_weight) * legacy + expert_weight * expert).astype("float32")


def grid_rows_for_type(
    target: np.ndarray,
    seasons: np.ndarray,
    game_types: np.ndarray,
    legacy: np.ndarray,
    expert: np.ndarray,
    game_type: str,
    grid: list[float],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    type_mask = game_types == game_type
    baseline_2024 = brier(
        target[type_mask & (seasons == 2024)], legacy[type_mask & (seasons == 2024)]
    )
    for weight in grid:
        prediction = blend_prediction(legacy, expert, weight)
        row: dict[str, Any] = {
            "game_type": game_type,
            "expert_weight": weight,
            "weighted_brier": weighted_season_brier(
                target, prediction, seasons, type_mask, SELECTION_WEIGHTS
            ),
            "aggressive_weighted_brier": weighted_season_brier(
                target, prediction, seasons, type_mask, AGGRESSIVE_WEIGHTS
            ),
            "legacy_brier_2024": baseline_2024,
            "v1_reference_brier_2024_r": V1_2024_R_REFERENCE,
        }
        for year in VALID_YEARS:
            current = type_mask & (seasons == year)
            row[f"brier_{year}"] = brier(target[current], prediction[current])
        row["protects_legacy_anchor"] = bool(
            game_type != "R" or row["brier_2024"] <= baseline_2024 + 1e-12
        )
        row["protects_2024_r"] = bool(
            game_type != "R" or row["brier_2024"] <= V1_2024_R_REFERENCE + 1e-12
        )
        rows.append(row)
    return rows


def select_grid_weight(rows: list[dict[str, Any]], game_type: str) -> float:
    eligible = rows
    if game_type == "R":
        eligible = [row for row in rows if row["protects_2024_r"]]
    if not eligible:
        # No expert correction satisfies the V1 guard; keep the anchor itself.
        return float(min(rows, key=lambda row: row["expert_weight"])["expert_weight"])
    return float(min(eligible, key=lambda row: (row["weighted_brier"], row["expert_weight"]))["expert_weight"])


def calibration_statistics(
    target: np.ndarray,
    prediction: np.ndarray,
    seasons: np.ndarray,
    game_types: np.ndarray,
    mask: np.ndarray,
) -> dict[str, float]:
    current_2024 = mask & (seasons == 2024)
    regular_2024 = current_2024 & (game_types == "R")
    futures_2024 = current_2024 & (game_types == "F")
    return {
        "weighted_cv_brier": weighted_season_brier(target, prediction, seasons, mask),
        "brier_2024": brier(target[current_2024], prediction[current_2024]),
        "brier_2024_r": brier(target[regular_2024], prediction[regular_2024]),
        "brier_2024_f": brier(target[futures_2024], prediction[futures_2024]),
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
    iterations = int(os.environ.get("BASEBALL_ITERATIONS", "1200"))
    depth = int(os.environ.get("BASEBALL_DEPTH", "8"))
    learning_rate = float(os.environ.get("BASEBALL_LEARNING_RATE", "0.04"))
    early_stopping = int(os.environ.get("BASEBALL_EARLY_STOPPING", "120"))
    task_type = os.environ.get("BASEBALL_TASK_TYPE", "GPU").upper()
    devices = os.environ.get("BASEBALL_GPU_DEVICES", "0")
    if fast_mode is None:
        fast_mode = os.environ.get("BASEBALL_FAST_MODE", "0") == "1"
    fast_rows = int(os.environ.get("BASEBALL_FAST_ROWS_PER_SEASON", "30000"))
    verbose = 0 if fast_mode else 100

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
    feature_config["version"] = 4
    feature_config["post_regime_f_prior_map"] = make_post_regime_f_prior_map(train)
    X = build_features(train, feature_config)
    feature_config["feature_columns"] = list(X.columns)
    all_feature_sets, all_categorical_sets = make_feature_sets_v21(
        list(X.columns), list(feature_config["categorical_columns"])
    )
    feature_sets = {name: all_feature_sets[name] for name in SOURCE_ORDER}
    categorical_sets = {name: all_categorical_sets[name] for name in SOURCE_ORDER}
    feature_config["model_feature_columns"] = feature_sets
    feature_config["model_categorical_columns"] = categorical_sets

    y = train[TARGET_COL].to_numpy(dtype="int8")
    seasons = train["season"].to_numpy(dtype="int16")
    game_types = train["game_type"].astype(str).to_numpy()
    n_rows = len(train)
    metadata = train[
        [ID_COL, "season", "game_type", "pitcher_id", "asof_pitcher_n", "asof_batter_n"]
    ].copy()
    oof = {
        "legacy_global": np.full(n_rows, np.nan, dtype="float32"),
        "regular_expert": np.full(n_rows, np.nan, dtype="float32"),
        "futures_expert": np.full(n_rows, np.nan, dtype="float32"),
    }
    best_iterations = {name: [] for name in oof}
    metric_rows: list[dict[str, Any]] = []
    sensitivity_rows: list[dict[str, Any]] = []

    legacy_parameters = model_parameters(
        iterations, depth, learning_rate, seed, task_type, devices, True, verbose
    )
    expert_parameters = model_parameters(
        iterations, depth, learning_rate, seed, task_type, devices, False, verbose
    )

    for valid_year in VALID_YEARS:
        fold_started = time.time()
        train_mask = seasons < valid_year
        valid_mask = seasons == valid_year
        baseline = float(y[train_mask].mean())

        prediction, best = fit_fold(
            X, y, train_mask, valid_mask,
            feature_sets["legacy_global"], categorical_sets["legacy_global"],
            recency_weights(seasons[train_mask], valid_year, 3.0),
            legacy_parameters, early_stopping,
        )
        oof["legacy_global"][valid_mask] = prediction
        best_iterations["legacy_global"].append(best)

        regular_train = train_mask & (game_types == "R")
        regular_valid = valid_mask & (game_types == "R")
        prediction, best = fit_fold(
            X, y, regular_train, regular_valid,
            feature_sets["game_type"], categorical_sets["game_type"],
            recency_weights(seasons[regular_train], valid_year, 1.5),
            expert_parameters, early_stopping,
        )
        oof["regular_expert"][regular_valid] = prediction
        best_iterations["regular_expert"].append(best)

        futures_train = train_mask & (game_types == "F")
        post_regime_train = futures_train & (seasons >= 2023)
        if post_regime_train.any():
            futures_train = post_regime_train
        futures_valid = valid_mask & (game_types == "F")
        prediction, best = fit_fold(
            X, y, futures_train, futures_valid,
            feature_sets["game_type"], categorical_sets["game_type"],
            recency_weights(seasons[futures_train], valid_year, 1.0),
            expert_parameters, early_stopping,
        )
        oof["futures_expert"][futures_valid] = prediction
        best_iterations["futures_expert"].append(best)

        for name, type_name, iteration_value in [
            ("legacy_global", None, best_iterations["legacy_global"][-1]),
            ("regular_expert", "R", best_iterations["regular_expert"][-1]),
            ("futures_expert", "F", best_iterations["futures_expert"][-1]),
        ]:
            current = valid_mask if type_name is None else valid_mask & (game_types == type_name)
            metric_rows.append(
                _model_metrics(
                    valid_year, name, y[current], oof[name][current], baseline, iteration_value
                )
            )

        # E224: change only the Legacy final iteration axis on 2023/2024.
        if valid_year in {2023, 2024}:
            for fixed_iterations in LEGACY_ITERATION_GRID:
                fixed = fit_fixed_iteration(
                    X, y, train_mask, valid_mask,
                    feature_sets["legacy_global"], categorical_sets["legacy_global"],
                    recency_weights(seasons[train_mask], valid_year, 3.0),
                    model_parameters(
                        fixed_iterations, depth, learning_rate, seed, task_type, devices, False, verbose
                    ),
                )
                sensitivity_rows.append(
                    {"family": "legacy_global", "season": valid_year,
                     "iterations": fixed_iterations, "brier": brier(y[valid_mask], fixed)}
                )

        # E226: evaluate Futures final iteration only on the current regime fold.
        if valid_year == 2024:
            for fixed_iterations in FUTURES_ITERATION_GRID:
                fixed = fit_fixed_iteration(
                    X, y, futures_train, futures_valid,
                    feature_sets["game_type"], categorical_sets["game_type"],
                    recency_weights(seasons[futures_train], valid_year, 1.0),
                    model_parameters(
                        fixed_iterations, depth, learning_rate, seed, task_type, devices, False, verbose
                    ),
                )
                sensitivity_rows.append(
                    {"family": "futures_expert", "season": valid_year,
                     "iterations": fixed_iterations, "brier": brier(y[futures_valid], fixed)}
                )
        print(f"V2.2 fold {valid_year} complete | elapsed={time.time() - fold_started:.1f}s")

    expert = oof["legacy_global"].copy()
    expert[game_types == "R"] = oof["regular_expert"][game_types == "R"]
    expert[game_types == "F"] = oof["futures_expert"][game_types == "F"]
    oof_mask = np.isfinite(oof["legacy_global"]) & np.isfinite(expert)

    grid_rows = grid_rows_for_type(
        y, seasons, game_types, oof["legacy_global"], expert, "R", R_WEIGHT_GRID
    ) + grid_rows_for_type(
        y, seasons, game_types, oof["legacy_global"], expert, "F", F_WEIGHT_GRID
    )
    regular_weight = select_grid_weight(
        [row for row in grid_rows if row["game_type"] == "R"], "R"
    )
    futures_weight = select_grid_weight(
        [row for row in grid_rows if row["game_type"] == "F"], "F"
    )

    blend = np.full(n_rows, np.nan, dtype="float32")
    for game_type, weight in [("R", regular_weight), ("F", futures_weight)]:
        current = oof_mask & (game_types == game_type)
        blend[current] = blend_prediction(oof["legacy_global"][current], expert[current], weight)

    # E227: only identity and a global affine calibrator are eligible.
    walk_affine = np.full(n_rows, np.nan, dtype="float32")
    calibration_folds: list[dict[str, float]] = []
    for valid_year in VALID_YEARS:
        current = oof_mask & (seasons == valid_year)
        history = oof_mask & (seasons < valid_year)
        if history.any():
            slope, intercept = fit_affine_calibrator(y[history], blend[history])
        else:
            slope, intercept = 1.0, 0.0
        walk_affine[current] = np.clip(
            slope * blend[current] + intercept, 1e-5, 1 - 1e-5
        ).astype("float32")
        calibration_folds.append(
            {"season": valid_year, "slope": slope, "intercept": intercept}
        )

    calibration_candidates = {"identity": blend, "affine": walk_affine}
    calibration_stats = {
        name: calibration_statistics(y, pred, seasons, game_types, oof_mask)
        for name, pred in calibration_candidates.items()
    }
    identity_stats = calibration_stats["identity"]
    conservative_methods = [
        name for name, stats in calibration_stats.items()
        if stats["brier_2024"] <= identity_stats["brier_2024"] + 1e-12
        and stats["brier_2024_r"] <= identity_stats["brier_2024_r"] + 1e-12
        and stats["brier_2024_f"] <= identity_stats["brier_2024_f"] + 1e-12
    ]
    calibration_method = min(
        conservative_methods or ["identity"],
        key=lambda name: calibration_stats[name]["weighted_cv_brier"],
    )
    calibrated = calibration_candidates[calibration_method]
    final_slope, final_intercept = fit_affine_calibrator(y[oof_mask], blend[oof_mask])

    for valid_year in VALID_YEARS:
        current = oof_mask & (seasons == valid_year)
        baseline = float(y[seasons < valid_year].mean())
        for name, prediction in [
            ("game_type", expert),
            ("blend", blend),
            ("calibrated", calibrated),
        ]:
            metric_rows.append(_model_metrics(valid_year, name, y[current], prediction[current], baseline))

    metrics = pd.DataFrame(metric_rows)
    metrics.to_csv(artifact_dir / "cv_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(grid_rows).to_csv(artifact_dir / "blend_grid.csv", index=False, encoding="utf-8")
    sensitivity = pd.DataFrame(sensitivity_rows)
    sensitivity.to_csv(artifact_dir / "iteration_sensitivity.csv", index=False, encoding="utf-8")

    oof_frame = metadata.loc[oof_mask].copy()
    oof_frame[TARGET_COL] = y[oof_mask]
    oof_frame["pred_legacy_global"] = oof["legacy_global"][oof_mask]
    oof_frame["pred_regular_expert"] = oof["regular_expert"][oof_mask]
    oof_frame["pred_futures_expert"] = oof["futures_expert"][oof_mask]
    oof_frame["pred_game_type"] = expert[oof_mask]
    oof_frame["pred_blend_walk_forward"] = blend[oof_mask]
    oof_frame["pred_calibrated_walk_forward"] = calibrated[oof_mask]
    final_fit_calibrated = blend[oof_mask].astype("float64")
    if calibration_method == "affine":
        final_fit_calibrated = np.clip(
            final_slope * final_fit_calibrated + final_intercept, 1e-5, 1 - 1e-5
        )
    oof_frame["pred_blend_final_fit"] = blend[oof_mask]
    oof_frame["pred_calibrated_final_fit"] = final_fit_calibrated.astype("float32")
    oof_frame.to_parquet(artifact_dir / "oof_predictions.parquet", index=False)

    legacy_sensitivity = sensitivity[sensitivity["family"] == "legacy_global"]
    legacy_scores = (
        legacy_sensitivity.assign(
            weighted=legacy_sensitivity.apply(
                lambda row: SELECTION_WEIGHTS[int(row["season"])] * row["brier"], axis=1
            )
        ).groupby("iterations", as_index=False)["weighted"].sum()
    )
    legacy_final_iterations = int(
        legacy_scores.sort_values(["weighted", "iterations"]).iloc[0]["iterations"]
    )
    futures_sensitivity = sensitivity[sensitivity["family"] == "futures_expert"]
    futures_final_iterations = int(
        futures_sensitivity.sort_values(["brier", "iterations"]).iloc[0]["iterations"]
    )
    final_iterations = {
        "legacy_global": legacy_final_iterations,
        "regular_expert": recent_two_iteration(best_iterations["regular_expert"]),
        "futures_expert": futures_final_iterations,
    }

    importance_parts: list[pd.DataFrame] = []
    final_year = int(seasons.max()) + 1

    def fit_final(
        filename: str,
        feature_name: str,
        mask: np.ndarray,
        sample_weight: np.ndarray,
        iteration_key: str,
    ) -> None:
        model = CatBoostClassifier(
            **model_parameters(
                final_iterations[iteration_key], depth, learning_rate, seed,
                task_type, devices, False, verbose,
            )
        )
        columns = feature_sets[feature_name]
        model.fit(
            X.loc[mask, columns], y[mask],
            cat_features=categorical_sets[feature_name], sample_weight=sample_weight,
        )
        model.save_model(str(model_dir / filename))
        importance_parts.append(
            pd.DataFrame({"feature": columns, filename.removesuffix(".cbm"): model.get_feature_importance()})
        )
        del model
        gc.collect()

    all_rows = np.ones(n_rows, dtype=bool)
    fit_final(
        "legacy_global.cbm", "legacy_global", all_rows,
        recency_weights(seasons, final_year, 3.0), "legacy_global",
    )
    regular_mask = game_types == "R"
    fit_final(
        "regular_expert.cbm", "game_type", regular_mask,
        recency_weights(seasons[regular_mask], final_year, 1.5), "regular_expert",
    )
    futures_mask = (game_types == "F") & (seasons >= 2023)
    fit_final(
        "futures_expert.cbm", "game_type", futures_mask,
        recency_weights(seasons[futures_mask], final_year, 1.0), "futures_expert",
    )

    importance = importance_parts[0]
    for part in importance_parts[1:]:
        importance = importance.merge(part, on="feature", how="outer")
    importance_columns = [column for column in importance if column != "feature"]
    importance[importance_columns] = importance[importance_columns].fillna(0.0)
    importance["mean_importance"] = importance[importance_columns].mean(axis=1)
    importance = importance.sort_values("mean_importance", ascending=False)
    importance.to_csv(artifact_dir / "feature_importance.csv", index=False, encoding="utf-8")

    weights_by_type = {
        "R": [1.0 - regular_weight, regular_weight],
        "F": [1.0 - futures_weight, futures_weight],
    }
    ensemble_config = {
        "version": 4,
        "source_order": SOURCE_ORDER,
        "weights": [1.0, 0.0],
        "weights_by_game_type": weights_by_type,
        "weight_map_by_game_type": {
            game_type: dict(zip(SOURCE_ORDER, values))
            for game_type, values in weights_by_type.items()
        },
        "sources": [
            {"name": "legacy_global", "kind": "single", "file": "legacy_global.cbm", "feature_set": "legacy_global"},
            {
                "name": "game_type", "kind": "game_type_router", "feature_set": "game_type",
                "fallback_source": "legacy_global",
                "routes": {"R": "regular_expert.cbm", "F": "futures_expert.cbm"},
            },
        ],
        "model_files": MODEL_FILES,
        "selected_regular": "regular_hl1_5",
        "selected_futures": "futures_post2023",
        "best_iterations": best_iterations,
        "final_iterations": final_iterations,
        "pitcher_expert": "experimental_only_train_v21",
    }
    calibration_config: dict[str, Any] = {
        "version": 4,
        "method": calibration_method,
        "selection_statistics": calibration_stats,
        "walk_forward_folds": calibration_folds,
        "trained_on_seasons": VALID_YEARS,
    }
    if calibration_method == "affine":
        calibration_config.update(slope=final_slope, intercept=final_intercept)
    save_json(model_dir / "feature_config.json", feature_config)
    save_json(model_dir / "ensemble_config.json", ensemble_config)
    save_json(model_dir / "calibration_config.json", calibration_config)

    selected_fold_brier = {
        str(year): brier(
            y[oof_mask & (seasons == year)], calibrated[oof_mask & (seasons == year)]
        )
        for year in VALID_YEARS
    }
    current_2024 = oof_mask & (seasons == 2024)
    regular_2024 = current_2024 & (game_types == "R")
    futures_2024 = current_2024 & (game_types == "F")
    selected_stats = calibration_stats[calibration_method]
    acceptance = {
        key: bool(selected_stats[key] <= limit)
        for key, limit in ACCEPTANCE_LIMITS.items()
    }
    summary = {
        "version": "2.2",
        "fast_mode": bool(fast_mode),
        "rows": n_rows,
        "features": X.shape[1],
        "selected_regular": "regular_hl1_5",
        "selected_futures": "futures_post2023",
        "selected_weights": {"R": regular_weight, "F": futures_weight},
        "selected_fold_brier": selected_fold_brier,
        "weighted_cv_brier": selected_stats["weighted_cv_brier"],
        "worst_fold_brier": float(max(selected_fold_brier.values())),
        "brier_2024": selected_stats["brier_2024"],
        "brier_2024_r": selected_stats["brier_2024_r"],
        "brier_2024_f": selected_stats["brier_2024_f"],
        "mean_gap_2024_r": float(calibrated[regular_2024].mean() - y[regular_2024].mean()),
        "mean_gap_2024_f": float(calibrated[futures_2024].mean() - y[futures_2024].mean()),
        "calibration_slope": final_slope if calibration_method == "affine" else 1.0,
        "calibration_intercept": final_intercept if calibration_method == "affine" else 0.0,
        "acceptance_limits": ACCEPTANCE_LIMITS,
        "acceptance": acceptance,
        "all_acceptance_passed": bool(all(acceptance.values())),
        "ensemble": ensemble_config,
        "calibration": calibration_config,
        "elapsed_seconds": time.time() - started,
    }
    save_json(artifact_dir / "training_summary.json", summary)
    save_json(
        artifact_dir / "adoption_criteria.json",
        {"limits": ACCEPTANCE_LIMITS, "results": acceptance, "all_passed": all(acceptance.values())},
    )
    submission_path = run_inference(
        root, data_dir=str(data_dir), model_dir=str(model_dir), output_dir=str(run_path / "output")
    )
    print(f"V2.2 training complete in {time.time() - started:.1f}s")
    return {
        "summary": summary,
        "metrics": metrics,
        "correlations": pd.DataFrame(
            {"legacy_global": oof["legacy_global"][oof_mask], "game_type": expert[oof_mask]}
        ).corr(),
        "blend_grid": pd.DataFrame(grid_rows),
        "iteration_sensitivity": sensitivity,
        "importance": importance,
        "submission_path": submission_path,
    }


if __name__ == "__main__":
    run_training(Path(__file__).resolve().parent)
