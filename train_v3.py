"""V3 integrated leakage-safe failure decomposition and cohort physics model."""

from __future__ import annotations

import gc
import json
import os
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import roc_auc_score

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
from train_v2 import recency_weights
from train_v21 import make_feature_sets_v21, recent_two_iteration
from v3.ensemble import (
    RECENCY_SCHEMES,
    apply_calibrator,
    apply_failure_combiner,
    brier,
    fit_affine,
    fit_beta,
    fit_failure_combiner,
    fit_simplex_brier,
    robust_score,
    season_metrics,
    season_sample_weights,
    structured_probability,
)
from v3.features import FAILURE_TARGETS, build_fold_eb_features
from v3.labels import failure_prevalence, reconstruct_auxiliary_labels
from v3.trackman import (
    build_trackman_features,
    dispose_model,
    prepare_trackman,
)


VALID_YEARS = [2022, 2023, 2024]
MODEL_FILES = [
    "direct_success.cbm",
    "reverse_expert.cbm",
    "middle_expert.cbm",
    "outside_expert.cbm",
    "overlap_expert.cbm",
    "physics_command.cbm",
    "futures_expert.cbm",
    "pitch_type_model.cbm",
    "trackman_profiles.parquet",
]
OUTER_SOURCE_SETS = {
    "direct": ["direct"],
    "failure": ["direct", "structured_formula", "structured_logit"],
    "physics": ["direct", "structured_formula", "structured_logit", "physics"],
    "full": ["direct", "structured_formula", "structured_logit", "physics", "futures_anchor"],
}
ADOPTION_LIMITS = {
    "recent_weighted_cv": 0.24740,
    "brier_2024": 0.24760,
    "brier_2024_r": 0.24790,
    "brier_2024_f": 0.24700,
    "auc_2024": 0.555,
}


def catboost_parameters(
    iterations: int,
    depth: int,
    learning_rate: float,
    seed: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> dict[str, Any]:
    config: dict[str, Any] = {
        "iterations": int(iterations),
        "depth": int(depth),
        "learning_rate": float(learning_rate),
        "l2_leaf_reg": 8.0,
        "random_strength": 0.5,
        "random_seed": int(seed),
        "loss_function": "Logloss",
        "eval_metric": "Logloss",
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": int(verbose),
        "task_type": task_type,
    }
    if task_type == "GPU":
        config["devices"] = devices
    return config


def fit_binary(
    X: pd.DataFrame,
    target: pd.Series | np.ndarray,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    columns: list[str],
    categorical: list[str],
    parameters: dict[str, Any],
    sample_weight: np.ndarray | None,
    early_stopping: int,
) -> tuple[np.ndarray, int]:
    y = np.asarray(target, dtype="float64")
    train_mask = np.asarray(train_mask) & np.isfinite(y)
    valid_predict_mask = np.asarray(valid_mask)
    valid_label_mask = valid_predict_mask & np.isfinite(y)
    if train_mask.sum() < 100 or len(np.unique(y[train_mask])) < 2:
        prior = float(np.nanmean(y[train_mask])) if train_mask.any() else 0.5
        return np.full(valid_predict_mask.sum(), prior, dtype="float32"), 1
    model = CatBoostClassifier(**parameters)
    fit_kwargs: dict[str, Any] = {
        "X": X.loc[train_mask, columns],
        "y": y[train_mask].astype("int8"),
        "cat_features": categorical,
    }
    if sample_weight is not None:
        fit_kwargs["sample_weight"] = sample_weight[np.asarray(train_mask)]
    if valid_label_mask.sum() >= 100 and len(np.unique(y[valid_label_mask])) > 1:
        fit_kwargs.update(
            eval_set=(X.loc[valid_label_mask, columns], y[valid_label_mask].astype("int8")),
            early_stopping_rounds=early_stopping,
            use_best_model=True,
        )
    model.fit(**fit_kwargs)
    prediction = np.clip(
        model.predict_proba(X.loc[valid_predict_mask, columns])[:, 1], 1e-5, 1 - 1e-5
    ).astype("float32")
    best_iteration = int(model.get_best_iteration()) + 1
    if best_iteration <= 0:
        best_iteration = int(parameters["iterations"])
    del model
    gc.collect()
    return prediction, best_iteration


def fit_final_binary(
    X: pd.DataFrame,
    target: pd.Series | np.ndarray,
    mask: np.ndarray,
    columns: list[str],
    categorical: list[str],
    parameters: dict[str, Any],
    sample_weight: np.ndarray | None,
    destination: Path,
) -> pd.DataFrame:
    y = np.asarray(target, dtype="float64")
    fitted_mask = np.asarray(mask) & np.isfinite(y)
    model = CatBoostClassifier(**parameters)
    weights = sample_weight[fitted_mask] if sample_weight is not None else None
    model.fit(
        X.loc[fitted_mask, columns], y[fitted_mask].astype("int8"),
        cat_features=categorical, sample_weight=weights,
    )
    model.save_model(str(destination))
    importance = pd.DataFrame(
        {"feature": columns, destination.stem: model.get_feature_importance()}
    )
    del model
    gc.collect()
    return importance


def row_metadata(train: pd.DataFrame) -> pd.DataFrame:
    return train[[ID_COL, "season", "game_type", "pitcher_id", "asof_pitcher_n"]].copy()


def failure_recency_weights(
    seasons: np.ndarray,
    game_types: np.ndarray,
    prediction_year: int,
    half_life: float,
) -> np.ndarray:
    weights = recency_weights(seasons, prediction_year, half_life).astype("float32")
    has_post_regime_f = np.any((game_types == "F") & (seasons >= 2023))
    if has_post_regime_f:
        weights[(game_types == "F") & (seasons < 2023)] *= 0.05
        weights /= weights.mean()
    return weights


def run_training(
    root: str | Path,
    run_dir: str | Path | None = None,
    fast_mode: bool | None = None,
) -> dict[str, Any]:
    started = time.time()
    root = Path(root).resolve()
    run_path = Path(run_dir).resolve() if run_dir else (root / "runs" / "v3").resolve()
    data_dir = root / "open" / "data"
    model_dir = run_path / "model"
    artifact_dir = run_path / "artifacts"
    output_dir = run_path / "output"
    model_dir.mkdir(parents=True, exist_ok=True)
    artifact_dir.mkdir(parents=True, exist_ok=True)

    seed = int(os.environ.get("BASEBALL_SEED", "42"))
    iterations = int(os.environ.get("BASEBALL_V3_ITERATIONS", "700"))
    pitch_iterations = int(os.environ.get("BASEBALL_V3_PITCH_ITERATIONS", "140"))
    early_stopping = int(os.environ.get("BASEBALL_EARLY_STOPPING", "80"))
    depth = int(os.environ.get("BASEBALL_DEPTH", "8"))
    learning_rate = float(os.environ.get("BASEBALL_LEARNING_RATE", "0.04"))
    task_type = os.environ.get("BASEBALL_TASK_TYPE", "GPU").upper()
    devices = os.environ.get("BASEBALL_GPU_DEVICES", "0")
    if fast_mode is None:
        fast_mode = os.environ.get("BASEBALL_FAST_MODE", "0") == "1"
    fast_rows = int(os.environ.get("BASEBALL_FAST_ROWS_PER_SEASON", "25000"))
    fast_trackman_rows = int(os.environ.get("BASEBALL_FAST_TRACKMAN_ROWS_PER_SEASON", "5000"))
    if fast_mode:
        iterations = min(iterations, 30)
        pitch_iterations = min(pitch_iterations, 20)
    verbose = 0 if fast_mode else 100

    train = read_compact_csv(data_dir / "train.csv", train=True)
    # Cumulative-rate differencing multiplies rates by large pitch counts;
    # retain the CSV's float64 precision for exact 0/1 reconstruction.
    reconstruction_rate_columns = [
        "asof_pitcher_success_rate", "asof_pitcher_reverse_rate",
        "asof_pitcher_middle_rate", "asof_pitcher_ball_rate",
        "asof_pitcher_strike_rate",
    ]
    precise_rates = pd.read_csv(
        data_dir / "train.csv", usecols=reconstruction_rate_columns,
        dtype={column: "float64" for column in reconstruction_rate_columns},
        encoding="utf-8-sig",
    )
    for column in reconstruction_rate_columns:
        train[column] = precise_rates[column]
    del precise_rates
    if fast_mode:
        # Contiguous season slices preserve same-pitcher n -> n+1 transitions.
        selected = []
        for _, part in train.groupby("season", sort=True):
            selected.extend(part.index[: min(len(part), fast_rows)].tolist())
        train = train.loc[sorted(selected)].reset_index(drop=True)
        print("FAST_MODE: V3 artifacts are diagnostic only and must not be submitted")

    feature_config = initial_feature_config(train)
    feature_config["version"] = 5
    feature_config["post_regime_f_prior_map"] = make_post_regime_f_prior_map(train)
    base_X = build_features(train, feature_config)
    feature_config["feature_columns"] = list(base_X.columns)

    print("Loading Trackman history", flush=True)
    trackman = pd.read_csv(data_dir / "trackman_history.csv", encoding="utf-8-sig")
    if fast_mode:
        trackman = pd.concat(
            [
                part.iloc[: min(len(part), fast_trackman_rows)]
                for _, part in trackman.groupby("season", sort=True)
            ],
            ignore_index=True,
        )
    prepared_trackman = prepare_trackman(trackman)
    del trackman
    gc.collect()

    trackman_extra = pd.DataFrame(index=train.index)
    trackman_logs: list[dict[str, Any]] = []
    for season in sorted(train["season"].unique()):
        current = train["season"].eq(season)
        features, pitch_model, mechanics_config, _ = build_trackman_features(
            train.loc[current], prepared_trackman, int(season),
            pitch_iterations, seed, task_type, devices, verbose,
            fit_pitch_model_enabled=bool(season >= 2022),
        )
        trackman_extra.loc[current, features.columns] = features.to_numpy()
        trackman_logs.append(
            {"prediction_season": int(season), "history_rows": int((prepared_trackman["season"] < season).sum()),
             "pitch_model": pitch_model is not None, "mechanics_profiles": len(mechanics_config.get("profiles", {}))}
        )
        dispose_model(pitch_model)
    trackman_extra = trackman_extra.astype("float32")

    final_prediction_season = int(train["season"].max()) + 1
    _, final_pitch_model, final_mechanics_config, final_profile_frame = build_trackman_features(
        train.iloc[: min(len(train), 1)], prepared_trackman, final_prediction_season,
        pitch_iterations, seed, task_type, devices, verbose,
    )
    if final_pitch_model is None:
        raise RuntimeError("Could not fit final pitch-type model")
    final_pitch_model.save_model(str(model_dir / "pitch_type_model.cbm"))
    final_profile_frame.to_parquet(model_dir / "trackman_profiles.parquet", index=False)
    final_profile_frame.to_csv(artifact_dir / "trackman_profiles.csv", index=False, encoding="utf-8")
    dispose_model(final_pitch_model)
    save_json(artifact_dir / "trackman_cutoff_log.json", {"folds": trackman_logs})

    y = train[TARGET_COL].to_numpy(dtype="int8")
    seasons = train["season"].to_numpy(dtype="int16")
    game_types = train["game_type"].astype(str).to_numpy()
    n_rows = len(train)
    base_feature_sets, base_categorical_sets = make_feature_sets_v21(
        list(base_X.columns), list(feature_config["categorical_columns"])
    )
    direct_columns = list(base_feature_sets["legacy_global"])
    direct_categorical = list(base_categorical_sets["legacy_global"])
    source_oof = {
        name: np.full(n_rows, np.nan, dtype="float32")
        for name in ["direct", *FAILURE_TARGETS, "physics", "futures_anchor"]
    }
    best_iterations: dict[str, list[int]] = {name: [] for name in source_oof}
    failure_metric_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    prevalence_parts: list[pd.DataFrame] = []
    fold_eb_configs: dict[int, dict[str, Any]] = {}

    common_parameters = catboost_parameters(
        iterations, depth, learning_rate, seed, task_type, devices, verbose
    )
    direct_parameters = dict(common_parameters)
    direct_parameters["eval_metric"] = "BrierScore"
    for valid_year in VALID_YEARS:
        fold_started = time.time()
        train_mask = seasons < valid_year
        valid_mask = seasons == valid_year
        train_labels, train_audit = reconstruct_auxiliary_labels(train.loc[train_mask])
        valid_labels, valid_audit = reconstruct_auxiliary_labels(train.loc[valid_mask])
        train_audit.update(fold=valid_year, split="train")
        valid_audit.update(fold=valid_year, split="valid")
        audit_rows.extend([train_audit, valid_audit])
        prevalence = failure_prevalence(train.loc[train_mask], train_labels)
        prevalence["fold"] = valid_year
        prevalence_parts.append(prevalence)

        eb_features, eb_config = build_fold_eb_features(
            train, train_mask, valid_mask, train_labels
        )
        fold_eb_configs[valid_year] = eb_config
        X = pd.concat([base_X, trackman_extra, eb_features], axis=1)
        all_columns = list(X.columns)
        categorical = [column for column in base_categorical_sets["legacy_global"] if column in all_columns]
        weights = np.ones(n_rows, dtype="float32")
        weights[train_mask] = recency_weights(seasons[train_mask], valid_year, 3.0)

        direct_prediction, best = fit_binary(
            X, y, train_mask, valid_mask, direct_columns, direct_categorical,
            direct_parameters, weights, early_stopping,
        )
        source_oof["direct"][valid_mask] = direct_prediction
        best_iterations["direct"].append(best)

        full_labels = pd.DataFrame(index=train.index, columns=FAILURE_TARGETS, dtype="float32")
        full_labels.loc[train_mask, FAILURE_TARGETS] = train_labels[FAILURE_TARGETS]
        full_labels.loc[valid_mask, FAILURE_TARGETS] = valid_labels[FAILURE_TARGETS]
        half_lives = {"reverse": 1.5, "middle": 1.0, "outside_only": 2.0, "reverse_middle": 1.5}
        for target in FAILURE_TARGETS:
            target_weights = np.ones(n_rows, dtype="float32")
            target_weights[train_mask] = failure_recency_weights(
                seasons[train_mask], game_types[train_mask], valid_year, half_lives[target]
            )
            prediction, best = fit_binary(
                X, full_labels[target], train_mask, valid_mask,
                all_columns, categorical, common_parameters, target_weights, early_stopping,
            )
            source_oof[target][valid_mask] = prediction
            best_iterations[target].append(best)
            known = valid_mask & full_labels[target].notna().to_numpy()
            failure_metric_rows.append(
                {"season": valid_year, "target": target, "rows": int(known.sum()),
                 "brier": brier(full_labels.loc[known, target], source_oof[target][known]),
                 "target_mean": float(full_labels.loc[known, target].mean()),
                 "prediction_mean": float(source_oof[target][known].mean()), "best_iteration": best}
            )

        physics_columns = list(trackman_extra.columns) + [
            column for column in X.columns if column.startswith("eb_")
        ] + [
            "balls_before", "strikes_before", "outs_before", "inning", "li",
            "asof_pitcher_n", "asof_pitcher_success_rate", "asof_pitcher_ball_rate",
            "asof_pitcher_middle_rate", "asof_pitcher_reverse_rate", "game_type",
            "pitcher_hand", "batter_hand", "count_state_code", "hand_matchup_code",
        ]
        physics_columns = list(dict.fromkeys(column for column in physics_columns if column in X))
        physics_categorical = [column for column in categorical if column in physics_columns]
        prediction, best = fit_binary(
            X, y, train_mask, valid_mask, physics_columns, physics_categorical,
            common_parameters, weights, early_stopping,
        )
        source_oof["physics"][valid_mask] = prediction
        best_iterations["physics"].append(best)

        futures_train = train_mask & (game_types == "F")
        post_regime = futures_train & (seasons >= 2023)
        if post_regime.any():
            futures_train = post_regime
        futures_valid = valid_mask & (game_types == "F")
        routed = direct_prediction.copy()
        if futures_valid.any():
            f_prediction, best = fit_binary(
                X, y, futures_train, futures_valid, all_columns, categorical,
                common_parameters, weights, early_stopping,
            )
            routed[game_types[valid_mask] == "F"] = f_prediction
        else:
            best = 1
        source_oof["futures_anchor"][valid_mask] = routed
        best_iterations["futures_anchor"].append(best)
        print(f"V3 fold {valid_year} complete | elapsed={time.time() - fold_started:.1f}s")
        del X, eb_features, full_labels
        gc.collect()

    oof_mask = np.logical_and.reduce([np.isfinite(values) for values in source_oof.values()])
    structured_formula = structured_probability(
        source_oof["reverse"], source_oof["middle"],
        source_oof["outside_only"], source_oof["reverse_middle"],
    )
    structured_logit = np.full(n_rows, np.nan, dtype="float64")
    failure_fold_configs: list[dict[str, Any]] = []
    for valid_year in VALID_YEARS:
        current = oof_mask & (seasons == valid_year)
        history = oof_mask & (seasons < valid_year)
        if history.sum() >= 1000:
            config = fit_failure_combiner(
                y[history], source_oof["reverse"][history], source_oof["middle"][history],
                source_oof["outside_only"][history], source_oof["reverse_middle"][history],
            )
            structured_logit[current] = apply_failure_combiner(
                config, source_oof["reverse"][current], source_oof["middle"][current],
                source_oof["outside_only"][current], source_oof["reverse_middle"][current],
            )
        else:
            config = {"method": "formula_fallback"}
            structured_logit[current] = structured_formula[current]
        failure_fold_configs.append({"season": valid_year, **config})
    final_failure_config = fit_failure_combiner(
        y[oof_mask], source_oof["reverse"][oof_mask], source_oof["middle"][oof_mask],
        source_oof["outside_only"][oof_mask], source_oof["reverse_middle"][oof_mask],
    )

    success_sources = {
        "direct": source_oof["direct"],
        "structured_formula": structured_formula,
        "structured_logit": structured_logit,
        "physics": source_oof["physics"],
        "futures_anchor": source_oof["futures_anchor"],
    }
    blend_rows: list[dict[str, Any]] = []
    walk_predictions: dict[str, np.ndarray] = {}
    walk_parameters: dict[str, list[dict[str, Any]]] = {}
    for candidate_name, source_names in OUTER_SOURCE_SETS.items():
        prediction = np.full(n_rows, np.nan, dtype="float64")
        parameters: list[dict[str, Any]] = []
        for valid_year in VALID_YEARS:
            current = oof_mask & (seasons == valid_year)
            history = oof_mask & (seasons < valid_year)
            if history.sum() >= 1000:
                history_matrix = np.column_stack([success_sources[name][history] for name in source_names])
                coefficients = fit_simplex_brier(
                    y[history], history_matrix, season_sample_weights(seasons[history])
                )
            else:
                coefficients = np.zeros(len(source_names), dtype="float64")
                coefficients[0] = 1.0
            current_matrix = np.column_stack([success_sources[name][current] for name in source_names])
            prediction[current] = current_matrix @ coefficients
            parameters.append({"season": valid_year, "weights": dict(zip(source_names, coefficients.tolist()))})
        metrics = season_metrics(y[oof_mask], prediction[oof_mask], seasons[oof_mask], game_types[oof_mask])
        row = {"candidate": candidate_name, "sources": ",".join(source_names), **metrics}
        for scheme in RECENCY_SCHEMES:
            row[f"score_{scheme}"] = robust_score(metrics, scheme)
            row[f"weighted_{scheme}"] = sum(
                RECENCY_SCHEMES[scheme][year] * metrics[f"brier_{year}"] for year in VALID_YEARS
            )
        blend_rows.append(row)
        walk_predictions[candidate_name] = prediction
        walk_parameters[candidate_name] = parameters
    selected_candidate = min(blend_rows, key=lambda row: row["score_primary"])["candidate"]
    selected_sources = OUTER_SOURCE_SETS[selected_candidate]
    walk_blend = walk_predictions[selected_candidate]
    full_matrix = np.column_stack([success_sources[name][oof_mask] for name in selected_sources])
    final_weights = fit_simplex_brier(
        y[oof_mask], full_matrix, season_sample_weights(seasons[oof_mask])
    )
    final_blend = full_matrix @ final_weights

    calibration_predictions: dict[str, np.ndarray] = {}
    calibration_fold_configs: dict[str, list[dict[str, Any]]] = {}
    for method in ["identity", "affine", "beta"]:
        for eta in [0.0, 0.25, 0.50, 0.75, 1.0]:
            name = f"{method}_eta_{eta:.2f}"
            prediction = np.full(n_rows, np.nan, dtype="float64")
            configs: list[dict[str, Any]] = []
            for valid_year in VALID_YEARS:
                current = oof_mask & (seasons == valid_year)
                history = oof_mask & (seasons < valid_year)
                if method == "identity" or history.sum() < 1000:
                    config: dict[str, Any] = {"method": "identity", "eta": 0.0}
                elif method == "affine":
                    config = {**fit_affine(y[history], walk_blend[history]), "eta": eta}
                else:
                    config = {**fit_beta(y[history], walk_blend[history]), "eta": eta}
                prediction[current] = apply_calibrator(walk_blend[current], config)
                configs.append({"season": valid_year, **config})
            calibration_predictions[name] = prediction
            calibration_fold_configs[name] = configs
    calibration_rows: list[dict[str, Any]] = []
    for name, prediction in calibration_predictions.items():
        metrics = season_metrics(y[oof_mask], prediction[oof_mask], seasons[oof_mask], game_types[oof_mask])
        calibration_rows.append(
            {"candidate": name, **metrics,
             "score_primary": robust_score(metrics),
             "weighted_primary": sum(RECENCY_SCHEMES["primary"][year] * metrics[f"brier_{year}"] for year in VALID_YEARS)}
        )
    selected_calibration = min(calibration_rows, key=lambda row: row["score_primary"])["candidate"]
    walk_calibrated = calibration_predictions[selected_calibration]
    method, eta_text = selected_calibration.split("_eta_")
    eta = float(eta_text)
    if method == "affine":
        final_calibration = {**fit_affine(y[oof_mask], final_blend), "eta": eta}
    elif method == "beta":
        final_calibration = {**fit_beta(y[oof_mask], final_blend), "eta": eta}
    else:
        final_calibration = {"method": "identity", "eta": 0.0}

    # Final leakage-safe feature maps are built from the entire 2019-2024 window.
    final_labels, final_audit = reconstruct_auxiliary_labels(train)
    all_rows = np.ones(n_rows, dtype=bool)
    no_rows = np.zeros(n_rows, dtype=bool)
    final_eb, final_eb_config = build_fold_eb_features(
        train, all_rows, no_rows, final_labels
    )
    final_X = pd.concat([base_X, trackman_extra, final_eb], axis=1)
    all_columns = list(final_X.columns)
    categorical = [column for column in base_categorical_sets["legacy_global"] if column in all_columns]
    physics_columns = list(trackman_extra.columns) + [column for column in all_columns if column.startswith("eb_")] + [
        "balls_before", "strikes_before", "outs_before", "inning", "li", "asof_pitcher_n",
        "asof_pitcher_success_rate", "asof_pitcher_ball_rate", "asof_pitcher_middle_rate",
        "asof_pitcher_reverse_rate", "game_type", "pitcher_hand", "batter_hand",
        "count_state_code", "hand_matchup_code",
    ]
    physics_columns = list(dict.fromkeys(column for column in physics_columns if column in final_X))
    physics_categorical = [column for column in categorical if column in physics_columns]
    final_iterations = {
        name: recent_two_iteration(values) for name, values in best_iterations.items()
    }
    final_year = int(seasons.max()) + 1
    importance_parts: list[pd.DataFrame] = []
    success_weights = recency_weights(seasons, final_year, 3.0)
    importance_parts.append(fit_final_binary(
        final_X, y, all_rows, direct_columns, direct_categorical,
        catboost_parameters(final_iterations["direct"], depth, learning_rate, seed, task_type, devices, verbose),
        success_weights, model_dir / "direct_success.cbm",
    ))
    filename_map = {
        "reverse": "reverse_expert.cbm", "middle": "middle_expert.cbm",
        "outside_only": "outside_expert.cbm", "reverse_middle": "overlap_expert.cbm",
    }
    half_lives = {"reverse": 1.5, "middle": 1.0, "outside_only": 2.0, "reverse_middle": 1.5}
    for target in FAILURE_TARGETS:
        importance_parts.append(fit_final_binary(
            final_X, final_labels[target], all_rows, all_columns, categorical,
            catboost_parameters(final_iterations[target], depth, learning_rate, seed, task_type, devices, verbose),
            failure_recency_weights(seasons, game_types, final_year, half_lives[target]),
            model_dir / filename_map[target],
        ))
    importance_parts.append(fit_final_binary(
        final_X, y, all_rows, physics_columns, physics_categorical,
        catboost_parameters(final_iterations["physics"], depth, learning_rate, seed, task_type, devices, verbose),
        success_weights, model_dir / "physics_command.cbm",
    ))
    futures_mask = (game_types == "F") & (seasons >= 2023)
    if futures_mask.sum() < 100 or len(np.unique(y[futures_mask])) < 2:
        futures_mask = game_types == "F"
    if futures_mask.sum() < 100 or len(np.unique(y[futures_mask])) < 2:
        # Only reachable in very small FAST_MODE slices.
        futures_mask = all_rows.copy()
    importance_parts.append(fit_final_binary(
        final_X, y, futures_mask, all_columns, categorical,
        catboost_parameters(final_iterations["futures_anchor"], depth, learning_rate, seed, task_type, devices, verbose),
        success_weights, model_dir / "futures_expert.cbm",
    ))

    importance = importance_parts[0]
    for part in importance_parts[1:]:
        importance = importance.merge(part, on="feature", how="outer")
    numeric_importance = [column for column in importance if column != "feature"]
    importance[numeric_importance] = importance[numeric_importance].fillna(0.0)
    importance["mean_importance"] = importance[numeric_importance].mean(axis=1)
    importance.sort_values("mean_importance", ascending=False).to_csv(
        artifact_dir / "feature_importance.csv", index=False, encoding="utf-8"
    )
    importance[
        importance["feature"].str.startswith(("expected_", "pitch_probability_", "pitch_type_", "mechanical_"))
    ].sort_values("mean_importance", ascending=False).to_csv(
        artifact_dir / "trackman_feature_importance.csv", index=False, encoding="utf-8"
    )

    feature_config.update(
        v3_eb_config=final_eb_config,
        v3_trackman_config=final_mechanics_config,
        v3_extra_columns=list(trackman_extra.columns) + list(final_eb.columns),
        model_feature_columns={
            "direct": direct_columns,
            "all": all_columns,
            "physics": physics_columns,
        },
        model_categorical_columns={
            "direct": direct_categorical,
            "all": categorical,
            "physics": physics_categorical,
        },
    )
    ensemble_config = {
        "version": 5,
        "model_files": MODEL_FILES,
        "selected_candidate": selected_candidate,
        "source_order": selected_sources,
        "weights": final_weights.tolist(),
        "weight_map": dict(zip(selected_sources, final_weights.tolist())),
        "failure_combiner": final_failure_config,
        "failure_fold_configs": failure_fold_configs,
        "walk_forward_parameters": walk_parameters[selected_candidate],
        "best_iterations": best_iterations,
        "final_iterations": final_iterations,
    }
    calibration_config = {
        "version": 5,
        **final_calibration,
        "selected_candidate": selected_calibration,
        "walk_forward_folds": calibration_fold_configs[selected_calibration],
    }
    save_json(model_dir / "feature_config.json", feature_config)
    save_json(model_dir / "ensemble_config.json", ensemble_config)
    save_json(model_dir / "calibration_config.json", calibration_config)

    pd.DataFrame(blend_rows).to_csv(artifact_dir / "blend_candidates.csv", index=False, encoding="utf-8")
    pd.DataFrame(calibration_rows).to_csv(artifact_dir / "calibration_metrics.csv", index=False, encoding="utf-8")
    pd.DataFrame(failure_metric_rows).to_csv(artifact_dir / "failure_metrics.csv", index=False, encoding="utf-8")
    cv_rows: list[dict[str, Any]] = []
    evaluation_sources = {
        **success_sources,
        "blend_walk_forward": walk_blend,
        "calibrated_walk_forward": walk_calibrated,
    }
    for valid_year in VALID_YEARS:
        current = oof_mask & (seasons == valid_year)
        for name, prediction in evaluation_sources.items():
            cv_rows.append(
                {
                    "season": valid_year,
                    "model": name,
                    "rows": int(current.sum()),
                    "brier": brier(y[current], prediction[current]),
                    "target_mean": float(y[current].mean()),
                    "prediction_mean": float(prediction[current].mean()),
                }
            )
    pd.DataFrame(cv_rows).to_csv(
        artifact_dir / "cv_metrics.csv", index=False, encoding="utf-8"
    )
    game_type_rows: list[dict[str, Any]] = []
    for valid_year in VALID_YEARS:
        for game_type in ["R", "F"]:
            current = oof_mask & (seasons == valid_year) & (game_types == game_type)
            if not current.any():
                continue
            for name, prediction in evaluation_sources.items():
                game_type_rows.append(
                    {"season": valid_year, "game_type": game_type, "model": name,
                     "rows": int(current.sum()), "brier": brier(y[current], prediction[current]),
                     "target_mean": float(y[current].mean()),
                     "prediction_mean": float(prediction[current].mean())}
                )
    pd.DataFrame(game_type_rows).to_csv(
        artifact_dir / "game_type_metrics.csv", index=False, encoding="utf-8"
    )
    final_prevalence = failure_prevalence(train, final_labels)
    final_prevalence["fold"] = "final"
    pd.concat([*prevalence_parts, final_prevalence], ignore_index=True).to_csv(
        artifact_dir / "failure_prevalence.csv", index=False, encoding="utf-8"
    )
    with (artifact_dir / "failure_reconstruction_audit.json").open("w", encoding="utf-8") as handle:
        json.dump({"folds": audit_rows, "final": final_audit}, handle, ensure_ascii=False, indent=2)

    oof_frame = row_metadata(train).loc[oof_mask].copy()
    oof_frame[TARGET_COL] = y[oof_mask]
    for name, values in source_oof.items():
        oof_frame[f"pred_{name}"] = values[oof_mask]
    oof_frame["pred_structured_formula"] = structured_formula[oof_mask]
    oof_frame["pred_structured_logit"] = structured_logit[oof_mask]
    oof_frame["pred_blend_walk_forward"] = walk_blend[oof_mask]
    oof_frame["pred_calibrated_walk_forward"] = walk_calibrated[oof_mask]
    oof_frame.to_parquet(artifact_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame({name: values[oof_mask] for name, values in success_sources.items()}).corr().to_csv(
        artifact_dir / "model_correlations.csv", encoding="utf-8"
    )

    selected_metrics = season_metrics(
        y[oof_mask], walk_calibrated[oof_mask], seasons[oof_mask], game_types[oof_mask]
    )
    recent_weighted = sum(
        RECENCY_SCHEMES["primary"][year] * selected_metrics[f"brier_{year}"]
        for year in VALID_YEARS
    )
    current_2024 = oof_mask & (seasons == 2024)
    auc_2024 = float(roc_auc_score(y[current_2024], walk_calibrated[current_2024]))
    acceptance = {
        "recent_weighted_cv": recent_weighted <= ADOPTION_LIMITS["recent_weighted_cv"],
        "brier_2024": selected_metrics["brier_2024"] <= ADOPTION_LIMITS["brier_2024"],
        "brier_2024_r": selected_metrics.get("brier_2024_r", float("inf")) <= ADOPTION_LIMITS["brier_2024_r"],
        "brier_2024_f": selected_metrics.get("brier_2024_f", float("inf")) <= ADOPTION_LIMITS["brier_2024_f"],
        "auc_2024": auc_2024 >= ADOPTION_LIMITS["auc_2024"],
    }
    selected_metrics.setdefault("brier_2024_r", float("nan"))
    selected_metrics.setdefault("brier_2024_f", float("nan"))
    summary = {
        "version": "3.0",
        "fast_mode": bool(fast_mode),
        "rows": n_rows,
        "base_features": base_X.shape[1],
        "total_features": final_X.shape[1],
        "selected_candidate": selected_candidate,
        "selected_sources": selected_sources,
        "selected_calibration": selected_calibration,
        "recent_weighted_cv": recent_weighted,
        **selected_metrics,
        "auc_2024": auc_2024,
        "acceptance_limits": ADOPTION_LIMITS,
        "acceptance": acceptance,
        "all_acceptance_passed": bool(all(acceptance.values())),
        "recommended_for_submission": bool(all(acceptance.values()) and not fast_mode),
        "ensemble": ensemble_config,
        "calibration": calibration_config,
        "elapsed_seconds": time.time() - started,
    }
    save_json(artifact_dir / "training_summary.json", summary)
    save_json(
        artifact_dir / "adoption_criteria.json",
        {"limits": ADOPTION_LIMITS, "results": acceptance,
         "all_passed": bool(all(acceptance.values())),
         "recommended_for_submission": summary["recommended_for_submission"]},
    )
    submission_path = run_inference(
        root, data_dir=str(data_dir), model_dir=str(model_dir), output_dir=str(output_dir)
    )
    print(f"V3 training complete in {time.time() - started:.1f}s")
    return {
        "summary": summary,
        "blend_candidates": pd.DataFrame(blend_rows),
        "calibration_metrics": pd.DataFrame(calibration_rows),
        "failure_metrics": pd.DataFrame(failure_metric_rows),
        "importance": importance.sort_values("mean_importance", ascending=False),
        "submission_path": submission_path,
    }


if __name__ == "__main__":
    run_training(Path(__file__).resolve().parent)
