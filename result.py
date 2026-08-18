"""Train/inference shared utilities and DACON submission entry point.

Local usage
-----------
python result.py
python result.py --build-submit

`--build-submit` copies this file to `script.py` inside submit.zip because the
DACON evaluator requires that filename.  Feature engineering is intentionally
row-wise: it never aggregates or otherwise uses other test rows.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


ID_COL = "row_id"
TARGET_COL = "control_success"
CONFIG_FILES = (
    "feature_config.json",
    "ensemble_config.json",
    "calibration_config.json",
)
LEGACY_MODEL_FILES = ("global_rmse.cbm", "global_logloss.cbm")
EPSILON = 1e-5

STRING_COLUMNS = [ID_COL, "top_bottom", "game_type", "base_state"]
INT8_COLUMNS = [
    "game_month",
    "game_dayofweek",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "pitcher_hand",
    "batter_hand",
]
INT16_COLUMNS = [
    "season",
    "inning",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "pitcher_team_id",
    "batter_team_id",
]
INT32_COLUMNS = [
    "pitcher_id",
    "batter_id",
    "asof_pitcher_n",
    "asof_batter_n",
    "asof_pitcher_pitchmix_n",
]
FLOAT32_COLUMNS = [
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

RAW_INPUT_COLUMNS = [
    ID_COL,
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "run_top_before",
    "run_bot_before",
    "run_total_before",
    "score_diff_home",
    "score_diff_pitcher_team",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "home_win_expectancy",
    "away_win_expectancy",
    "li",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "asof_pitcher_n",
    "asof_pitcher_success_rate",
    "asof_pitcher_reverse_rate",
    "asof_pitcher_middle_rate",
    "asof_pitcher_ball_rate",
    "asof_pitcher_strike_rate",
    "asof_pitcher_prev1_game_success_rate",
    "asof_pitcher_prev3_game_success_rate",
    "asof_pitcher_prev5_game_success_rate",
    "asof_pitcher_prev1_game_middle_rate",
    "asof_pitcher_prev3_game_middle_rate",
    "asof_pitcher_prev5_game_middle_rate",
    "asof_batter_n",
    "asof_batter_success_rate",
    "asof_batter_middle_rate",
    "asof_pitcher_pitchmix_n",
    "asof_pitcher_fastball_rate",
    "asof_pitcher_breaking_rate",
    "asof_pitcher_offspeed_rate",
]

BASE_CATEGORICAL_COLUMNS = [
    "game_month",
    "game_dayofweek",
    "top_bottom",
    "game_type",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    "count_state_code",
    "hand_matchup_code",
]


def read_compact_csv(path: str | Path, train: bool = False) -> pd.DataFrame:
    """Read the official schema with memory-conscious dtypes."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"CSV not found: {path}")

    dtypes: dict[str, str] = {}
    dtypes.update({column: "string" for column in STRING_COLUMNS})
    dtypes.update({column: "int8" for column in INT8_COLUMNS})
    dtypes.update({column: "int16" for column in INT16_COLUMNS})
    dtypes.update({column: "int32" for column in INT32_COLUMNS})
    dtypes.update({column: "float32" for column in FLOAT32_COLUMNS})
    if train:
        dtypes[TARGET_COL] = "int8"

    frame = pd.read_csv(path, dtype=dtypes, encoding="utf-8-sig")
    required = set(RAW_INPUT_COLUMNS)
    if train:
        required.add(TARGET_COL)
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing required CSV columns: {missing}")
    return frame


def make_season_prior_map(
    train: pd.DataFrame,
    default_prior: float = 0.5,
    future_season: int | None = None,
) -> dict[str, float]:
    """Build leakage-safe expanding league priors.

    The prior for season Y uses target rows whose season is strictly less than
    Y.  The extra future-season entry is therefore safe for 2025 inference.
    """
    if TARGET_COL not in train.columns:
        raise ValueError("Target is required to build season priors")
    seasons = sorted(int(value) for value in train["season"].unique())
    if not seasons:
        raise ValueError("Training data has no seasons")
    if future_season is None:
        future_season = max(seasons) + 1

    priors: dict[str, float] = {}
    for season in seasons + [int(future_season)]:
        history = train.loc[train["season"] < season, TARGET_COL]
        prior = float(history.mean()) if len(history) else float(default_prior)
        priors[str(season)] = prior
    return priors


def make_recent_prior_maps(
    train: pd.DataFrame,
    half_life: float = 1.0,
    default_prior: float = 0.5,
    future_season: int | None = None,
) -> tuple[dict[str, float], dict[str, dict[str, float]]]:
    """Build target-only recency priors without reading evaluation rows.

    For prediction season Y, only labeled seasons earlier than Y contribute.
    Counts and target sums are exponentially downweighted by season age.
    """
    if TARGET_COL not in train.columns:
        raise ValueError("Target is required to build recent priors")
    if half_life <= 0:
        raise ValueError("half_life must be positive")
    seasons = sorted(int(value) for value in train["season"].unique())
    if not seasons:
        raise ValueError("Training data has no seasons")
    if future_season is None:
        future_season = max(seasons) + 1

    by_season = train.groupby("season", observed=True)[TARGET_COL].agg(["sum", "count"])
    by_type = train.groupby(["season", "game_type"], observed=True)[TARGET_COL].agg(
        ["sum", "count"]
    )
    game_types = sorted(str(value) for value in train["game_type"].dropna().unique())
    global_priors: dict[str, float] = {}
    type_priors: dict[str, dict[str, float]] = {}

    for prediction_season in seasons + [int(future_season)]:
        historical_years = [year for year in seasons if year < prediction_season]
        weighted_sum = 0.0
        weighted_count = 0.0
        for year in historical_years:
            weight = 0.5 ** (((prediction_season - 1) - year) / half_life)
            weighted_sum += weight * float(by_season.loc[year, "sum"])
            weighted_count += weight * float(by_season.loc[year, "count"])
        global_prior = (
            weighted_sum / weighted_count if weighted_count else float(default_prior)
        )
        global_priors[str(prediction_season)] = float(global_prior)

        type_priors[str(prediction_season)] = {}
        for game_type in game_types:
            typed_sum = 0.0
            typed_count = 0.0
            for year in historical_years:
                key = (year, game_type)
                if key not in by_type.index:
                    continue
                weight = 0.5 ** (((prediction_season - 1) - year) / half_life)
                typed_sum += weight * float(by_type.loc[key, "sum"])
                typed_count += weight * float(by_type.loc[key, "count"])
            type_priors[str(prediction_season)][game_type] = float(
                typed_sum / typed_count if typed_count else global_prior
            )
    return global_priors, type_priors


def initial_feature_config(train: pd.DataFrame) -> dict[str, Any]:
    """Create the preprocessing configuration later persisted for inference."""
    future_season = int(train["season"].max()) + 1
    recent_global_prior_map, recent_game_type_prior_map = make_recent_prior_maps(
        train,
        half_life=1.0,
        default_prior=0.5,
        future_season=future_season,
    )
    return {
        "version": 2,
        "id_column": ID_COL,
        "target_column": TARGET_COL,
        "raw_input_columns": RAW_INPUT_COLUMNS,
        "drop_columns": [
            "run_total_before",
            "away_win_expectancy",
            "asof_pitcher_pitchmix_n",
        ],
        "categorical_columns": BASE_CATEGORICAL_COLUMNS,
        "shrinkage_k": [50, 200, 1000],
        "batter_shrinkage_k": [50, 200],
        "default_prior": 0.5,
        "future_season": future_season,
        "season_prior_map": make_season_prior_map(
            train, default_prior=0.5, future_season=future_season
        ),
        "recent_prior_half_life": 1.0,
        "recent_global_prior_map": recent_global_prior_map,
        "recent_game_type_prior_map": recent_game_type_prior_map,
        "prediction_clip": [EPSILON, 1.0 - EPSILON],
    }


def make_post_regime_f_prior_map(
    train: pd.DataFrame,
    start_season: int = 2023,
    half_life: float = 1.0,
) -> dict[str, float]:
    """Build an F-only prior emphasizing the post-2023 regime."""
    future_season = int(train["season"].max()) + 1
    recent_global, recent_by_type = make_recent_prior_maps(
        train, half_life=half_life, future_season=future_season
    )
    f_rows = train.loc[train["game_type"].astype(str) == "F"]
    result: dict[str, float] = {}
    for prediction_season in sorted(int(value) for value in train["season"].unique()) + [future_season]:
        history = f_rows.loc[
            (f_rows["season"] >= start_season)
            & (f_rows["season"] < prediction_season)
        ]
        if history.empty:
            result[str(prediction_season)] = float(
                recent_by_type.get(str(prediction_season), {}).get(
                    "F", recent_global[str(prediction_season)]
                )
            )
            continue
        latest = prediction_season - 1
        age = latest - history["season"].to_numpy(dtype="float64")
        weights = np.power(0.5, age / half_life)
        result[str(prediction_season)] = float(
            np.average(history[TARGET_COL].to_numpy(dtype="float64"), weights=weights)
        )
    return result


def _season_prior(series: pd.Series, config: dict[str, Any]) -> pd.Series:
    prior_map = {
        int(key): float(value)
        for key, value in config["season_prior_map"].items()
    }
    fallback = prior_map.get(
        int(config.get("future_season", -1)),
        float(config.get("default_prior", 0.5)),
    )
    return series.map(prior_map).fillna(fallback).astype("float32")


def _recent_global_prior(series: pd.Series, config: dict[str, Any]) -> pd.Series:
    prior_map = {
        int(key): float(value)
        for key, value in config.get(
            "recent_global_prior_map", config["season_prior_map"]
        ).items()
    }
    fallback = prior_map.get(
        int(config.get("future_season", -1)),
        float(config.get("default_prior", 0.5)),
    )
    return series.map(prior_map).fillna(fallback).astype("float32")


def _recent_game_type_prior(raw: pd.DataFrame, config: dict[str, Any]) -> pd.Series:
    nested = config.get("recent_game_type_prior_map", {})
    global_prior = _recent_global_prior(raw["season"], config)
    values = []
    for season, game_type, fallback in zip(
        raw["season"], raw["game_type"], global_prior, strict=False
    ):
        value = nested.get(str(int(season)), {}).get(str(game_type), fallback)
        values.append(float(value))
    return pd.Series(values, index=raw.index, dtype="float32")


def build_features(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Create deterministic row-level model features.

    This function never groups, sorts, rolls, or computes frequencies from the
    provided frame, so using it on test data respects row independence.
    """
    expected_raw = set(config.get("raw_input_columns", RAW_INPUT_COLUMNS))
    missing_raw = sorted(expected_raw.difference(raw.columns))
    if missing_raw:
        raise ValueError(f"Missing raw columns for feature engineering: {missing_raw}")

    excluded = {ID_COL, TARGET_COL, *config.get("drop_columns", [])}
    base_columns = [
        column for column in RAW_INPUT_COLUMNS if column not in excluded
    ]
    features = raw.loc[:, base_columns].copy()
    feature_version = int(config.get("version", 1))
    is_v2 = feature_version >= 2
    is_v21 = feature_version >= 3
    historical_prior = _season_prior(raw["season"], config)
    recent_global_prior = _recent_global_prior(raw["season"], config)
    recent_game_type_prior = _recent_game_type_prior(raw, config)
    modeling_prior = recent_game_type_prior if is_v2 else historical_prior
    post_regime_f_map = {
        int(key): float(value)
        for key, value in config.get("post_regime_f_prior_map", {}).items()
    }
    post_regime_f_fallback = post_regime_f_map.get(
        int(config.get("future_season", -1)), float(recent_global_prior.mean())
    )
    recent_f_prior = raw["season"].map(post_regime_f_map).fillna(
        post_regime_f_fallback
    ).astype("float32")

    balls = raw["balls_before"]
    strikes = raw["strikes_before"]
    features["count_state_code"] = (balls * 3 + strikes).astype("int8")
    features["is_full_count"] = ((balls == 3) & (strikes == 2)).astype("int8")
    features["is_three_ball"] = (balls == 3).astype("int8")
    features["is_two_strike"] = (strikes == 2).astype("int8")

    features["risp"] = (
        (raw["runner_on_2b"] == 1) | (raw["runner_on_3b"] == 1)
    ).astype("int8")
    features["bases_loaded"] = (raw["num_runners_on"] == 3).astype("int8")
    features["late_inning"] = (raw["inning"] >= 7).astype("int8")
    features["extra_inning"] = (raw["inning"] >= 10).astype("int8")
    features["close_game"] = (
        raw["score_diff_pitcher_team"].abs() <= 1
    ).astype("int8")

    same_hand = raw["pitcher_hand"] == raw["batter_hand"]
    features["same_hand"] = same_hand.astype("int8")
    features["hand_matchup_code"] = (
        raw["pitcher_hand"].astype("int16") * 10
        + raw["batter_hand"].astype("int16")
    ).astype("int16")

    features["log_pitcher_n"] = np.log1p(raw["asof_pitcher_n"]).astype("float32")
    features["log_batter_n"] = np.log1p(raw["asof_batter_n"]).astype("float32")
    features["pitcher_cold_start"] = (raw["asof_pitcher_n"] == 0).astype("int8")
    features["batter_cold_start"] = (raw["asof_batter_n"] == 0).astype("int8")
    features["recent_history_missing"] = raw[
        "asof_pitcher_prev3_game_success_rate"
    ].isna().astype("int8")

    pitcher_rate = raw["asof_pitcher_success_rate"].fillna(modeling_prior)
    batter_rate = raw["asof_batter_success_rate"].fillna(modeling_prior)
    pitcher_n = raw["asof_pitcher_n"].astype("float32")
    batter_n = raw["asof_batter_n"].astype("float32")
    if is_v2:
        if is_v21:
            features["season_prior_rate"] = historical_prior
        features["historical_expanding_prior"] = historical_prior
        features["recent_global_prior"] = recent_global_prior
        features["recent_game_type_prior"] = recent_game_type_prior
        if is_v21:
            features["recent_f_prior"] = recent_f_prior
        features["pitcher_relative_historical"] = (
            pitcher_rate - historical_prior
        ).astype("float32")
    else:
        features["season_prior_rate"] = historical_prior
    features["pitcher_relative_success"] = (pitcher_rate - modeling_prior).astype(
        "float32"
    )
    features["batter_relative_success"] = (batter_rate - modeling_prior).astype(
        "float32"
    )
    if is_v21:
        legacy_pitcher_rate = raw["asof_pitcher_success_rate"].fillna(
            historical_prior
        )
        legacy_batter_rate = raw["asof_batter_success_rate"].fillna(
            historical_prior
        )
        features["legacy_pitcher_relative_success"] = (
            legacy_pitcher_rate - historical_prior
        ).astype("float32")
        features["legacy_batter_relative_success"] = (
            legacy_batter_rate - historical_prior
        ).astype("float32")
        features["pitcher_relative_game_type_success"] = (
            pitcher_rate - recent_game_type_prior
        ).astype("float32")
        features["pitcher_relative_to_f_prior"] = (
            pitcher_rate - recent_f_prior
        ).astype("float32")
        features["recent_success_relative"] = (
            raw["asof_pitcher_prev3_game_success_rate"] - recent_game_type_prior
        ).astype("float32")

    for k in config.get("shrinkage_k", [50, 200, 1000]):
        k_value = float(k)
        features[f"pitcher_reliability_{k}"] = (
            pitcher_n / (pitcher_n + k_value)
        ).astype("float32")
        features[f"pitcher_success_shrink_{k}"] = (
            (pitcher_n * pitcher_rate + k_value * modeling_prior)
            / (pitcher_n + k_value)
        ).astype("float32")
        if is_v21:
            features[f"legacy_pitcher_success_shrink_{k}"] = (
                (pitcher_n * raw["asof_pitcher_success_rate"].fillna(historical_prior)
                 + k_value * historical_prior)
                / (pitcher_n + k_value)
            ).astype("float32")

    for k in config.get("batter_shrinkage_k", [50, 200]):
        k_value = float(k)
        features[f"batter_success_shrink_{k}"] = (
            (batter_n * batter_rate + k_value * modeling_prior)
            / (batter_n + k_value)
        ).astype("float32")
        if is_v21:
            features[f"legacy_batter_success_shrink_{k}"] = (
                (batter_n * raw["asof_batter_success_rate"].fillna(historical_prior)
                 + k_value * historical_prior)
                / (batter_n + k_value)
            ).astype("float32")

    career_success = raw["asof_pitcher_success_rate"]
    for window in (1, 3, 5):
        recent = raw[f"asof_pitcher_prev{window}_game_success_rate"]
        features[f"pitcher_success_form_{window}"] = (
            recent - career_success
        ).astype("float32")
    features["pitcher_success_prev1_minus_prev5"] = (
        raw["asof_pitcher_prev1_game_success_rate"]
        - raw["asof_pitcher_prev5_game_success_rate"]
    ).astype("float32")
    features["pitcher_success_prev3_minus_prev5"] = (
        raw["asof_pitcher_prev3_game_success_rate"]
        - raw["asof_pitcher_prev5_game_success_rate"]
    ).astype("float32")

    career_middle = raw["asof_pitcher_middle_rate"]
    for window in (1, 3, 5):
        recent = raw[f"asof_pitcher_prev{window}_game_middle_rate"]
        features[f"pitcher_middle_form_{window}"] = (
            recent - career_middle
        ).astype("float32")

    mix_columns = [
        "asof_pitcher_fastball_rate",
        "asof_pitcher_breaking_rate",
        "asof_pitcher_offspeed_rate",
    ]
    mix = raw[mix_columns].clip(lower=EPSILON, upper=1.0)
    features["pitchmix_entropy"] = (
        -(mix * np.log(mix)).sum(axis=1, min_count=len(mix_columns))
    ).astype("float32")
    features["fastball_minus_breaking"] = (
        raw["asof_pitcher_fastball_rate"]
        - raw["asof_pitcher_breaking_rate"]
    ).astype("float32")
    features["fastball_minus_offspeed"] = (
        raw["asof_pitcher_fastball_rate"]
        - raw["asof_pitcher_offspeed_rate"]
    ).astype("float32")

    features["li_x_runners"] = (
        raw["li"] * raw["num_runners_on"]
    ).astype("float32")
    features["li_x_score_diff"] = (
        raw["li"] * raw["score_diff_pitcher_team"]
    ).astype("float32")
    features["inning_x_score_diff"] = (
        raw["inning"] * raw["score_diff_pitcher_team"]
    ).astype("float32")
    features["runner3_x_outs"] = (
        raw["runner_on_3b"] * raw["outs_before"]
    ).astype("int8")
    features["li_x_three_ball"] = (
        raw["li"] * (balls == 3).astype("int8")
    ).astype("float32")

    categorical = config.get("categorical_columns", BASE_CATEGORICAL_COLUMNS)
    for column in categorical:
        if column not in features.columns:
            raise ValueError(f"Configured categorical feature is missing: {column}")
        if pd.api.types.is_numeric_dtype(features[column]):
            features[column] = features[column].fillna(-1).astype("int32")
        else:
            features[column] = features[column].fillna("__MISSING__").astype(str)

    features.replace([np.inf, -np.inf], np.nan, inplace=True)
    expected_features = config.get("feature_columns")
    if expected_features:
        missing_features = sorted(set(expected_features).difference(features.columns))
        if missing_features:
            raise ValueError(f"Engineered features are missing: {missing_features}")
        features = features.loc[:, expected_features]
    return features


def clip_predictions(predictions: np.ndarray, config: dict[str, Any]) -> np.ndarray:
    low, high = config.get("prediction_clip", [EPSILON, 1.0 - EPSILON])
    predictions = np.asarray(predictions, dtype="float64")
    if not np.isfinite(predictions).all():
        raise ValueError("Predictions contain NaN or infinity")
    return np.clip(predictions, float(low), float(high))


def apply_calibration(
    predictions: np.ndarray,
    calibration: dict[str, Any],
    feature_config: dict[str, Any],
    game_type: pd.Series | np.ndarray | None = None,
) -> np.ndarray:
    method = calibration.get("method", "identity")
    raw_predictions = np.asarray(predictions, dtype="float64")
    if method == "identity":
        calibrated = predictions
    elif method == "affine":
        calibrated = (
            float(calibration["slope"]) * predictions
            + float(calibration["intercept"])
        )
    elif method == "type_affine":
        if game_type is None:
            raise ValueError("type_affine calibration requires game_type")
        types = np.asarray(game_type).astype(str)
        offsets = calibration.get("game_type_offsets", {})
        type_offset = np.asarray(
            [float(offsets.get(value, 0.0)) for value in types], dtype="float64"
        )
        calibrated = (
            float(calibration["slope"]) * predictions
            + float(calibration["intercept"])
            + type_offset
        )
    elif method == "beta":
        probability = np.clip(raw_predictions, EPSILON, 1.0 - EPSILON)
        linear = (
            float(calibration["a"]) * np.log(probability)
            + float(calibration["b"]) * (-np.log1p(-probability))
            + float(calibration["c"])
        )
        linear = np.clip(linear, -30.0, 30.0)
        calibrated = 1.0 / (1.0 + np.exp(-linear))
    else:
        raise ValueError(f"Unsupported calibration method: {method}")
    eta = float(calibration.get("eta", 1.0))
    calibrated = (1.0 - eta) * raw_predictions + eta * np.asarray(calibrated)
    return clip_predictions(np.asarray(calibrated), feature_config)


def load_json(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        return json.load(handle)


def save_json(path: str | Path, payload: dict[str, Any]) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def resolve_data_dir(root: Path, explicit: str | None = None) -> Path:
    candidates = []
    if explicit:
        candidates.append(Path(explicit))
    candidates.extend(
        [root / "data", root / "open" / "data", Path.cwd() / "data", Path.cwd() / "open" / "data"]
    )
    for candidate in candidates:
        if (candidate / "test.csv").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find test.csv. Checked: " + ", ".join(str(path) for path in candidates)
    )


def required_model_artifacts(model_path: Path) -> tuple[str, ...]:
    """Return config and model files required by either V1 or V2 inference."""
    ensemble_path = model_path / "ensemble_config.json"
    if not ensemble_path.exists():
        return CONFIG_FILES + LEGACY_MODEL_FILES
    ensemble = load_json(ensemble_path)
    if int(ensemble.get("version", 1)) < 2:
        return CONFIG_FILES + LEGACY_MODEL_FILES
    model_files = tuple(str(value) for value in ensemble.get("model_files", []))
    if not model_files:
        raise ValueError("V2 ensemble_config.json has no model_files")
    return CONFIG_FILES + model_files


def _predict_v2_sources(
    model_path: Path,
    raw: pd.DataFrame,
    features: pd.DataFrame,
    feature_config: dict[str, Any],
    ensemble_config: dict[str, Any],
) -> dict[str, np.ndarray]:
    from catboost import CatBoostClassifier

    feature_sets = feature_config.get("model_feature_columns", {})
    source_specs = ensemble_config.get("sources", [])
    predictions: dict[str, np.ndarray] = {}

    def predict_file(filename: str, feature_set: str, mask: np.ndarray | None = None) -> np.ndarray:
        columns = feature_sets.get(feature_set)
        if not columns:
            raise ValueError(f"Missing feature set in feature_config: {feature_set}")
        model = CatBoostClassifier()
        model.load_model(str(model_path / filename))
        frame = features.loc[:, columns] if mask is None else features.loc[mask, columns]
        return clip_predictions(model.predict_proba(frame)[:, 1], feature_config)

    for spec in source_specs:
        name = str(spec["name"])
        kind = spec.get("kind", "single")
        if kind == "single":
            predictions[name] = predict_file(str(spec["file"]), str(spec["feature_set"]))
            continue
        if kind != "game_type_router":
            raise ValueError(f"Unsupported V2 source kind: {kind}")

        fallback_name = str(spec["fallback_source"])
        if fallback_name not in predictions:
            raise ValueError(f"Router fallback source is not available: {fallback_name}")
        routed = predictions[fallback_name].copy()
        feature_set = str(spec["feature_set"])
        game_type_values = raw["game_type"].astype(str).to_numpy()
        for game_type, filename in spec.get("routes", {}).items():
            mask = game_type_values == str(game_type)
            if mask.any():
                routed[mask] = predict_file(str(filename), feature_set, mask)
        predictions[name] = routed
    return predictions


def _predict_v3_blend(
    model_path: Path,
    raw: pd.DataFrame,
    base_features: pd.DataFrame,
    feature_config: dict[str, Any],
    ensemble_config: dict[str, Any],
) -> np.ndarray:
    """Run the V3 stack using only row-wise and stored historical features."""
    from catboost import CatBoostClassifier
    from v3.ensemble import apply_failure_combiner, structured_probability
    from v3.features import attach_eb_from_config
    from v3.trackman import mechanics_features, predict_pitch_probabilities

    pitch_model = CatBoostClassifier()
    pitch_model.load_model(str(model_path / "pitch_type_model.cbm"))
    pitch_probabilities = predict_pitch_probabilities(pitch_model, raw)
    trackman_features = mechanics_features(
        raw, pitch_probabilities, feature_config["v3_trackman_config"]
    )
    eb_features = attach_eb_from_config(raw, feature_config["v3_eb_config"])
    features = pd.concat([base_features, trackman_features, eb_features], axis=1)
    missing_extra = sorted(
        set(feature_config.get("v3_extra_columns", [])).difference(features.columns)
    )
    if missing_extra:
        raise ValueError(f"V3 engineered features are missing: {missing_extra}")
    feature_sets = feature_config["model_feature_columns"]

    def predict(filename: str, feature_set: str, mask: np.ndarray | None = None) -> np.ndarray:
        columns = list(feature_sets[feature_set])
        missing = sorted(set(columns).difference(features.columns))
        if missing:
            raise ValueError(f"Missing V3 model features for {filename}: {missing}")
        model = CatBoostClassifier()
        model.load_model(str(model_path / filename))
        frame = features.loc[:, columns] if mask is None else features.loc[mask, columns]
        return clip_predictions(model.predict_proba(frame)[:, 1], feature_config)

    direct = predict("direct_success.cbm", "direct")
    reverse = predict("reverse_expert.cbm", "all")
    middle = predict("middle_expert.cbm", "all")
    outside = predict("outside_expert.cbm", "all")
    overlap = predict("overlap_expert.cbm", "all")
    physics = predict("physics_command.cbm", "physics")
    structured_formula = structured_probability(reverse, middle, outside, overlap)
    structured_logit = apply_failure_combiner(
        ensemble_config["failure_combiner"], reverse, middle, outside, overlap
    )
    futures_anchor = direct.copy()
    futures_mask = raw["game_type"].astype(str).to_numpy() == "F"
    if futures_mask.any():
        futures_anchor[futures_mask] = predict(
            "futures_expert.cbm", "all", futures_mask
        )
    available = {
        "direct": direct,
        "structured_formula": structured_formula,
        "structured_logit": structured_logit,
        "physics": physics,
        "futures_anchor": futures_anchor,
    }
    source_order = [str(value) for value in ensemble_config["source_order"]]
    weights = np.asarray(ensemble_config["weights"], dtype="float64")
    if len(weights) != len(source_order) or not math.isclose(
        float(weights.sum()), 1.0, abs_tol=1e-8
    ):
        raise ValueError("V3 ensemble weights must match sources and sum to one")
    matrix = np.column_stack([available[name] for name in source_order])
    return matrix @ weights


def run_inference(
    root: Path,
    data_dir: str | None = None,
    model_dir: str | None = None,
    output_dir: str | None = None,
) -> Path:
    """Run model inference and create output/submission.csv."""
    try:
        from catboost import CatBoostClassifier, CatBoostRegressor
    except ImportError as exc:
        raise RuntimeError(
            "catboost is required for inference. Install requirements.txt first."
        ) from exc

    data_path = resolve_data_dir(root, data_dir)
    model_path = Path(model_dir).resolve() if model_dir else (root / "model").resolve()
    output_path = Path(output_dir).resolve() if output_dir else (root / "output").resolve()
    for filename in required_model_artifacts(model_path):
        if not (model_path / filename).exists():
            raise FileNotFoundError(f"Missing model artifact: {model_path / filename}")

    feature_config = load_json(model_path / "feature_config.json")
    ensemble_config = load_json(model_path / "ensemble_config.json")
    calibration_config = load_json(model_path / "calibration_config.json")

    test = read_compact_csv(data_path / "test.csv", train=False)
    if test[ID_COL].duplicated().any():
        raise ValueError("test.csv contains duplicate row_id values")
    model_features = build_features(test, feature_config)

    if int(ensemble_config.get("version", 1)) >= 5:
        blended = _predict_v3_blend(
            model_path, test, model_features, feature_config, ensemble_config
        )
    elif int(ensemble_config.get("version", 1)) >= 2:
        source_predictions = _predict_v2_sources(
            model_path, test, model_features, feature_config, ensemble_config
        )
        source_order = [str(value) for value in ensemble_config["source_order"]]
        matrix = np.column_stack([source_predictions[name] for name in source_order])
        weights_by_type = ensemble_config.get("weights_by_game_type")
        if weights_by_type:
            default_weights = np.asarray(
                ensemble_config.get("weights", [1.0 / len(source_order)] * len(source_order)),
                dtype="float64",
            )
            if len(default_weights) != len(source_order):
                raise ValueError("Default ensemble weights do not match source count")
            blended = matrix @ default_weights
            type_values = test["game_type"].astype(str).to_numpy()
            for game_type, values in weights_by_type.items():
                weights = np.asarray(values, dtype="float64")
                if len(weights) != len(source_order) or not math.isclose(
                    float(weights.sum()), 1.0, abs_tol=1e-8
                ):
                    raise ValueError(f"Invalid ensemble weights for game_type={game_type}")
                mask = type_values == str(game_type)
                blended[mask] = matrix[mask] @ weights
        else:
            weights = np.asarray(ensemble_config["weights"], dtype="float64")
            if len(weights) != len(source_order) or not math.isclose(
                float(weights.sum()), 1.0, abs_tol=1e-8
            ):
                raise ValueError("V2 ensemble weights must match sources and sum to one")
            blended = matrix @ weights
    else:
        rmse_model = CatBoostRegressor()
        rmse_model.load_model(str(model_path / "global_rmse.cbm"))
        logloss_model = CatBoostClassifier()
        logloss_model.load_model(str(model_path / "global_logloss.cbm"))
        pred_rmse = clip_predictions(rmse_model.predict(model_features), feature_config)
        pred_logloss = clip_predictions(
            logloss_model.predict_proba(model_features)[:, 1], feature_config
        )
        weight_rmse = float(ensemble_config["rmse_weight"])
        weight_logloss = float(ensemble_config["logloss_weight"])
        if not math.isclose(weight_rmse + weight_logloss, 1.0, abs_tol=1e-8):
            raise ValueError("Ensemble weights must sum to one")
        blended = weight_rmse * pred_rmse + weight_logloss * pred_logloss
    predictions = apply_calibration(
        blended, calibration_config, feature_config, test["game_type"]
    )

    predicted = pd.DataFrame({ID_COL: test[ID_COL].astype(str), TARGET_COL: predictions})
    sample_path = data_path / "sample_submission.csv"
    if sample_path.exists():
        sample = pd.read_csv(sample_path, encoding="utf-8-sig", dtype={ID_COL: "string"})
        if sample[ID_COL].duplicated().any():
            raise ValueError("sample_submission.csv contains duplicate row_id values")
        if set(sample[ID_COL].astype(str)) != set(predicted[ID_COL]):
            raise ValueError("test.csv and sample_submission.csv row_id sets do not match")
        submission = sample[[ID_COL]].astype({ID_COL: str}).merge(
            predicted, on=ID_COL, how="left", validate="one_to_one"
        )
    else:
        submission = predicted

    if len(submission) != len(test) or submission[TARGET_COL].isna().any():
        raise ValueError("Submission row validation failed")
    values = submission[TARGET_COL].to_numpy(dtype="float64")
    if not np.isfinite(values).all() or ((values < 0) | (values > 1)).any():
        raise ValueError("Submission probabilities must be finite and within [0, 1]")

    output_path.mkdir(parents=True, exist_ok=True)
    destination = output_path / "submission.csv"
    submission.to_csv(destination, index=False, encoding="utf-8")
    print(
        f"Saved {destination} | rows={len(submission):,} "
        f"mean={values.mean():.6f} min={values.min():.6f} max={values.max():.6f}"
    )
    return destination


def build_submit_zip(root: Path, model_dir: str | None, destination: str | None) -> Path:
    """Create the evaluator-compatible submit.zip without training code."""
    model_path = Path(model_dir).resolve() if model_dir else (root / "model").resolve()
    required_files = required_model_artifacts(model_path)
    missing = [filename for filename in required_files if not (model_path / filename).exists()]
    if missing:
        raise FileNotFoundError(
            "Run main.ipynb before packaging. Missing model files: " + ", ".join(missing)
        )
    destination_path = (
        Path(destination).resolve() if destination else (root / "submit.zip").resolve()
    )
    requirements = "\n".join(
        [
            "catboost==1.2.10",
            "numpy==2.3.2",
            "pandas==2.3.3",
            "scipy>=1.14",
            "",
        ]
    )

    with tempfile.TemporaryDirectory(prefix="baseball_submit_") as temp_name:
        staging = Path(temp_name)
        staged_model = staging / "model"
        staged_model.mkdir(parents=True)
        shutil.copy2(Path(__file__).resolve(), staging / "script.py")
        if int(load_json(model_path / "ensemble_config.json").get("version", 1)) >= 5:
            source_package = Path(__file__).resolve().parent / "v3"
            staged_package = staging / "v3"
            staged_package.mkdir(parents=True, exist_ok=True)
            for source_file in source_package.glob("*.py"):
                shutil.copy2(source_file, staged_package / source_file.name)
        for filename in required_files:
            shutil.copy2(model_path / filename, staged_model / filename)
        (staging / "requirements.txt").write_text(requirements, encoding="utf-8")

        destination_path.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(
            destination_path, mode="w", compression=zipfile.ZIP_DEFLATED
        ) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging).as_posix())

    with zipfile.ZipFile(destination_path) as archive:
        names = set(archive.namelist())
    required_entries = {"script.py", "requirements.txt"} | {
        f"model/{filename}" for filename in required_files
    }
    if not required_entries.issubset(names):
        raise RuntimeError("submit.zip validation failed")
    print(f"Saved {destination_path} | files={len(names)}")
    return destination_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Baseball control probability inference")
    parser.add_argument("--data-dir", default=None, help="Directory containing test.csv")
    parser.add_argument("--model-dir", default=None, help="Directory containing model artifacts")
    parser.add_argument("--output-dir", default=None, help="Directory for submission.csv")
    parser.add_argument(
        "--build-submit", action="store_true", help="Build submit.zip instead of inference"
    )
    parser.add_argument("--submit-path", default=None, help="Optional submit.zip destination")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parent
    if args.build_submit:
        build_submit_zip(root, args.model_dir, args.submit_path)
    else:
        run_inference(root, args.data_dir, args.model_dir, args.output_dir)


if __name__ == "__main__":
    main()
