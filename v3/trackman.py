"""Transferable Trackman cohort mechanics and latent pitch-type features."""

from __future__ import annotations

import gc
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier


PITCH_GROUPS = ["fastball", "breaking", "offspeed", "other"]
MECHANIC_COLUMNS = [
    "rel_speed",
    "spin_rate",
    "induced_vert_break",
    "horz_break",
    "extension",
    "rel_height",
    "rel_side",
    "zone_speed",
]
MECHANIC_STATISTICS = ["mean", "std", "median", "iqr", "mad", "drift"]
PROFILE_SCALARS = ["release_cov", "release_det", "release_trace", "release_logdet"]
PITCH_INPUT_COLUMNS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "balls_before",
    "strikes_before",
    "outs_before",
    "pitcher_hand",
    "batter_hand",
    "pitchmix_n",
    "fastball_rate",
    "breaking_rate",
    "offspeed_rate",
]
PITCH_CATEGORICAL = ["top_bottom", "pitcher_hand", "batter_hand"]


def prepare_trackman(trackman: pd.DataFrame) -> pd.DataFrame:
    frame = trackman.copy()
    frame["pitch_type_group"] = frame["pitch_type_group"].astype(str).str.lower()
    frame.loc[~frame["pitch_type_group"].isin(PITCH_GROUPS), "pitch_type_group"] = "other"
    frame["pitcher_hand"] = frame["pitcher_hand"].map({"Left": "1", "Right": "2"}).fillna("0")
    frame["batter_hand"] = frame["batter_hand"].map({"Left": "1", "Right": "2"}).fillna("0")
    frame["top_bottom"] = frame["top_bottom"].map({"Top": "T", "Bottom": "B"}).fillna(
        frame["top_bottom"].astype(str)
    )
    frame["_date"] = pd.to_datetime(frame["game_date"], errors="coerce")
    frame.sort_values(
        ["season", "_date", "trackman_game_id", "pitch_no", "trackman_id"],
        kind="stable",
        inplace=True,
    )
    pitcher = frame["pitcher_trackman_id"]
    order = frame.groupby("pitcher_trackman_id", sort=False).cumcount()
    frame["pitchmix_n"] = order.astype("int32")
    for group in PITCH_GROUPS[:3]:
        current = frame["pitch_type_group"].eq(group).astype("int32")
        prior_sum = current.groupby(pitcher, sort=False).cumsum() - current
        frame[f"{group}_rate"] = np.where(order > 0, prior_sum / order, np.nan).astype("float32")
    frame.drop(columns=["_date"], inplace=True)
    return frame


def pitch_input_from_main(raw: pd.DataFrame) -> pd.DataFrame:
    frame = pd.DataFrame(index=raw.index)
    for column in [
        "season", "game_month", "game_dayofweek", "inning", "balls_before",
        "strikes_before", "outs_before",
    ]:
        frame[column] = raw[column]
    frame["top_bottom"] = raw["top_bottom"].astype(str)
    frame["pitcher_hand"] = raw["pitcher_hand"].astype(str)
    frame["batter_hand"] = raw["batter_hand"].astype(str)
    frame["pitchmix_n"] = raw["asof_pitcher_pitchmix_n"]
    frame["fastball_rate"] = raw["asof_pitcher_fastball_rate"]
    frame["breaking_rate"] = raw["asof_pitcher_breaking_rate"]
    frame["offspeed_rate"] = raw["asof_pitcher_offspeed_rate"]
    return frame.loc[:, PITCH_INPUT_COLUMNS]


def pitch_input_from_trackman(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[:, PITCH_INPUT_COLUMNS].copy()


def fallback_pitch_probabilities(raw: pd.DataFrame) -> np.ndarray:
    matrix = np.column_stack(
        [
            pd.to_numeric(raw["asof_pitcher_fastball_rate"], errors="coerce"),
            pd.to_numeric(raw["asof_pitcher_breaking_rate"], errors="coerce"),
            pd.to_numeric(raw["asof_pitcher_offspeed_rate"], errors="coerce"),
        ]
    ).astype("float64")
    other = np.maximum(0.0, 1.0 - np.nansum(matrix, axis=1))
    matrix = np.column_stack([matrix, other])
    prior = np.asarray([0.48, 0.34, 0.15, 0.03], dtype="float64")
    missing = ~np.isfinite(matrix).all(axis=1) | (matrix.sum(axis=1) <= 0)
    matrix[missing] = prior
    matrix = np.clip(matrix, 1e-5, None)
    return matrix / matrix.sum(axis=1, keepdims=True)


def fit_pitch_type_model(
    history: pd.DataFrame,
    iterations: int,
    seed: int,
    task_type: str,
    devices: str,
    verbose: int,
) -> CatBoostClassifier | None:
    if len(history) < 1000 or history["pitch_type_group"].nunique() < 2:
        return None
    parameters: dict[str, Any] = {
        "iterations": int(iterations),
        "depth": 7,
        "learning_rate": 0.07,
        "l2_leaf_reg": 10.0,
        "random_strength": 0.5,
        "loss_function": "MultiClass",
        "random_seed": int(seed),
        "allow_writing_files": False,
        "thread_count": -1,
        "verbose": int(verbose),
        "task_type": task_type,
    }
    if task_type == "GPU":
        parameters["devices"] = devices
    model = CatBoostClassifier(**parameters)
    model.fit(
        pitch_input_from_trackman(history),
        history["pitch_type_group"].astype(str),
        cat_features=PITCH_CATEGORICAL,
    )
    return model


def predict_pitch_probabilities(
    model: CatBoostClassifier | None,
    raw: pd.DataFrame,
) -> np.ndarray:
    fallback = fallback_pitch_probabilities(raw)
    if model is None:
        return fallback
    predicted = model.predict_proba(pitch_input_from_main(raw))
    output = np.full((len(raw), len(PITCH_GROUPS)), 1e-5, dtype="float64")
    for index, label in enumerate(model.classes_):
        if str(label) in PITCH_GROUPS:
            output[:, PITCH_GROUPS.index(str(label))] = predicted[:, index]
    output /= output.sum(axis=1, keepdims=True)
    history_n = raw["asof_pitcher_pitchmix_n"].to_numpy(dtype="float64")
    trust = np.clip(history_n / 200.0, 0.0, 1.0)[:, None]
    # The official as-of pitch mix is a strong individual prior.  The model
    # supplies context adjustment learned in the separate ID namespace.
    output = trust * np.sqrt(output * fallback) + (1.0 - trust) * output
    return output / output.sum(axis=1, keepdims=True)


def build_mechanics_config(history: pd.DataFrame) -> tuple[dict[str, Any], pd.DataFrame]:
    if history.empty:
        return {"profiles": {}, "defaults": {}}, pd.DataFrame()
    recent_season = int(history["season"].max())
    recent = history[history["season"] == recent_season]
    profiles: dict[str, Any] = {}
    rows: list[dict[str, Any]] = []

    def aggregate(part: pd.DataFrame, recent_part: pd.DataFrame) -> dict[str, float]:
        values: dict[str, float] = {"count": float(len(part))}
        for column in MECHANIC_COLUMNS:
            career = pd.to_numeric(part[column], errors="coerce")
            current = pd.to_numeric(recent_part[column], errors="coerce")
            mean = float(career.mean())
            std = float(career.std(ddof=0))
            recent_mean = float(current.mean()) if current.notna().any() else mean
            values[f"{column}_mean"] = mean
            values[f"{column}_std"] = std
            values[f"{column}_median"] = float(career.median())
            values[f"{column}_iqr"] = float(career.quantile(0.75) - career.quantile(0.25))
            median = float(career.median())
            values[f"{column}_mad"] = float((career - median).abs().median())
            values[f"{column}_drift"] = float((recent_mean - mean) / (std + 1e-6))
        release = part[["rel_height", "rel_side"]].apply(pd.to_numeric, errors="coerce").dropna()
        if len(release) >= 2:
            covariance = np.cov(release.to_numpy(dtype="float64"), rowvar=False, ddof=0)
            values["release_cov"] = float(covariance[0, 1])
            values["release_det"] = float(np.linalg.det(covariance))
            values["release_trace"] = float(np.trace(covariance))
            values["release_logdet"] = float(np.log(max(values["release_det"], 1e-12)))
        else:
            values.update({name: 0.0 for name in PROFILE_SCALARS})
        return values

    defaults: dict[str, Any] = {}
    for group in PITCH_GROUPS:
        part = history[history["pitch_type_group"] == group]
        current = recent[recent["pitch_type_group"] == group]
        if not part.empty:
            defaults[group] = aggregate(part, current)
    for (hand, group), part in history.groupby(["pitcher_hand", "pitch_type_group"], sort=False):
        current = recent[(recent["pitcher_hand"] == hand) & (recent["pitch_type_group"] == group)]
        values = aggregate(part, current)
        key = f"{hand}|{group}"
        profiles[key] = values
        rows.append({"pitcher_hand": hand, "pitch_type_group": group, **values})
    return {
        "recent_season": recent_season,
        "hand_mapping": {"Left": "1", "Right": "2"},
        "id_join_policy": "cohort_only_no_player_id_mapping",
        "profiles": profiles,
        "defaults": defaults,
        "pitch_groups": PITCH_GROUPS,
        "mechanic_columns": MECHANIC_COLUMNS,
        "mechanic_statistics": MECHANIC_STATISTICS,
        "profile_scalars": PROFILE_SCALARS,
    }, pd.DataFrame(rows)


def mechanics_features(
    raw: pd.DataFrame,
    probabilities: np.ndarray,
    config: dict[str, Any],
) -> pd.DataFrame:
    result = pd.DataFrame(index=raw.index)
    for index, group in enumerate(PITCH_GROUPS):
        result[f"pitch_probability_{group}"] = probabilities[:, index].astype("float32")
    result["pitch_type_entropy"] = (
        -np.sum(probabilities * np.log(np.clip(probabilities, 1e-8, 1.0)), axis=1)
    ).astype("float32")
    result["max_pitch_probability"] = probabilities.max(axis=1).astype("float32")

    hands = raw["pitcher_hand"].astype(str).to_numpy()
    profiles = config.get("profiles", {})
    defaults = config.get("defaults", {})
    for mechanic in MECHANIC_COLUMNS:
        for statistic in MECHANIC_STATISTICS:
            output = np.zeros(len(raw), dtype="float64")
            for group_index, group in enumerate(PITCH_GROUPS):
                values = np.empty(len(raw), dtype="float64")
                default = float(defaults.get(group, {}).get(f"{mechanic}_{statistic}", 0.0))
                for row_index, hand in enumerate(hands):
                    values[row_index] = float(
                        profiles.get(f"{hand}|{group}", {}).get(
                            f"{mechanic}_{statistic}", default
                        )
                    )
                output += probabilities[:, group_index] * values
            result[f"expected_{mechanic}_{statistic}"] = output.astype("float32")
    for scalar in PROFILE_SCALARS:
        output = np.zeros(len(raw), dtype="float64")
        for group_index, group in enumerate(PITCH_GROUPS):
            default = float(defaults.get(group, {}).get(scalar, 0.0))
            values = np.asarray(
                [float(profiles.get(f"{hand}|{group}", {}).get(scalar, default)) for hand in hands],
                dtype="float64",
            )
            output += probabilities[:, group_index] * values
        result[f"expected_{scalar}"] = output.astype("float32")
    std_columns = [f"expected_{column}_std" for column in MECHANIC_COLUMNS]
    drift_columns = [f"expected_{column}_drift" for column in MECHANIC_COLUMNS]
    result["mechanical_instability"] = np.sqrt(
        np.mean(np.square(result[std_columns].fillna(0.0)), axis=1)
    ).astype("float32")
    result["mechanical_drift"] = np.sqrt(
        np.mean(np.square(result[drift_columns].fillna(0.0)), axis=1)
    ).astype("float32")
    return result


def build_trackman_features(
    raw: pd.DataFrame,
    prepared_trackman: pd.DataFrame,
    prediction_season: int,
    iterations: int,
    seed: int,
    task_type: str,
    devices: str,
    verbose: int,
    fit_pitch_model_enabled: bool = True,
) -> tuple[pd.DataFrame, CatBoostClassifier | None, dict[str, Any], pd.DataFrame]:
    history = prepared_trackman[prepared_trackman["season"] < prediction_season]
    model = (
        fit_pitch_type_model(history, iterations, seed, task_type, devices, verbose)
        if fit_pitch_model_enabled
        else None
    )
    probabilities = predict_pitch_probabilities(model, raw)
    mechanics_config, profile_frame = build_mechanics_config(history)
    features = mechanics_features(raw, probabilities, mechanics_config)
    return features, model, mechanics_config, profile_frame


def dispose_model(model: CatBoostClassifier | None) -> None:
    if model is not None:
        del model
    gc.collect()
