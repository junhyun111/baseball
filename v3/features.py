"""Leakage-safe empirical-Bayes features for reconstructed failure modes."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


FAILURE_TARGETS = ["reverse", "middle", "outside_only", "reverse_middle"]
EB_LEVELS = {
    "pitcher": ["pitcher_id"],
    "pitcher_game_type": ["pitcher_id", "game_type"],
    "pitcher_batter_hand": ["pitcher_id", "batter_hand"],
}


def _key_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    if len(columns) == 1:
        return frame[columns[0]].astype(str)
    return frame[columns].astype(str).agg("|".join, axis=1)


def build_fold_eb_features(
    raw: pd.DataFrame,
    train_mask: np.ndarray,
    valid_mask: np.ndarray,
    train_labels: pd.DataFrame,
    strength: float = 80.0,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Build shifted train features and train-only maps for validation."""
    result = pd.DataFrame(index=raw.index)
    train_index = raw.index[np.asarray(train_mask)]
    valid_index = raw.index[np.asarray(valid_mask)]
    train_raw = raw.loc[train_index]
    valid_raw = raw.loc[valid_index]
    config: dict[str, Any] = {"strength": float(strength), "targets": {}}

    for target in FAILURE_TARGETS:
        label = train_labels[target].reindex(train_index).astype("float64")
        known = label.notna()
        prior = float(label.loc[known].mean()) if known.any() else 0.1
        target_config: dict[str, Any] = {"prior": prior, "levels": {}}
        for level, columns in EB_LEVELS.items():
            train_key = _key_series(train_raw, columns)
            filled = label.fillna(0.0)
            observed = known.astype("int64")
            cumulative_sum = filled.groupby(train_key, sort=False).cumsum() - filled
            cumulative_count = observed.groupby(train_key, sort=False).cumsum() - observed
            feature_name = f"eb_{target}_{level}"
            count_name = f"eb_{target}_{level}_n"
            result.loc[train_index, feature_name] = (
                (cumulative_sum + strength * prior) / (cumulative_count + strength)
            ).astype("float32")
            result.loc[train_index, count_name] = np.log1p(cumulative_count).astype("float32")

            stats = pd.DataFrame({"key": train_key, "value": label}).dropna(subset=["value"])
            grouped = stats.groupby("key", sort=False)["value"].agg(["sum", "count"])
            posterior = (grouped["sum"] + strength * prior) / (grouped["count"] + strength)
            posterior_map = posterior.to_dict()
            count_map = grouped["count"].astype(int).to_dict()
            valid_key = _key_series(valid_raw, columns)
            result.loc[valid_index, feature_name] = valid_key.map(posterior_map).fillna(prior).astype("float32")
            result.loc[valid_index, count_name] = np.log1p(valid_key.map(count_map).fillna(0)).astype("float32")
            target_config["levels"][level] = {
                "columns": columns,
                "posterior": {str(key): float(value) for key, value in posterior_map.items()},
                "count": {str(key): int(value) for key, value in count_map.items()},
            }
        config["targets"][target] = target_config
    result.replace([np.inf, -np.inf], np.nan, inplace=True)
    return result, config


def attach_eb_from_config(raw: pd.DataFrame, config: dict[str, Any]) -> pd.DataFrame:
    """Attach final train-derived EB maps to independent inference rows."""
    result = pd.DataFrame(index=raw.index)
    for target, target_config in config.get("targets", {}).items():
        prior = float(target_config["prior"])
        for level, level_config in target_config["levels"].items():
            key = _key_series(raw, list(level_config["columns"]))
            feature_name = f"eb_{target}_{level}"
            count_name = f"eb_{target}_{level}_n"
            result[feature_name] = key.map(level_config["posterior"]).fillna(prior).astype("float32")
            result[count_name] = np.log1p(key.map(level_config["count"]).fillna(0)).astype("float32")
    return result
