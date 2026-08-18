"""Recover auxiliary pitch outcomes from strictly in-window cumulative rates."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


RATE_COLUMNS = {
    "success_reconstructed": "asof_pitcher_success_rate",
    "reverse": "asof_pitcher_reverse_rate",
    "middle": "asof_pitcher_middle_rate",
    "ball": "asof_pitcher_ball_rate",
    "strike": "asof_pitcher_strike_rate",
}


def reconstruct_auxiliary_labels(
    frame: pd.DataFrame,
    tolerance: float = 2e-2,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Recover current-row labels using the next row of the same pitcher.

    ``frame`` must already be restricted to the allowed temporal window.  The
    function never joins across a missing row or across a caller-excluded fold.
    Returned labels preserve the input index; unavailable transitions are NaN.
    """
    if frame.index.has_duplicates:
        raise ValueError("Auxiliary reconstruction requires a unique index")
    required = {"pitcher_id", "asof_pitcher_n", *RATE_COLUMNS.values()}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError(f"Missing reconstruction columns: {missing}")

    working = frame.loc[:, sorted(required)].copy()
    next_n = working.groupby("pitcher_id", sort=False)["asof_pitcher_n"].shift(-1)
    consecutive = next_n.eq(working["asof_pitcher_n"] + 1)
    labels = pd.DataFrame(index=frame.index)
    diagnostics: dict[str, Any] = {
        "rows": int(len(frame)),
        "consecutive_transitions": int(consecutive.sum()),
        "consecutive_rate": float(consecutive.mean()) if len(frame) else 0.0,
        "targets": {},
    }
    n = working["asof_pitcher_n"].astype("float64")
    for target, rate_column in RATE_COLUMNS.items():
        rate = pd.to_numeric(working[rate_column], errors="coerce")
        next_rate = rate.groupby(working["pitcher_id"], sort=False).shift(-1)
        delta = next_n.astype("float64") * next_rate - n * rate
        rounded = np.rint(delta)
        valid = (
            consecutive
            & delta.notna()
            & rounded.isin([0.0, 1.0])
            & (delta - rounded).abs().le(tolerance)
        )
        output = pd.Series(np.nan, index=frame.index, dtype="float32")
        output.loc[valid] = rounded.loc[valid].astype("float32")
        labels[target] = output
        diagnostics["targets"][target] = {
            "recovered": int(valid.sum()),
            "coverage": float(valid.mean()) if len(frame) else 0.0,
            "positive_rate": float(output.loc[valid].mean()) if valid.any() else None,
            "max_rounding_error": float((delta.loc[valid] - rounded.loc[valid]).abs().max())
            if valid.any()
            else None,
        }

    reverse_known = labels["reverse"].notna()
    middle_known = labels["middle"].notna()
    target_known = reverse_known & middle_known
    labels["reverse_middle"] = np.where(
        target_known,
        ((labels["reverse"] == 1) & (labels["middle"] == 1)).astype("float32"),
        np.nan,
    ).astype("float32")
    if "control_success" in frame:
        success = frame["control_success"].astype("float32")
        labels["outside_only"] = np.where(
            target_known,
            ((success == 0) & (labels["reverse"] == 0) & (labels["middle"] == 0)).astype("float32"),
            np.nan,
        ).astype("float32")
        comparable = labels["success_reconstructed"].notna()
        diagnostics["success_match_rate"] = float(
            (labels.loc[comparable, "success_reconstructed"] == success.loc[comparable]).mean()
        ) if comparable.any() else None
    else:
        labels["outside_only"] = np.nan

    for target in ["reverse_middle", "outside_only"]:
        known = labels[target].notna()
        diagnostics["targets"][target] = {
            "recovered": int(known.sum()),
            "coverage": float(known.mean()) if len(frame) else 0.0,
            "positive_rate": float(labels.loc[known, target].mean()) if known.any() else None,
        }
    return labels, diagnostics


def failure_prevalence(
    frame: pd.DataFrame,
    labels: pd.DataFrame,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for season, indices in frame.groupby("season", sort=True).groups.items():
        part = labels.loc[indices]
        row: dict[str, Any] = {"season": int(season), "rows": int(len(indices))}
        for target in ["reverse", "middle", "outside_only", "reverse_middle"]:
            known = part[target].notna()
            row[f"{target}_coverage"] = float(known.mean())
            row[f"{target}_rate"] = float(part.loc[known, target].mean()) if known.any() else np.nan
        rows.append(row)
    return pd.DataFrame(rows)
