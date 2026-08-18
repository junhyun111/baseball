"""Reproducible, leakage-aware EDA for the LG Aimers pitch-control dataset.

The two large CSV files are processed in chunks so the script can run on a
normal laptop.  Every statistic written by this script uses train data or the
2019--2024 Trackman history only.  The five-row public test sample is used only
for schema and train-coverage checks.
"""

from __future__ import annotations

import argparse
import json
import math
from collections import defaultdict
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns


TARGET = "control_success"
ID_COL = "row_id"

BASIC_GROUPS = [
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "num_runners_on",
    "base_state",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
]

ASOF_COLS = [
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

EXACT_UNIQUE_COLS = {
    "season",
    "game_month",
    "game_dayofweek",
    "inning",
    "top_bottom",
    "game_type",
    "balls_before",
    "strikes_before",
    "outs_before",
    "runner_on_1b",
    "runner_on_2b",
    "runner_on_3b",
    "num_runners_on",
    "base_state",
    "pitcher_id",
    "batter_id",
    "pitcher_hand",
    "batter_hand",
    "pitcher_team_id",
    "batter_team_id",
    TARGET,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).parents[1] / "data")
    parser.add_argument("--output-dir", type=Path, default=Path(__file__).parent / "outputs")
    parser.add_argument("--chunksize", type=int, default=150_000)
    parser.add_argument("--sample-frac", type=float, default=0.12)
    parser.add_argument("--trackman-sample-frac", type=float, default=0.08)
    return parser.parse_args()


def add_group_stats(
    store: dict[str, dict[str, list[int]]], feature: str, values: pd.Series, y: pd.Series
) -> None:
    frame = pd.DataFrame({"value": values.astype("string").fillna("<NA>"), "y": y})
    grouped = frame.groupby("value", dropna=False, observed=True)["y"].agg(["count", "sum"])
    feature_store = store[feature]
    for value, row in grouped.iterrows():
        bucket = feature_store[str(value)]
        bucket[0] += int(row["count"])
        bucket[1] += int(row["sum"])


def experience_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-1, 0, 9, 49, 199, 999, 4_999, np.inf],
        labels=["0", "1-9", "10-49", "50-199", "200-999", "1k-4,999", "5k+"],
        include_lowest=True,
    )


def score_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, -6, -4, -2, -1, 0, 2, 4, 6, np.inf],
        labels=["<=-7", "-6:-5", "-4:-3", "-2", "-1", "0:1", "2:3", "4:5", "6+"],
        right=False,
    )


def li_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 0.5, 1.0, 1.5, 2.0, 3.0, 5.0, np.inf],
        labels=["<0.5", "0.5-<1", "1-<1.5", "1.5-<2", "2-<3", "3-<5", "5+"],
        right=False,
    )


def inning_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[0, 3, 6, 9, np.inf],
        labels=["1-3", "4-6", "7-9", "10+"],
        include_lowest=True,
    )


def rate_bin(values: pd.Series) -> pd.Series:
    return pd.cut(
        values,
        bins=[-np.inf, 0.40, 0.45, 0.50, 0.55, 0.60, 0.65, np.inf],
        labels=["<0.40", "0.40-<0.45", "0.45-<0.50", "0.50-<0.55", "0.55-<0.60", "0.60-<0.65", "0.65+"],
        right=False,
    )


def update_moments(store: dict[str, dict[str, float]], frame: pd.DataFrame, numeric_cols: list[str]) -> None:
    for col in numeric_cols:
        x = pd.to_numeric(frame[col], errors="coerce")
        valid = x.dropna().astype("float64")
        if valid.empty:
            continue
        s = store[col]
        s["n"] += int(valid.size)
        s["sum"] += float(valid.sum())
        s["sum2"] += float(np.square(valid).sum())
        s["min"] = min(s["min"], float(valid.min()))
        s["max"] = max(s["max"], float(valid.max()))


def update_correlations(
    store: dict[str, dict[str, float]], frame: pd.DataFrame, numeric_cols: list[str]
) -> None:
    y_all = frame[TARGET].astype("float64")
    for col in numeric_cols:
        if col == TARGET:
            continue
        x_all = pd.to_numeric(frame[col], errors="coerce")
        mask = x_all.notna() & y_all.notna()
        if not mask.any():
            continue
        x = x_all[mask].astype("float64")
        y = y_all[mask]
        s = store[col]
        s["n"] += int(mask.sum())
        s["sx"] += float(x.sum())
        s["sy"] += float(y.sum())
        s["sxx"] += float(np.square(x).sum())
        s["syy"] += float(np.square(y).sum())
        s["sxy"] += float((x * y).sum())


def update_season_numeric(
    store: dict[tuple[int, str], dict[str, float]], frame: pd.DataFrame, columns: list[str]
) -> None:
    for season, part in frame.groupby("season", observed=True):
        for col in columns:
            x = pd.to_numeric(part[col], errors="coerce").dropna().astype("float64")
            if x.empty:
                continue
            s = store[(int(season), col)]
            s["n"] += int(x.size)
            s["sum"] += float(x.sum())
            s["sum2"] += float(np.square(x).sum())


def scan_train(path: Path, chunksize: int, sample_frac: float) -> dict[str, object]:
    header = pd.read_csv(path, nrows=0)
    columns = header.columns.tolist()
    dtype_probe = pd.read_csv(path, nrows=2_000)
    numeric_cols = dtype_probe.select_dtypes(include=np.number).columns.tolist()

    rows = 0
    target_sum = 0
    missing = pd.Series(0, index=columns, dtype="int64")
    moments = defaultdict(lambda: {"n": 0, "sum": 0.0, "sum2": 0.0, "min": math.inf, "max": -math.inf})
    correlations = defaultdict(
        lambda: {"n": 0, "sx": 0.0, "sy": 0.0, "sxx": 0.0, "syy": 0.0, "sxy": 0.0}
    )
    unique_values: dict[str, set] = {c: set() for c in EXACT_UNIQUE_COLS if c in columns}
    group_stats: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    missing_by_season: dict[int, pd.Series] = {}
    season_numeric = defaultdict(lambda: {"n": 0, "sum": 0.0, "sum2": 0.0})
    season_numeric_cols = [
        "li",
        "home_win_expectancy",
        "asof_pitcher_n",
        "asof_pitcher_success_rate",
        "asof_pitcher_prev3_game_success_rate",
        "asof_batter_n",
        "asof_batter_success_rate",
        "asof_pitcher_pitchmix_n",
    ]
    samples: list[pd.DataFrame] = []
    seen_row_numbers: set[int] = set()
    duplicate_row_ids = 0
    invalid_row_ids = 0
    invariants = defaultdict(
        lambda: {
            "eligible": 0,
            "violations": 0,
            "abs_error_sum": 0.0,
            "max_abs_error": 0.0,
            "tolerance": 0.0,
        }
    )

    for chunk_idx, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, low_memory=False)):
        n = len(chunk)
        rows += n
        y = chunk[TARGET].astype("int64")
        target_sum += int(y.sum())
        missing = missing.add(chunk.isna().sum(), fill_value=0).astype("int64")
        update_moments(moments, chunk, numeric_cols)
        update_correlations(correlations, chunk, numeric_cols)
        update_season_numeric(season_numeric, chunk, season_numeric_cols)

        for col, values in unique_values.items():
            values.update(chunk[col].dropna().unique().tolist())

        row_numbers = pd.to_numeric(chunk[ID_COL].str.rsplit("_", n=1).str[-1], errors="coerce")
        invalid_row_ids += int(row_numbers.isna().sum())
        unique_chunk_ids = set(row_numbers.dropna().astype("int64").tolist())
        duplicate_row_ids += n - int(row_numbers.isna().sum()) - len(unique_chunk_ids)
        duplicate_row_ids += len(seen_row_numbers.intersection(unique_chunk_ids))
        seen_row_numbers.update(unique_chunk_ids)

        for col in BASIC_GROUPS:
            add_group_stats(group_stats, col, chunk[col], y)
        add_group_stats(group_stats, "pitcher_id", chunk["pitcher_id"], y)
        add_group_stats(group_stats, "batter_id", chunk["batter_id"], y)
        add_group_stats(
            group_stats,
            "count_state",
            chunk["balls_before"].astype(str) + "-" + chunk["strikes_before"].astype(str),
            y,
        )
        add_group_stats(
            group_stats,
            "hand_matchup",
            "P" + chunk["pitcher_hand"].astype(str) + "-B" + chunk["batter_hand"].astype(str),
            y,
        )
        add_group_stats(group_stats, "inning_group", inning_bin(chunk["inning"]), y)
        add_group_stats(group_stats, "li_bin", li_bin(chunk["li"]), y)
        add_group_stats(
            group_stats, "pitcher_score_diff_bin", score_bin(chunk["score_diff_pitcher_team"]), y
        )
        add_group_stats(group_stats, "pitcher_history_n_bin", experience_bin(chunk["asof_pitcher_n"]), y)
        add_group_stats(group_stats, "batter_history_n_bin", experience_bin(chunk["asof_batter_n"]), y)
        add_group_stats(
            group_stats, "pitchmix_history_n_bin", experience_bin(chunk["asof_pitcher_pitchmix_n"]), y
        )
        add_group_stats(
            group_stats,
            "pitcher_success_rate_bin",
            rate_bin(chunk["asof_pitcher_success_rate"]),
            y,
        )
        add_group_stats(
            group_stats,
            "batter_success_rate_bin",
            rate_bin(chunk["asof_batter_success_rate"]),
            y,
        )
        add_group_stats(
            group_stats,
            "recent3_success_rate_bin",
            rate_bin(chunk["asof_pitcher_prev3_game_success_rate"]),
            y,
        )
        add_group_stats(
            group_stats,
            "season_game_type",
            chunk["season"].astype(str) + "-" + chunk["game_type"].astype(str),
            y,
        )
        add_group_stats(
            group_stats,
            "season_month",
            chunk["season"].astype(str) + "-" + chunk["game_month"].astype(str).str.zfill(2),
            y,
        )
        add_group_stats(
            group_stats,
            "pitcher_history_missing",
            chunk["asof_pitcher_success_rate"].isna().map({True: "missing", False: "present"}),
            y,
        )
        add_group_stats(
            group_stats,
            "recent_history_missing",
            chunk["asof_pitcher_prev3_game_success_rate"].isna().map(
                {True: "missing", False: "present"}
            ),
            y,
        )

        for season, part in chunk.groupby("season", observed=True):
            season = int(season)
            cur = part.isna().sum()
            if season not in missing_by_season:
                missing_by_season[season] = cur
            else:
                missing_by_season[season] = missing_by_season[season].add(cur, fill_value=0)

        checks: dict[str, tuple[pd.Series, pd.Series, float]] = {
            "run_total_equals_parts": (
                pd.Series(True, index=chunk.index),
                (chunk["run_total_before"] - chunk["run_top_before"] - chunk["run_bot_before"]).abs(),
                0.0,
            ),
            "runner_count_equals_flags": (
                pd.Series(True, index=chunk.index),
                (
                    chunk["num_runners_on"]
                    - chunk[["runner_on_1b", "runner_on_2b", "runner_on_3b"]].sum(axis=1)
                ).abs(),
                0.0,
            ),
            "win_expectancies_sum_100": (
                chunk[["home_win_expectancy", "away_win_expectancy"]].notna().all(axis=1),
                (chunk["home_win_expectancy"] + chunk["away_win_expectancy"] - 100).abs(),
                0.11,
            ),
            "home_score_diff_matches_scores": (
                pd.Series(True, index=chunk.index),
                (chunk["score_diff_home"] - (chunk["run_bot_before"] - chunk["run_top_before"])).abs(),
                0.0,
            ),
            "pitcher_n_equals_pitchmix_n": (
                pd.Series(True, index=chunk.index),
                (chunk["asof_pitcher_n"] - chunk["asof_pitcher_pitchmix_n"]).abs(),
                0.0,
            ),
        }
        expected_pitcher_diff = np.where(
            chunk["top_bottom"].eq("T"), chunk["score_diff_home"], -chunk["score_diff_home"]
        )
        checks["pitcher_score_diff_matches_half"] = (
            pd.Series(True, index=chunk.index),
            (chunk["score_diff_pitcher_team"] - expected_pitcher_diff).abs(),
            0.0,
        )
        expected_base = (
            chunk["runner_on_1b"].map({0: "_", 1: "1"})
            + chunk["runner_on_2b"].map({0: "_", 1: "2"})
            + chunk["runner_on_3b"].map({0: "_", 1: "3"})
        )
        checks["base_state_matches_flags"] = (
            expected_base.notna() & chunk["base_state"].notna(),
            expected_base.ne(chunk["base_state"]).astype(float),
            0.0,
        )
        pitchmix_cols = [
            "asof_pitcher_fastball_rate",
            "asof_pitcher_breaking_rate",
            "asof_pitcher_offspeed_rate",
        ]
        checks["pitchmix_rates_sum_1"] = (
            chunk[pitchmix_cols].notna().all(axis=1),
            (chunk[pitchmix_cols].sum(axis=1) - 1).abs(),
            1.1e-6,
        )
        for name, (eligible, error, tolerance) in checks.items():
            eligible = eligible.fillna(False)
            err = pd.to_numeric(error[eligible], errors="coerce").fillna(np.inf)
            s = invariants[name]
            s["eligible"] += int(eligible.sum())
            s["violations"] += int((err > tolerance).sum())
            s["tolerance"] = tolerance
            finite = err[np.isfinite(err)]
            s["abs_error_sum"] += float(finite.sum())
            if not finite.empty:
                s["max_abs_error"] = max(s["max_abs_error"], float(finite.max()))

        sampled = chunk.sample(frac=sample_frac, random_state=20260812 + chunk_idx)
        samples.append(sampled)

    sample = pd.concat(samples, ignore_index=True)
    return {
        "columns": columns,
        "numeric_cols": numeric_cols,
        "dtype_probe": dtype_probe.dtypes.astype(str).to_dict(),
        "rows": rows,
        "target_sum": target_sum,
        "missing": missing,
        "moments": moments,
        "correlations": correlations,
        "unique_values": unique_values,
        "group_stats": group_stats,
        "missing_by_season": missing_by_season,
        "season_numeric": season_numeric,
        "sample": sample,
        "duplicate_row_ids": duplicate_row_ids,
        "invalid_row_ids": invalid_row_ids,
        "unique_row_ids": len(seen_row_numbers),
        "invariants": invariants,
    }


def moments_table(
    columns: list[str],
    dtypes: dict[str, str],
    rows: int,
    missing: pd.Series,
    moments: dict[str, dict[str, float]],
    unique_values: dict[str, set],
    unique_row_ids: int,
) -> pd.DataFrame:
    records = []
    for col in columns:
        s = moments.get(col)
        record: dict[str, object] = {
            "column": col,
            "dtype": dtypes.get(col, "object"),
            "rows": rows,
            "missing_count": int(missing[col]),
            "missing_rate": float(missing[col] / rows),
            "unique_count": (
                unique_row_ids if col == ID_COL else len(unique_values[col]) if col in unique_values else np.nan
            ),
            "mean": np.nan,
            "std": np.nan,
            "min": np.nan,
            "max": np.nan,
        }
        if s and s["n"]:
            mean = s["sum"] / s["n"]
            var = max(0.0, (s["sum2"] - s["sum"] ** 2 / s["n"]) / max(1, s["n"] - 1))
            record.update(mean=mean, std=math.sqrt(var), min=s["min"], max=s["max"])
        records.append(record)
    return pd.DataFrame(records)


def correlation_table(correlations: dict[str, dict[str, float]]) -> pd.DataFrame:
    records = []
    for col, s in correlations.items():
        n = s["n"]
        numerator = n * s["sxy"] - s["sx"] * s["sy"]
        denom_x = n * s["sxx"] - s["sx"] ** 2
        denom_y = n * s["syy"] - s["sy"] ** 2
        corr = numerator / math.sqrt(denom_x * denom_y) if denom_x > 0 and denom_y > 0 else np.nan
        records.append({"feature": col, "n_complete": n, "pearson_with_target": corr})
    result = pd.DataFrame(records)
    result["abs_correlation"] = result["pearson_with_target"].abs()
    return result.sort_values("abs_correlation", ascending=False).reset_index(drop=True)


def group_table(
    group_stats: dict[str, dict[str, list[int]]], global_rate: float
) -> pd.DataFrame:
    records = []
    for feature, values in group_stats.items():
        for value, (count, successes) in values.items():
            rate = successes / count
            se = math.sqrt(max(rate * (1 - rate), 0.0) / count)
            records.append(
                {
                    "feature": feature,
                    "value": value,
                    "count": count,
                    "successes": successes,
                    "success_rate": rate,
                    "rate_minus_global": rate - global_rate,
                    "ci95_low_wald": max(0.0, rate - 1.96 * se),
                    "ci95_high_wald": min(1.0, rate + 1.96 * se),
                }
            )
    return pd.DataFrame(records).sort_values(["feature", "count"], ascending=[True, False])


def season_numeric_table(store: dict[tuple[int, str], dict[str, float]]) -> pd.DataFrame:
    records = []
    for (season, feature), s in store.items():
        mean = s["sum"] / s["n"]
        var = max(0.0, (s["sum2"] - s["sum"] ** 2 / s["n"]) / max(s["n"] - 1, 1))
        records.append(
            {"season": season, "feature": feature, "count": s["n"], "mean": mean, "std": math.sqrt(var)}
        )
    return pd.DataFrame(records).sort_values(["feature", "season"])


def scan_trackman(path: Path, chunksize: int, sample_frac: float) -> dict[str, object]:
    probe = pd.read_csv(path, nrows=2_000)
    columns = probe.columns.tolist()
    numeric_cols = probe.select_dtypes(include=np.number).columns.tolist()
    rows = 0
    missing = pd.Series(0, index=columns, dtype="int64")
    moments = defaultdict(lambda: {"n": 0, "sum": 0.0, "sum2": 0.0, "min": math.inf, "max": -math.inf})
    unique_cols = [
        "season",
        "game_month",
        "game_dayofweek",
        "top_bottom",
        "pitcher_hand",
        "batter_hand",
        "pitcher_team",
        "batter_team",
        "tagged_pitch_type",
        "auto_pitch_type",
        "pitch_type_group",
        "pitcher_trackman_id",
        "batter_trackman_id",
        "trackman_game_id",
    ]
    unique_values = {c: set() for c in unique_cols}
    group_store = defaultdict(lambda: defaultdict(lambda: {"n": 0, "sum": 0.0, "sum2": 0.0}))
    count_store = defaultdict(int)
    samples: list[pd.DataFrame] = []
    metrics = [
        "rel_speed",
        "spin_rate",
        "induced_vert_break",
        "horz_break",
        "extension",
        "rel_height",
        "rel_side",
        "zone_speed",
    ]
    quality_counts = defaultdict(int)

    for chunk_idx, chunk in enumerate(pd.read_csv(path, chunksize=chunksize, low_memory=False)):
        rows += len(chunk)
        missing = missing.add(chunk.isna().sum(), fill_value=0).astype("int64")
        update_moments(moments, chunk, numeric_cols)
        for col in unique_cols:
            unique_values[col].update(chunk[col].dropna().unique().tolist())

        quality_counts["inning_lt_1"] += int(chunk["inning"].lt(1).sum())
        quality_counts["balls_outside_0_3"] += int((~chunk["balls_before"].between(0, 3)).sum())
        quality_counts["strikes_outside_0_2"] += int((~chunk["strikes_before"].between(0, 2)).sum())
        quality_counts["outs_outside_0_2"] += int((~chunk["outs_before"].between(0, 2)).sum())
        quality_counts["extension_le_0"] += int(chunk["extension"].le(0).sum())
        quality_counts["release_speed_lt_80"] += int(chunk["rel_speed"].lt(80).sum())
        quality_counts["spin_rate_lt_500"] += int(chunk["spin_rate"].lt(500).sum())

        for (season, pitch_group), part in chunk.groupby(["season", "pitch_type_group"], dropna=False):
            key = (int(season), str(pitch_group))
            count_store[key] += len(part)
            for metric in metrics:
                x = pd.to_numeric(part[metric], errors="coerce").dropna().astype("float64")
                if x.empty:
                    continue
                s = group_store[key][metric]
                s["n"] += int(x.size)
                s["sum"] += float(x.sum())
                s["sum2"] += float(np.square(x).sum())
        samples.append(chunk.sample(frac=sample_frac, random_state=20260812 + chunk_idx))

    records = []
    for key, n_rows in count_store.items():
        season, pitch_group = key
        row: dict[str, object] = {"season": season, "pitch_type_group": pitch_group, "rows": n_rows}
        for metric in metrics:
            s = group_store[key].get(metric)
            if not s or not s["n"]:
                row[f"{metric}_mean"] = np.nan
                row[f"{metric}_std"] = np.nan
                continue
            mean = s["sum"] / s["n"]
            var = max(0.0, (s["sum2"] - s["sum"] ** 2 / s["n"]) / max(s["n"] - 1, 1))
            row[f"{metric}_mean"] = mean
            row[f"{metric}_std"] = math.sqrt(var)
        records.append(row)
    return {
        "columns": columns,
        "dtypes": probe.dtypes.astype(str).to_dict(),
        "numeric_cols": numeric_cols,
        "rows": rows,
        "missing": missing,
        "moments": moments,
        "unique_values": unique_values,
        "by_group": pd.DataFrame(records).sort_values(["season", "pitch_type_group"]),
        "sample": pd.concat(samples, ignore_index=True),
        "quality_counts": quality_counts,
    }


def make_plots(
    output_dir: Path,
    groups: pd.DataFrame,
    correlations: pd.DataFrame,
    profile: pd.DataFrame,
    trackman_sample: pd.DataFrame,
    train_sample: pd.DataFrame,
    global_rate: float,
) -> None:
    plot_dir = output_dir / "figures"
    plot_dir.mkdir(parents=True, exist_ok=True)
    sns.set_theme(style="whitegrid", context="notebook")

    season = groups[groups["feature"].eq("season")].copy()
    season["value_num"] = pd.to_numeric(season["value"])
    season = season.sort_values("value_num")
    fig, ax = plt.subplots(figsize=(8, 4.8))
    sns.lineplot(data=season, x="value_num", y="success_rate", marker="o", ax=ax)
    ax.axhline(global_rate, color="gray", ls="--", lw=1, label="overall")
    ax.set(title="Control success rate by season", xlabel="Season", ylabel="Success rate")
    ax.legend()
    fig.tight_layout()
    fig.savefig(plot_dir / "01_success_rate_by_season.png", dpi=160)
    plt.close(fig)

    counts = groups[groups["feature"].eq("count_state")].copy()
    split = counts["value"].str.split("-", expand=True).astype(int)
    counts["balls"] = split[0]
    counts["strikes"] = split[1]
    heat = counts.pivot(index="balls", columns="strikes", values="success_rate")
    fig, ax = plt.subplots(figsize=(6.2, 5.0))
    sns.heatmap(heat, annot=True, fmt=".3f", cmap="RdYlBu", center=global_rate, ax=ax)
    ax.set(title="Control success rate by count", xlabel="Strikes before", ylabel="Balls before")
    fig.tight_layout()
    fig.savefig(plot_dir / "02_count_state_heatmap.png", dpi=160)
    plt.close(fig)

    miss = profile[profile["missing_count"].gt(0)].nlargest(18, "missing_rate").sort_values("missing_rate")
    fig, ax = plt.subplots(figsize=(9, 6.5))
    sns.barplot(data=miss, x="missing_rate", y="column", color="#4C78A8", ax=ax)
    ax.set(title="Highest missing rates in train", xlabel="Missing rate", ylabel="")
    ax.xaxis.set_major_formatter(lambda x, _: f"{x:.0%}")
    fig.tight_layout()
    fig.savefig(plot_dir / "03_missingness.png", dpi=160)
    plt.close(fig)

    corr = correlations.dropna().head(18).sort_values("pearson_with_target")
    colors = ["#E45756" if x < 0 else "#54A24B" for x in corr["pearson_with_target"]]
    fig, ax = plt.subplots(figsize=(9, 6.5))
    ax.barh(corr["feature"], corr["pearson_with_target"], color=colors)
    ax.axvline(0, color="black", lw=0.8)
    ax.set(title="Top numeric correlations with target", xlabel="Pearson correlation", ylabel="")
    fig.tight_layout()
    fig.savefig(plot_dir / "04_numeric_target_correlations.png", dpi=160)
    plt.close(fig)

    exp = groups[groups["feature"].eq("pitcher_history_n_bin")].copy()
    order = ["0", "1-9", "10-49", "50-199", "200-999", "1k-4,999", "5k+"]
    exp["value"] = pd.Categorical(exp["value"], order, ordered=True)
    exp = exp.sort_values("value")
    fig, ax1 = plt.subplots(figsize=(9, 5))
    ax2 = ax1.twinx()
    ax1.plot(exp["value"].astype(str), exp["success_rate"], color="#4C78A8", marker="o")
    ax2.bar(exp["value"].astype(str), exp["count"], alpha=0.22, color="#F58518")
    ax1.set(title="Target and volume by pitcher history size", xlabel="Prior pitch count", ylabel="Success rate")
    ax2.set_ylabel("Rows")
    fig.tight_layout()
    fig.savefig(plot_dir / "05_pitcher_history_bins.png", dpi=160)
    plt.close(fig)

    pitchers = groups[groups["feature"].eq("pitcher_id") & groups["count"].ge(300)].copy()
    pitchers["smoothed_rate"] = (
        pitchers["successes"] + 200 * global_rate
    ) / (pitchers["count"] + 200)
    fig, ax = plt.subplots(figsize=(8, 5.5))
    sns.scatterplot(
        data=pitchers,
        x="count",
        y="success_rate",
        hue="smoothed_rate",
        palette="viridis",
        s=35,
        alpha=0.65,
        legend=False,
        ax=ax,
    )
    ax.axhline(global_rate, color="gray", ls="--", lw=1)
    ax.set(xscale="log", title="Pitcher volume vs observed success rate (n >= 300)", xlabel="Train pitches (log)", ylabel="Success rate")
    fig.tight_layout()
    fig.savefig(plot_dir / "06_pitcher_heterogeneity.png", dpi=160)
    plt.close(fig)

    tm = trackman_sample[
        trackman_sample["pitch_type_group"].isin(["fastball", "breaking", "offspeed"])
    ].copy()
    tm = tm.dropna(subset=["rel_speed", "spin_rate"])
    if len(tm) > 100_000:
        tm = tm.sample(100_000, random_state=42)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8))
    sns.boxplot(data=tm, x="pitch_type_group", y="rel_speed", showfliers=False, ax=axes[0])
    sns.boxplot(data=tm, x="pitch_type_group", y="spin_rate", showfliers=False, ax=axes[1])
    axes[0].set(title="Release speed by pitch group", xlabel="", ylabel="km/h")
    axes[1].set(title="Spin rate by pitch group", xlabel="", ylabel="rpm")
    fig.tight_layout()
    fig.savefig(plot_dir / "07_trackman_pitch_groups.png", dpi=160)
    plt.close(fig)

    situation_features = ["inning_group", "num_runners_on", "li_bin", "pitcher_score_diff_bin"]
    situation_orders = {
        "inning_group": ["1-3", "4-6", "7-9", "10+"],
        "num_runners_on": ["0", "1", "2", "3"],
        "li_bin": ["<0.5", "0.5-<1", "1-<1.5", "1.5-<2", "2-<3", "3-<5", "5+"],
        "pitcher_score_diff_bin": ["<=-7", "-6:-5", "-4:-3", "-2", "-1", "0:1", "2:3", "4:5", "6+"],
    }
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    for ax, feature in zip(axes.flat, situation_features):
        part = groups[groups["feature"].eq(feature)].copy()
        part["value"] = pd.Categorical(part["value"], situation_orders[feature], ordered=True)
        part = part.sort_values("value")
        sns.barplot(data=part, x="value", y="rate_minus_global", color="#72B7B2", ax=ax)
        ax.axhline(0, color="gray", ls="--", lw=1)
        ax.set(title=feature, xlabel="", ylabel="Difference from overall rate")
        ax.yaxis.set_major_formatter(lambda x, _: f"{x * 100:+.1f}pp")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(plot_dir / "08_situation_rates.png", dpi=160)
    plt.close(fig)

    season_type = groups[groups["feature"].eq("season_game_type")].copy()
    split = season_type["value"].str.split("-", expand=True)
    season_type["season"] = split[0].astype(int)
    season_type["game_type"] = split[1]
    season_type = season_type.sort_values("season")
    fig, ax = plt.subplots(figsize=(8.5, 5.2))
    sns.lineplot(
        data=season_type,
        x="season",
        y="success_rate",
        hue="game_type",
        marker="o",
        ax=ax,
    )
    ax.set(title="Success-rate drift differs sharply by game type", xlabel="Season", ylabel="Success rate")
    fig.tight_layout()
    fig.savefig(plot_dir / "09_season_game_type_drift.png", dpi=160)
    plt.close(fig)

    calibration = train_sample.dropna(subset=["asof_pitcher_success_rate"]).copy()
    calibration["history_rate_bin"] = pd.cut(
        calibration["asof_pitcher_success_rate"],
        bins=np.linspace(0.35, 0.70, 15),
        include_lowest=True,
    )
    calibration = (
        calibration.groupby(["season", "history_rate_bin"], observed=True)
        .agg(
            history_rate=("asof_pitcher_success_rate", "mean"),
            observed_rate=(TARGET, "mean"),
            rows=(TARGET, "size"),
        )
        .reset_index()
    )
    calibration = calibration[calibration["rows"].ge(150)]
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.lineplot(
        data=calibration,
        x="history_rate",
        y="observed_rate",
        hue="season",
        palette="viridis",
        marker="o",
        ax=ax,
    )
    low = min(calibration["history_rate"].min(), calibration["observed_rate"].min())
    high = max(calibration["history_rate"].max(), calibration["observed_rate"].max())
    ax.plot([low, high], [low, high], color="gray", ls="--", lw=1, label="identity")
    ax.set(
        title="Prior pitcher rate vs next-pitch outcome by season",
        xlabel="Mean as-of pitcher success rate",
        ylabel="Observed target rate",
    )
    fig.tight_layout()
    fig.savefig(plot_dir / "10_asof_rate_temporal_calibration.png", dpi=160)
    plt.close(fig)


def build_test_coverage(test: pd.DataFrame, unique_values: dict[str, set]) -> pd.DataFrame:
    records = []
    for col in [
        "pitcher_id",
        "batter_id",
        "pitcher_team_id",
        "batter_team_id",
        "game_type",
        "base_state",
        "pitcher_hand",
        "batter_hand",
    ]:
        seen = test[col].isin(unique_values[col])
        records.append(
            {
                "feature": col,
                "test_sample_rows": len(test),
                "seen_in_train_rows": int(seen.sum()),
                "unseen_in_train_rows": int((~seen).sum()),
                "seen_rate": float(seen.mean()),
                "note": "Public test has only 5 format-check rows; not a drift estimate.",
            }
        )
    return pd.DataFrame(records)


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    train_path = args.data_dir / "train.csv"
    test_path = args.data_dir / "test.csv"
    trackman_path = args.data_dir / "trackman_history.csv"
    print(f"Scanning train: {train_path}", flush=True)
    train = scan_train(train_path, args.chunksize, args.sample_frac)
    print(f"Train rows: {train['rows']:,}", flush=True)

    global_rate = train["target_sum"] / train["rows"]
    profile = moments_table(
        train["columns"],
        train["dtype_probe"],
        train["rows"],
        train["missing"],
        train["moments"],
        train["unique_values"],
        train["unique_row_ids"],
    )
    correlations = correlation_table(train["correlations"])
    groups = group_table(train["group_stats"], global_rate)
    season_numeric = season_numeric_table(train["season_numeric"])

    profile.to_csv(args.output_dir / "train_column_profile.csv", index=False, encoding="utf-8-sig")
    correlations.to_csv(args.output_dir / "numeric_target_correlations.csv", index=False, encoding="utf-8-sig")
    groups.to_csv(args.output_dir / "group_target_rates.csv", index=False, encoding="utf-8-sig")
    season_numeric.to_csv(args.output_dir / "season_numeric_stats.csv", index=False, encoding="utf-8-sig")

    missing_season = pd.DataFrame(train["missing_by_season"]).T.sort_index().astype("float64")
    missing_season.index.name = "season"
    season_counts = groups[groups["feature"].eq("season")].set_index("value")["count"]
    for season in missing_season.index:
        missing_season.loc[season] = missing_season.loc[season] / int(season_counts.loc[str(season)])
    missing_season.to_csv(args.output_dir / "missing_rate_by_season.csv", encoding="utf-8-sig")

    invariant_rows = []
    for name, s in train["invariants"].items():
        invariant_rows.append(
            {
                "check": name,
                "eligible_rows": s["eligible"],
                "tolerance": s["tolerance"],
                "violations": s["violations"],
                "violation_rate": s["violations"] / s["eligible"] if s["eligible"] else np.nan,
                "mean_abs_error": s["abs_error_sum"] / s["eligible"] if s["eligible"] else np.nan,
                "max_abs_error": s["max_abs_error"],
            }
        )
    pd.DataFrame(invariant_rows).to_csv(
        args.output_dir / "invariant_checks.csv", index=False, encoding="utf-8-sig"
    )

    test = pd.read_csv(test_path)
    coverage = build_test_coverage(test, train["unique_values"])
    coverage.to_csv(args.output_dir / "test_sample_train_coverage.csv", index=False, encoding="utf-8-sig")

    print(f"Scanning Trackman: {trackman_path}", flush=True)
    trackman = scan_trackman(trackman_path, args.chunksize, args.trackman_sample_frac)
    print(f"Trackman rows: {trackman['rows']:,}", flush=True)
    trackman_profile = moments_table(
        trackman["columns"],
        trackman["dtypes"],
        trackman["rows"],
        trackman["missing"],
        trackman["moments"],
        trackman["unique_values"],
        unique_row_ids=trackman["rows"],
    )
    trackman_profile.to_csv(
        args.output_dir / "trackman_column_profile.csv", index=False, encoding="utf-8-sig"
    )
    trackman["by_group"].to_csv(
        args.output_dir / "trackman_by_season_pitch_group.csv", index=False, encoding="utf-8-sig"
    )
    pd.DataFrame(
        [
            {"check": name, "violations": count, "rate": count / trackman["rows"]}
            for name, count in trackman["quality_counts"].items()
        ]
    ).to_csv(args.output_dir / "trackman_quality_checks.csv", index=False, encoding="utf-8-sig")

    make_plots(
        args.output_dir,
        groups,
        correlations,
        profile,
        trackman["sample"],
        train["sample"],
        global_rate,
    )

    joinability = {
        "main_pitcher_ids": len(train["unique_values"]["pitcher_id"]),
        "trackman_pitcher_ids": len(trackman["unique_values"]["pitcher_trackman_id"]),
        "direct_pitcher_id_intersection": len(
            train["unique_values"]["pitcher_id"].intersection(
                trackman["unique_values"]["pitcher_trackman_id"]
            )
        ),
        "main_batter_ids": len(train["unique_values"]["batter_id"]),
        "trackman_batter_ids": len(trackman["unique_values"]["batter_trackman_id"]),
        "direct_batter_id_intersection": len(
            train["unique_values"]["batter_id"].intersection(
                trackman["unique_values"]["batter_trackman_id"]
            )
        ),
        "interpretation": "The identifier namespaces do not directly overlap; no row/player direct join key is provided.",
    }
    (args.output_dir / "trackman_joinability.json").write_text(
        json.dumps(joinability, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    summary = {
        "train_rows": train["rows"],
        "train_columns": len(train["columns"]),
        "positive_rows": train["target_sum"],
        "negative_rows": train["rows"] - train["target_sum"],
        "global_success_rate": global_rate,
        "unique_row_ids": train["unique_row_ids"],
        "duplicate_row_ids": train["duplicate_row_ids"],
        "invalid_row_ids": train["invalid_row_ids"],
        "trackman_rows": trackman["rows"],
        "trackman_columns": len(trackman["columns"]),
        "public_test_rows": len(test),
        "sampling": {
            "train_fraction_for_visual_only": args.sample_frac,
            "trackman_fraction_for_visual_only": args.trackman_sample_frac,
            "all_csv_tables_are_exact": True,
        },
    }
    (args.output_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(f"Saved outputs to: {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
