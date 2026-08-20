"""Cost-controlled strict-temporal V4 direct-model experiments EXP017-EXP019.

Outer validation predictions never use their validation labels for early stopping.
Tree count is selected on the latest earlier season, then a fresh model is fit on
all outer-training seasons with that fixed tree count.
"""
from __future__ import annotations

import gc
import json
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier
from sklearn.metrics import log_loss, roc_auc_score

from result import (
    TARGET_COL, build_features, initial_feature_config, make_post_regime_f_prior_map,
    read_compact_csv,
)
from train_v2 import recency_weights
from train_v21 import make_feature_sets_v21

try:
    import resource
except ModuleNotFoundError:  # Windows does not provide the Unix-only module.
    resource = None

YEARS = (2022, 2023, 2024)
WEIGHTS = {2022: 0.15, 2023: 0.30, 2024: 0.55}
SEED = 42


def peak_memory_mb() -> float:
    """Return peak resident memory when the platform exposes it."""
    if resource is None:
        return float("nan")
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return rss / (1024 * 1024) if os.name == "darwin" else rss / 1024


@dataclass(frozen=True)
class Plan:
    experiment_id: str
    parent_experiment: str
    description: str
    hypothesis: str
    exact_change: str
    expected_mechanism: str
    feature_set: str


PLANS = [
    Plan(
        "EXP017", "EXP002", "Strict-temporal V3 direct baseline",
        "Removing outer-validation early stopping gives an honest reference while retaining V3's strongest simple architecture.",
        "Use the same 87 legacy-global columns and CatBoost defaults. Select tree count on the latest earlier season, refit on the complete outer training window, then predict the untouched outer season.",
        "Eliminate validation-label iteration leakage.", "v3_legacy_87",
    ),
    Plan(
        "EXP018", "EXP017", "Strict direct model with all 102 existing base features",
        "V3 unnecessarily excludes 15 already-built temporal-prior, reliability, and shrinkage features from the direct model.",
        "Replace the 87-column list with all 102 base features. Keep the strict temporal fitting protocol and all CatBoost settings fixed.",
        "Expose sample-size-aware command skill and prior-relative effects directly to the target model.", "base_all_102",
    ),
    Plan(
        "EXP019", "EXP018", "Expanded reliability and recent-career direct model",
        "Rates other than success need explicit reliability, posterior shrinkage, and uncertainty; recent-career disagreements should be discounted for low-history pitchers.",
        "Start from all 102 base features. Add generic-prior posterior/deviation/uncertainty features for pitcher reverse, middle, ball, strike, batter middle/success, and pitch mix, plus reliability-weighted recent-career form and disagreement features.",
        "Estimate latent skill under unequal sample sizes and suppress noisy recent-form deviations.", "base_102_plus_reliability",
    ),
]


def add_reliability_features(raw: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    out = base.copy()
    specs = [
        ("pitcher_reverse", "asof_pitcher_reverse_rate", "asof_pitcher_n", 0.23, 200.0),
        ("pitcher_middle", "asof_pitcher_middle_rate", "asof_pitcher_n", 0.15, 200.0),
        ("pitcher_ball", "asof_pitcher_ball_rate", "asof_pitcher_n", 0.37, 200.0),
        ("pitcher_strike", "asof_pitcher_strike_rate", "asof_pitcher_n", 0.44, 200.0),
        ("batter_success", "asof_batter_success_rate", "asof_batter_n", 0.50, 200.0),
        ("batter_middle", "asof_batter_middle_rate", "asof_batter_n", 0.15, 200.0),
        ("pitchmix_fastball", "asof_pitcher_fastball_rate", "asof_pitcher_pitchmix_n", 1/3, 200.0),
        ("pitchmix_breaking", "asof_pitcher_breaking_rate", "asof_pitcher_pitchmix_n", 1/3, 200.0),
        ("pitchmix_offspeed", "asof_pitcher_offspeed_rate", "asof_pitcher_pitchmix_n", 1/3, 200.0),
    ]
    for name, rate_col, n_col, prior, alpha in specs:
        n = raw[n_col].astype("float32").clip(lower=0)
        rate = raw[rate_col].astype("float32").fillna(prior).clip(0, 1)
        reliability = n / (n + alpha)
        out[f"v4_{name}_posterior"] = (reliability * rate + (1 - reliability) * prior).astype("float32")
        out[f"v4_{name}_deviation"] = (rate - prior).astype("float32")
        out[f"v4_{name}_uncertainty"] = np.sqrt((rate * (1 - rate) + 0.25) / (n + alpha + 1)).astype("float32")
    pitcher_reliability = (raw["asof_pitcher_n"].astype("float32") / (raw["asof_pitcher_n"].astype("float32") + 200.0)).fillna(0)
    career_success = raw["asof_pitcher_success_rate"].astype("float32")
    career_middle = raw["asof_pitcher_middle_rate"].astype("float32")
    success_deltas = []
    middle_deltas = []
    for window in (1, 3, 5):
        sd = raw[f"asof_pitcher_prev{window}_game_success_rate"].astype("float32") - career_success
        md = raw[f"asof_pitcher_prev{window}_game_middle_rate"].astype("float32") - career_middle
        out[f"v4_success_form_{window}_reliable"] = (sd * pitcher_reliability).astype("float32")
        out[f"v4_middle_form_{window}_reliable"] = (md * pitcher_reliability).astype("float32")
        success_deltas.append(sd)
        middle_deltas.append(md)
    out["v4_success_career_recent_disagreement"] = (pd.concat(success_deltas, axis=1).abs().mean(axis=1) * pitcher_reliability).astype("float32")
    out["v4_middle_career_recent_disagreement"] = (pd.concat(middle_deltas, axis=1).abs().mean(axis=1) * pitcher_reliability).astype("float32")
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    return out


def parameters(iterations: int, task_type: str, verbose: int) -> dict[str, Any]:
    p: dict[str, Any] = {
        "iterations": int(iterations), "depth": 8, "learning_rate": 0.04,
        "l2_leaf_reg": 8.0, "random_strength": 0.5, "random_seed": SEED,
        "loss_function": "Logloss", "eval_metric": "BrierScore",
        "allow_writing_files": False, "thread_count": -1,
        "verbose": verbose, "task_type": task_type,
    }
    if task_type == "GPU":
        p["devices"] = os.environ.get("BASEBALL_GPU_DEVICES", "0")
    return p


def select_iteration(X, y, seasons, columns, categorical, outer_year, sample_weight, task_type) -> tuple[int, dict]:
    inner_year = outer_year - 1
    train_mask = seasons < inner_year
    valid_mask = seasons == inner_year
    model = CatBoostClassifier(**parameters(700, task_type, 100))
    model.fit(
        X.loc[train_mask, columns], y[train_mask], cat_features=categorical,
        sample_weight=sample_weight[train_mask],
        eval_set=(X.loc[valid_mask, columns], y[valid_mask]),
        early_stopping_rounds=80, use_best_model=True,
    )
    best = model.get_best_iteration() + 1
    if best <= 0:
        best = 700
    info = {"inner_train_rows": int(train_mask.sum()), "inner_valid_year": inner_year,
            "inner_valid_rows": int(valid_mask.sum()), "selected_iterations": int(best),
            "inner_best_score": model.get_best_score()}
    del model
    gc.collect()
    return int(best), info


def fit_outer(X, y, seasons, columns, categorical, outer_year, iterations, sample_weight, task_type, model_path):
    train_mask = seasons < outer_year
    valid_mask = seasons == outer_year
    model = CatBoostClassifier(**parameters(iterations, task_type, 100))
    model.fit(X.loc[train_mask, columns], y[train_mask], cat_features=categorical,
              sample_weight=sample_weight[train_mask])
    pred = model.predict_proba(X.loc[valid_mask, columns])[:, 1].astype("float32")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(str(model_path))
    del model
    gc.collect()
    return pred


def metrics(y, p, seasons, types) -> dict[str, float]:
    out = {}
    for year in YEARS:
        mask = (seasons == year) & np.isfinite(p)
        if mask.any(): out[f"brier_{year}"] = float(np.mean((p[mask] - y[mask]) ** 2))
    if all(f"brier_{year}" in out for year in YEARS):
        out["weighted_brier"] = sum(WEIGHTS[year] * out[f"brier_{year}"] for year in YEARS)
    latest = (seasons == 2024) & np.isfinite(p)
    if latest.any():
        for kind in ("R", "F"):
            mask = latest & (types == kind)
            out[f"brier_{kind}"] = float(np.mean((p[mask] - y[mask]) ** 2))
        out["logloss_2024"] = float(log_loss(y[latest], p[latest], labels=[0, 1]))
        out["auc_2024"] = float(roc_auc_score(y[latest], p[latest]))
        out["prediction_mean_2024"] = float(p[latest].mean())
        out["target_mean_2024"] = float(y[latest].mean())
    return out


def run(root: Path, data_dir: Path) -> pd.DataFrame:
    started = time.time()
    run_dir = root / "runs/research/strict_direct"
    run_dir.mkdir(parents=True, exist_ok=True)
    # Record hypotheses before fitting.
    with (run_dir / "experiment_plan.jsonl").open("w", encoding="utf-8") as f:
        for plan in PLANS: f.write(json.dumps(asdict(plan), ensure_ascii=False) + "\n")

    train = read_compact_csv(data_dir / "train.csv", train=True)
    config = initial_feature_config(train)
    config["version"] = 5
    config["post_regime_f_prior_map"] = make_post_regime_f_prior_map(train)
    base = build_features(train, config)
    legacy_sets, legacy_cats = make_feature_sets_v21(list(base.columns), list(config["categorical_columns"]))
    feature_frames = {"EXP017": base, "EXP018": base}
    columns = {"EXP017": legacy_sets["legacy_global"], "EXP018": list(base.columns)}
    categorical = {
        "EXP017": legacy_cats["legacy_global"],
        "EXP018": [c for c in config["categorical_columns"] if c in base.columns],
    }
    y = train[TARGET_COL].to_numpy(dtype="int8")
    seasons = train["season"].to_numpy(dtype="int16")
    types = train["game_type"].astype(str).to_numpy()
    task_type = os.environ.get("BASEBALL_TASK_TYPE", "GPU").upper()
    predictions = {p.experiment_id: np.full(len(train), np.nan, dtype="float32") for p in PLANS}
    records = []
    fold_logs = {}

    # Cheap 2024 screens for all three experiments.
    screen_metrics = {}
    for plan in PLANS:
        exp_started = time.time()
        if plan.experiment_id == "EXP019":
            feature_frames["EXP019"] = add_reliability_features(train, base)
            columns["EXP019"] = list(feature_frames["EXP019"].columns)
            categorical["EXP019"] = categorical["EXP018"]
        X = feature_frames[plan.experiment_id]
        sample_weight = np.ones(len(train), dtype="float32")
        outer_train = seasons < 2024
        sample_weight[outer_train] = recency_weights(seasons[outer_train], 2024, 3.0)
        selected, inner_info = select_iteration(X, y, seasons, columns[plan.experiment_id], categorical[plan.experiment_id], 2024, sample_weight, task_type)
        pred = fit_outer(X, y, seasons, columns[plan.experiment_id], categorical[plan.experiment_id], 2024, selected, sample_weight, task_type, run_dir/f"models/{plan.experiment_id}_2024.cbm")
        predictions[plan.experiment_id][seasons == 2024] = pred
        screen_metrics[plan.experiment_id] = metrics(y, predictions[plan.experiment_id], seasons, types)
        fold_logs[plan.experiment_id] = {"2024": inner_info}
        records.append({"experiment_id": plan.experiment_id, "stage": "screen_2024", **asdict(plan),
                        "feature_count": len(columns[plan.experiment_id]), **screen_metrics[plan.experiment_id],
                        "runtime_seconds": time.time()-exp_started,
                        "peak_memory_mb": peak_memory_mb()})

    baseline_2024 = screen_metrics["EXP017"]["brier_2024"]
    # Full CV only candidates that improve the strict 2024 baseline by at least 1e-5.
    promoted = ["EXP017"] + [p.experiment_id for p in PLANS[1:] if screen_metrics[p.experiment_id]["brier_2024"] <= baseline_2024 - 1e-5]
    for exp_id in promoted:
        plan = next(p for p in PLANS if p.experiment_id == exp_id)
        exp_started = time.time()
        X = feature_frames[exp_id]
        for outer_year in (2022, 2023):
            sample_weight = np.ones(len(train), dtype="float32")
            outer_train = seasons < outer_year
            sample_weight[outer_train] = recency_weights(seasons[outer_train], outer_year, 3.0)
            selected, inner_info = select_iteration(X, y, seasons, columns[exp_id], categorical[exp_id], outer_year, sample_weight, task_type)
            pred = fit_outer(X, y, seasons, columns[exp_id], categorical[exp_id], outer_year, selected, sample_weight, task_type, run_dir/f"models/{exp_id}_{outer_year}.cbm")
            predictions[exp_id][seasons == outer_year] = pred
            fold_logs[exp_id][str(outer_year)] = inner_info
        result = metrics(y, predictions[exp_id], seasons, types)
        decision = "strict_baseline" if exp_id == "EXP017" else ("keep" if result.get("weighted_brier", 1) < 1 else "evaluated")
        records.append({"experiment_id": exp_id, "stage": "full_temporal_cv", **asdict(plan),
                        "feature_count": len(columns[exp_id]), **result,
                        "runtime_seconds": time.time()-exp_started,
                        "peak_memory_mb": peak_memory_mb(),
                        "decision": decision})

    for plan in PLANS:
        if plan.experiment_id not in promoted:
            records.append({"experiment_id": plan.experiment_id, "stage": "decision", **asdict(plan),
                            "feature_count": len(columns[plan.experiment_id]),
                            "brier_2024": screen_metrics[plan.experiment_id]["brier_2024"],
                            "decision": "reject_after_2024_screen",
                            "notes": f"Did not beat strict EXP017 by 0.00001; delta={screen_metrics[plan.experiment_id]['brier_2024']-baseline_2024:+.9f}."})

    oof = pd.DataFrame({"row_id": train["row_id"], "season": seasons, "game_type": types,
                        TARGET_COL: y, **{f"pred_{k}": v for k,v in predictions.items()}})
    oof.to_parquet(run_dir / "oof_predictions.parquet", index=False)
    pd.DataFrame(records).to_csv(run_dir / "experiments.csv", index=False)
    (run_dir / "fold_parameters.json").write_text(json.dumps(fold_logs, ensure_ascii=False, indent=2))
    summary = {"promoted_to_full_cv": promoted, "baseline_brier_2024": baseline_2024,
               "elapsed_seconds": time.time()-started, "task_type": task_type,
               "data_dir": str(data_dir), "strict_outer_predictions": True}
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    return pd.DataFrame(records)


if __name__ == "__main__":
    root = Path(__file__).resolve().parent
    data_dir = Path(os.environ.get("BASEBALL_DATA_DIR", "/mnt/d/baseball/open/data"))
    run(root, data_dir)
