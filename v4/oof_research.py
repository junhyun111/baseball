"""Evaluate stored temporal OOF predictions and maintain the research registry.

These experiments are post-hoc diagnostics over the committed 2022-2024 OOF
predictions. They do not turn the selected result into an unbiased holdout
estimate. Fresh V4 training uses a separate, nested temporal protocol.
"""
from __future__ import annotations

import json
import resource
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import log_loss, roc_auc_score

from v3.ensemble import (
    apply_calibrator,
    fit_affine,
    fit_beta,
    fit_simplex_brier,
    season_sample_weights,
)

YEARS = (2022, 2023, 2024)
YEAR_WEIGHTS = {2022: 0.15, 2023: 0.30, 2024: 0.55}
EPSILON = 1e-5


@dataclass(frozen=True)
class Experiment:
    experiment_id: str
    parent_experiment: str
    description: str
    hypothesis: str
    feature_set: str
    model_type: str
    parameters: dict
    expected_mechanism: str
    expected_affected_segments: str
    sources: tuple[str, ...]
    blend: str = "simplex"
    calibration: str = "identity"


def brier(y: np.ndarray, p: np.ndarray) -> float:
    return float(np.mean(np.square(np.asarray(y) - np.asarray(p))))


def _fit_logit_blend(y: np.ndarray, matrix: np.ndarray, penalty: float = 0.02) -> np.ndarray:
    """Fit a conservative logistic stack on logits of source probabilities."""
    x = np.log(np.clip(matrix, EPSILON, 1 - EPSILON) / np.clip(1 - matrix, EPSILON, 1))
    target = np.asarray(y, dtype="float64")

    def objective(params: np.ndarray) -> tuple[float, np.ndarray]:
        z = np.clip(params[0] + x @ params[1:], -30, 30)
        p = 1.0 / (1.0 + np.exp(-z))
        loss = float(np.mean((p - target) ** 2) + penalty * np.sum((params[1:] - 1 / x.shape[1]) ** 2))
        common = 2 * (p - target) * p * (1 - p) / len(target)
        grad = np.r_[common.sum(), x.T @ common + 2 * penalty * (params[1:] - 1 / x.shape[1])]
        return loss, grad

    initial = np.r_[0.0, np.full(x.shape[1], 1 / x.shape[1])]
    fitted = minimize(lambda z: objective(z)[0], initial, jac=lambda z: objective(z)[1], method="L-BFGS-B")
    return fitted.x if fitted.success else initial


def _apply_logit_blend(matrix: np.ndarray, params: np.ndarray) -> np.ndarray:
    x = np.log(np.clip(matrix, EPSILON, 1 - EPSILON) / np.clip(1 - matrix, EPSILON, 1))
    z = np.clip(params[0] + x @ params[1:], -30, 30)
    return np.clip(1 / (1 + np.exp(-z)), EPSILON, 1 - EPSILON)


def _fit_type_affine(y: np.ndarray, p: np.ndarray, game_type: np.ndarray) -> dict:
    base = fit_affine(y, p)
    calibrated = base["slope"] * p + base["intercept"]
    offsets = {}
    for kind in ("R", "F"):
        mask = game_type == kind
        offsets[kind] = float(np.clip(np.mean(y[mask] - calibrated[mask]), -0.04, 0.04)) if mask.any() else 0.0
    return {**base, "method": "type_affine", "game_type_offsets": offsets}


def _apply_calibration(p: np.ndarray, config: dict, game_type: np.ndarray) -> np.ndarray:
    if config.get("method") != "type_affine":
        return apply_calibrator(p, config)
    offsets = np.array([config["game_type_offsets"].get(str(x), 0.0) for x in game_type])
    raw = config["slope"] * p + config["intercept"] + offsets
    eta = float(config.get("eta", 1.0))
    return np.clip((1 - eta) * p + eta * raw, EPSILON, 1 - EPSILON)


def load_oof(root: Path) -> pd.DataFrame:
    v3 = pd.read_parquet(root / "runs/v3/artifacts/oof_predictions.parquet")
    v2 = pd.read_parquet(root / "artifacts/oof_predictions.parquet")
    v2_columns = [
        "row_id", "control_success", "pred_legacy_global", "pred_regular_expert",
        "pred_futures_expert", "pred_game_type",
    ]
    merged = v3.merge(v2[v2_columns], on="row_id", how="left", validate="one_to_one", suffixes=("", "_v2"))
    if not np.array_equal(merged["control_success"], merged["control_success_v2"]):
        raise ValueError("V2 and V3 OOF targets do not match")
    merged.drop(columns="control_success_v2", inplace=True)
    if merged[["pred_legacy_global", "pred_game_type"]].isna().any().any():
        raise ValueError("Missing matched V2 OOF predictions")
    return merged


def walk_forward_prediction(frame: pd.DataFrame, experiment: Experiment) -> tuple[np.ndarray, list[dict]]:
    sources = list(experiment.sources)
    matrix = frame[sources].to_numpy(dtype="float64")
    y = frame["control_success"].to_numpy(dtype="float64")
    seasons = frame["season"].to_numpy()
    types = frame["game_type"].astype(str).to_numpy()
    output = np.full(len(frame), np.nan, dtype="float64")
    fold_parameters: list[dict] = []

    for year in YEARS:
        current = seasons == year
        history = seasons < year
        if not history.any() or len(sources) == 1:
            blended = matrix[current, 0]
            blend_config: dict = {"method": "fallback_first_source", "weights": [1.0] + [0.0] * (len(sources) - 1)}
        elif experiment.blend == "mean":
            blended = matrix[current].mean(axis=1)
            blend_config = {"method": "mean", "weights": [1 / len(sources)] * len(sources)}
        elif experiment.blend == "type_simplex":
            blended = np.empty(current.sum(), dtype="float64")
            current_types = types[current]
            weight_map = {}
            for kind in ("R", "F"):
                hist_kind = history & (types == kind)
                cur_kind = current_types == kind
                if not hist_kind.any():
                    weights = np.r_[1.0, np.zeros(len(sources) - 1)]
                else:
                    weights = fit_simplex_brier(y[hist_kind], matrix[hist_kind], season_sample_weights(seasons[hist_kind]))
                blended[cur_kind] = matrix[current][cur_kind] @ weights
                weight_map[kind] = weights.tolist()
            blend_config = {"method": "type_simplex", "weights_by_type": weight_map}
        elif experiment.blend == "logit":
            params = _fit_logit_blend(y[history], matrix[history])
            blended = _apply_logit_blend(matrix[current], params)
            blend_config = {"method": "logit", "parameters": params.tolist()}
        else:
            weights = fit_simplex_brier(y[history], matrix[history], season_sample_weights(seasons[history]))
            blended = matrix[current] @ weights
            blend_config = {"method": "simplex", "weights": weights.tolist()}

        calibration_config: dict = {"method": "identity", "eta": 0.0}
        if experiment.calibration != "identity" and history.any():
            # Fit the calibrator on historical walk-forward predictions already available.
            usable = history & np.isfinite(output)
            if usable.sum() >= 1000:
                if experiment.calibration == "affine":
                    calibration_config = {**fit_affine(y[usable], output[usable]), "eta": 1.0}
                elif experiment.calibration == "affine_075":
                    calibration_config = {**fit_affine(y[usable], output[usable]), "eta": 0.75}
                elif experiment.calibration == "beta":
                    calibration_config = {**fit_beta(y[usable], output[usable]), "eta": 1.0}
                elif experiment.calibration == "type_affine":
                    calibration_config = {**_fit_type_affine(y[usable], output[usable], types[usable]), "eta": 0.75}
                else:
                    raise ValueError(f"Unknown calibration: {experiment.calibration}")
        output[current] = _apply_calibration(blended, calibration_config, types[current])
        fold_parameters.append({"season": year, "blend": blend_config, "calibration": calibration_config})
    return output, fold_parameters


def evaluate(frame: pd.DataFrame, prediction: np.ndarray) -> dict:
    y = frame["control_success"].to_numpy(dtype="float64")
    seasons = frame["season"].to_numpy()
    types = frame["game_type"].astype(str).to_numpy()
    metrics: dict[str, float] = {}
    for year in YEARS:
        mask = seasons == year
        metrics[f"brier_{year}"] = brier(y[mask], prediction[mask])
    metrics["weighted_brier"] = sum(YEAR_WEIGHTS[year] * metrics[f"brier_{year}"] for year in YEARS)
    latest = seasons == 2024
    for kind in ("R", "F"):
        mask = latest & (types == kind)
        metrics[f"brier_{kind}"] = brier(y[mask], prediction[mask])
    p = np.clip(prediction[latest], EPSILON, 1 - EPSILON)
    metrics["logloss_2024"] = float(log_loss(y[latest], p, labels=[0, 1]))
    metrics["auc_2024"] = float(roc_auc_score(y[latest], p))
    metrics["prediction_mean_2024"] = float(np.mean(p))
    metrics["target_mean_2024"] = float(np.mean(y[latest]))
    return metrics


def _plans() -> list[Experiment]:
    direct = "pred_direct"
    formula = "pred_structured_formula"
    logit = "pred_structured_logit"
    physics = "pred_physics"
    futures = "pred_futures_anchor"
    plans = [
        Experiment("EXP001", "", "Exact committed V3 calibrated walk-forward reproduction", "The committed artifacts reproduce the claimed V3 reference exactly.", "V3 full", "stored prediction", {}, "Artifact consistency check", "all folds", ("pred_calibrated_walk_forward",), calibration="identity"),
        Experiment("EXP002", "EXP001", "V3 direct-only reference", "Most V3 complexity adds little independent Brier signal; direct-only will be competitive.", "87-column legacy direct", "CatBoost OOF", {}, "Remove expert variance", "R and F", (direct,)),
        Experiment("EXP003", "EXP002", "Direct plus structured failure formula", "Failure decomposition contains weak residual signal not present in direct probabilities.", "stored OOF sources", "simplex blend", {}, "Blend different failure-mode errors", "high predicted failure", (direct, formula)),
        Experiment("EXP004", "EXP002", "Direct plus learned failure-logit source", "A learned failure combiner is better calibrated than the deterministic formula.", "stored OOF sources", "simplex blend", {}, "Correct formula misspecification", "failure-heavy pitches", (direct, logit)),
        Experiment("EXP005", "EXP002", "Direct plus physics expert", "Cohort TrackMan mechanics correct direct-model residuals.", "stored OOF sources", "simplex blend", {}, "Independent mechanics signal", "pitchers with mechanical drift", (direct, physics)),
        Experiment("EXP006", "EXP002", "Direct plus Futures router", "A post-regime F specialist improves F without changing R.", "stored OOF sources", "simplex blend", {}, "Domain adaptation", "F", (direct, futures)),
        Experiment("EXP007", "EXP002", "Direct plus formula, physics, and Futures", "Only the plausible nonzero V3 sources are needed; dropping failure-logit reduces noise.", "stored OOF sources", "simplex blend", {}, "Diverse residual correction", "F and failure-heavy pitches", (direct, formula, physics, futures)),
        Experiment("EXP008", "EXP001", "Rebuilt full V3 five-source walk-forward blend", "Refitting the exact full source set reproduces the pre-calibration V3 blend.", "stored OOF sources", "simplex blend", {}, "Exact source ablation control", "all folds", (direct, formula, logit, physics, futures)),
        Experiment("EXP009", "EXP002", "Temporally affine-calibrated direct", "The direct model's annual mean drift can be corrected without experts.", "87-column legacy direct", "CatBoost + affine", {}, "Correct slope/intercept drift", "2024 and F", (direct,), calibration="affine"),
        Experiment("EXP010", "EXP007", "Affine-calibrated compact V3 blend", "Calibration supplies most of the full stack's Brier gain after weak sources are removed.", "stored OOF sources", "simplex + affine", {}, "Blend resolution plus mean correction", "2024", (direct, formula, physics, futures), calibration="affine"),
        Experiment("EXP011", "EXP002", "V3 direct plus V2 legacy direct", "Two independently trained direct baselines have useful seed/training-path diversity.", "stored V2/V3 OOF", "simplex blend", {}, "Average uncorrelated training noise", "all folds", (direct, "pred_legacy_global")),
        Experiment("EXP012", "EXP011", "V3 direct plus V2 game-type expert", "The older R/F expert contains domain signal missed by V3 direct.", "stored V2/V3 OOF", "simplex blend", {}, "Cross-version domain diversity", "R/F", (direct, "pred_game_type")),
        Experiment("EXP013", "EXP012", "Game-type-specific source weights", "R and F require different ensemble weights after the 2023 F regime change.", "stored V2/V3 OOF", "per-type simplex", {}, "Domain-specific pooling", "F", (direct, formula, futures, "pred_game_type"), blend="type_simplex"),
        Experiment("EXP014", "EXP007", "Conservative logit-space stack", "Logit blending captures source scale differences better than a probability simplex.", "stored OOF sources", "regularized logit stack", {"penalty": 0.02}, "Correct source scale", "probability tails", (direct, formula, physics, futures), blend="logit"),
        Experiment("EXP015", "EXP010", "Game-type affine calibration", "Residual calibration bias differs between R and F and merits partial type offsets.", "stored OOF sources", "simplex + type affine", {"eta": 0.75}, "Correct domain prior shift conservatively", "F", (direct, formula, physics, futures), calibration="type_affine"),
        Experiment("EXP016", "EXP007", "Simple mean of plausible V3 sources", "A fixed equal blend is more stable than fold-fitted weights under drift.", "stored OOF sources", "mean blend", {}, "Reduce meta-model selection variance", "all folds", (direct, formula, physics, futures), blend="mean"),
    ]
    return plans


def run(root: Path) -> pd.DataFrame:
    started = time.time()
    research = root / "runs/research"
    research.mkdir(parents=True, exist_ok=True)
    plans = _plans()
    # Persist hypotheses before executing any experiment.
    with (research / "experiment_plan.jsonl").open("w", encoding="utf-8") as handle:
        for plan in plans:
            handle.write(json.dumps(asdict(plan), ensure_ascii=False) + "\n")

    frame = load_oof(root)
    rows = []
    predictions: dict[str, np.ndarray] = {}
    parameter_log: dict[str, list[dict]] = {}
    direct_metrics = None
    champion_weighted = float("inf")
    champion_id = ""
    for plan in plans:
        experiment_started = time.time()
        if plan.experiment_id == "EXP001":
            prediction = frame["pred_calibrated_walk_forward"].to_numpy(dtype="float64")
            fold_parameters = [{"method": "stored_exact"}]
        else:
            prediction, fold_parameters = walk_forward_prediction(frame, plan)
        metrics = evaluate(frame, prediction)
        if plan.experiment_id == "EXP002":
            direct_metrics = metrics
        delta_direct = metrics["weighted_brier"] - direct_metrics["weighted_brier"] if direct_metrics else np.nan
        delta_champion = metrics["weighted_brier"] - champion_weighted if np.isfinite(champion_weighted) else np.nan
        if plan.experiment_id == "EXP001":
            status = "reference"
        elif delta_direct > 0.00005:
            status = "rejected"
        elif delta_direct <= -0.00005 and metrics["brier_2024"] <= direct_metrics["brier_2024"] + 0.00002:
            status = "promoted_diagnostic"
        else:
            status = "interesting_diagnostic"
        if metrics["weighted_brier"] < champion_weighted:
            champion_weighted = metrics["weighted_brier"]
            champion_id = plan.experiment_id
        notes = "Post-hoc stored-OOF diagnostic; base folds used validation early stopping and candidate comparison is not an unbiased holdout."
        rows.append({
            **{k: v for k, v in asdict(plan).items() if k not in {"sources", "hypothesis", "expected_mechanism", "expected_affected_segments"}},
            "hypothesis": plan.hypothesis,
            "expected_mechanism": plan.expected_mechanism,
            "expected_affected_segments": plan.expected_affected_segments,
            "source_predictions": ",".join(plan.sources),
            **metrics,
            "runtime_seconds": time.time() - experiment_started,
            "peak_memory_mb": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024,
            "delta_weighted_vs_direct": delta_direct,
            "delta_weighted_vs_running_champion": delta_champion,
            "status": status,
            "notes": notes,
        })
        predictions[plan.experiment_id] = prediction.astype("float32")
        parameter_log[plan.experiment_id] = fold_parameters

    registry = pd.DataFrame(rows)
    registry.to_csv(research / "experiments.csv", index=False)
    pd.DataFrame({"row_id": frame["row_id"], **predictions}).to_parquet(research / "phase0_predictions.parquet", index=False)
    (research / "phase0_parameters.json").write_text(json.dumps(parameter_log, ensure_ascii=False, indent=2))
    summary = {
        "scope": "post-hoc stored OOF diagnostics",
        "rows": len(frame),
        "years": list(YEARS),
        "year_weights": YEAR_WEIGHTS,
        "best_diagnostic": champion_id,
        "best_weighted_brier": champion_weighted,
        "elapsed_seconds": time.time() - started,
        "warning": "Not an unbiased holdout because base folds use validation early stopping and experiments reuse 2022-2024 OOF.",
    }
    (research / "phase0_summary.json").write_text(json.dumps(summary, indent=2))
    return registry


if __name__ == "__main__":
    run(Path(__file__).resolve().parents[1])
