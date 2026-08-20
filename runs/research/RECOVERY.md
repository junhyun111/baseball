# Research recovery snapshot

Recovered without rerunning committed or Phase-0 experiments.

## Completed work
- Committed V3 full run: 1,475,092 rows, 186 total features, 8 CatBoost models, 1,179.55 seconds.
- Stored V3 calibrated walk-forward Brier: 2022 0.2434340861; 2023 0.2499957711; 2024 0.2479638457; weighted 0.2478939594.
- V3 direct Brier: 2022 0.2434340861; 2023 0.2499766754; 2024 0.2480897615; weighted 0.2479580741.
- Leakage audit: outer validation is used for base-model early stopping; blend/calibration candidates are selected on the reported folds; EB training priors use full training-window labels. Stored V3 is a 2025-selection reference, not an unbiased holdout estimate.
- Phase-0 post-hoc source ablation EXP001-EXP016 completed. Artifacts are in `runs/research/`.
- Best weighted diagnostic EXP013 (0.2478689) worsened 2024 to 0.2482419 and was not promoted.
- EXP007 compact sources (direct + formula + physics + Futures) scored 0.2480328 in 2024 and 0.2479179 weighted before calibration.
- Learned failure-logit, physics alone, V2 game-type expert, equal blend, and regularized logit blend were rejected or non-beneficial.

## Reusable artifacts
- `runs/v3/artifacts/oof_predictions.parquet`
- `artifacts/oof_predictions.parquet`
- `runs/research/phase0_predictions.parquet`
- `runs/research/experiments.csv`
- all committed V2/V3 model and metric artifacts

## Recovered data
Ignored raw data exists at `/mnt/d/baseball/open/data`. TrackMan will not be recomputed in this continuation.

## Interruption point
Phase-0 OOF ablations had just completed. Strict V4 direct-model training had not started.
