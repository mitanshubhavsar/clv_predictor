"""Optuna tuning helper shared by every step-7 candidate.

The rule that picks hyperparameters (and later the winning model) is:
  maximise R2 on the ORIGINAL GBP scale, from 5-fold cross-validation on the
  TRAIN snapshot only. The test snapshot is never used here.

Predictions are pooled across the 5 held-out folds (out-of-fold), then R2 is
computed once on all 4,242 train customers. Same folds for every trial and
every model (KFold seed 42, defined in step6_baselines.cv_predictions), so
differences between models are not fold luck.
"""
import optuna
import pandas as pd

from metrics import r2
from step6_baselines import cv_predictions

optuna.logging.set_verbosity(optuna.logging.WARNING)


def tune(make_model, suggest, X: pd.DataFrame, y: pd.Series,
         n_trials: int = 30, seed: int = 42) -> optuna.Study:
    """make_model(**params) -> unfitted model with fit/predict.
    suggest(trial) -> dict of params for that trial."""
    y_arr = y.to_numpy()

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial)
        oof = cv_predictions(lambda: make_model(**params), X, y)
        return r2(y_arr, oof)

    study = optuna.create_study(direction="maximize",
                                sampler=optuna.samplers.TPESampler(seed=seed))
    study.optimize(objective, n_trials=n_trials)
    return study
