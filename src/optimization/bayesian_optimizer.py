import logging

import optuna
from src.optimization.objective_function import svm_objective

logger = logging.getLogger(__name__)

N_STARTUP_TRIALS = 10


def optimize_svm_hyperparameters(X, y, n_trials: int = 20, cv: int = 3):
    # Adjust CV folds: StratifiedKFold requires at least 2 samples per class
    min_class_count = y.value_counts().min()
    cv = min(cv, max(2, min_class_count))
    if cv < 2:
        cv = 2

    logger.info("Bayesian optimization: %d trials, %d CV folds, %d startup (random) trials", n_trials, cv, N_STARTUP_TRIALS)

    sampler = optuna.samplers.TPESampler(seed=42, n_startup_trials=N_STARTUP_TRIALS)
    pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=0)
    study = optuna.create_study(direction="maximize", sampler=sampler, pruner=pruner)
    study.optimize(lambda trial: svm_objective(trial, X, y, cv=cv), n_trials=n_trials)
    return study.best_params, study
