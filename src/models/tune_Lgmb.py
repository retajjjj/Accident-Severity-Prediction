"""
tune_lightgbm.py
Optuna hyperparameter search for LGBMClassifier, optimising macro F1.
Usage:  poetry run python src/models/tune_lightgbm.py
"""

import sys, json, pickle, logging
from pathlib import Path

import numpy as np
import optuna
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from sklearn.metrics import f1_score

PROJECT_ROOT  = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "optuna"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

LABEL_ORDER  = ["Fatal", "Serious", "Slight"]
N_TRIALS     = 60
RANDOM_STATE = 42

optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("lightgbm").setLevel(logging.WARNING)

# ── same normalise_labels as train.py ─────────────────────────────────────────
_INT_TO_LABEL = {"0": "Slight", "1": "Serious", "2": "Fatal"}

def normalise_labels(y):
    y = np.array(y, dtype=str)
    if set(y).issubset(_INT_TO_LABEL.keys()):
        y = np.array([_INT_TO_LABEL[v] for v in y], dtype=str)
    unknown = set(y) - set(LABEL_ORDER)
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}")
    return y

def load(name):
    with open(PROCESSED_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)

# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial, X_train, y_train, X_val, y_val):
    num_leaves        = trial.suggest_int("num_leaves",          15, 255)
    learning_rate     = trial.suggest_float("learning_rate",     0.005, 0.15,  log=True)
    n_estimators      = trial.suggest_categorical("n_estimators",[500, 1000, 2000, 3000])
    min_child_samples = trial.suggest_int("min_child_samples",   5, 100)
    subsample         = trial.suggest_float("subsample",         0.5, 1.0)
    colsample_bytree  = trial.suggest_float("colsample_bytree",  0.5, 1.0)
    reg_alpha         = trial.suggest_float("reg_alpha",         1e-3, 10.0, log=True)
    reg_lambda        = trial.suggest_float("reg_lambda",        1e-3, 10.0, log=True)
    fatal_w           = trial.suggest_float("fatal_weight",      5.0, 80.0)
    serious_w         = trial.suggest_float("serious_weight",    1.5, 10.0)

    model = LGBMClassifier(
        num_leaves=num_leaves,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        min_child_samples=min_child_samples,
        subsample=subsample,
        subsample_freq=1,
        colsample_bytree=colsample_bytree,
        reg_alpha=reg_alpha,
        reg_lambda=reg_lambda,
        class_weight={"Fatal": fatal_w, "Serious": serious_w, "Slight": 1.0},
        objective="multiclass",
        metric="multi_logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=-1,
    )

    model.fit(
        X_train, y_train,
        eval_set=[(X_val, y_val)],
        callbacks=[
            early_stopping(stopping_rounds=50, verbose=False),
            log_evaluation(period=-1),
        ],
    )

    internal_order = list(model.classes_)
    col_idx = [internal_order.index(c) for c in LABEL_ORDER if c in internal_order]
    probs   = model.predict_proba(X_val)[:, col_idx]

    # Raw argmax predictions — train.py handles thresholding separately
    preds = np.array(LABEL_ORDER)[np.argmax(probs, axis=1)]
    return f1_score(y_val, preds, average="macro", zero_division=0)

# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  LightGBM Optuna Hyperparameter Search")
    print(f"  Trials: {N_TRIALS}  |  Optimising: macro F1 (with thresholds)")
    print("=" * 60)

    X_train = load("X_train_balanced")   # use SMOTE-balanced set
    y_train = normalise_labels(load("y_train_balanced"))
    X_val   = load("X_val")
    y_val   = normalise_labels(load("y_val"))
    X_test  = load("X_test")
    y_test  = normalise_labels(load("y_test"))

    print(f"\n  X_train: {X_train.shape}  |  X_val: {X_val.shape}")
    print(f"  Train dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val   dist: {dict(zip(*np.unique(y_val,   return_counts=True)))}\n")

    # ── drop "balanced" class_weight since SMOTE already rebalanced ──
    # The explicit fatal_weight/serious_weight in the trial handles this.

    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study   = optuna.create_study(direction="maximize", sampler=sampler)
    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    best = study.best_trial
    print(f"\n  Best trial #{best.number}  val macro F1 = {best.value:.4f}")
    for k, v in best.params.items():
        print(f"    {k:25s} = {v}")
    print(f"  Thresholds: Fatal={best.user_attrs['fatal_thresh']:.3f}  "
          f"Serious={best.user_attrs['serious_thresh']:.3f}  "
          f"Slight={best.user_attrs['slight_thresh']:.3f}")

    # ── save ──
    output = {
        "best_trial": best.number,
        "best_val_f1": best.value,
        "params": best.params,
        "thresholds": {
            "Fatal":   best.user_attrs["fatal_thresh"],
            "Serious": best.user_attrs["serious_thresh"],
            "Slight":  best.user_attrs["slight_thresh"],
        },
    }
    with open(REPORTS_DIR / "lightgbm_best_params.json", "w") as f:
        json.dump(output, f, indent=2)
    with open(REPORTS_DIR / "lightgbm_study.pkl", "wb") as f:
        pickle.dump(study, f)
    print(f"\n  ✓ Saved to {REPORTS_DIR}")

if __name__ == "__main__":
    main()