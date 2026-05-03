"""
tune_catboost.py
================
Optuna hyperparameter search for CatBoostClassifier, optimising macro F1.

Search space
------------
  depth          : {4, 6, 8, 10}
  learning_rate  : [0.01, 0.15]   (log-uniform)
  iterations     : {500, 1000, 2000, 3000}
  l2_leaf_reg    : [1, 30]        (log-uniform)
  min_data_in_leaf: {1, 5, 10, 20, 50}
  fatal_weight   : [5, 60]        manual class weight multiplier for Fatal
  serious_weight : [1.5, 10]      manual class weight multiplier for Serious

Usage
-----
    poetry run python src/models/tune_catboost.py

Outputs
-------
  reports/optuna/catboost_best_params.json   ← best params
  reports/optuna/catboost_study.pkl          ← full Optuna study (for analysis)
  Printed: top-10 trials
"""

import sys
import json
import pickle
import logging
from pathlib import Path

import numpy as np
import optuna
from catboost import CatBoostClassifier
from sklearn.metrics import f1_score

# ── paths ─────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
REPORTS_DIR   = PROJECT_ROOT / "reports" / "optuna"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# ── constants ─────────────────────────────────────────────────────────────────
LABEL_ORDER  = ["Fatal", "Serious", "Slight"]
N_TRIALS     = 60       # increase to 100+ for a deeper search (costs time)
RANDOM_STATE = 42
EARLY_STOPPING_ROUNDS = 50   # stop if no improvement for N rounds on val set

# Silence CatBoost and Optuna INFO logs
optuna.logging.set_verbosity(optuna.logging.WARNING)
logging.getLogger("catboost").setLevel(logging.WARNING)

_INT_TO_LABEL = {"0": "Slight", "1": "Serious", "2": "Fatal"}

def normalise_labels(y: np.ndarray) -> np.ndarray:
    y = np.array(y, dtype=str)
    if set(y).issubset(_INT_TO_LABEL.keys()):
        print(f"  [INFO] Remapping integers → {_INT_TO_LABEL}")
        y = np.array([_INT_TO_LABEL[v] for v in y], dtype=str)
    unknown = set(y) - {"Fatal", "Serious", "Slight"}
    if unknown:
        raise ValueError(f"Unexpected labels after remap: {unknown}")
    return y
# ── data loading ──────────────────────────────────────────────────────────────
def load(name: str):
    with open(PROCESSED_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# ── threshold helper (vectorised — fast) ──────────────────────────────────────
def apply_margin_thresholds(probs: np.ndarray, thresholds: list, classes: list) -> np.ndarray:
    """
    For each sample pick the class with the highest (prob - threshold).
    Falls back to argmax when all margins are negative.
    """
    thresh = np.array(thresholds)                       # (n_classes,)
    margins = probs - thresh[np.newaxis, :]             # (n_samples, n_classes)
    best_margin_idx = np.argmax(margins, axis=1)        # (n_samples,)
    fallback_idx    = np.argmax(probs,   axis=1)        # (n_samples,)
    has_positive    = margins.max(axis=1) > 0           # (n_samples,)
    chosen_idx      = np.where(has_positive, best_margin_idx, fallback_idx)
    return np.array(classes)[chosen_idx]


def best_thresholds_fast(probs: np.ndarray, y_true: np.ndarray, classes: list) -> tuple:
    """
    Vectorised grid search over per-class thresholds.
    Returns (best_thresholds_list, best_f1).
    """
    fatal_range   = np.arange(0.05, 0.55, 0.05)
    serious_range = np.arange(0.10, 0.60, 0.05)
    slight_range  = np.arange(0.20, 0.80, 0.10)

    best_f1     = 0.0
    best_thresh = [0.33, 0.33, 0.33]

    for tf in fatal_range:
        for ts in serious_range:
            for tsl in slight_range:
                preds = apply_margin_thresholds(probs, [tf, ts, tsl], classes)
                f1 = f1_score(y_true, preds, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1     = f1
                    best_thresh = [float(tf), float(ts), float(tsl)]

    return best_thresh, best_f1


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial, X_train, y_train, X_val, y_val):
    depth          = trial.suggest_categorical("depth",           [4, 6, 8, 10])
    learning_rate  = trial.suggest_float("learning_rate",         0.01, 0.15, log=True)
    iterations     = trial.suggest_categorical("iterations",      [500, 1000, 2000, 3000])
    l2_leaf_reg    = trial.suggest_float("l2_leaf_reg",           1.0, 30.0, log=True)
    min_data_leaf  = trial.suggest_categorical("min_data_in_leaf",[1, 5, 10, 20, 50])
    fatal_w        = trial.suggest_float("fatal_weight",          5.0, 60.0)
    serious_w      = trial.suggest_float("serious_weight",        1.5, 10.0)

    class_weights = {
        "Fatal":   fatal_w,
        "Serious": serious_w,
        "Slight":  1.0,
    }

    model = CatBoostClassifier(
        depth=depth,
        learning_rate=learning_rate,
        iterations=iterations,
        l2_leaf_reg=l2_leaf_reg,
        min_data_in_leaf=min_data_leaf,
        class_weights=class_weights,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=RANDOM_STATE,
        verbose=0,
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        use_best_model=True,
    )

    model.fit(
        X_train, y_train,
        eval_set=(X_val, y_val),
        verbose=False,
    )

    # Get probabilities in [Fatal, Serious, Slight] order
    internal_order = list(model.classes_)
    col_idx = [internal_order.index(c) for c in LABEL_ORDER if c in internal_order]
    probs = model.predict_proba(X_val)[:, col_idx]

    # Optimise thresholds on val set
    best_thresh, best_f1 = best_thresholds_fast(probs, y_val, LABEL_ORDER)

    # Store thresholds as trial user attributes for retrieval later
    trial.set_user_attr("fatal_thresh",   best_thresh[0])
    trial.set_user_attr("serious_thresh", best_thresh[1])
    trial.set_user_attr("slight_thresh",  best_thresh[2])

    return best_f1


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    print("=" * 60)
    print("  CatBoost Optuna Hyperparameter Search")
    print(f"  Trials: {N_TRIALS}  |  Optimising: macro F1 (with thresholds)")
    print("=" * 60)

    # ── load data ──
    X_train = load("X_train")
    X_val   = load("X_val")
    X_test  = load("X_test")    
    y_train = normalise_labels(load("y_train"))
    y_val   = normalise_labels(load("y_val"))
    y_test  = normalise_labels(load("y_test"))

    print(f"\n  X_train: {X_train.shape}  |  X_val: {X_val.shape}  |  X_test: {X_test.shape}")
    print(f"  Train class dist: {dict(zip(*np.unique(y_train, return_counts=True)))}")
    print(f"  Val   class dist: {dict(zip(*np.unique(y_val, return_counts=True)))}\n")

    # ── run study ──
    sampler = optuna.samplers.TPESampler(seed=RANDOM_STATE)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    study.optimize(
        lambda trial: objective(trial, X_train, y_train, X_val, y_val),
        n_trials=N_TRIALS,
        show_progress_bar=True,
    )

    # ── results ──
    best = study.best_trial
    print("\n" + "=" * 60)
    print(f"  Best trial #{best.number}")
    print(f"  Val macro F1 (with thresholds): {best.value:.4f}")
    print("  Params:")
    for k, v in best.params.items():
        print(f"    {k:25s} = {v}")
    print("  Thresholds:")
    print(f"    Fatal   = {best.user_attrs['fatal_thresh']:.3f}")
    print(f"    Serious = {best.user_attrs['serious_thresh']:.3f}")
    print(f"    Slight  = {best.user_attrs['slight_thresh']:.3f}")
    print("=" * 60)

    # ── retrain best model on full train data ──
    print("\n  Retraining best model on full train+val data...")

    import pandas as pd
    X_trainval = pd.concat([
        pd.DataFrame(X_train) if not hasattr(X_train, 'columns') else X_train,
        pd.DataFrame(X_val)   if not hasattr(X_val,   'columns') else X_val,
    ], ignore_index=True)
    y_trainval = np.concatenate([y_train, y_val])

    best_class_weights = {
        "Fatal":   best.params["fatal_weight"],
        "Serious": best.params["serious_weight"],
        "Slight":  1.0,
    }

    final_model = CatBoostClassifier(
        depth=best.params["depth"],
        learning_rate=best.params["learning_rate"],
        iterations=best.params["iterations"],
        l2_leaf_reg=best.params["l2_leaf_reg"],
        min_data_in_leaf=best.params["min_data_in_leaf"],
        class_weights=best_class_weights,
        loss_function="MultiClass",
        eval_metric="TotalF1:average=Macro",
        random_seed=RANDOM_STATE,
        verbose=100,
    )

    final_model.fit(X_trainval, y_trainval)

    # ── evaluate on test set ──
    internal_order = list(final_model.classes_)
    col_idx = [internal_order.index(c) for c in LABEL_ORDER if c in internal_order]
    probs_test = final_model.predict_proba(X_test)[:, col_idx]

    # Use thresholds found on val set
    best_thresh = [
        best.user_attrs["fatal_thresh"],
        best.user_attrs["serious_thresh"],
        best.user_attrs["slight_thresh"],
    ]
    y_pred_thresh = apply_margin_thresholds(probs_test, best_thresh, LABEL_ORDER)
    y_pred_raw    = np.array(final_model.predict(X_test), dtype=str).flatten()

    from sklearn.metrics import classification_report
    print("\n  ── Test Results (raw argmax) ──")
    print(classification_report(y_test, y_pred_raw, target_names=LABEL_ORDER, zero_division=0))

    print("  ── Test Results (with tuned thresholds) ──")
    print(classification_report(y_test, y_pred_thresh, target_names=LABEL_ORDER, zero_division=0))

    raw_f1    = f1_score(y_test, y_pred_raw,    average="macro", zero_division=0)
    thresh_f1 = f1_score(y_test, y_pred_thresh, average="macro", zero_division=0)
    print(f"  Macro F1 raw     : {raw_f1:.4f}")
    print(f"  Macro F1 thresh  : {thresh_f1:.4f}")

    # ── save artifacts ──
    output = {
        "best_trial":   best.number,
        "best_val_f1":  best.value,
        "test_macro_f1_raw":    raw_f1,
        "test_macro_f1_thresh": thresh_f1,
        "params":       best.params,
        "class_weights": best_class_weights,
        "thresholds": {
            "Fatal":   best.user_attrs["fatal_thresh"],
            "Serious": best.user_attrs["serious_thresh"],
            "Slight":  best.user_attrs["slight_thresh"],
        },
    }

    params_path = REPORTS_DIR / "catboost_best_params.json"
    with open(params_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"\n  ✓ Best params saved to: {params_path}")

    study_path = REPORTS_DIR / "catboost_study.pkl"
    with open(study_path, "wb") as f:
        pickle.dump(study, f)
    print(f"  ✓ Full study saved to:  {study_path}")

    model_path = REPORTS_DIR / "catboost_best_model.pkl"
    with open(model_path, "wb") as f:
        pickle.dump(final_model, f)
    print(f"  ✓ Retrained model saved: {model_path}")

    # ── print top-10 trials ──
    print("\n  Top 10 trials by val macro F1:")
    print(f"  {'#':>4}  {'val_f1':>8}  {'depth':>5}  {'lr':>6}  {'iters':>5}  {'l2':>6}  {'fatal_w':>7}  {'serious_w':>9}")
    print("  " + "-" * 65)
    top10 = sorted(study.trials, key=lambda t: t.value or 0, reverse=True)[:10]
    for t in top10:
        p = t.params
        print(f"  {t.number:>4}  {t.value:>8.4f}  {p['depth']:>5}  "
              f"{p['learning_rate']:>6.3f}  {p['iterations']:>5}  "
              f"{p['l2_leaf_reg']:>6.2f}  {p['fatal_weight']:>7.1f}  "
              f"{p['serious_weight']:>9.2f}")


if __name__ == "__main__":
    main()