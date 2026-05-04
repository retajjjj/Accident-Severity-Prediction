"""
train.py — with corrected label mapping.

The correct mapping based on empirical class frequencies is:
  0 → Slight  (86% majority — was wrongly mapped to Fatal)
  1 → Serious (12%)
  2 → Fatal   (1.2% minority — was wrongly mapped to Slight)

sklearn LabelEncoder uses alphabetical order on the ORIGINAL string labels.
If the original Accident_Severity was stored as integers 1/2/3 in STATS19
(1=Fatal, 2=Serious, 3=Slight), LabelEncoder encodes them as:
  "1" → 0,  "2" → 1,  "3" → 2
meaning: 0=Fatal, 1=Serious, 2=Slight  ← this would be CORRECT alphabetically.

BUT the empirical distribution shows 0 has 86% of samples = Slight.
So either:
  (a) the original data already had string labels where Slight came first
      alphabetically, OR
  (b) the encoding was done on frequency order, not alphabetical.
Either way, 0=Slight, 1=Serious, 2=Fatal is what the data shows.
"""

import sys
import time
import pickle
from pathlib import Path

import numpy as np
import mlflow
from sklearn.metrics import f1_score

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import Evaluate
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.LightGBM import LGBMModel
from models.baseline import BaselineModel
PROCESSED_DIR  = PROJECT_ROOT / "data" / "processed"
CORRECT_LABELS = ["Fatal", "Serious", "Slight"]

# ─────────────────────────────────────────────────────────────────────────────
# CORRECTED label mapping — empirically verified from class frequencies:
#   0 → Slight  (86% of data = majority = Slight)
#   1 → Serious (12%)
#   2 → Fatal   (1.2% = minority = Fatal)
# ─────────────────────────────────────────────────────────────────────────────
_INT_TO_LABEL = {"0": "Slight", "1": "Serious", "2": "Fatal"}


def normalise_labels(y) -> np.ndarray:
    y = np.array(y, dtype=str)
    unique = set(y)

    if unique.issubset(_INT_TO_LABEL.keys()):
        print(f"  [INFO] Remapping integers → {_INT_TO_LABEL}")
        y = np.array([_INT_TO_LABEL[v] for v in y], dtype=str)

    unknown = set(y) - set(CORRECT_LABELS)
    if unknown:
        raise ValueError(
            f"Unexpected labels after normalisation: {unknown}\n"
            f"Expected {CORRECT_LABELS}."
        )

    counts = dict(zip(*np.unique(y, return_counts=True)))
    # Sanity check: Slight must be majority
    if counts.get("Slight", 0) < counts.get("Fatal", 0):
        print("  [WARN] After mapping, Slight is still not the majority class.")
        print("         Check _INT_TO_LABEL in train.py — mapping may need adjustment.")

    return y


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_split(name: str):
    with open(PROCESSED_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# ─────────────────────────────────────────────────────────────────────────────
# Probability diagnostics
# ─────────────────────────────────────────────────────────────────────────────

def print_prob_diagnostics(model, X_val, y_val):
    try:
        estimator = getattr(model, "model", model)
        probs = estimator.predict_proba(X_val)
    except Exception:
        print("  [SKIP] predict_proba not available")
        return

    print("\n  ── Probability diagnostics ──")
    print(f"  {'Class':<10} {'mean':>7} {'p95':>7} {'max':>7}  class_freq  signal?")
    for i, cls in enumerate(CORRECT_LABELS):
        col  = probs[:, i]
        freq = (np.array(y_val) == cls).mean()
        has_signal = abs(col.mean() - freq) > 0.01
        flag = "YES" if has_signal else "NO — mean≈freq, threshold wont help"
        print(f"  {cls:<10} {col.mean():>7.4f} {np.percentile(col,95):>7.4f} "
              f"{col.max():>7.4f}  {freq:.4f}      {flag}")
    print()


# ─────────────────────────────────────────────────────────────────────────────
# Vectorised threshold search
# ─────────────────────────────────────────────────────────────────────────────

def _vectorised_predict(probs, thresh_array, classes):
    margins  = probs - thresh_array[np.newaxis, :]
    best     = np.argmax(margins, axis=1)
    fallback = np.argmax(probs,   axis=1)
    has_pos  = margins.max(axis=1) > 0
    chosen   = np.where(has_pos, best, fallback)
    return np.array(classes)[chosen]


def find_best_thresholds_fast(model, X_val, y_val):
    try:
        estimator = getattr(model, "model", model)
        probs = estimator.predict_proba(X_val)
    except Exception:
        print("  [SKIP] No predict_proba")
        return {c: 0.33 for c in CORRECT_LABELS}

    y_val      = np.array(y_val, dtype=str)
    cls_to_idx = {c: i for i, c in enumerate(CORRECT_LABELS)}
    y_int      = np.array([cls_to_idx[c] for c in y_val], dtype=np.int32)

    # Fatal is column 0, Serious is 1, Slight is 2
    fatal_range   = np.arange(0.02, 0.52, 0.05)   # 10 values
    serious_range = np.arange(0.05, 0.55, 0.05)   # 10 values
    slight_range  = np.arange(0.10, 0.80, 0.10)   #  7 values

    best_f1     = 0.0
    best_thresh = np.array([0.33, 0.33, 0.33])

    t0 = time.time()
    for t_f in fatal_range:
        for t_s in serious_range:
            for t_sl in slight_range:
                thresh   = np.array([t_f, t_s, t_sl])
                margins  = probs - thresh[np.newaxis, :]
                best_m   = np.argmax(margins, axis=1)
                fallback = np.argmax(probs,   axis=1)
                has_pos  = margins.max(axis=1) > 0
                chosen   = np.where(has_pos, best_m, fallback).astype(np.int32)
                f1 = f1_score(y_int, chosen, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1     = f1
                    best_thresh = thresh.copy()

    print(f"  Threshold search: {time.time()-t0:.1f}s  "
          f"Fatal={best_thresh[0]:.2f}  Serious={best_thresh[1]:.2f}  "
          f"Slight={best_thresh[2]:.2f}  val macro F1={best_f1:.4f}")

    return {CORRECT_LABELS[i]: float(best_thresh[i]) for i in range(3)}


def predict_with_thresholds(model, X, thresholds):
    estimator  = getattr(model, "model", model)
    probs      = estimator.predict_proba(X)
    thresh_arr = np.array([thresholds[c] for c in CORRECT_LABELS])
    return _vectorised_predict(probs, thresh_arr, CORRECT_LABELS)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    X_train = load_split("X_train")
    # X_val   = load_split("X_val")
    # X_test  = load_split("X_test")

    y_train = normalise_labels(load_split("y_train"))
    # y_val   = normalise_labels(load_split("y_val"))
    # y_test  = normalise_labels(load_split("y_test"))

    print(f"\n  X_train : {X_train.shape}")
    #print(f"  X_val   : {X_val.shape}")
    #print(f"  X_test  : {X_test.shape}")

    counts = dict(zip(*np.unique(y_train, return_counts=True)))
    print(f"\n  y_train class distribution (CORRECTED):")
    for cls in CORRECT_LABELS:
        n    = counts.get(cls, 0)
        pct  = 100 * n / len(y_train)
        smote_ok = ""
        if cls == "Fatal" and pct < 5:
            smote_ok = "  ← SMOTE DID NOT RUN — re-run preprocessing!"
        elif cls == "Fatal" and pct > 30:
            smote_ok = "  ← SMOTE balanced correctly"
        print(f"    {cls:<10}: {n:>8,}  ({pct:.1f}%){smote_ok}")

    # mlflow.set_experiment("Accident_Severity_Pipeline")

    models = [
        # ("baseline_constant",BaselineModel(strategy="constant", constant="Slight"))
        # ("baseline_stratified" ,BaselineModel(strategy="stratified"))
        # ("baseline_frequent") ,BaselineModel(strategy="most_frequent")
        ("LogReg",   LogisticRegressionModel()),
        ("RF",       RandomForestModel()),
        ("XGB",      XGBoostModel()),
        ("CatBoost", CatBoostModel()),
        ("LightGBM", LGBMModel()),
    ]

    # for run_name, model in models:
    #     print(f"\n{'='*55}")
    #     print(f"  Training {run_name}...")
    #     t0 = time.time()
    #     model.fit(X_train, y_train)
    #     print(f"  Fit: {time.time()-t0:.1f}s")

    #     evaluator = Evaluate(X_test, y_test, model, CORRECT_LABELS)
    #     evaluator.evaluate(run_name)

    #     y_pred_base = np.array(model.predict(X_test), dtype=str)
    #     base_f1     = f1_score(y_test, y_pred_base, average="macro", zero_division=0)

    #     print_prob_diagnostics(model, X_val, y_val)

    #     print("  Running threshold search on val set...")
    #     thresholds    = find_best_thresholds_fast(model, X_val, y_val)
    #     y_pred_thresh = predict_with_thresholds(model, X_test, thresholds)
    #     thresh_f1     = f1_score(y_test, y_pred_thresh, average="macro", zero_division=0)

    #     print(f"\n  Macro F1 — base: {base_f1:.4f}  thresholded: {thresh_f1:.4f}"
    #           f"  delta={thresh_f1-base_f1:+.4f}")

    #     with mlflow.start_run(run_name=run_name + "_thresholded"):
    #         mlflow.log_params(thresholds)
    #         mlflow.log_metric("macro_f1_before", base_f1)
    #         mlflow.log_metric("macro_f1_after",  thresh_f1)
    #         evaluator.evaluate(
    #             run_name=run_name + "_thresholded",
    #             y_pred=y_pred_thresh,
    #             log_to_mlflow=False,
    #         )


if __name__ == "__main__":
    main()