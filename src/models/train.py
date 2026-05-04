"""
train.py — loads data, optionally applies SMOTETomek, trains models.

SMOTE rules:
  - Applied to training data only if Fatal class < 5% of total samples.
  - Result is cached to data/processed/X_train_balanced.pkl.
    Delete that file to force a re-run.
  - X_val and X_test are NEVER resampled — they must reflect real-world distribution.
"""

import sys
import time
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
from sklearn.metrics import f1_score
from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.models.evaluate import Evaluate
from models.baseline import BaselineModel
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.LightGBM import LGBMModel


PROCESSED_DIR   = PROJECT_ROOT / "data" / "processed"
BALANCED_X_PATH = PROCESSED_DIR / "X_train_balanced.pkl"
BALANCED_Y_PATH = PROCESSED_DIR / "y_train_balanced.pkl"


CORRECT_LABELS = ["Fatal", "Serious", "Slight"]
RANDOM_STATE   = 42

FATAL_IMBALANCE_THRESHOLD = 5.0   # % — trigger SMOTE below this

# Desired absolute class counts after resampling.
SMOTE_STRATEGY = {
    "Fatal":   150_000,   # 10K  → 150K  (~15×)
    "Serious": 450_000,   # 107K → 450K  (~4×)
}

USE_SMOTE_TOMEK = True   # False = vanilla SMOTE (faster, slightly noisier)

# Integer → string label mapping (used when y is stored as 0/1/2).
_INT_TO_LABEL = {"0": "Slight", "1": "Serious", "2": "Fatal"}


def normalise_labels(y) -> np.ndarray:
    """Convert integer-encoded labels to string labels if needed."""
    y = np.array(y, dtype=str)

    converted = [_INT_TO_LABEL.get(label, label) for label in y]
    if converted != list(y):
        print(f"  [INFO] Remapped integer labels → {_INT_TO_LABEL}")
    y = np.array(converted, dtype=str)

    unknown = set(y) - set(CORRECT_LABELS)
    if unknown:
        raise ValueError(f"Unexpected labels: {unknown}. Expected: {CORRECT_LABELS}.")

    return y


def class_counts(y: np.ndarray) -> dict:
    counts, values = np.unique(y, return_counts=True)
    return dict(zip(counts, values))


def load_split(name: str):
    with open(PROCESSED_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


def save_pkl(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(obj, f)


def _print_distribution(y: np.ndarray, label: str = "") -> None:
    n_total = len(y)
    if label:
        print(f"  [{label}] class distribution:")
    for cls in CORRECT_LABELS:
        n   = (y == cls).sum()
        pct = 100 * n / n_total
        print(f"    {cls:<10}: {n:>8,}  ({pct:.1f}%)")


def _is_balanced(y: np.ndarray) -> bool:
    """Return True if Fatal class meets the minimum threshold."""
    fatal_count = (y == "Fatal").sum()
    fatal_pct   = 100 * fatal_count / len(y)
    return fatal_pct >= FATAL_IMBALANCE_THRESHOLD


def validate_cached_balance(y_bal: np.ndarray) -> None:
    """
    Confirm the cached balanced data still meets balance requirements.
    Raises RuntimeError if the cache is stale or corrupted.
    """
    if not _is_balanced(y_bal):
        raise RuntimeError(
            "Cached balanced training data does not meet balance requirements "
            f"(Fatal < {FATAL_IMBALANCE_THRESHOLD}%). "
            "Delete the cache files and re-run to regenerate:\n"
            f"  {BALANCED_X_PATH}\n"
            f"  {BALANCED_Y_PATH}"
        )

    counts = dict(zip(*np.unique(y_bal, return_counts=True)))
    for cls, target in SMOTE_STRATEGY.items():
        actual = counts.get(cls, 0)
        if actual < target * 0.90:
            raise RuntimeError(
                f"Cached balanced data has {actual:,} '{cls}' samples "
                f"but expected at least {int(target * 0.90):,} (90% of target {target:,}). "
                "Delete the cache and re-run."
            )

    print("  [SMOTE] Cache balance check passed.")


def _drop_non_numeric_columns(X: pd.DataFrame) -> pd.DataFrame:
    """Remove datetime and string columns that SMOTE cannot process."""
    drop_cols = X.select_dtypes(include=["datetime64", "object", "string"]).columns.tolist()
    if drop_cols:
        print(f"  [SMOTE] Dropping non-numeric columns: {drop_cols}")
        X = X.drop(columns=drop_cols)
    return X


def load_or_create_balanced_train(
    X_train: pd.DataFrame,
    y_train: np.ndarray,
) -> tuple[pd.DataFrame, np.ndarray]:
    """
    Load cached balanced training data if available and valid.
    Otherwise run SMOTETomek (or SMOTE) and cache the result.
    """
    if BALANCED_X_PATH.exists() and BALANCED_Y_PATH.exists():
        print("  [SMOTE] Loading cached balanced training data...")
        t0    = time.time()
        X_bal = load_split("X_train_balanced")
        y_bal = np.array(load_split("y_train_balanced"), dtype=str)
        print(f"  [SMOTE] Loaded in {time.time() - t0:.1f}s  shape={X_bal.shape}")
        _print_distribution(y_bal, label="SMOTE cache")
        validate_cached_balance(y_bal)
        return X_bal, y_bal

    if not isinstance(X_train, pd.DataFrame):
        X_train = pd.DataFrame(X_train)

    X_train      = _drop_non_numeric_columns(X_train)
    numeric_cols = X_train.columns.tolist()
    X_train      = X_train.astype(float)

    sampler_name = "SMOTETomek" if USE_SMOTE_TOMEK else "SMOTE"
    print(f"\n  [SMOTE] Running {sampler_name}...")
    print(f"  [SMOTE] Input  shape    : {X_train.shape}")
    print(f"  [SMOTE] Target strategy : {SMOTE_STRATEGY}")
    _print_distribution(y_train, label="input")

    t0 = time.time()

    base_smote = SMOTE(sampling_strategy=SMOTE_STRATEGY, random_state=RANDOM_STATE)
    sampler    = SMOTETomek(smote=base_smote, random_state=RANDOM_STATE) if USE_SMOTE_TOMEK else base_smote
    X_bal, y_bal = sampler.fit_resample(X_train, y_train)

    X_bal = pd.DataFrame(X_bal, columns=numeric_cols).astype(float)
    y_bal = np.array(y_bal, dtype=str)

    print(f"\n  [SMOTE] Done in {time.time() - t0:.1f}s  shape={X_bal.shape}")
    _print_distribution(y_bal, label="output")

    print(f"  [SMOTE] Saving cache to {BALANCED_X_PATH}...")
    save_pkl(X_bal, BALANCED_X_PATH)
    save_pkl(y_bal, BALANCED_Y_PATH)

    return X_bal, y_bal


def _vectorised_predict(
    probs: np.ndarray,
    thresh_array: np.ndarray,
    classes: list[str],
) -> np.ndarray:
    margins  = probs - thresh_array[np.newaxis, :]
    best     = np.argmax(margins, axis=1)
    fallback = np.argmax(probs, axis=1)
    has_pos  = margins.max(axis=1) > 0
    chosen   = np.where(has_pos, best, fallback)
    return np.array(classes)[chosen]


def find_best_thresholds(
    model,
    X_val: pd.DataFrame,
    y_val: np.ndarray,
) -> dict[str, float]:
    """
    Grid-search per-class probability thresholds that maximise macro F1 on X_val.
    Falls back to equal thresholds (0.33) if predict_proba is unavailable.
    """
    estimator = getattr(model, "model", model)

    try:
        probs = estimator.predict_proba(X_val)
    except Exception:
        print("  [THRESHOLD] predict_proba unavailable — using default 0.33 thresholds.")
        return {c: 0.33 for c in CORRECT_LABELS}

    y_val      = np.array(y_val, dtype=str)
    cls_to_idx = {c: i for i, c in enumerate(CORRECT_LABELS)}
    y_int      = np.array([cls_to_idx[c] for c in y_val], dtype=np.int32)

    fatal_range   = np.arange(0.02, 0.52, 0.05)
    serious_range = np.arange(0.05, 0.55, 0.05)
    slight_range  = np.arange(0.10, 0.80, 0.10)

    best_f1     = 0.0
    best_thresh = np.array([0.33, 0.33, 0.33])

    t0 = time.time()
    for t_fatal in fatal_range:
        for t_serious in serious_range:
            for t_slight in slight_range:
                thresh   = np.array([t_fatal, t_serious, t_slight])
                margins  = probs - thresh[np.newaxis, :]
                best_m   = np.argmax(margins, axis=1)
                fallback = np.argmax(probs, axis=1)
                has_pos  = margins.max(axis=1) > 0
                chosen   = np.where(has_pos, best_m, fallback).astype(np.int32)
                f1       = f1_score(y_int, chosen, average="macro", zero_division=0)
                if f1 > best_f1:
                    best_f1     = f1
                    best_thresh = thresh.copy()

    print(
        f"  [THRESHOLD] Search: {time.time() - t0:.1f}s | "
        f"Fatal={best_thresh[0]:.2f}  Serious={best_thresh[1]:.2f}  "
        f"Slight={best_thresh[2]:.2f}  val macro F1={best_f1:.4f}"
    )
    return {CORRECT_LABELS[i]: float(best_thresh[i]) for i in range(3)}


def predict_with_thresholds(
    model,
    X: pd.DataFrame,
    thresholds: dict[str, float],
) -> np.ndarray:
    estimator  = getattr(model, "model", model)
    probs      = estimator.predict_proba(X)
    thresh_arr = np.array([thresholds[c] for c in CORRECT_LABELS])
    return _vectorised_predict(probs, thresh_arr, CORRECT_LABELS)


def print_prob_diagnostics(model, X_val: pd.DataFrame, y_val: np.ndarray) -> None:
    """Print per-class probability statistics to surface calibration issues."""
    estimator = getattr(model, "model", model)
    try:
        probs = estimator.predict_proba(X_val)
    except Exception:
        print("  [DIAG] predict_proba not available — skipping diagnostics.")
        return

    header = f"  {'Class':<10} {'mean':>7} {'p95':>7} {'max':>7}  {'freq':>7}  signal?"
    print(f"\n  ── Probability diagnostics ──\n{header}")
    for i, cls in enumerate(CORRECT_LABELS):
        col    = probs[:, i]
        freq   = (np.array(y_val) == cls).mean()
        signal = "YES" if abs(col.mean() - freq) > 0.01 else "NO (mean ≈ freq)"
        print(
            f"  {cls:<10} {col.mean():>7.3f} {np.percentile(col, 95):>7.3f} "
            f"{col.max():>7.3f}  {freq:>7.1%}  {signal}"
        )
    print()



def _log_model_artifact(model, run_name: str) -> None:
    """
    Save the trained model as an MLflow artifact using the correct flavour.
    Dispatches on the inner estimator type so the correct MLflow integration
    (xgboost / lightgbm / sklearn) handles serialisation.
    """
    estimator = getattr(model, "model", model)
    name      = run_name.lower()

    try:
        if "xgb" in name:
            mlflow.xgboost.log_model(estimator, artifact_path="model")
        elif "lightgbm" in name or "lgbm" in name:
            mlflow.lightgbm.log_model(estimator, artifact_path="model")
        else:
            mlflow.sklearn.log_model(estimator, artifact_path="model")
        print(f"  [MLFLOW] Model artifact saved for {run_name}.")
    except Exception as exc:
        print(f"  [WARN] Could not save model artifact for {run_name}: {exc}")


def main() -> None:
    
    X_train_raw = load_split("X_train")
    X_val       = load_split("X_val")
    X_test      = load_split("X_test")

    y_train_raw = normalise_labels(load_split("y_train"))
    y_val       = normalise_labels(load_split("y_val"))
    y_test      = normalise_labels(load_split("y_test"))

    print(f"\n  X_train (raw) : {X_train_raw.shape}")
    print(f"  X_val         : {X_val.shape}")
    print(f"  X_test        : {X_test.shape}")

    _print_distribution(y_train_raw, label="raw y_train")

    if _is_balanced(y_train_raw):
        print(
            f"\n  [SMOTE] Fatal class >= {FATAL_IMBALANCE_THRESHOLD}% — "
            "data already balanced, skipping SMOTE."
        )
        X_train, y_train = X_train_raw, y_train_raw
    else:
        print(
            f"\n  [SMOTE] Fatal class < {FATAL_IMBALANCE_THRESHOLD}% — "
            "applying SMOTETomek to training data."
        )
        X_train, y_train = load_or_create_balanced_train(X_train_raw, y_train_raw)

    print(f"\n  X_train (final): {X_train.shape}")

    mlflow.set_experiment("Accident_Severity_Pipeline")

    models = [
        ("baseline_constant",BaselineModel(strategy="constant", constant="Slight"))
        ("baseline_stratified" ,BaselineModel(strategy="stratified"))
        ("baseline_frequent") ,BaselineModel(strategy="most_frequent")
        ("LogReg",   LogisticRegressionModel()),
        ("RF",       RandomForestModel()),
        ("XGB",      XGBoostModel()),
        ("CatBoost", CatBoostModel()),
        ("LightGBM", LGBMModel()),
    ]

    for run_name, model in models:
        print(f"\n{'=' * 55}")
        print(f"  Training: {run_name}")

        t0 = time.time()
        model.fit(X_train, y_train)
        print(f"  Fit time: {time.time() - t0:.1f}s")

        evaluator   = Evaluate(X_test, y_test, model, CORRECT_LABELS)
        evaluator.evaluate(run_name)

        y_pred_base = np.array(model.predict(X_test), dtype=str)
        base_f1     = f1_score(y_test, y_pred_base, average="macro", zero_division=0)

        print_prob_diagnostics(model, X_val, y_val)

        print("  Running threshold search on val set...")
        thresholds    = find_best_thresholds(model, X_val, y_val)
        y_pred_thresh = predict_with_thresholds(model, X_test, thresholds)
        thresh_f1     = f1_score(y_test, y_pred_thresh, average="macro", zero_division=0)

        print(
            f"\n  Macro F1 — base: {base_f1:.4f}  "
            f"thresholded: {thresh_f1:.4f}  "
            f"delta={thresh_f1 - base_f1:+.4f}"
        )

        with mlflow.start_run(run_name=f"{run_name}_thresholded"):
            mlflow.log_params(thresholds)
            mlflow.log_metric("macro_f1_before", base_f1)
            mlflow.log_metric("macro_f1_after",  thresh_f1)

            _log_model_artifact(model, run_name)

            evaluator.evaluate(
                run_name=f"{run_name}_thresholded",
                y_pred=y_pred_thresh,
                log_to_mlflow=True,
            )


if __name__ == "__main__":
    main()