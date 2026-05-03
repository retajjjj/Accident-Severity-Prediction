import sys
from pathlib import Path
import pickle
import numpy as np
import mlflow

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from sklearn.metrics import f1_score

from models.evaluate import Evaluate
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.LightGBM import LGBMModel
from models.baseline import BaselineModel


PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CORRECT_LABELS = ["Fatal", "Serious", "Slight"]


def load_split(name: str):
    with open(PROCESSED_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)


# Threshold tuning
def find_best_thresholds(model, X_val, y_val, classes):
    """
    Find per-class thresholds that maximize macro F1 on the validation set.
    Uses margin-based prediction: predict the class furthest above its threshold.
    """
    estimator = getattr(model, "model", model)
    probs = estimator.predict_proba(X_val)  # shape (n, n_classes)
    classes = list(classes)

    best_thresholds = {cls: 0.33 for cls in classes}
    best_combined_f1 = 0.0

    # Grid search over all class thresholds jointly (coarse grid)
    # Fatal threshold matters most — search it more finely
    from itertools import product
    fatal_range   = np.arange(0.05, 0.50, 0.05)
    serious_range = np.arange(0.10, 0.60, 0.05)
    slight_range  = np.arange(0.20, 0.80, 0.10)

    for t_fatal, t_serious, t_slight in product(fatal_range, serious_range, slight_range):
        t_map = {
            classes[0]: t_fatal,    # Fatal
            classes[1]: t_serious,  # Serious
            classes[2]: t_slight,   # Slight
        }
        preds = _apply_margin_thresholds(probs, t_map, classes)
        f1 = f1_score(y_val, preds, average='macro', zero_division=0)
        if f1 > best_combined_f1:
            best_combined_f1 = f1
            best_thresholds = dict(t_map)

    print(f"Best threshold combo → Fatal={best_thresholds[classes[0]]:.2f}, "
          f"Serious={best_thresholds[classes[1]]:.2f}, "
          f"Slight={best_thresholds[classes[2]]:.2f} → macro F1={best_combined_f1:.4f}")
    return best_thresholds


def _apply_margin_thresholds(probs, thresholds, classes):
    """
    For each sample: pick the class with the highest (prob - threshold).
    If all margins are negative, fall back to argmax.
    This avoids the Fatal-first priority bug.
    """
    classes = list(classes)
    thresh_array = np.array([thresholds[c] for c in classes])  # shape (n_classes,)
    margins = probs - thresh_array[np.newaxis, :]               # shape (n_samples, n_classes)

    preds = []
    for i in range(len(probs)):
        if margins[i].max() > 0:
            # At least one class clears its threshold — pick highest margin
            preds.append(classes[np.argmax(margins[i])])
        else:
            # Nothing clears threshold — fall back to argmax probability
            preds.append(classes[np.argmax(probs[i])])

    return np.array(preds)


def predict_with_thresholds(model, X, thresholds, classes):
    estimator = getattr(model, "model", model)
    probs = estimator.predict_proba(X)
    return _apply_margin_thresholds(probs, thresholds, list(classes))

# MAIN

def main():

    X_train = load_split("X_train")
    y_train = load_split("y_train")

    X_val = load_split("X_val")
    y_val = load_split("y_val")

    X_test = load_split("X_test")
    y_test = load_split("y_test")

    y_train = np.array(y_train, dtype=str)
    y_val = np.array(y_val, dtype=str)
    y_test = np.array(y_test, dtype=str)
    
    mlflow.set_experiment("Accident_Severity_Pipeline")

    models = [
        # ("BaselineModel",BaselineModel(strategy="constant", constant="Slight")),
        # ("Baseline_stratified",BaselineModel(strategy="stratified"))
        # ("Baseline_frequent",BaselineModel(strategy="most_frequent"))
        ("LogReg", LogisticRegressionModel()),
        ("RF", RandomForestModel()),
        ("XGB", XGBoostModel()),
        ("CatBoost", CatBoostModel()),
        ("LightGBM", LGBMModel()),
    ]

    for run_name, model in models:

        print(f"\nTraining {run_name}...")

        model.fit(X_train, y_train)

        evaluator = Evaluate(X_test, y_test, model, CORRECT_LABELS)

        
        evaluator.evaluate(run_name)

        y_pred_base = model.predict(X_test)
        y_pred_base = np.array(y_pred_base, dtype=str)

        base_f1 = f1_score(y_test, y_pred_base, average="macro")

        
        thresholds = find_best_thresholds(model, X_val, y_val, CORRECT_LABELS)

        print(f"{run_name} thresholds:", thresholds)

        
        y_pred_thresh = predict_with_thresholds(
            model, X_test, thresholds, CORRECT_LABELS
        )

        thresh_f1 = f1_score(y_test, y_pred_thresh, average="macro")

        
        with mlflow.start_run(run_name=run_name + "_thresholded"):

            mlflow.log_param("thresholds", thresholds)
            mlflow.log_metric("macro_f1_before", base_f1)
            mlflow.log_metric("macro_f1_after", thresh_f1)

            evaluator.evaluate(
                run_name=run_name + "_thresholded",
                y_pred=y_pred_thresh,
                log_to_mlflow=False
            )


if __name__ == "__main__":
    main()