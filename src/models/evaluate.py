"""
evaluate.py — consistent evaluation for all models.

IMPORTANT: always call self.model.predict() (the wrapper), never the raw inner
estimator directly. The wrapper's predict() applies XGBoost's integer-to-label
mapping and any other post-processing the wrapper owns.
_get_estimator() is used only for logging hyperparameters to MLflow.
"""

import json
import pickle
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.lightgbm
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    f1_score,
)

ARTIFACTS_DIR  = Path("reports") / "mlflow_artifacts"
CORRECT_LABELS = ["Fatal", "Serious", "Slight"]

# Business-metric cost matrix: cost[true_label][predicted_label].
# Fatal predicted as Slight is the worst possible error (value 100) —
# a real fatality is treated as a minor incident.
# Slight predicted as Fatal wastes emergency resources but costs no lives (value 2).
_MISCLASSIFICATION_COST = {
    "Fatal":   {"Fatal": 0, "Serious": 20, "Slight": 100},
    "Serious": {"Fatal": 5, "Serious": 0,  "Slight": 15},
    "Slight":  {"Fatal": 2, "Serious": 1,  "Slight": 0},
}


class Evaluate:
    """
    Wraps model evaluation, metric logging, and MLflow artifact creation.

    Parameters
    ----------
    X_test      : Feature matrix for the held-out test set.
    y_test      : True labels (string or int-encoded; converted to str internally).
    model       : Trained model wrapper exposing a predict() method.
    class_names : Ordered list of class labels. Defaults to CORRECT_LABELS.
    """

    def __init__(self, X_test, y_test, model, class_names=None):
        if len(X_test) == 0:
            raise ValueError("X_test must be non-empty.")
        if len(y_test) == 0:
            raise ValueError("y_test must be non-empty.")
        if len(X_test) != len(y_test):
            raise ValueError(
                f"X_test ({len(X_test)}) and y_test ({len(y_test)}) must be the same length."
            )
        if not hasattr(model, "predict"):
            raise ValueError("model must expose a predict() method.")
        if class_names is not None:
            if len(class_names) == 0:
                raise ValueError("class_names must not be empty.")
            if len(class_names) != len(set(class_names)):
                raise ValueError("class_names must be unique.")

        self.X_test      = X_test
        self.y_test      = np.array(y_test, dtype=str)
        self.model       = model
        self.class_names = list(class_names) if class_names is not None else CORRECT_LABELS

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_estimator(self):
        """Return the inner sklearn estimator (used for hyperparameter logging only)."""
        inner = getattr(self.model, "model", None)
        return inner if inner is not None else self.model

    @staticmethod
    def _compute_business_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute road-safety business metrics from predictions.

        fatal_recall_business     : Of all fatal accidents, fraction correctly identified.
        serious_recall_business   : Of all serious accidents, fraction correctly identified.
        critical_case_recall      : Of all Fatal+Serious cases, fraction not downgraded to Slight.
                                    Measures the pipeline's ability to flag high-risk incidents.
        critical_undertriage_rate : Fraction of Fatal+Serious cases wrongly predicted as Slight.
                                    Primary safety KPI — must be driven as close to 0 as possible.
        weighted_cost             : Mean misclassification cost per sample via _MISCLASSIFICATION_COST.
                                    Converts all error types into a single comparable number.
                                    Lower is better.
        """
        fatal_mask    = y_true == "Fatal"
        serious_mask  = y_true == "Serious"
        critical_mask = fatal_mask | serious_mask

        fatal_recall = (
            float((y_pred[fatal_mask] == "Fatal").mean()) if fatal_mask.any() else 0.0
        )
        serious_recall = (
            float((y_pred[serious_mask] == "Serious").mean()) if serious_mask.any() else 0.0
        )
        critical_recall = (
            float((y_pred[critical_mask] != "Slight").mean()) if critical_mask.any() else 0.0
        )
        undertriage_rate = (
            float((y_pred[critical_mask] == "Slight").mean()) if critical_mask.any() else 0.0
        )

        total_cost = sum(
            _MISCLASSIFICATION_COST.get(t, {}).get(p, 0)
            for t, p in zip(y_true, y_pred)
        )
        weighted_cost = float(total_cost / len(y_true))

        return {
            "fatal_recall_business":     fatal_recall,
            "serious_recall_business":   serious_recall,
            "critical_case_recall":      critical_recall,
            "critical_undertriage_rate": undertriage_rate,
            "weighted_cost":             weighted_cost,
        }

    def _log_model_artifact(self, estimator, run_name: str) -> None:
        """
        Persist the fitted estimator as an MLflow model artifact.

        Dispatches to the correct MLflow flavour (xgboost / lightgbm / sklearn).
        Falls back to a raw pickle artifact if no flavour applies or fails.
        """
        name = run_name.lower()
        try:
            if "xgb" in name:
                mlflow.xgboost.log_model(estimator, artifact_path="model")
            elif "lightgbm" in name or "lgbm" in name:
                mlflow.lightgbm.log_model(estimator, artifact_path="model")
            else:
                mlflow.sklearn.log_model(estimator, artifact_path="model")
            print(f"  [MLFLOW] Model artifact saved for {run_name}.")
            return
        except Exception as exc:
            print(f"  [WARN] Native MLflow flavour failed for {run_name}: {exc}. Falling back to pickle.")

        # Pickle fallback — ensures the model is always persisted even if the
        # MLflow flavour integration is unavailable or misconfigured.
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        pkl_path = ARTIFACTS_DIR / f"{name.replace(' ', '_')}_model.pkl"
        try:
            with open(pkl_path, "wb") as f:
                pickle.dump(self.model, f)
            mlflow.log_artifact(str(pkl_path), artifact_path="model")
            print(f"  [MLFLOW] Pickle fallback saved: {pkl_path}")
        except Exception as exc:
            print(f"  [WARN] Could not save model artifact for {run_name}: {exc}")

    def _save_confusion_matrix(self, y_true: np.ndarray, y_pred: np.ndarray, run_name: str) -> Path:
        """Save a confusion matrix PNG to ARTIFACTS_DIR and return its path."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(figsize=(7, 6))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred,
            labels=self.class_names,
            display_labels=self.class_names,
            cmap="Blues",
            values_format="d",
            ax=ax,
            colorbar=False,
        )
        ax.set_title(run_name)
        safe    = run_name.lower().replace(" ", "_")
        cm_path = ARTIFACTS_DIR / f"{safe}_cm.png"
        fig.savefig(cm_path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        return cm_path

    def _save_report_json(self, report_dict: dict, run_name: str) -> Path:
        """Save the classification report dict as JSON and return its path."""
        ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
        safe        = run_name.lower().replace(" ", "_")
        report_path = ARTIFACTS_DIR / f"{safe}_report.json"
        with open(report_path, "w") as f:
            json.dump(report_dict, f, indent=2)
        return report_path

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def evaluate(
        self,
        run_name: str,
        y_pred: Optional[np.ndarray] = None,
        log_to_mlflow: bool = True,
    ) -> dict:
        """
        Evaluate the model and optionally log everything to an MLflow run.

        Parameters
        ----------
        run_name      : Human-readable name for the MLflow run and artifact filenames.
        y_pred        : Pre-computed predictions (string labels). If None, calls
                        self.model.predict(self.X_test).
        log_to_mlflow : When False, prints only — no MLflow run is opened.
                        Pass False when calling from inside an existing run.

        Returns
        -------
        dict with all logged metrics (standard + business).
        """
        # ── Predictions ───────────────────────────────────────────────────────
        if y_pred is None:
            y_pred = self.model.predict(self.X_test)

        y_pred = np.array(y_pred, dtype=str)
        y_true = self.y_test

        if len(y_pred) != len(y_true):
            raise ValueError(
                f"Prediction length ({len(y_pred)}) does not match "
                f"test set length ({len(y_true)})."
            )

        unexpected = set(y_pred) - set(self.class_names)
        if unexpected:
            print(f"  [WARN] predict() returned unexpected labels: {unexpected}")

        # ── Metrics ───────────────────────────────────────────────────────────
        acc      = accuracy_score(y_true, y_pred)
        macro_f1 = f1_score(y_true, y_pred, average="macro",    zero_division=0)
        w_f1     = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        report_dict: dict = classification_report(
            y_true, y_pred,
            labels=self.class_names,
            target_names=self.class_names,
            output_dict=True,
            zero_division=0,
        )
        business = self._compute_business_metrics(y_true, y_pred)

        # ── Console output ────────────────────────────────────────────────────
        print(f"\n{'=' * 42}")
        print(f"  {run_name}")
        print(f"{'=' * 42}")
        print(f"  Accuracy               : {acc:.4f}")
        print(f"  Macro F1               : {macro_f1:.4f}")
        print(f"  Weighted F1            : {w_f1:.4f}")
        print(f"  Fatal recall           : {business['fatal_recall_business']:.4f}")
        print(f"  Serious recall         : {business['serious_recall_business']:.4f}")
        print(f"  Critical case recall   : {business['critical_case_recall']:.4f}")
        print(f"  Critical undertriage   : {business['critical_undertriage_rate']:.4f}  ← primary safety KPI")
        print(f"  Weighted cost/sample   : {business['weighted_cost']:.2f}")
        print()
        print(classification_report(
            y_true, y_pred,
            labels=self.class_names,
            target_names=self.class_names,
            zero_division=0,
        ))

        # ── Artifacts (always saved — independent of MLflow) ──────────────────
        cm_path     = self._save_confusion_matrix(y_true, y_pred, run_name)
        report_path = self._save_report_json(report_dict, run_name)

        # ── MLflow logging ────────────────────────────────────────────────────
        ctx = mlflow.start_run(run_name=run_name) if log_to_mlflow else nullcontext()

        with ctx:
            if log_to_mlflow:
                # Log model identity explicitly — satisfies spec requirement.
                mlflow.log_param("model_name",  run_name)
                mlflow.log_param("model_class", type(self.model).__name__)

                estimator = self._get_estimator()
                if hasattr(estimator, "get_params"):
                    try:
                        mlflow.log_params({k: str(v) for k, v in estimator.get_params().items()})
                    except Exception:
                        pass

                # Standard metrics
                mlflow.log_metric("accuracy",    float(acc))
                mlflow.log_metric("macro_f1",    float(macro_f1))
                mlflow.log_metric("weighted_f1", float(w_f1))

                for cls in self.class_names:
                    if cls in report_dict:
                        mlflow.log_metric(f"{cls}_f1",        float(report_dict[cls]["f1-score"]))
                        mlflow.log_metric(f"{cls}_recall",    float(report_dict[cls]["recall"]))
                        mlflow.log_metric(f"{cls}_precision", float(report_dict[cls]["precision"]))

                # Business metrics
                for metric_name, metric_value in business.items():
                    mlflow.log_metric(metric_name, metric_value)

                # Artifacts
                mlflow.log_artifact(str(cm_path),     artifact_path="plots")
                mlflow.log_artifact(str(report_path), artifact_path="reports")

                # Model artifact — dispatches by framework, pickle fallback if needed
                self._log_model_artifact(estimator, run_name)

        return {
            "accuracy":    float(acc),
            "macro_f1":    float(macro_f1),
            "weighted_f1": float(w_f1),
            **business,
        }