"""
evaluate.py — consistent evaluation for all models.

IMPORTANT: always call self.model.predict() (the wrapper), never the raw inner
estimator directly. The wrapper's predict() applies XGBoost's integer-to-label
mapping and any other post-processing the wrapper owns.
_get_estimator() is used only for logging hyperparameters to MLflow.
"""

import json
from contextlib import nullcontext
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import mlflow
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ARTIFACTS_DIR  = Path("reports") / "mlflow_artifacts"
CORRECT_LABELS = ["Fatal", "Serious", "Slight"]

# Business-metric cost matrix: cost[true][predicted].
# Fatal predicted as Slight is the worst error (value 100).
# Slight predicted as Fatal wastes resources but risks no lives (value 2).
_MISCLASSIFICATION_COST = {
    "Fatal":   {"Fatal": 0, "Serious": 20,  "Slight": 100},
    "Serious": {"Fatal": 5, "Serious": 0,   "Slight": 15},
    "Slight":  {"Fatal": 2, "Serious": 1,   "Slight": 0},
}


class Evaluate:
    """
    Wraps model evaluation, metric logging, and MLflow artifact creation.

    Parameters
    ----------
    X_test      : Feature matrix for the held-out test set.
    y_test      : True labels (string or int-encoded; converted to str internally).
    model       : Trained model wrapper exposing a `predict()` method.
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

        self.X_test     = X_test
        self.y_test     = np.array(y_test, dtype=str)
        self.model      = model
        self.class_names = list(class_names) if class_names is not None else CORRECT_LABELS

    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers
    # ──────────────────────────────────────────────────────────────────────────

    def _get_estimator(self):
        """Return the inner sklearn estimator (for hyperparameter logging only)."""
        inner = getattr(self.model, "model", None)
        return inner if inner is not None else self.model

    @staticmethod
    def _compute_business_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
        """
        Compute road-safety business metrics from predictions.

        fatal_miss_rate   : P(predicted = Slight | true = Fatal).
                            Real fatalities classified as minor — the costliest error.
        false_fatal_rate  : P(predicted = Fatal  | true = Slight).
                            Minor incidents over-escalated — wastes emergency resources.
        weighted_cost     : Mean misclassification cost per sample using
                            _MISCLASSIFICATION_COST. Lower is better.
        """
        n_fatal = (y_true == "Fatal").sum()
        n_slight = (y_true == "Slight").sum()

        fatal_miss_rate = (
            ((y_true == "Fatal") & (y_pred == "Slight")).sum() / n_fatal
            if n_fatal > 0 else 0.0
        )
        false_fatal_rate = (
            ((y_true == "Slight") & (y_pred == "Fatal")).sum() / n_slight
            if n_slight > 0 else 0.0
        )

        total_cost = sum(
            _MISCLASSIFICATION_COST.get(true, {}).get(pred, 0)
            for true, pred in zip(y_true, y_pred)
        )
        weighted_cost = total_cost / len(y_true)

        return {
            "fatal_miss_rate":  float(fatal_miss_rate),
            "false_fatal_rate": float(false_fatal_rate),
            "weighted_cost":    float(weighted_cost),
        }

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
        safe     = run_name.lower().replace(" ", "_")
        cm_path  = ARTIFACTS_DIR / f"{safe}_cm.png"
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
        dict with keys: accuracy, macro_f1, weighted_f1, fatal_miss_rate,
                        false_fatal_rate, weighted_cost.
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

        # ── Standard metrics ──────────────────────────────────────────────────
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

        # ── Business metrics ──────────────────────────────────────────────────
        business = self._compute_business_metrics(y_true, y_pred)

        # ── Console output ────────────────────────────────────────────────────
        print(f"\n{'=' * 42}")
        print(f"  {run_name}")
        print(f"{'=' * 42}")
        print(f"  Accuracy        : {acc:.4f}")
        print(f"  Macro F1        : {macro_f1:.4f}")
        print(f"  Weighted F1     : {w_f1:.4f}")
        print(f"  Fatal miss rate : {business['fatal_miss_rate']:.4f}  (Fatal→Slight errors / all Fatal)")
        print(f"  False fatal rate: {business['false_fatal_rate']:.4f}  (Slight→Fatal errors / all Slight)")
        print(f"  Weighted cost   : {business['weighted_cost']:.2f}  (mean misclassification cost/sample)")
        print()
        print(classification_report(
            y_true, y_pred,
            labels=self.class_names,
            target_names=self.class_names,
            zero_division=0,
        ))

        # ── Artifacts ─────────────────────────────────────────────────────────
        cm_path     = self._save_confusion_matrix(y_true, y_pred, run_name)
        report_path = self._save_report_json(report_dict, run_name)

        # ── MLflow logging ────────────────────────────────────────────────────
        ctx = mlflow.start_run(run_name=run_name) if log_to_mlflow else nullcontext()

        with ctx:
            if log_to_mlflow:
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
                mlflow.log_metric("fatal_miss_rate",  business["fatal_miss_rate"])
                mlflow.log_metric("false_fatal_rate", business["false_fatal_rate"])
                mlflow.log_metric("weighted_cost",    business["weighted_cost"])

                # Artifact files
                mlflow.log_artifact(str(cm_path),     artifact_path="plots")
                mlflow.log_artifact(str(report_path), artifact_path="reports")

        all_metrics = {
            "accuracy":         float(acc),
            "macro_f1":         float(macro_f1),
            "weighted_f1":      float(w_f1),
            **business,
        }
        return all_metrics