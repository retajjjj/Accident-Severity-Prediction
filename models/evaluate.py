"""
evaluate.py — consistent evaluation class for all models.

Key fix: always calls self.model.predict() (the wrapper), never the raw
inner estimator. This ensures XGBoost's integer-to-label mapping is applied.
"""

import mlflow
import json
from pathlib import Path
from contextlib import contextmanager
from typing import Any, Dict, cast

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

ARTIFACTS_DIR = Path("reports") / "mlflow_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)

# Ordered class list used everywhere in this project
CORRECT_LABELS = ["Fatal", "Serious", "Slight"]


@contextmanager
def _nullcontext():
    """No-op context manager for log_to_mlflow=False branches."""
    yield


class Evaluate:
    def __init__(self, X_test, y_test, model, class_names=None):
        self.X_test = X_test
        self.y_test = np.array(y_test, dtype=str)
        self.model = model
        self.class_names = list(class_names) if class_names is not None else CORRECT_LABELS

    # ------------------------------------------------------------------
    # IMPORTANT: never call _get_estimator().predict() — it bypasses the
    # wrapper's label-decoding step (critical for XGBoost).
    # Use self.model.predict() for predictions at all times.
    # _get_estimator() is kept only for logging params to MLflow.
    # ------------------------------------------------------------------
    def _get_estimator(self):
        return getattr(self.model, "model", self.model)

    def evaluate(self, run_name: str, y_pred=None, log_to_mlflow: bool = True):
        """
        Evaluate the model on self.X_test / self.y_test.

        Parameters
        ----------
        run_name      : MLflow run name (also used for artifact filenames).
        y_pred        : Optional pre-computed predictions (string labels).
                        If None, calls self.model.predict(self.X_test).
        log_to_mlflow : If False, prints only — no MLflow run is created.
                        Use this when calling from inside an existing run.
        """
        ctx = mlflow.start_run(run_name=run_name) if log_to_mlflow else _nullcontext()

        with ctx:
            # ── Predictions ───────────────────────────────────────────
            if y_pred is None:
                # Always call the wrapper's predict(), not the raw estimator.
                y_pred = self.model.predict(self.X_test)

            y_pred = np.array(y_pred, dtype=str)
            y_true = self.y_test  # already str from __init__

            # Sanity check: warn if unexpected labels slip through
            unexpected = set(y_pred) - set(self.class_names)
            if unexpected:
                print(f"[WARN] predict() returned unexpected labels: {unexpected}")

            # ── Metrics ───────────────────────────────────────────────
            acc       = accuracy_score(y_true, y_pred)
            macro_f1  = f1_score(y_true, y_pred, average="macro",    zero_division=0)
            w_f1      = f1_score(y_true, y_pred, average="weighted", zero_division=0)

            report_dict: Dict[str, Any] = cast(
                Dict[str, Any],
                classification_report(
                    y_true, y_pred,
                    labels=self.class_names,
                    target_names=self.class_names,
                    output_dict=True,
                    zero_division=0,
                ),
            )

            # ── Print ─────────────────────────────────────────────────
            print(f"\n{'=' * 42}")
            print(f"  {run_name}")
            print(f"{'=' * 42}")
            print(f"  Accuracy  : {acc:.4f}")
            print(f"  Macro F1  : {macro_f1:.4f}")
            print(f"  Weighted F1: {w_f1:.4f}")
            print()
            print(classification_report(
                y_true, y_pred,
                labels=self.class_names,
                target_names=self.class_names,
                zero_division=0,
            ))

            # ── MLflow logging ────────────────────────────────────────
            if log_to_mlflow:
                estimator = self._get_estimator()
                if hasattr(estimator, "get_params"):
                    try:
                        mlflow.log_params({k: str(v) for k, v in estimator.get_params().items()})
                    except Exception:
                        pass

                mlflow.log_metric("accuracy",    float(acc))
                mlflow.log_metric("macro_f1",    float(macro_f1))
                mlflow.log_metric("weighted_f1", float(w_f1))

                for cls in self.class_names:
                    if cls in report_dict:
                        mlflow.log_metric(f"{cls}_f1",        float(report_dict[cls]["f1-score"]))
                        mlflow.log_metric(f"{cls}_recall",    float(report_dict[cls]["recall"]))
                        mlflow.log_metric(f"{cls}_precision", float(report_dict[cls]["precision"]))

            # ── Confusion matrix artifact ─────────────────────────────
            cm = confusion_matrix(y_true, y_pred, labels=self.class_names)
            fig, ax = plt.subplots(figsize=(7, 6))
            ConfusionMatrixDisplay(cm, display_labels=self.class_names).plot(
                cmap="Blues", values_format="d", ax=ax, colorbar=False
            )
            ax.set_title(run_name)
            safe = run_name.lower().replace(" ", "_")
            cm_path = ARTIFACTS_DIR / f"{safe}_cm.png"
            fig.savefig(cm_path, bbox_inches="tight", dpi=150)
            plt.close(fig)

            if log_to_mlflow:
                mlflow.log_artifact(str(cm_path), artifact_path="plots")

                report_path = ARTIFACTS_DIR / f"{safe}_report.json"
                with open(report_path, "w") as f:
                    json.dump(report_dict, f, indent=2)
                mlflow.log_artifact(str(report_path), artifact_path="reports")