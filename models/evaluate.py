import mlflow
import mlflow.sklearn
import mlflow.xgboost
import mlflow.catboost
import pickle
import json
from pathlib import Path
from typing import Any, Dict, cast
import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    classification_report,
    confusion_matrix,
)

from models.baseline import BaselineModel
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel

ARTIFACTS_DIR = Path("reports") / "mlflow_artifacts"
ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)


def load_processed_split(split_name: str):
    file_path = f"data/processed/{split_name}.pkl"
    with open(file_path, "rb") as f:
        return pickle.load(f)


class Evaluate:
    def __init__(self, X_test, y_test, model, class_names=['Fatal', 'Serious', 'Slight']):
        self.X_test = X_test
        self.y_test = y_test
        self.model = model
        self.class_names = class_names

    def _get_estimator(self):
        """Support wrapper classes that store the sklearn estimator in `.model`."""
        return getattr(self.model, "model", self.model)

    def evaluate(self, run_name):
        with mlflow.start_run(run_name=run_name):
            estimator = self._get_estimator()

            # Log estimator parameters when available.
            if hasattr(estimator, "get_params"):
                params = estimator.get_params()
                mlflow.log_params({k: str(v) for k, v in params.items()})

            y_pred = self.model.predict(self.X_test)

            print(f"\n{'=' * 40}")
            print(f"Evaluating {self.model.__class__.__name__}...")
            print(f"{'=' * 40}")

            # 1. Accuracy and summary metrics
            acc = accuracy_score(self.y_test, y_pred)
            report_dict: Dict[str, Any] = cast(
                Dict[str, Any],
                classification_report(
                    self.y_test,
                    y_pred,
                    labels=self.class_names,
                    target_names=self.class_names,
                    zero_division=0,
                    output_dict=True,
                ),
            )

            macro_f1: float = report_dict["macro avg"]["f1-score"]
            weighted_f1: float = report_dict["weighted avg"]["f1-score"]
            mlflow.log_metric("accuracy", float(acc))
            mlflow.log_metric("macro_f1", float(macro_f1))
            mlflow.log_metric("weighted_f1", float(weighted_f1))

            for class_name in self.class_names:
                if class_name in report_dict:
                    mlflow.log_metric(f"{class_name}_precision", float(report_dict[class_name]["precision"]))
                    mlflow.log_metric(f"{class_name}_recall", float(report_dict[class_name]["recall"]))
                    mlflow.log_metric(f"{class_name}_f1", float(report_dict[class_name]["f1-score"]))

            print(f"Overall Accuracy: {acc:.4f}  <-- (Caution: Check minority class recall!)\n")

            # 2. Per-class metrics
            print("--- Classification Report ---")
            print(classification_report(
                self.y_test,
                y_pred,
                labels=self.class_names,
                target_names=self.class_names,
                zero_division=0,
            ))

            # 3. Confusion matrix artifact
            print("--- Confusion Matrix ---")
            cm = confusion_matrix(self.y_test, y_pred, labels=self.class_names)
            fig, ax = plt.subplots(figsize=(7, 6))
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=self.class_names)
            disp.plot(cmap="Blues", values_format="d", ax=ax, colorbar=False)
            ax.set_title(f"{self.model.__class__.__name__} Confusion Matrix")

            safe_run_name = run_name.lower().replace(" ", "_")
            cm_path = ARTIFACTS_DIR / f"{safe_run_name}_confusion_matrix.png"
            fig.savefig(cm_path, dpi=150, bbox_inches="tight")
            plt.close(fig)
            mlflow.log_artifact(str(cm_path), artifact_path="plots")

            report_path = ARTIFACTS_DIR / f"{safe_run_name}_classification_report.json"
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report_dict, f, indent=2)
            mlflow.log_artifact(str(report_path), artifact_path="reports")

            # 4. Model artifact — use correct MLflow flavor per model type
            try:
                from xgboost import XGBClassifier
                from catboost import CatBoostClassifier
                if isinstance(estimator, XGBClassifier):
                    mlflow.xgboost.log_model(estimator, artifact_path="model")
                elif isinstance(estimator, CatBoostClassifier):
                    mlflow.catboost.log_model(estimator, artifact_path="model")
                else:
                    mlflow.sklearn.log_model(estimator, artifact_path="model")
            except Exception:
                mlflow.sklearn.log_model(estimator, artifact_path="model")


if __name__ == "__main__":
    X_train = load_processed_split("X_train")
    y_train = load_processed_split("y_train")
    X_test = load_processed_split("X_test")
    y_test = load_processed_split("y_test")

    mlflow.set_experiment("Accident_Severity_Pipeline")
    correct_labels = ['Fatal', 'Serious', 'Slight']

    models = [
        ("Baseline_Constant_Slight", BaselineModel(strategy="constant", constant="Slight")),
        ("Baseline_Stratified", BaselineModel(strategy="stratified")),
        ("Baseline_Most_Frequent", BaselineModel(strategy="most_frequent")),
        ("Logistic_Regression", LogisticRegressionModel()),
        ("Random_Forest", RandomForestModel()),
        ("XGBoost", XGBoostModel()),
        ("CatBoost", CatBoostModel()),
    ]

    for run_name, model in models:
        print(f"\nTraining {run_name}...")
        model.fit(X_train, y_train)
        evaluator = Evaluate(X_test, y_test, model, class_names=correct_labels)
        evaluator.evaluate(run_name=run_name)