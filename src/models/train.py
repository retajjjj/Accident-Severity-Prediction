import sys
from pathlib import Path
import pickle

import mlflow
import mlflow.sklearn

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from models.baseline import BaselineModel
from models.logistic_regression import LogisticRegressionModel
from models.random_forest import RandomForestModel
from models.xgboost_model import XGBoostModel
from models.catboost_model import CatBoostModel
from models.evaluate import Evaluate

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
CORRECT_LABELS = ["Fatal", "Serious", "Slight"]


def load_split(name: str):
    path = PROCESSED_DIR / f"{name}.pkl"
    with open(path, "rb") as f:
        return pickle.load(f)


def main():
    X_train = load_split("X_train")
    y_train = load_split("y_train")
    X_test = load_split("X_test")
    y_test = load_split("y_test")

    mlflow.set_experiment("Accident_Severity_Pipeline")

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
        evaluator = Evaluate(X_test, y_test, model, class_names=CORRECT_LABELS)
        evaluator.evaluate(run_name=run_name)


if __name__ == "__main__":
    main()
