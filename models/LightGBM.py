"""
LightGBM.py

Contract (shared across all models):
  fit(X_train, y_train)   → trains; y_train must be string labels
  predict(X_test)         → np.ndarray of string labels
  predict_proba(X_test)   → (n_samples, 3) float array,
                            columns always in ["Fatal", "Serious", "Slight"] order
"""

import numpy as np
from lightgbm import LGBMClassifier

LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class LGBMModel:
    """
    LightGBM multiclass classifier.

    NOTE: Assumes training data is already class-balanced (e.g. via SMOTE).
          class_weight is intentionally left as None.
    """

    def __init__(
        self,
        n_estimators: int = 1000,
        learning_rate: float = 0.05,
        num_leaves: int = 63,
        min_child_samples: int = 20,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        reg_alpha: float = 0.1,
        reg_lambda: float = 1.0,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            class_weight=None,
            objective="multiclass",
            metric="multi_logloss",
            random_state=random_state,
            n_jobs=n_jobs,
            verbose=-1,
        )
        self._proba_cols: list[int] | None = None

    def fit(self, X_train, y_train):
        y_train = np.array(y_train, dtype=str)
        self.model.fit(X_train, y_train)

        internal = list(self.model.classes_)
        missing = [c for c in LABEL_ORDER if c not in internal]
        if missing:
            raise ValueError(f"Classes missing from training data: {missing}")

        self._proba_cols = [internal.index(c) for c in LABEL_ORDER]
        return self

    def predict(self, X_test) -> np.ndarray:
        """Returns string labels."""
        return np.array(self.model.predict(X_test), dtype=str)

    def predict_proba(self, X_test) -> np.ndarray:
        """Returns (n_samples, 3) in [Fatal, Serious, Slight] order."""
        raw = self.model.predict_proba(X_test)
        return raw[:, self._proba_cols]