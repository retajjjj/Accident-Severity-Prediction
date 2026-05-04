"""
random_forest.py

Contract (shared across all models):
  fit(X_train, y_train)   → trains; y_train must be string labels
  predict(X_test)         → np.ndarray of string labels
  predict_proba(X_test)   → (n_samples, 3) float array,
                            columns always in ["Fatal", "Serious", "Slight"] order
"""

import numpy as np
from sklearn.ensemble import RandomForestClassifier

LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class RandomForestModel:
    """
    Random Forest classifier.

    NOTE: Assumes training data is already class-balanced (e.g. via SMOTE).
          class_weight is intentionally left as None.
    """

    def __init__(
        self,
        n_estimators: int = 200,
        max_depth: int = 20,
        min_samples_split: int = 5,
        min_samples_leaf: int = 2,
        random_state: int = 42,
        n_jobs: int = -1,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._proba_cols: list[int] | None = None

    def fit(self, X_train, y_train):
        y_train = np.array(y_train, dtype=str)
        self.model.fit(X_train, y_train)

        internal = list(self.model.classes_)
        self._proba_cols = [internal.index(c) for c in LABEL_ORDER if c in internal]
        return self

    def predict(self, X_test) -> np.ndarray:
        """Returns string labels."""
        return np.array(self.model.predict(X_test), dtype=str)

    def predict_proba(self, X_test) -> np.ndarray:
        """Returns (n_samples, 3) in [Fatal, Serious, Slight] order."""
        raw = self.model.predict_proba(X_test)
        return raw[:, self._proba_cols]