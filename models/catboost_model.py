"""
catboost_model.py

Contract (shared across all models):
  fit(X_train, y_train)   → trains; y_train must be string labels
  predict(X_test)         → np.ndarray of string labels
  predict_proba(X_test)   → (n_samples, 3) float array,
                            columns always in ["Fatal", "Serious", "Slight"] order
"""

import numpy as np
from catboost import CatBoostClassifier

LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class CatBoostModel:
    """
    CatBoost multiclass classifier.

    NOTE: Assumes training data is already class-balanced (e.g. via SMOTE).
          class_weights is intentionally left as None.
    """

    def __init__(
        self,
        iterations: int = 1000,
        depth: int = 6,
        learning_rate: float = 0.05,
        l2_leaf_reg: float = 5.0,
        min_data_in_leaf: int = 10,
        random_seed: int = 42,
        verbose: int = 0,
    ):
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            min_data_in_leaf=min_data_in_leaf,
            random_seed=random_seed,
            verbose=verbose,
            loss_function="MultiClass",
            eval_metric="TotalF1:average=Macro",
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
        raw = self.model.predict(X_test)
        return np.array(raw, dtype=str).flatten()

    def predict_proba(self, X_test) -> np.ndarray:
        """Returns (n_samples, 3) in [Fatal, Serious, Slight] order."""
        raw = self.model.predict_proba(X_test)
        return raw[:, self._proba_cols]