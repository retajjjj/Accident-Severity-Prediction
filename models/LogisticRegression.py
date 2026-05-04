"""
logistic_regression.py

Contract (shared across all models):
  fit(X_train, y_train)   → trains; y_train must be string labels
  predict(X_test)         → np.ndarray of string labels
  predict_proba(X_test)   → (n_samples, 3) float array,
                            columns always in ["Fatal", "Serious", "Slight"] order
"""

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class LogisticRegressionModel:
    """
    Multinomial logistic regression with SAGA solver.

    NOTE: Assumes training data is already class-balanced (e.g. via SMOTE).
          class_weight is intentionally left as None.
    """

    def __init__(self, max_iter: int = 2000, C: float = 0.1):
        self.model = LogisticRegression(
            max_iter=max_iter,
            C=C,
            solver="saga",
            multi_class="multinomial",
            n_jobs=-1,
        )
        self._le = LabelEncoder()
        self._proba_cols: list[int] | None = None

    def fit(self, X_train, y_train):
        y_train = np.array(y_train, dtype=str)
        y_enc = self._le.fit_transform(y_train)
        self.model.fit(X_train, y_enc)

        internal = list(self._le.classes_)
        self._proba_cols = [internal.index(c) for c in LABEL_ORDER if c in internal]
        return self

    def predict(self, X_test) -> np.ndarray:
        """Returns string labels."""
        enc_preds = self.model.predict(X_test)
        return np.array(self._le.inverse_transform(enc_preds), dtype=str)

    def predict_proba(self, X_test) -> np.ndarray:
        """Returns (n_samples, 3) in [Fatal, Serious, Slight] order."""
        raw = self.model.predict_proba(X_test)
        return raw[:, self._proba_cols]
