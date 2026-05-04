"""
xgboost_model.py

Contract (shared across all models):
  fit(X_train, y_train)   → trains; y_train must be string labels
  predict(X_test)         → np.ndarray of string labels
  predict_proba(X_test)   → (n_samples, 3) float array,
                            columns always in ["Fatal", "Serious", "Slight"] order
"""

import numpy as np
from xgboost import XGBClassifier

LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class XGBoostModel:
    """
    XGBoost multiclass classifier.

    NOTE: Assumes training data is already class-balanced (e.g. via SMOTE).
          sample_weight and scale_pos_weight are intentionally omitted.
    """

    def __init__(
        self,
        n_estimators: int = 500,
        max_depth: int = 10,
        learning_rate: float = 0.1,
        subsample: float = 0.8,
        colsample_bytree: float = 0.8,
        min_child_weight: int = 30,
        gamma: float = 0.5,
        random_state: int = 42,
    ):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            min_child_weight=min_child_weight,
            gamma=gamma,
            random_state=random_state,
            objective="multi:softprob",
            eval_metric="mlogloss",
            tree_method="hist",
        )

        self.classes_: list[str] | None = None
        self._to_idx: dict[str, int] | None = None
        self._to_label: dict[int, str] | None = None
        self._proba_cols: list[int] | None = None

    def fit(self, X_train, y_train):
        y_train = np.array(y_train, dtype=str)

        # Alphabetical sort → deterministic label encoding across runs
        self.classes_ = sorted(np.unique(y_train).tolist())
        self._to_idx = {c: i for i, c in enumerate(self.classes_)}
        self._to_label = {i: c for i, c in enumerate(self.classes_)}

        y_enc = np.array([self._to_idx[c] for c in y_train], dtype=np.int64)
        self.model.fit(X_train, y_enc)

        self._proba_cols = [self._to_idx[c] for c in LABEL_ORDER if c in self._to_idx]
        return self

    def predict(self, X_test) -> np.ndarray:
        """Returns string labels."""
        enc_preds = self.model.predict(X_test)
        return np.array([self._to_label[int(p)] for p in enc_preds], dtype=str)

    def predict_proba(self, X_test) -> np.ndarray:
        """Returns (n_samples, 3) in [Fatal, Serious, Slight] order."""
        raw = self.model.predict_proba(X_test)
        return raw[:, self._proba_cols]