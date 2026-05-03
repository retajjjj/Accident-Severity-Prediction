"""
xgboost_model.py

Contract (same for every model in this project):
  fit(X_train, y_train)          → trains; y_train must be string labels
  predict(X_test)                → returns np.ndarray of string labels
  predict_proba(X_test)          → returns (n_samples, 3) float array,
                                columns ALWAYS in ["Fatal","Serious","Slight"] order
"""

import numpy as np
from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

# Canonical class order used by every model and the threshold tuner
LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class XGBoostModel:
    def __init__(
        self,
        n_estimators=500,
        max_depth=6,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=10,        # key param for imbalance — prevents overfitting to rare Fatal leaves
        gamma=0.5,                  # min loss reduction for a split
        random_state=42,
        eval_metric="mlogloss",
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
            eval_metric=eval_metric,
            objective="multi:softprob",
            tree_method="hist",     # faster on large datasets
            use_label_encoder=False,
        )

        # Set after fit() — maps between integer indices and string labels
        self.classes_     = None    # ordered string labels e.g. ['Fatal','Serious','Slight']
        self._to_idx      = None    # str → int
        self._to_label    = None    # int → str
        self._proba_cols  = None    # column indices to reorder predict_proba output

    # ------------------------------------------------------------------

    def fit(self, X_train, y_train):
        y_train = np.array(y_train, dtype=str)

        # Sort alphabetically so indices are deterministic across runs
        self.classes_ = sorted(np.unique(y_train).tolist())
        self._to_idx   = {c: i for i, c in enumerate(self.classes_)}
        self._to_label = {i: c for i, c in enumerate(self.classes_)}

        y_enc = np.array([self._to_idx[c] for c in y_train], dtype=np.int64)

        sample_weights = compute_sample_weight(class_weight="balanced", y=y_enc)

        self.model.fit(X_train, y_enc, sample_weight=sample_weights)

        # Precompute column reorder so predict_proba is always [Fatal, Serious, Slight]
        self._proba_cols = [self._to_idx[c] for c in LABEL_ORDER if c in self._to_idx]

        return self

    def predict(self, X_test):
        """Returns string labels, never integers."""
        enc_preds = self.model.predict(X_test)                        # int64 array
        return np.array([self._to_label[int(p)] for p in enc_preds], dtype=str)

    def predict_proba(self, X_test):
        """
        Returns (n_samples, 3) with columns in [Fatal, Serious, Slight] order.
        This is mandatory for threshold tuning to work correctly.
        """
        raw = self.model.predict_proba(X_test)                        # (n, n_classes)
        return raw[:, self._proba_cols]