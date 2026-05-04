"""
lightgbm_model.py  (save as LightGBM.py to match your current import)

Contract (same for every model in this project):
  fit(X_train, y_train)          → trains; y_train must be string labels
  predict(X_test)                → returns np.ndarray of string labels
  predict_proba(X_test)          → returns (n_samples, 3) float array,
                                   columns ALWAYS in ["Fatal","Serious","Slight"] order
"""

import numpy as np
from lightgbm import LGBMClassifier

LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class LGBMModel:
    def __init__(
        self,
        n_estimators=1000,
        learning_rate=0.05,
        num_leaves=63,              # key param: controls model complexity (2^max_depth - 1)
        min_child_samples=20,       # min samples in a leaf — critical for Fatal imbalance
        class_weight="balanced",
        subsample=0.8,
        colsample_bytree=0.8,
        reg_alpha=0.1,              # L1 regularization
        reg_lambda=1.0,             # L2 regularization
        random_state=42,
        n_jobs=-1,
        verbose=-1,                 # silence training output
    ):
        self.model = LGBMClassifier(
            n_estimators=n_estimators,
            learning_rate=learning_rate,
            num_leaves=num_leaves,
            min_child_samples=min_child_samples,
            class_weight=class_weight,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            reg_alpha=reg_alpha,
            reg_lambda=reg_lambda,
            random_state=random_state,
            n_jobs=n_jobs,
            objective="multiclass",
            metric="multi_logloss",
            verbose=verbose,
        )

        self._proba_cols = None     # column reorder for predict_proba

    # ------------------------------------------------------------------

    def fit(self, X_train, y_train):
        y_train = np.array(y_train, dtype=str)
        self.model.fit(X_train, y_train)

        # LightGBM stores classes in self.model.classes_ after fit
        internal_order = list(self.model.classes_)

        # Safety: if a class is absent from training data, skip it
        missing = [c for c in LABEL_ORDER if c not in internal_order]
        if missing:
            print(f"[WARN] LGBMModel: classes missing from training data: {missing}")

        self._proba_cols = [internal_order.index(c) for c in LABEL_ORDER
                            if c in internal_order]
        return self

    def predict(self, X_test):
        """Returns string labels."""
        raw = self.model.predict(X_test)
        return np.array(raw, dtype=str)

    def predict_proba(self, X_test):
        """
        Returns (n_samples, 3) with columns in [Fatal, Serious, Slight] order.
        """
        raw = self.model.predict_proba(X_test)          # (n, n_classes)
        return raw[:, self._proba_cols]