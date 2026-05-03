"""
catboost_model.py

Contract (same for every model in this project):
  fit(X_train, y_train)          → trains; y_train must be string labels
  predict(X_test)                → returns np.ndarray of string labels
  predict_proba(X_test)          → returns (n_samples, 3) float array,
                                   columns ALWAYS in ["Fatal","Serious","Slight"] order
"""

import numpy as np
from catboost import CatBoostClassifier

LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class CatBoostModel:
    def __init__(
        self,
        iterations=1000,
        depth=6,
        learning_rate=0.05,
        l2_leaf_reg=5,              # L2 regularization on leaves — tune this for Fatal
        min_data_in_leaf=10,        # prevents leaf overfitting on rare Fatal samples
        random_seed=42,
        verbose=0,
    ):
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            l2_leaf_reg=l2_leaf_reg,
            min_data_in_leaf=min_data_in_leaf,
            random_seed=random_seed,
            verbose=verbose,
            auto_class_weights="Balanced",  # handles Fatal imbalance automatically
            eval_metric="TotalF1:average=Macro",  # optimize for what we measure
            loss_function="MultiClass",
        )

        self._proba_cols = None     # column reorder for predict_proba

    # ------------------------------------------------------------------

    def fit(self, X_train, y_train):
        y_train = np.array(y_train, dtype=str)
        self.model.fit(X_train, y_train)

        # CatBoost stores classes in self.model.classes_
        # Precompute reorder indices so predict_proba → [Fatal, Serious, Slight]
        internal_order = list(self.model.classes_)
        self._proba_cols = [internal_order.index(c) for c in LABEL_ORDER
                            if c in internal_order]
        return self

    def predict(self, X_test):
        """Returns string labels."""
        raw = self.model.predict(X_test)
        # CatBoost returns shape (n,1) for multiclass — flatten it
        return np.array(raw, dtype=str).flatten()

    def predict_proba(self, X_test):
        """
        Returns (n_samples, 3) with columns in [Fatal, Serious, Slight] order.
        """
        raw = self.model.predict_proba(X_test)          # (n, n_classes)
        return raw[:, self._proba_cols]