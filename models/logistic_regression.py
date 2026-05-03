import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class LogisticRegressionModel:
    def __init__(self, max_iter=2000, C=0.1):
        self.model = LogisticRegression(
            max_iter=max_iter,
            class_weight="balanced",
            C=C,
            solver="saga",
        )
        self._le          = LabelEncoder()
        self._proba_cols  = None   # column reorder for predict_proba

    def fit(self, X_train, y_train):
        y_train = np.array(y_train, dtype=str)
        y_enc   = self._le.fit_transform(y_train)   # str → int, remembers mapping
        self.model.fit(X_train, y_enc)

        # Precompute column order so predict_proba → [Fatal, Serious, Slight]
        internal = list(self._le.classes_)           # e.g. ['Fatal','Serious','Slight'] alphabetical
        self._proba_cols = [internal.index(c) for c in LABEL_ORDER if c in internal]
        return self

    def predict(self, X_test):
        """Returns string labels, never integers."""
        enc_preds = self.model.predict(X_test)
        return np.array(self._le.inverse_transform(enc_preds), dtype=str)

    def predict_proba(self, X_test):
        """(n_samples, 3) in [Fatal, Serious, Slight] order."""
        raw = self.model.predict_proba(X_test)
        return raw[:, self._proba_cols]