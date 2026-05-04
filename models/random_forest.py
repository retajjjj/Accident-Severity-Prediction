import numpy as np
from sklearn.ensemble import RandomForestClassifier

LABEL_ORDER = ["Fatal", "Serious", "Slight"]


class RandomForestModel:
    def __init__(
        self,
        n_estimators=200,
        max_depth=20,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            class_weight=class_weight,
            random_state=random_state,
            n_jobs=n_jobs,
        )
        self._proba_cols = None

    def fit(self, X_train, y_train):
        y_train = np.array(y_train, dtype=str)
        self.model.fit(X_train, y_train)   

        internal = list(self.model.classes_)
        self._proba_cols = [internal.index(c) for c in LABEL_ORDER if c in internal]
        return self

    def predict(self, X_test):
        return np.array(self.model.predict(X_test), dtype=str)

    def predict_proba(self, X_test):
        """(n_samples, 3) in [Fatal, Serious, Slight] order."""
        raw = self.model.predict_proba(X_test)
        return raw[:, self._proba_cols]