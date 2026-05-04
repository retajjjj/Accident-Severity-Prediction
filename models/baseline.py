"""
baseline.py
Dummy classifier used as a sanity-check lower bound.
"""

from sklearn.dummy import DummyClassifier


class BaselineModel:
    """Thin wrapper around sklearn's DummyClassifier."""

    def __init__(self, strategy: str = "most_frequent", constant=None):
        """
        Parameters
        ----------
        strategy : str
            Any strategy accepted by DummyClassifier
            ('most_frequent', 'stratified', 'uniform', 'constant').
        constant : label, optional
            Required when strategy='constant'.
        """
        kwargs = {"strategy": strategy}
        if strategy == "constant":
            if constant is None:
                raise ValueError("'constant' must be set when strategy='constant'.")
            kwargs["constant"] = constant

        self.model = DummyClassifier(**kwargs)

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test)