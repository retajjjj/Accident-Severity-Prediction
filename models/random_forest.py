from sklearn.ensemble import RandomForestClassifier


class RandomForestModel:
    def __init__(self, n_estimators=200, max_depth=20, min_samples_split=5,
                 min_samples_leaf=2, random_state=42, n_jobs=-1):
        self.model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            min_samples_split=min_samples_split,
            min_samples_leaf=min_samples_leaf,
            random_state=random_state,
            n_jobs=n_jobs,
        )

    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
