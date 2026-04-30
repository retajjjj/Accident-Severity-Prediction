from catboost import CatBoostClassifier


class CatBoostModel:
    def __init__(self, iterations=500, depth=6, learning_rate=0.1,
                 random_seed=42, verbose=0):
        self.model = CatBoostClassifier(
            iterations=iterations,
            depth=depth,
            learning_rate=learning_rate,
            random_seed=random_seed,
            verbose=verbose,
            auto_class_weights="Balanced",
        )

    def fit(self, X_train, y_train):
        self.model.fit(X_train, y_train)
        return self

    def predict(self, X_test):
        return self.model.predict(X_test).flatten()
