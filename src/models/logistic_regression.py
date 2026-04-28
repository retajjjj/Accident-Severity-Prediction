from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, max_iter=1000):
        self.model = LogisticRegression(max_iter=max_iter)

    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)
