from sklearn.linear_model import LogisticRegression

class LogisticRegressionModel:
    def __init__(self, max_iter=2000, C=0.1):
        self.model = LogisticRegression(
            max_iter=max_iter,
            class_weight='balanced',   # ← was missing
            C=C,                       # regularization — tune this
            solver='saga',             # handles large datasets better
            multi_class='multinomial'
        )

    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)

    def predict(self, X_test):
        return self.model.predict(X_test)