from sklearn.dummy import DummyClassifier

class BaselineModel:
    # Set default to 'most_frequent' but allow overrides
    def __init__(self, strategy="most_frequent", constant=None):
        self.strategy = strategy
        self.constant = constant
        
        # If strategy is constant, we must pass the constant value
        if strategy == "constant":
            self.model = DummyClassifier(strategy=strategy, constant=constant)
        else:
            self.model = DummyClassifier(strategy=strategy)

    def fit(self, X_train, y_train):
        return self.model.fit(X_train, y_train)
    
    def predict(self, X_test):
        return self.model.predict(X_test)