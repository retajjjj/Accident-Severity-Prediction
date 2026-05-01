from xgboost import XGBClassifier
from sklearn.utils.class_weight import compute_sample_weight

class XGBoostModel:
    def __init__(self, n_estimators=200, max_depth=6, learning_rate=0.1,
                 subsample=0.8, colsample_bytree=0.8, random_state=42,
                 eval_metric="mlogloss"):
        self.model = XGBClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            subsample=subsample,
            colsample_bytree=colsample_bytree,
            random_state=random_state,
            eval_metric=eval_metric,
        )
        self._label_map = None
        self._inverse_label_map = None

    def fit(self, X_train, y_train):
        labels = sorted(y_train.unique())
        self._label_map = {label: idx for idx, label in enumerate(labels)}
        self._inverse_label_map = {idx: label for label, idx in self._label_map.items()}
        y_encoded = y_train.map(self._label_map)
        sample_weights = compute_sample_weight(class_weight='balanced', y=y_encoded)
        self.model.fit(X_train, y_encoded, sample_weight=sample_weights) 
        return self

    def predict(self, X_test):
        import pandas as pd
        y_pred_encoded = self.model.predict(X_test)
        return pd.Series(y_pred_encoded).map(self._inverse_label_map).values
