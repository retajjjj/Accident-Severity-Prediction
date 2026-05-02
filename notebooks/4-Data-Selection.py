from sklearn.feature_selection import RFECV
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier
import pandas as pd
import numpy as np
import pickle


df = pd.read_pickle("../data/interim/engineered_data.pkl")

X = df.drop('Accident_Severity', axis=1) # Features
y = df['Accident_Severity']              # Target
X = X.fillna(X.mean())

model = LogisticRegression()

min_features_to_select = 10 
clf = LogisticRegression()
cv = StratifiedKFold(5)

rfecv = RFECV(
    estimator=clf,
    step=1,
    cv=cv,
    scoring="accuracy",
    min_features_to_select=min_features_to_select,
    n_jobs=2,
)
rfecv.fit(X, y)

print(f"Optimal number of features: {rfecv.n_features_}")
# Returns an array of the selected feature names
selected_features = rfecv.get_feature_names_out()
print(selected_features)


X_Selectkbest = X[selected_features]
X_Selectkbest.head()
X_train, X_test, y_train, y_test = train_test_split(X_Selectkbest, y, test_size=0.20, random_state=0)


with open("../data/interim/x_train.pkl", "wb") as f:
    pickle.dump(X_train, f)
    
with open("../data/interim/X_test.pkl", "wb") as f:
    pickle.dump(X_test, f)
    
with open("../data/interim/y_train.pkl", "wb") as f:
    pickle.dump(y_train, f)

with open("../data/interim/y_test.pkl", "wb") as f:
    pickle.dump(y_test, f)
    
    
#model testing
from sklearn.utils.class_weight import compute_class_weight

classes_array = np.unique(y_train)
weights = compute_class_weight(
    class_weight="balanced",
    classes= classes_array,
    y=y_train
)

class_weight_dict = dict(zip([0, 1, 2], weights))

model = RandomForestClassifier(class_weight=class_weight_dict)
model.fit(X_train, y_train)
y_pred = model.predict(X_train)

Lg_score_seen = accuracy_score(y_train, y_pred)
print('The Accuracy score for Random forest on seen data : ', Lg_score_seen)

y_test_pred = model.predict(X_test)
Lg_score_unseen = accuracy_score(y_test, y_test_pred)
print('The Accuracy score for Random forest on unseen data : ', Lg_score_unseen)


matrix = confusion_matrix(y_test, y_test_pred)


print(classification_report(y_test, y_test_pred))
##########################################################################
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

undersample = RandomUnderSampler(sampling_strategy={
    0: 400000,   # reduce from 1.1M to 400k
    1: 134175,   
    2: 13103,   
})

oversample = SMOTE(sampling_strategy={
    0:  400000,    
    1:  134175,   
    2:  80000,   
}, k_neighbors=5)

pipeline = Pipeline([
    ("under", undersample),
    ("over",  oversample)
])

X_resampled, y_resampled = pipeline.fit_resample(X_train, y_train)
print(pd.Series(y_resampled).value_counts())

model.fit(X_resampled, y_resampled)
y_pred = model.predict(X_resampled)

Lg_score_seen = accuracy_score(y_resampled, y_pred)
print('The Accuracy score for Random forest on seen data : ', Lg_score_seen)

y_test_pred = model.predict(X_test)
Lg_score_unseen = accuracy_score(y_test, y_test_pred)
print('The Accuracy score for Random forest on unseen data : ', Lg_score_unseen)

matrix = confusion_matrix(y_test, y_test_pred)
print(classification_report(y_test, y_test_pred))


with open("../data/interim/transformed_data.pkl", "wb") as f:
    pickle.dump(df, f)