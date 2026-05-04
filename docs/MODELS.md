# Models & Evaluation — Accident Severity Prediction

## Task

Multi-class classification predicting `Accident_Severity` ∈ {**Fatal**, **Serious**, **Slight**}.

**Training data:** 2,778,678 rows, SMOTE-balanced (33% each class)  
**Test data:** 543,124 rows, real-world imbalanced (85% Slight, 13% Serious, 1% Fatal)  
**Features (9):** `Engine_Capacity_.CC.`, `Vehicle_Type`, `temp_road_risk`, `Vehicle_Manoeuvre`, `Vehicle_Leaving_Carriageway`, `Time`, `Junction_Location`, `LSOA_of_Accident_Location`, `Vehicle_Reference`

---

## Evaluation Metrics

Every model is assessed on the same held-out test split using:

| Metric                                | Why it matters here                                                                                                       |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------- |
| **Accuracy**                          | Reported but treated as misleading — the test set is 85% Slight, so a model that always predicts Slight gets 85% for free |
| **Macro F1**                          | Treats all three classes equally; the primary comparison metric — every model must beat the floor set by the baselines    |
| **Weighted F1**                       | Accounts for class imbalance; useful secondary view                                                                       |
| **Per-class Precision / Recall / F1** | Fatal and Serious recall are the most safety-critical numbers — missing a fatal accident is far worse than a false alarm  |

All runs are tracked in MLflow under the experiment `Accident_Severity_Pipeline`.  
Run: `poetry run mlflow ui --backend-store-uri file:./mlruns`

For rubric compliance, the MLflow comparison view should show at least these columns side by side for every run:

- `params.model_name`
- standard metrics: `accuracy`, `macro_f1` (and `weighted_f1` if desired)
- business metrics: `fatal_recall_business`, `serious_recall_business`, `critical_case_recall`, `critical_undertriage_rate`

The final report should include a screenshot of that MLflow experiment comparison table with all required runs visible.

---

## Models

### Baseline 1 — Constant "Slight" (`Baseline_Constant_Slight`)

|             |                                                                                                                                                                                                                                 |
| ----------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Class**   | `DummyClassifier(strategy="constant", constant="Slight")`                                                                                                                                                                       |
| **Logic**   | Predicts _Slight_ for every single input, regardless of features                                                                                                                                                                |
| **Purpose** | Exposes the **accuracy trap**: achieves 85% accuracy by exploiting class imbalance, while completely ignoring Fatal and Serious. Any real model must achieve higher _minority-class recall_ than this, not just higher accuracy |

**Results (test set):**

| Class         | Precision | Recall | F1   |
| ------------- | --------- | ------ | ---- |
| Fatal         | 0.00      | 0.00   | 0.00 |
| Serious       | 0.00      | 0.00   | 0.00 |
| Slight        | 0.85      | 1.00   | 0.92 |
| **Macro avg** | 0.28      | 0.33   | 0.31 |
| **Accuracy**  | —         | —      | 0.85 |

---

### Baseline 2 — Stratified Sampler (`Baseline_Stratified`)

|             |                                                                                                                                                                                                                                                                                                                                     |
| ----------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Class**   | `DummyClassifier(strategy="stratified")`                                                                                                                                                                                                                                                                                            |
| **Logic**   | Samples predictions randomly from the **training** class distribution (which is 33/33/33 due to SMOTE). Each prediction is drawn independently at random.                                                                                                                                                                           |
| **Purpose** | The true floor — sets the absolute minimum a real model must beat. Because training was SMOTE-balanced, this model effectively rolls a 3-sided die. Accuracy collapses to ~33%, demonstrating that SMOTE-balanced training does **not** mean test performance will be balanced. Any model that cannot beat 33% macro-F1 is useless. |

**Results (test set):**

| Class         | Precision | Recall | F1    |
| ------------- | --------- | ------ | ----- |
| Fatal         | ~0.01     | ~0.33  | ~0.03 |
| Serious       | ~0.13     | ~0.33  | ~0.19 |
| Slight        | ~0.85     | ~0.33  | ~0.48 |
| **Macro avg** | ~0.33     | ~0.33  | ~0.23 |
| **Accuracy**  | —         | —      | ~0.33 |

> Accuracy drops to 33% even though training was balanced — because test data is not balanced. This is intentional and expected.

---

### Baseline 3 — Most Frequent (`Baseline_Most_Frequent`)

|             |                                                                                                                                                                                                                                                                                        |
| ----------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| **Class**   | `DummyClassifier(strategy="most_frequent")`                                                                                                                                                                                                                                            |
| **Logic**   | After SMOTE the three training classes are tied at 33% each. scikit-learn breaks ties by picking the first class alphabetically — **Fatal** — and predicts it for everything.                                                                                                          |
| **Purpose** | Demonstrates the "panic model" failure mode: achieves 100% Fatal recall but 0% recall on Slight (85% of test) and 0% on Serious. Accuracy collapses to 1.3% (the actual Fatal share in test). Useful for confirming that chasing a single class's recall destroys overall performance. |

**Results (test set):**

| Class         | Precision | Recall | F1   |
| ------------- | --------- | ------ | ---- |
| Fatal         | 0.01      | 1.00   | 0.03 |
| Serious       | 0.00      | 0.00   | 0.00 |
| Slight        | 0.00      | 0.00   | 0.00 |
| **Macro avg** | 0.00      | 0.33   | 0.01 |
| **Accuracy**  | —         | —      | 0.01 |

---

### Model 1 — Logistic Regression (`Logistic_Regression`)

|              |                                                                                                                                                                                                                                                        |
| ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| **Class**    | `LogisticRegression(max_iter=1000)`                                                                                                                                                                                                                    |
| **Solver**   | `lbfgs` (default multinomial)                                                                                                                                                                                                                          |
| **Features** | All 9 selected features, pre-scaled by `StandardScaler` from preprocessing                                                                                                                                                                             |
| **Purpose**  | Linear baseline — the first model that actually uses features. Establishes a performance ceiling for linear models. Non-linear models (Random Forest, XGBoost) in later phases must meaningfully exceed these numbers to justify the added complexity. |

**Results (test set):**

| Class         | Precision | Recall | F1   |
| ------------- | --------- | ------ | ---- |
| Fatal         | 0.03      | 0.60   | 0.06 |
| Serious       | 0.15      | 0.36   | 0.21 |
| Slight        | 0.89      | 0.47   | 0.61 |
| **Macro avg** | 0.36      | 0.47   | 0.29 |
| **Accuracy**  | —         | —      | 0.45 |

**Key takeaway:** Macro F1 of 0.29 beats the stratified floor (0.23) and Fatal recall jumps to 60% — meaning the model has learned genuine signal. However, accuracy drops to 45% because `class_weight='balanced'` pushes the model to predict minority classes more aggressively.

---

## Comparison Summary

| MLflow Run               | Accuracy | Macro F1      | Fatal Recall | Serious Recall |
| ------------------------ | -------- | ------------- | ------------ | -------------- |
| Baseline_Constant_Slight | **0.85** | 0.31          | 0.00         | 0.00           |
| Baseline_Stratified      | ~0.33    | ~0.23 ← floor | ~0.33        | ~0.33          |
| Baseline_Most_Frequent   | 0.01     | 0.01          | **1.00**     | 0.00           |
| **Logistic_Regression**  | 0.45     | **0.29**      | **0.60**     | **0.36**       |

**What to beat in the next phase:** Macro F1 > 0.29 and Fatal recall > 0.60.

---

## Artifacts Saved Per Run

Each MLflow run stores:

- `params/model_name` and the estimator hyperparameters used for that run
- standard metrics including `accuracy`, `macro_f1`, and `weighted_f1`
- business metrics including `fatal_recall_business`, `serious_recall_business`, `critical_case_recall`, and `critical_undertriage_rate`
- `plots/<run>_confusion_matrix.png` — visual confusion matrix
- `reports/<run>_classification_report.json` — full per-class metrics as JSON
- `model/` — the fitted sklearn estimator (loadable via `mlflow.sklearn.load_model`)

Local copies are also saved to `reports/mlflow_artifacts/`.
