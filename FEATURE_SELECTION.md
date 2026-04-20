# Feature Selection Steps Applied — Phase 3 Preprocessing

All feature selection steps are implemented in `select_features()` and `select_features_model_based()` in `accident_severity_predictor/preprocess.py` and `accident_severity_predictor/features.py`.

---

## 1. BUILD CANDIDATE FEATURE SET

**Method:** `select_features()`

**What happened:**

- Start from preprocessed dataframe after cleaning, feature engineering, missing-value handling, outlier clipping, and encoding.

- Remove target from predictors:

- Target: `Accident_Severity`

- Keep only numeric columns for model-based ranking.

**Justification:**

- Random Forest feature importance requires numeric predictors.

- Excluding target prevents information leakage from label to predictors.

**Impact:**

- Produces a model-ready candidate pool for ranking.

---

## 2. APPLY MODEL-BASED RANKING (RANDOM FOREST IMPORTANCE)

**Method:** `select_features_model_based()`

**Model configuration:**

- `RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=42)`

- Features are ranked by impurity-based importance.

**Justification:**

- Tree-based importance captures non-linear and interaction effects.

- Works well for mixed engineered + encoded accident features.

**Impact:**

- Produces an ordered feature-importance table used for selection.

---

## 3. LEAKAGE PREVENTION FILTER (MANDATORY)

**Method:** Inside `select_features_model_based()` before final top-k selection

**Features explicitly excluded:**

- `Number_of_Casualties`

- `Did_Police_Officer_Attend_Scene_of_Accident`

**Justification:**

- These are post-accident variables and are not available at real prediction time.

- Keeping them would artificially inflate metrics and invalidate deployment performance.

**Impact:**

- Ensures selected features are deployment-valid and causally appropriate.

---

## 4. SELECT TOP 7 FEATURES (PROJECT REQUIREMENT)

**Method:** `importance_scores.head(top_k)` with `top_k = 7`

**Project setting:**

- `CONFIG['n_features_to_select'] = 7`

**Justification:**

- Meets assignment requirement of selecting >= 7 best features.

- Keeps model compact and interpretable while retaining predictive signal.

**Impact:**

- Final model is trained on exactly 7 features.

---

## 5. FINAL SELECTED FEATURES (LATEST RUN)

Source: `data/processed/preprocessing_metadata.json`

Timestamp: `2026-04-20T22:58:26.361930`

Selected 7 features:

1. `Vehicle_Type`

2. `Engine_Capacity_.CC.`

3. `road_risk_score`

4. `Vehicle_Manoeuvre`

5. `Speed_limit`

6. `Vehicle_Leaving_Carriageway`

7. `LSOA_of_Accident_Location`

Top-7 importances from latest run:

- `Vehicle_Type`: 0.1322

- `Engine_Capacity_.CC.`: 0.1025

- `road_risk_score`: 0.0932

- `Vehicle_Manoeuvre`: 0.0865

- `Speed_limit`: 0.0706

- `Vehicle_Leaving_Carriageway`: 0.0594

- `LSOA_of_Accident_Location`: 0.0388

---

## 6. WHY THESE 7 FEATURES ARE STRONG

- `Vehicle_Type`

- Captures vulnerability and crash dynamics differences across vehicle classes.

- `Engine_Capacity_.CC.`

- Proxies vehicle power/performance class and likely travel context.

- `road_risk_score`
  - **Multicollinearity Note:** Post-processing validation identified a high correlation (|r| > 0.85) between this feature and `Speed_limit`.
  - **Justification:** `road_risk_score` was retained because it adds specific "Road Type" context (e.g., Roundabout vs. Single Carriageway) that the raw speed limit lacks. Tree-based models (XGBoost/RF) are robust to this correlation and will utilize the most informative split between the two.

- `Vehicle_Manoeuvre`

- Encodes maneuver context linked to conflict severity (turning, overtaking, etc.).

- `Speed_limit`

- Represents exposure/risk regime and stopping-distance context.

- `Vehicle_Leaving_Carriageway`

- Indicates loss-of-control patterns associated with severe outcomes.

- `LSOA_of_Accident_Location`

- Captures localized environmental/road-network risk effects.

---

## 7. FEATURES EXCLUDED ON PURPOSE

### A) Leakage-based exclusions (hard rule)

- `Number_of_Casualties`

- `Did_Police_Officer_Attend_Scene_of_Accident`

### B) Not in final top-7

- Other candidate features with lower importance than the selected set.

**Justification:**

- The goal is to keep only the strongest non-leaky predictors.

---

## 8. OUTPUTS SAVED FOR TRACEABILITY

Saved artifacts:

- `data/processed/feature_names.pkl`

- `data/processed/preprocessing_metadata.json` (contains full feature selection results)

**Why this matters:**

- Reproducibility: same selected features can be reused for training/inference.

- Auditability: decisions and scores are preserved for reporting.

---

## Summary

Feature selection follows a strict, leakage-safe workflow:

1. Prepare numeric candidate set

2. Rank with Random Forest importance

3. Remove post-accident leakage variables

4. Keep top 7 by importance

5. Persist selected features and scores

This balances predictive power, deployment realism, and explainability for accident severity modeling.
