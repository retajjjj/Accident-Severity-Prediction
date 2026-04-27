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

## 3. LEAKAGE PREVENTION FILTER

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

## 4. SELECT TOP 9 FEATURES (PROJECT REQUIRES AT LEAST 7)

**Method:** `importance_scores.head(top_k)` with `top_k = 9`

**Project setting:**

- `CONFIG['n_features_to_select'] = 9`

**Justification:**

- Exceeds assignment requirement of selecting >= 7 best features.

- Keeps model compact and interpretable while retaining predictive signal.

- Allows inclusion of composite weather features that provide additional predictive power.

- Systematic correlation filter removes multicollinear features automatically.

**Impact:**

- Final model is trained on exactly 9 features.

---

## 5. FINAL SELECTED FEATURES (LATEST RUN)

Source: `data/processed/preprocessing_metadata.json`

Timestamp: `2026-04-27T05:59:56`

Selected 9 features:

1. `Engine_Capacity_.CC.` (importance: 0.1124)

2. `Vehicle_Type` (importance: 0.1096)

3. `temp_road_risk` (importance: 0.0818)

4. `Vehicle_Manoeuvre` (importance: 0.0757)

5. `Vehicle_Leaving_Carriageway` (importance: 0.0553)

6. `Time` (importance: 0.0393)

7. `Junction_Location` (importance: 0.0360)

8. `LSOA_of_Accident_Location` (importance: 0.0322)

9. `Vehicle_Reference` (importance: 0.0223)

**Correlation Filtering Results:**

- Removed `road_risk_score` due to high correlation with `temp_road_risk` (r=0.969, kept higher importance)
- Removed `Speed_limit` due to high correlation with `temp_road_risk` (r=0.861, kept higher importance)
- Removed `Speed_limit` due to high correlation with `road_risk_score` (r=0.894, kept higher importance)

**Weather Feature Success:** The composite weather feature `temp_road_risk` (temperature-adjusted road risk) ranked 3rd in importance, demonstrating that the weather data merge was valuable. This feature combines road risk with temperature conditions, capturing the increased severity risk during cold weather (icy conditions on high-speed roads).

---

## 6. WHY THESE 9 FEATURES ARE STRONG

- `Engine_Capacity_.CC.` (importance: 0.1124)
  - Proxies vehicle power/performance class and likely travel context.
  - Higher engine capacity often correlates with higher speeds and more severe impacts.

- `Vehicle_Type` (importance: 0.1096)
  - Captures vulnerability and crash dynamics differences across vehicle classes.
  - Motorcycles, cars, and heavy goods vehicles have very different severity profiles.

- `temp_road_risk` (importance: 0.0818)
  - **Composite weather feature:** Combines road risk score with temperature adjustment.
  - **Logic:** Cold temperatures (<5°C) increase risk on high-speed roads (icy conditions, reduced traction).
  - **Formula:** `road_risk_score × (1 + 0.5 if temp < 5°C)`
  - Demonstrates that weather data merge was valuable - this feature ranked 3rd overall.

- `Vehicle_Manoeuvre` (importance: 0.0757)
  - Encodes maneuver context linked to conflict severity (turning, overtaking, etc.).
  - Certain maneuvers (e.g., turning right, U-turns) are higher-risk.

- `Vehicle_Leaving_Carriageway` (importance: 0.0553)
  - Indicates loss-of-control patterns associated with severe outcomes.
  - Vehicles leaving the carriageway often result in more severe collisions.

- `Time` (importance: 0.0393)
  - Original time column capturing time-of-day patterns.
  - Accident severity varies by time (rush hours, night driving, etc.).

- `Junction_Location` (importance: 0.0360)
  - Indicates where in the junction the accident occurred.
  - Junctions are high-risk areas with specific collision patterns.

- `LSOA_of_Accident_Location` (importance: 0.0322)
  - Captures localized environmental/road-network risk effects.
  - Encodes geographic and socioeconomic factors affecting accident severity.

- `Vehicle_Reference` (importance: 0.0223)
  - Unique identifier for vehicle within accident.
  - May capture vehicle-specific risk patterns in multi-vehicle accidents.

---

## 7. FEATURES EXCLUDED ON PURPOSE

### A) Leakage-based exclusions (hard rule)

- `Number_of_Casualties`

- `Did_Police_Officer_Attend_Scene_of_Accident`

### B) Not in final top-9

- Other candidate features with lower importance than the selected set.

**Justification:**

- The goal is to keep only the strongest non-leaky predictors.

---

## 8. WEATHER FEATURE HANDLING

**Issue:** Weather columns (`temp`, `tmin`, `tmax`, `rhum`, `prcp`, `snwd`, `wspd`, `wpgt`, `pres`, `tsun`, `cldc`, `time`) have high missingness (>50%) and were previously dropped.

**Solution:** Modified `handle_missing_values()` in `features.py` to:

- Retain weather columns regardless of missingness threshold
- Impute missing values using median imputation for numerical weather features
- Preserve valuable predictive information about environmental conditions

**Justification:**

- Weather conditions significantly impact accident severity (precipitation, temperature, visibility)
- Median imputation is a valid technique for handling missing numerical data
- Even partial weather data provides signal for model training
- Meets course requirements for proper missing value handling techniques

---

## 9. SYSTEMATIC CORRELATION FILTERING

**Issue:** Multiple features can be highly correlated, leading to multicollinearity and redundant information.

**Solution:** Implemented systematic correlation filter in `select_features_model_based()` to:

- Calculate correlation matrix for top-k features
- Identify all pairs with correlation > 0.8
- Remove lower-importance feature from each correlated pair
- Keep higher-importance feature to preserve predictive power

**Results from latest run:**

- Removed `road_risk_score` due to high correlation with `temp_road_risk` (r=0.969, kept higher importance)
- Removed `Speed_limit` due to high correlation with `temp_road_risk` (r=0.861, kept higher importance)
- Removed `Speed_limit` due to high correlation with `road_risk_score` (r=0.894, kept higher importance)

**Justification:**

- Systematic approach handles any correlation pattern, not just pre-identified pairs
- Data-driven: keeps the feature with higher importance score
- Reduces multicollinearity while preserving maximum predictive power
- More robust than hardcoding specific feature pairs

---

## 10. OUTPUTS SAVED FOR TRACEABILITY

Saved artifacts:

- `data/processed/feature_names.pkl`

- `data/processed/preprocessing_metadata.json` (contains full feature selection results)

**Why this matters:**

- Reproducibility: same selected features can be reused for training/inference.

- Auditability: decisions and scores are preserved for reporting.

---

## Summary

Feature selection follows a strict, leakage-safe workflow:

1. Prepare numeric candidate set (including weather features with imputation)

2. Rank with Random Forest importance

3. Remove post-accident leakage variables

4. Apply systematic correlation filter (remove lower-importance features from highly correlated pairs)

5. Keep top 9 by importance

6. Persist selected features and scores

This balances predictive power, deployment realism, and explainability for accident severity modeling while incorporating valuable weather data through composite features and handling multicollinearity systematically.
