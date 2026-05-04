# Feature Selection Approaches Comparison

## Overview

This document compares two feature selection approaches implemented for the UK Road Accidents Severity Prediction project:

1. **RFECV - Recursive Feature Elimination with Cross-Validation** (Optimized approach)
2. **Random Forest Feature Importance** (Tree-based approach)

Both approaches have been integrated into the preprocessing pipeline to allow for comparison and documentation.

## Key Features

### 1. Leakage Prevention ✓

- **Both methods** automatically exclude post-accident variables not available at prediction time:
  - `Number_of_Casualties` — directly caused by accident severity
  - `Did_Police_Officer_Attend_Scene_of_Accident` — recorded after incident
- **Implementation**: Exclusion happens before feature selection

### 2. Missing Value Handling

- Multi-stage imputation: forward fill → backward fill → median
- Pre-filter: Drop features with >20% missing before selection
- Outlier detection: IQR-based clipping (iqr=1.5)
- Categorical encoding: LabelEncoder for 8 categorical features

### 3. Feature Engineering

- 18 feature engineering functions creating temporal, lighting, road, vehicle, driver, and interaction features:
  - Temporal: hour_of_day, is_weekend, season
  - Lighting: is_dark
  - Road features: Road_Type, is_wet_road, is_urban, road_risk_score
  - Weather: is_adverse_weather, temp_road_risk (composite)
  - Vehicle/Driver: Vehicle_Type, engine_size_category, is_male, age
  - Administrative: Target encoding for make, district, highway
  - Interaction: speed_x_road_risk, wet_road_speed, young_driver_night

### 4. Feature Selection Methodology

#### RFECV Approach (Primary Method)

- **Method**: Recursive Feature Elimination with Cross-Validation using Logistic Regression
- **Base Estimator**: LogisticRegression with class_weight='balanced'
- **CV Strategy**: StratifiedKFold(5) with fallback to 3-fold if single-class fold error
- **Elimination**: step=1 (removes one feature at a time for fine-grained selection)
- **Features Selected**: 15 (after correlation filtering)
- **Correlation Filter**: Removes pairs with r>0.8, keeping lower-ranked feature
- **Advantages**:
  - Systematic, exhaustive feature elimination
  - Cross-validation ensures robustness and generalization
  - Automatically determines optimal feature count
  - Leakage prevention built-in
- **Limitations**:
  - Computationally intensive (requires multiple CV folds)
  - Linear model assumptions

#### Random Forest Approach (Comparison)

- **Method**: Tree-based feature importance using RandomForestClassifier
- **Configuration**: n_estimators=50, max_depth=10, class_weight='balanced'
- **Selection**: Top-k by importance scores
- **Features Selected**: 15
- **Correlation Filter**: Applied same as RFECV for consistency
- **Advantages**:
  - Fast computation
  - Handles non-linear relationships
  - Provides interpretable importance scores
- **Limitations**:
  - May miss linear relationships
  - Can select redundant correlated features

## RFECV Selected Features (15)

The RFECV approach selected the following 15 optimal features (after leakage prevention and correlation filtering):

1. `1st_Road_Number` — Primary road identifier
2. `Location_Easting_OSGR` — Geographic easting coordinate
3. `Location_Northing_OSGR` — Geographic northing coordinate
4. `LSOA_of_Accident_Location` — Local Super Output Area (neighborhood)
5. `Time` — Hour of day (categorical)
6. `Year_x` — Year of accident
7. `Engine_Capacity_.CC.` — Vehicle engine size
8. `model` — Vehicle model (target-encoded)
9. `Year_y` — Year of weather observation
10. `pres` — Atmospheric pressure
11. `temp_road_risk` — **Composite feature**: temperature-adjusted road risk
12. `junction_control_encoded` — Junction control type
13. `district_accident_volume` — Administrative severity metric
14. `highway_accident_volume` — Administrative severity metric
15. `speed_x_road_risk` — **Interaction feature**: speed limit × road risk

### Key Insights

- **Geographic features dominate**: 4 features (easting, northing, LSOA, road number) represent spatial information
- **Weather integration**: `temp_road_risk` is a composite feature combining temperature with road risk
- **Administrative metrics**: Target-encoded features (`model`, `district_accident_volume`, `highway_accident_volume`) capture local patterns
- **Interaction effects**: `speed_x_road_risk` captures non-linear relationship between speed and road conditions

### Correlation Filtering Results

Features removed due to multicollinearity (r > 0.8):

- `Speed_limit` (r=1.000 with `road_risk_score`, r=0.972 with `temp_road_risk`) — redundant
- `road_risk_score` (r=0.972 with `temp_road_risk`) — kept `temp_road_risk` (lower RFECV rank = more robust)

## Random Forest Selected Features (15)

The Random Forest approach selected the following 15 features by importance:

| Rank | Feature                     | Importance | Type                          |
| ---- | --------------------------- | ---------- | ----------------------------- |
| 1    | Vehicle_Type                | 0.1494     | Vehicle attribute             |
| 2    | make                        | 0.0752     | Vehicle make (target-encoded) |
| 3    | district_severity_rate      | 0.0648     | Administrative metric         |
| 4    | Vehicle_Leaving_Carriageway | 0.0475     | Vehicle behavior              |
| 5    | Engine*Capacity*.CC.        | 0.0416     | Vehicle attribute             |
| 6    | manoeuvre_encoded           | 0.0388     | Driver behavior (encoded)     |
| 7    | X1st_Point_of_Impact        | 0.0335     | Accident characteristic       |
| 8    | speed_x_road_risk           | 0.0323     | **Interaction feature**       |
| 9    | road_risk_score             | 0.0302     | Road condition                |
| 10   | Time                        | 0.0271     | Temporal (hour of day)        |
| 11   | junction_detail_encoded     | 0.0259     | Junction detail (encoded)     |
| 12   | is_urban                    | 0.0249     | Urban/rural classification    |
| 13   | Junction_Location           | 0.0215     | Junction location             |
| 14   | Vehicle_Reference           | 0.0182     | Vehicle sequence              |
| 15   | is_dark                     | 0.0147     | Lighting condition            |

### Key Insights

- **Vehicle attributes dominate**: Top 3 features are vehicle-related (type, make, leaving carriageway)
- **Administrative metrics**: Target-encoded features rank high for capturing local patterns
- **Behavioral factors**: Manoeuvre and point of impact reflect driver/accident dynamics
- **Environmental factors**: Lower-ranked but included (is_dark, is_urban, Time)

### Correlation Filtering Results

Features removed due to multicollinearity (r > 0.8):

- `highway_severity_rate` (r=0.907 with `district_severity_rate`) — removed
- `temp_road_risk` (r=0.972 with `road_risk_score`) — removed, kept `road_risk_score` (higher importance)
- `Speed_limit` (r=1.000 with `road_risk_score`) — removed

## Feature Selection Results Summary

### Latest Run: 2026-05-04 10:54:41

Both feature selection methods independently arrived at **15 features** after applying:

1. Leakage prevention (excluding post-accident variables)
2. Missing value filtering (>20% threshold)
3. Correlation filtering (r > 0.8)
4. Class balance handling (class_weight='balanced')

### Feature Overlap Analysis

**RFECV-only features (7)**:

- Location_Easting_OSGR, Location_Northing_OSGR, 1st_Road_Number
- LSOA_of_Accident_Location, Year_x, Year_y, pres
- highway_accident_volume, district_accident_volume

**Random Forest-only features (8)**:

- Vehicle_Type, make, district_severity_rate
- Vehicle_Leaving_Carriageway, manoeuvre_encoded
- X1st_Point_of_Impact, is_urban, is_dark, Vehicle_Reference, junction_detail_encoded

**Shared features (7)**:

- Engine*Capacity*.CC., Time, temp_road_risk\*, junction_control_encoded
- speed_x_road_risk, road_risk_score\*

\*Shared: RFECV selected temp_road_risk; RF selected road_risk_score (correlation r=0.972)

### Method Comparison

| Aspect                     | RFECV                        | Random Forest         |
| -------------------------- | ---------------------------- | --------------------- |
| **Approach**               | Linear elimination + CV      | Non-linear importance |
| **Features Selected**      | 15                           | 15                    |
| **Computation Time**       | ~60 minutes                  | ~33 seconds           |
| **Optimization Criterion** | Cross-validation accuracy    | Feature importance    |
| **Feature Focus**          | Geographic + administrative  | Vehicle + behavioral  |
| **CV Accuracy**            | 0.4214 (mean across 5 folds) | N/A (tree-based)      |
| **Leakage Prevention**     | ✓ Built-in                   | ✓ Built-in            |

## Implementation Details

### Configuration (from preprocess.py)

```json
{
  "test_size": 0.2,
  "random_state": 42,
  "feature_selection_method": "rfecv",
  "rfecv_min_features": 15,
  "rfecv_cv_folds": 5,
  "compare_feature_selection": true,
  "apply_smote": true,
  "smote_sampling_strategy": "not majority"
}
```

### Train/Val/Test Split Results

```
Training:   1,086,247 samples (40%) → After SMOTE: 2,778,678 (perfectly balanced)
Validation: 1,086,248 samples (40%)
Test:         543,124 samples (20%)
```

### Class Distribution

**Before SMOTE**:

- Training: 926,226 Slight (85.3%) | 145,830 Serious (13.4%) | 14,191 Fatal (1.3%)
- Validation: 926,226 Slight (85.3%) | 145,831 Serious (13.4%) | 14,191 Fatal (1.3%)
- Test: 463,113 Slight (85.3%) | 72,916 Serious (13.4%) | 7,095 Fatal (1.3%)

**After SMOTE (Training only)**:

- 926,226 Slight (33.3%) | 926,226 Serious (33.3%) | 926,226 Fatal (33.3%)
