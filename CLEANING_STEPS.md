# Data Cleaning Steps Applied — Phase 3 Preprocessing

All cleaning steps are implemented in the `DataCleaner` class in `preprocess.py`.

---

## 1. DROP HIGH-MISSINGNESS COLUMNS (>20% threshold)

**Method:** `drop_high_missingness_columns()`

**Columns Dropped:** 7 columns

- `Carriageway_Hazards` (98.1% missing)
- `Special_Conditions_at_Site` (97.5% missing)
- `Hit_Object_in_Carriageway` (95.9% missing)
- `Hit_Object_off_Carriageway` (91.4% missing)
- `Skidding_and_Overturning` (87.2% missing)
- `2nd_Road_Class` (41.2% missing)
- `Driver_IMD_Decile` (33.8% missing)

**Justification (Phase 2 Finding):**

- Phase 2 Validation found these columns to be >20% missing
- Columns with >80% missing data are not predictive (too sparse)
- Imputing ~98% of values would introduce artificial patterns
- Dropping entirely preserves data integrity over guessing

**Impact:**

- Removes non-informative features
- Reduces noise in model training
- Prevents overfitting to imputed values

---

## 2. REMOVE INVALID SPEED_LIMIT VALUES

**Method:** `remove_invalid_speed_limits()`

**Criteria:** Speed_limit must be in `{20, 30, 40, 50, 60, 70}`  
(Valid UK road speed limits in mph)

**Records Removed:** 36 rows (Phase 2 finding)

**Justification:**

- Business rule validation: Speed limits on UK roads are standardized
- Values outside this set are data entry errors
- Cannot be imputed without additional context
- Retaining them would corrupt model training

**Impact:**

- Ensures feature values match real-world constraints
- Low data loss (36 of 2M+ records = 0.002%)
- Improves model reliability

---

## 3. REMOVE ROWS WITH NULL REQUIRED FIELDS

**Method:** `remove_rows_with_required_nulls()`

**Required Columns:**

- `Latitude` (174 nulls, 0.0%)
- `Longitude` (175 nulls, 0.0%)
- `Speed_Limit` (37 nulls, 0.0%)
- `Accident_Index` (Primary key)

**Records Removed:** ~386 rows (Phase 2 finding)

**Justification:**

- Geographic coordinates (Lat/Long) cannot be imputed — they locate the accident
- Accident_Index is the primary key — missing values break data integrity
- Speed_Limit is critical for both safety analysis and model prediction
- <0.1% data loss is acceptable for maintaining data quality

**Impact:**

- Preserves referential integrity
- Ensures all records are locatable and identifiable
- Prevents models from learning from incomplete safety information

---

## 4. FIX DAY_OF_WEEK CONSISTENCY VIOLATIONS

**Method:** `correct_day_of_week()`

**Approach:** Overwrite `Day_of_Week` with correct values derived from `Date` column

**Records Affected:** ~2 rows (Phase 2 finding)

**Justification (Phase 2 Finding):**

- Phase 2 Validation found: "Day_of_Week does not always match the day derived from the Date column"
- Date is the source-of-truth (machine-generated timestamp)
- Day_of_Week can be calculated deterministically from Date
- Inconsistencies = data entry errors or ETL bugs

**Correction Logic:**

```
Day_of_Week = Date.day_name()  # Monday, Tuesday, etc.
```

**Impact:**

- Ensures logical consistency across temporal features
- Allows models to learn genuine day-of-week patterns
- Prevents contradictory inputs to the model

---

## 5. HANDLE INVALID AGE_BAND_OF_DRIVER CODES

**Method:** `handle_invalid_age_bands()`

**Valid Format:** Numeric range (e.g., "0-5", "6-10", "11-15", etc.)  
**Pattern:** `^\d+-\d+$`

**Records Affected:** 4,664 invalid codes (Phase 2 finding)

**Action:** Mark invalid codes as NaN for later imputation

**Justification:**

- Phase 2 Validation found 4,664 vehicle records with invalid age band codes
- Invalid codes cannot be mapped to age ranges
- Cannot be used for stratification or analysis
- Marking as NaN preserves the vehicle record while indicating missing information

**Impact:**

- Converts unusable data into properly flagged missing values
- Later imputation step can fill these intelligently
- Prevents incorrect age categorization in model

---

## 6. HANDLE INVALID SEX_OF_DRIVER CODES

**Method:** `handle_sex_of_driver_codes()`

**Valid Values:**

- Standard: `{Male, Female, Not known}`
- Codes: `{M, F, 1, 2, 3}`

**Records Affected:** 76,119 invalid vehicle records (Phase 2 finding)

**Actions:**

1. Remap valid code variants to standard names:
   - `M, 1` → `Male`
   - `F, 2` → `Female`
   - `3` → `Not known`
2. Replace truly invalid codes with `Not known`

**Justification:**

- Phase 2 Validation found 76,119 vehicle records with invalid sex codes
- Different systems may use different encodings (M/F vs 1/2 vs text)
- Standardization enables consistent feature engineering
- "Not known" is a valid category (missing != invalid)

**Impact:**

- Consolidates scattered valid values into single category
- Preserves all 76,119 records instead of dropping them
- Creates clean, categorical feature for modeling

---

## 7. REMOVE EXACT DUPLICATE RECORDS

**Method:** `check_and_remove_duplicates()`

**Criteria:** Rows with all identical values across all columns

**Records Removed:** ~0 found (Phase 2 finding indicated no exact duplicates, but check is performed)

**Justification:**

- Exact duplicates don't add information value
- Can skew model training by over-weighting identical accidents
- May indicate ETL errors or data processing bugs
- Removing preserves statistical independence of samples

**Impact:**

- Ensures each record represents a unique accident event
- Maintains proper sample size for train/val/test splits
- Improves model generalization

---

## 8. WEATHER FEATURE HANDLING

**Method:** `handle_missing_values()` in `features.py`

**Weather Columns Retained:** 12 columns

- `time`, `temp`, `tmin`, `tmax`, `rhum`, `prcp`, `snwd`, `wspd`, `wpgt`, `pres`, `tsun`, `cldc`

**Issue:** Weather columns have high missingness (>50%) and would normally be dropped by the >50% threshold.

**Solution:** Modified `handle_missing_values()` to:

- Retain weather columns regardless of missingness threshold
- Impute missing values using median imputation for numerical weather features
- Preserve valuable predictive information about environmental conditions

**Justification:**

- Weather conditions significantly impact accident severity (precipitation, temperature, visibility)
- Median imputation is a valid technique for handling missing numerical data
- Even partial weather data provides signal for model training
- Meets course requirements for proper missing value handling techniques
- Composite weather features (e.g., `temp_road_risk`) proved valuable - ranked 3rd in feature importance

**Impact:**

- Weather data is available for feature engineering
- Composite weather features can be created (e.g., temperature-adjusted road risk)
- Improves model predictive power by incorporating environmental context

---

## Summary: All Phase 2 Validation Issues Handled

| Dimension          | Issue Count | Cleaning Step                                                                                 | Records Affected             |
| ------------------ | ----------- | --------------------------------------------------------------------------------------------- | ---------------------------- |
| **ACCURACY**       | 3           | Remove invalid Speed_Limit<br>Mark invalid Age_Band codes as NaN<br>Standardize Sex_of_Driver | 36<br>4,664<br>76,119        |
| **CONSISTENCY**    | 1           | Fix Day_of_Week from Date                                                                     | ~2                           |
| **COMPLETENESS**   | 7           | Drop 7 high-missing columns<br>Remove rows with null required fields                          | 7 cols<br>~386 rows          |
| **UNIQUENESS**     | 1           | Remove exact duplicates                                                                       | ~0                           |
| **JOIN INTEGRITY** | 2           | (Already handled by inner merge in Phase 2)                                                   | —                            |
| **TOTAL**          | **15**      | **All addressed**                                                                             | **~81,207 records modified** |

---

## Data Loss Summary

**Starting Records:** ~2,047,256 accidents (after Phase 2 merge)

**Records Removed by Step:**

1. Invalid Speed_Limit: -36
2. Null required fields: -386
3. Total dropped: **-422 rows (0.02% data loss)**

**Records Modified (Not Removed):**

1. Invalid Age_Band codes: 4,664 → marked as NaN (recoverable via imputation)
2. Invalid Sex_of_Driver codes: 76,119 → remapped to standard categories
3. Day_of_Week mismatches: ~2 → corrected
4. High-missing columns: 7 columns → removed but no row loss

**Final Dataset:** ~2,047,000 records (99.98% retained)

---
