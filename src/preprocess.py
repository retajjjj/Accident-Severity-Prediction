"""
Data Preprocessing Pipeline — UK Road Accidents Project
CMPS344 Applied Data Science — Phase 3

This script orchestrates the complete data preprocessing workflow:
    1. Load merged raw data (from Phase 2)
    2. Clean Data (ALL 15 issues from Phase 2 Validation Report handled; check CLEANING_STEPS.md for rationale)
    3. Engineer features (hour_of_day, day_of_week, is_weekend, month, season, is_dark, is_adverse_weather, road_risk_score, max_driver_age, min_driver_age, num_motorcycles, involves_pedestrian)
    4. Handle missing values and outliers
    5. Encode categorical variables
    6. Select best features using RFECV or Random Forest (or BOTH for comparison)
    7. Apply SMOTE to training split only
    8. Scale numerical features
    9. Save train/test/val splits
    
FEATURE SELECTION MODES:
    Single Method (default):
        - Set feature_selection_method='rfecv' for RFECV-based selection
        - Set feature_selection_method='model_based' for Random Forest importance
    
    Comparison Mode:
        - Set compare_feature_selection=True to run BOTH methods
        - Primary method (feature_selection_method) is used for train/val/test splits
        - Alternative feature sets saved to data/processed/feature_selection_comparison.pkl
        - Compare model performance (recall, accuracy, etc.) in model evaluation phase
    
Usage:
    poetry run python accident_severity_predictor/preprocess.py
    make preprocess

Outputs (saved to data/processed/):
    - X_train.pkl, y_train.pkl
    - X_val.pkl, y_val.pkl
    - X_test.pkl, y_test.pkl
    - scaler.pkl, feature_names.pkl
    - preprocessing_report.txt (summary)
    - feature_selection_comparison.pkl (if compare_feature_selection=True)
"""

import logging
import sys
from pathlib import Path
from typing import Tuple, List, Dict
import pickle
import json
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from features import (
        encode_target_variable,
        create_temporal_features,
        create_lighting_features,
        encode_road_type_features,
        encode_road_surface_features,
        encode_weather_condition_features,
        encode_urban_rural_features,
        create_weather_features,
        create_weather_composite_features,
        create_road_risk_features,
        create_vehicle_features,
        encode_vehicle_attributes,
        encode_driver_features,
        encode_manoeuvre_features,
        encode_junction_features,
        encode_journey_features,
        encode_administrative_features,
        create_interaction_features,
        handle_missing_values,
        detect_and_handle_outliers,
        encode_categorical_features,
        select_features_rfecv,
        select_features_model_based,
        apply_smote,
    )


# ── Setup ────────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('preprocessing.log', encoding='utf-8')
    ],
    encoding='utf-8'
)
logger = logging.getLogger(__name__)
# Ensure console output uses UTF-8
if hasattr(sys.stdout, 'reconfigure'):
    sys.stdout.reconfigure(encoding='utf-8')
if hasattr(sys.stderr, 'reconfigure'):
    sys.stderr.reconfigure(encoding='utf-8')

# Paths
PROJECT_ROOT = Path(__file__).parent.parent  # Go up to Accident-Severity-Prediction root
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
REPORTS_DIR = PROJECT_ROOT / "reports"

# Create directories
for d in [PROCESSED_DIR, REPORTS_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# Configuration
CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'missing_threshold': 50.0,
    'feature_selection_method': 'model_based',  # Primary method: 'rfecv' or 'model_based'
    'rfecv_min_features': 15,  # Minimum features for RFECV (requirement: ≥7)
    'rfecv_cv_folds': 5,  # Cross-validation folds for RFECV
    'compare_feature_selection': False,  # If True, run both RFECV and Random Forest for comparison
    'apply_smote': True,
    'smote_sampling_strategy': 'not majority',
}

# ═══════════════════════════════════════════════════════════════════════════════
# DATA CLEANING CLASS (Per Phase 2 Validation Report)
# ═══════════════════════════════════════════════════════════════════════════════

class DataCleaner:
    """
    Implements ALL data cleaning steps from Phase 2 Validation Report.
    
    Handles 15 issues across 8 validation dimensions:
    
    ACCURACY (3 issues):
        1. 36 rows with invalid Speed_limit
        2. 4,664 rows with invalid Age_Band_of_Driver
        3. 76,119 rows with invalid Sex_of_Driver
    
    CONSISTENCY (1 issue):
        4. 2 rows with Day_of_Week/Date mismatches
    
    COMPLETENESS (7 issues):
        5-7.   Drop columns >97% missing (3 cols)
        8-10.  Drop columns >30% missing (2 cols)
        11-13. Remove rows with null required fields (3 cols)
    
    UNIQUENESS: No issues (duplicates handled)
    
    RELATIONSHIPS: No issues found
    
    JOIN INTEGRITY (2 issues):
        14. 657,532 orphaned accidents (no vehicle record)
        15. 99,257 orphaned vehicles (no accident record)
    """
    
    # High-missingness columns to drop (>20% threshold per Phase 2)
    HIGH_MISSING_COLS = [
        # EXTREME missingness (>85%)
        'Carriageway_Hazards',                # ~98.1% - MUST DROP
        'Special_Conditions_at_Site',         # ~97.5% - MUST DROP
        'Hit_Object_in_Carriageway',          # ~95.9% - MUST DROP
        'Hit_Object_off_Carriageway',         # ~91.4% - MUST DROP
        'Skidding_and_Overturning',           # ~87.2% - MUST DROP
        # MODERATE missingness (>30%)
        '2nd_Road_Class',                     # ~41.2% - MUST DROP
        'Driver_IMD_Decile',                  # ~33.8% - MUST DROP
    ]
    
    # Valid UK speed limits (accuracy check)
    VALID_SPEED_LIMITS = {20, 30, 40, 50, 60, 70}
    
    # Required columns (cannot have nulls)
    REQUIRED_COLS = ['Latitude', 'Longitude', 'Speed_limit', 'Accident_Index']
    
    def __init__(self):
        self.cleaning_log = []
        self.initial_shape = None
        self.final_shape = None
    
    def log_step(self, msg: str):
        """Log cleaning step."""
        logger.info(msg)
        self.cleaning_log.append(msg)
    
    def drop_high_missingness_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Drop columns with >97% missing values (Phase 2 Report).
        """
        df = df.copy()
        cols_to_drop = [col for col in self.HIGH_MISSING_COLS if col in df.columns]
        
        if cols_to_drop:
            df = df.drop(columns=cols_to_drop)
            self.log_step(f"  ✓ Dropped {len(cols_to_drop)} high-missingness columns")
            for col in cols_to_drop:
                self.log_step(f"    - {col}")
        
        return df
    
    def remove_invalid_Speed_limits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove records where Speed_limit not in valid UK set {20,30,40,50,60,70}.
        
        Phase 2 Report: 36 invalid records found.
        """
        df = df.copy()
        
        if 'Speed_limit' in df.columns:
            initial_len = len(df)
            df = df[df['Speed_limit'].isin(self.VALID_SPEED_LIMITS)]
            dropped = initial_len - len(df)
            
            if dropped > 0:
                self.log_step(f"  ✓ Removed {dropped} records with invalid Speed_limit values")
        
        return df
    
    def correct_day_of_week(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Correct Day_of_Week inconsistencies.
        
        Phase 2 Report: Day_of_Week does not always match the day derived 
        from the Date column.
        """
        df = df.copy()
        
        if 'Date' in df.columns and 'Day_of_Week' in df.columns:
            if not pd.api.types.is_datetime64_any_dtype(df['Date']):
                df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            
            # Overwrite with correct values from Date
            df['Day_of_Week'] = df['Date'].dt.day_name()
            self.log_step(f"  ✓ Corrected Day_of_Week consistency violations")
        
        return df
    
    def handle_invalid_age_bands(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Mark invalid Age_Band_of_Driver values as missing.
        
        Phase 2 Report: 4,664 invalid codes found.
        """
        df = df.copy()
        
        if 'Age_Band_of_Driver' in df.columns:
            # Valid patterns: "0-5", "6-10", etc.
            valid_pattern = r'^\d+-\d+$'
            mask = df['Age_Band_of_Driver'].astype(str).str.match(valid_pattern)
            invalid_count = (~mask).sum()
            
            if invalid_count > 0:
                df.loc[~mask, 'Age_Band_of_Driver'] = None
                self.log_step(f"  ✓ Marked {invalid_count} invalid Age_Band_of_Driver values as NaN")
        
        return df
    
    def handle_sex_of_driver_codes(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Standardize Sex_of_Driver codes (76,119 invalid records).
        
        Phase 2 Report: 76,119 vehicle records contain invalid codes.
        Standardize to: {Male, Female, Not known}
        """
        df = df.copy()
        
        if 'Sex_of_Driver' in df.columns:
            # Valid patterns
            valid_codes = {'Male', 'Female', 'Not known', 'M', 'F', '1', '2', '3'}
            mask = df['Sex_of_Driver'].astype(str).isin(valid_codes)
            invalid_count = (~mask).sum()
            
            if invalid_count > 0:
                df.loc[~mask, 'Sex_of_Driver'] = 'Not known'
                self.log_step(f"  ✓ Remapped {invalid_count} invalid Sex_of_Driver codes to 'Not known'")
            
            # Standardize to full names
            mapping = {'M': 'Male', 'F': 'Female', '1': 'Male', '2': 'Female', '3': 'Not known'}
            df['Sex_of_Driver'] = df['Sex_of_Driver'].map(mapping).fillna(df['Sex_of_Driver'])
        
        return df
    
    def remove_rows_with_required_nulls(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Remove rows with missing values in required columns.
        
        Phase 2 Report:
            - Latitude: 174 nulls (0.0%)
            - Longitude: 175 nulls (0.0%)
            - Speed_limit: 37 nulls (0.0%)
        """
        df = df.copy()
        initial_len = len(df)
        
        # Check required columns
        required_present = [col for col in self.REQUIRED_COLS if col in df.columns]
        
        if required_present:
            df = df.dropna(subset=required_present)
            dropped = initial_len - len(df)
            
            if dropped > 0:
                self.log_step(f"  ✓ Removed {dropped} rows with null values in required columns")
                for col in required_present:
                    null_count = (df[col].isna().sum())
                    if null_count > 0:
                        self.log_step(f"    - {col}: {null_count} nulls")
        
        return df
    
    def handle_join_integrity(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Handle orphaned records from join.
        
        Phase 2 Report found that inner join on Accident_Index loses:
            - 657,532 accidents (32.1%) with no vehicle record
            - 99,257 vehicle records (6.7%) with no matching accident
        
        This function assumes the data is already merged.
        We just document the expected data loss.
        """
        df = df.copy()
        
        # If this is merged data, check for inconsistencies
        if 'Accident_Index' in df.columns:
            # Verify Accident_Index uniqueness patterns
            accident_counts = df['Accident_Index'].value_counts()
            
            # Report statistics
            unique_accidents = accident_counts.shape[0]
            vehicles_per_accident = accident_counts.mean()
            
            self.log_step(f"  ℹ Join integrity check:")
            self.log_step(f"    - Unique Accident_Index: {unique_accidents:,}")
            self.log_step(f"    - Avg vehicles per accident: {vehicles_per_accident:.2f}")
            self.log_step(f"    - Expected data loss: ~32% (per Phase 2 join integrity)")
        
        return df
    
    def check_and_remove_duplicates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Check for and remove duplicate records.
        
        Phase 2 Report: No full duplicates were found, but verify after cleaning.
        """
        df = df.copy()
        
        initial_len = len(df)
        
        # Check exact duplicates
        duplicates = df.duplicated()
        if duplicates.any():
            df = df.drop_duplicates()
            dropped = duplicates.sum()
            self.log_step(f"  ✓ Removed {dropped} exact duplicate rows")
        else:
            self.log_step(f"  ✓ No exact duplicates found")
        
        return df
    
    def clean(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Run complete cleaning pipeline per Phase 2 Validation Report.
        
        Handles all 15 issues across 8 validation dimensions:
        - ACCURACY (3): Invalid Speed_limit, Age_Band_of_Driver, Sex_of_Driver
        - CONSISTENCY (1): Day_of_Week mismatches
        - COMPLETENESS (7): High-missing columns + null values in required fields
        - UNIQUENESS (0): Duplicates
        - JOIN INTEGRITY (2): Orphaned records (already handled by inner join in Phase 2)
        """
        self.initial_shape = df.shape
        
        self.log_step(f"\n{'='*70}")
        self.log_step(f"DATA CLEANING PIPELINE (All 15 Phase 2 Validation Issues)")
        self.log_step(f"{'='*70}")
        self.log_step(f"  Initial shape: {self.initial_shape[0]:,} rows × {self.initial_shape[1]} cols\n")
        
        # ─── COMPLETENESS: Drop high-missingness columns ───────────────────
        self.log_step("1. COMPLETENESS - Dropping high-missingness columns (>20%):")
        df = self.drop_high_missingness_columns(df)
        
        # ─── ACCURACY: Remove invalid Speed_limit values ──────────────────
        self.log_step("\n2. ACCURACY - Removing invalid Speed_limit values:")
        df = self.remove_invalid_Speed_limits(df)
        
        # ─── COMPLETENESS: Remove rows with null required fields ──────────
        self.log_step("\n3. COMPLETENESS - Removing null values in required fields:")
        df = self.remove_rows_with_required_nulls(df)
        
        # ─── CONSISTENCY: Fix Day_of_Week mismatches ─────────────────────
        self.log_step("\n4. CONSISTENCY - Fixing Day_of_Week inconsistencies:")
        df = self.correct_day_of_week(df)
        
        # ─── ACCURACY: Handle invalid Age_Band_of_Driver ────────────────
        self.log_step("\n5. ACCURACY - Handling invalid Age_Band_of_Driver codes:")
        df = self.handle_invalid_age_bands(df)
        
        # ─── ACCURACY: Handle invalid Sex_of_Driver codes ──────────────
        self.log_step("\n6. ACCURACY - Handling invalid Sex_of_Driver codes:")
        df = self.handle_sex_of_driver_codes(df)
        
        # ─── UNIQUENESS: Remove duplicates ──────────────────────────────
        self.log_step("\n7. UNIQUENESS - Checking for duplicate records:")
        df = self.check_and_remove_duplicates(df)
        
        # ─── JOIN INTEGRITY: Log join integrity stats ──────────────────
        self.log_step("\n8. JOIN INTEGRITY - Merge statistics:")
        df = self.handle_join_integrity(df)
        
        self.final_shape = df.shape
        
        records_removed = self.initial_shape[0] - self.final_shape[0]
        pct_removed = (records_removed / self.initial_shape[0]) * 100
        
        self.log_step(f"\n{'='*70}")
        self.log_step(f"CLEANING COMPLETE - All 15 Issues Handled")
        self.log_step(f"{'='*70}")
        self.log_step(f"  Final shape: {self.final_shape[0]:,} rows × {self.final_shape[1]} cols")
        self.log_step(f"  Records removed: {records_removed:,} ({pct_removed:.2f}%)")
        self.log_step(f"\nSummary of issues fixed:")
        self.log_step(f"  [ACCURACY]     3 issues - Speed_limit, Age_Band, Sex_of_Driver")
        self.log_step(f"  [CONSISTENCY]  1 issue  - Day_of_Week mismatches")
        self.log_step(f"  [COMPLETENESS] 7 issues - High-missing columns + null fields")
        self.log_step(f"  [UNIQUENESS]   1 issue  - Exact duplicates removed")
        self.log_step(f"  [JOIN_INTEGRITY] 2 issues - Documented orphaned records")
        self.log_step(f"  [TOTAL] 15 issues from Phase 2 Validation Report")
        self.log_step(f"{'='*70}\n")
        
        return df




def load_data() -> pd.DataFrame:
    """
    Load the merged dataset from Phase 2 data acquisition.
    
    Returns:
        pd.DataFrame: Merged accident + vehicle + weather data
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 1: LOADING DATA")
    logger.info("=" * 70 + "\n")
    
    merged_pkl = INTERIM_DIR / "merged_data.pkl"
    
    if not merged_pkl.exists():
        logger.error(f"Merged data not found at {merged_pkl}")
        logger.info("Please ensure Phase 2 data acquisition is complete.")
        raise FileNotFoundError(f"{merged_pkl}")
    
    with open(merged_pkl, 'rb') as f:
        df = pickle.load(f)
    
    logger.info(f"✓ Loaded merged data")
    logger.info(f"  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    logger.info(f"  Memory: {df.memory_usage(deep=True).sum() / 1e6:.2f} MB\n")
    
    return df


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all data cleaning steps per Phase 2 Validation Report.
    
    This includes:
        - Dropping high-missingness columns (>97%)
        - Removing accuracy violations (invalid Speed_limit)
        - Fixing consistency violations (Day_of_Week, invalid codes)
        - Removing duplicate records
    
    Args:
        df: Raw merged dataframe from Phase 2
    
    Returns:
        Cleaned dataframe ready for feature engineering
    """
    cleaner = DataCleaner()
    return cleaner.clean(df)


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create all engineered features for modeling.
    
    Args:
        df: Cleaned dataframe
    
    Returns:
        DataFrame with engineered features added
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 3: FEATURE ENGINEERING")
    logger.info("=" * 70 + "\n")
    
    initial_cols = df.shape[1]
    
    # Temporal features
    df = create_temporal_features(df)
    
    # Lighting features (creates is_dark, not is_daylight to avoid redundancy)
    df = create_lighting_features(df)
    
    # ─── Basic categorical encoding (must be before Road_Type usage) ────────
    # Encode Road_Type (required before road_risk_features)
    df = encode_road_type_features(df)
    
    # Encode road surface to is_wet_road
    df = encode_road_surface_features(df)
    
    # Encode weather conditions to is_adverse_weather
    df = encode_weather_condition_features(df)
    
    # Encode urban/rural to is_urban
    df = encode_urban_rural_features(df)
    
    # Weather features
    df = create_weather_features(df)
    
    # Road risk features (uses encoded Road_Type)
    df = create_road_risk_features(df)
    
    # Composite weather features
    df = create_weather_composite_features(df)
    
    # Vehicle aggregation (for multi-vehicle accidents)
    df = create_vehicle_features(df)
    
    # ─── Feature-specific encodings ─────────────────────────────────────────
    # Encode vehicle attributes
    df = encode_vehicle_attributes(df)
    
    # Encode driver features
    df = encode_driver_features(df)
    
    # Encode manoeuvre
    df = encode_manoeuvre_features(df)
    
    # Encode junction features
    df = encode_junction_features(df)
    
    # Encode journey features
    df = encode_journey_features(df)
    
    # Encode administrative/location features
    df = encode_administrative_features(df)
    
    # Create interaction features (must be after base features exist)
    df = create_interaction_features(df)
    
    # ─────────────────────────────────────────────────────────────────────────
    
    new_cols = df.shape[1] - initial_cols
    logger.info(f"✓ Feature engineering complete")
    logger.info(f"  New features created: {new_cols}")
    logger.info(f"  Total columns: {df.shape[1]}\n")
    
    return df


def preprocess_features(df: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
    """
    Preprocess features: handle missing values, encode categoricals, prepare for modeling.
    
    Args:
        df: DataFrame with engineered features
    
    Returns:
        (preprocessed_df, preprocessing_metadata)
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 4: FEATURE PREPROCESSING")
    logger.info("=" * 70 + "\n")
    
    metadata = {}
    
    # Handle missing values
    logger.info("Handling missing values...")
    logger.info(f"Before handle_missing_values: Accident_Severity classes={df['Accident_Severity'].unique()}")
    df = handle_missing_values(df, missing_threshold=CONFIG['missing_threshold'])
    logger.info(f"After handle_missing_values: Accident_Severity classes={df['Accident_Severity'].unique()}")
    
    # Detect and handle outliers
    logger.info("\nDetecting outliers in numerical features...")
    logger.info(f"Before outlier_detection: Accident_Severity classes={df['Accident_Severity'].unique()}")
    df, outlier_stats = detect_and_handle_outliers(
        df,
        method='iqr',
        iqr_multiplier=1.5,
        action='clip'
    )
    logger.info(f"After outlier_detection: Accident_Severity classes={df['Accident_Severity'].unique()}")
    metadata['outlier_stats'] = outlier_stats
    
    # Encode categorical features
    logger.info("\nEncoding categorical features...")
    logger.info(f"Before categorical_encoding: Accident_Severity classes={df['Accident_Severity'].unique()}")
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    # Remove non-predictive columns
    skip_cols = ['Accident_Severity', 'Accident_Index', 'Date', 'Day_of_Week']
    categorical_cols = [c for c in categorical_cols if c not in skip_cols]
    
    df, encoders = encode_categorical_features(
        df, categorical_cols=categorical_cols, method='label', max_categories=20
    )
    logger.info(f"After categorical_encoding: Accident_Severity classes={df['Accident_Severity'].unique()}")
    metadata['encoders'] = encoders
    
    logger.info(f"✓ Preprocessing complete")
    logger.info(f"  Final shape: {df.shape}\n")
    
    return df, metadata


def select_features(X: pd.DataFrame, 
                   y: pd.Series,
                   method: str = None) -> Tuple[List[str], dict]:
    """
    Perform feature selection using RFECV and/or Random Forest methods.
    
    Supports two methods for comparison:
    1. RFECV: Recursive Feature Elimination with Cross-Validation
       - Automatically determines optimal number of features via CV
       - More robust to overfitting
    
    2. Random Forest: Tree-based feature importance
       - Fast and interpretable
       - Good for understanding feature relationships
    
    When compare_feature_selection is True, runs BOTH methods and saves results
    for later comparison during model evaluation. Uses the primary method
    (feature_selection_method in CONFIG) for train/val/test split.
    
    Args:
        X: Feature matrix (should be numeric after preprocessing)
        y: Target variable
        method: Feature selection method (uses CONFIG if None). Options: 'rfecv', 'model_based', 'both'
    
    Returns:
        (selected_feature_names, selection_results_dict)
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 5: FEATURE SELECTION")
    logger.info("=" * 70 + "\n")
    
    if method is None:
        method = CONFIG.get('feature_selection_method', 'rfecv')
    
    # Check if we should run both methods for comparison
    run_comparison = CONFIG.get('compare_feature_selection', False)
    if run_comparison:
        method = 'both'
    
    logger.info(f"Feature selection mode: {method.upper()}")
    if run_comparison:
        logger.info(f"Primary method for train/val/test split: {CONFIG.get('feature_selection_method', 'rfecv').upper()}\n")
    else:
        logger.info("")  # Empty line for readability
    
    results = {}
    
    # Diagnostic: Check target variable
    logger.info(f"Target variable (y) diagnostic:")
    logger.info(f"  - Type: {type(y)}")
    logger.info(f"  - Length: {len(y) if y is not None else 'None'}")
    if y is not None:
        logger.info(f"  - Unique values: {y.unique() if hasattr(y, 'unique') else set(y)}")
        logger.info(f"  - Value counts:\n{y.value_counts().sort_index() if hasattr(y, 'value_counts') else {}}")
    
    logger.info(f"Feature matrix (X) diagnostic:")
    logger.info(f"  - Shape: {X.shape}")
    logger.info(f"  - Columns: {list(X.columns)[:5]}... (showing first 5)")
    
    # Filter to numeric columns only (drop any remaining string/object columns)
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    X_numeric = X[numeric_cols]
    
    if len(X_numeric.columns) == 0:
        raise ValueError("No numeric columns found after preprocessing")
    
    logger.info(f"Using {len(X_numeric.columns)} numeric features for selection\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # RFECV Feature Selection
    # ─────────────────────────────────────────────────────────────────────────
    if method in ['rfecv', 'both']:
        min_features = CONFIG.get('rfecv_min_features', 15)
        cv_folds = CONFIG.get('rfecv_cv_folds', 5)
        
        logger.info("=" * 70)
        logger.info("METHOD 1: RFECV (Recursive Feature Elimination + Cross-Validation)")
        logger.info("=" * 70)
        logger.info(f"Parameters:")
        logger.info(f"  - Base estimator: LogisticRegression")
        logger.info(f"  - CV folds: {cv_folds}")
        logger.info(f"  - Min features to select: {min_features}\n")
        
        rfecv_features, rfecv_ranking = select_features_rfecv(
            X_numeric, 
            y,
            min_features_to_select=min_features,
            random_state=CONFIG['random_state'],
            cv_folds=cv_folds
        )
        
        results['rfecv'] = {
            'selected_features': rfecv_features,
            'n_features': len(rfecv_features),
            'ranking_scores': rfecv_ranking.to_dict('records')
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Random Forest Feature Selection
    # ─────────────────────────────────────────────────────────────────────────
    if method in ['model_based', 'both']:
        if method == 'both':
            logger.info("\n" + "=" * 70)
        logger.info("METHOD 2: RANDOM FOREST (Tree-based Feature Importance)")
        logger.info("=" * 70)
        logger.info(f"Parameters:")
        logger.info(f"  - Estimator: RandomForestClassifier(n_estimators=50, max_depth=10)")
        logger.info(f"  - Selection: Top-k by importance\n")
        
        # Use same number of features as RFECV for fair comparison
        if method == 'both' and 'rfecv' in results:
            n_features = results['rfecv']['n_features']
        else:
            n_features = CONFIG.get('rfecv_min_features', 25)
        
        rf_features, rf_importance = select_features_model_based(
            X_numeric, y, top_k=n_features
        )
        
        results['model_based'] = {
            'selected_features': rf_features,
            'n_features': len(rf_features),
            'importance_scores': rf_importance.head(n_features).to_dict('records')
        }
    
    # ─────────────────────────────────────────────────────────────────────────
    # Select Primary Method Features for Train/Val/Test Split
    # ─────────────────────────────────────────────────────────────────────────
    primary_method = CONFIG.get('feature_selection_method', 'rfecv')
    
    if method == 'both':
        logger.info("\n" + "=" * 70)
        logger.info("COMPARISON SUMMARY")
        logger.info("=" * 70)
        
        if primary_method in results:
            selected_features = results[primary_method]['selected_features']
            logger.info(f"\n✓ Using {primary_method.upper()} features for train/val/test split")
            logger.info(f"  Number of features: {len(selected_features)}\n")
            
            # Show comparison
            logger.info(f"RFECV features: {results['rfecv']['n_features']}")
            logger.info(f"Random Forest features: {results['model_based']['n_features']}")
            logger.info(f"\nBoth feature sets saved for later comparison during model evaluation")
        else:
            raise ValueError(f"Primary method '{primary_method}' not found in results")
    else:
        if primary_method == 'rfecv':
            selected_features = results['rfecv']['selected_features']
        else:
            selected_features = results['model_based']['selected_features']
        
        logger.info(f"\n✓ Selected {len(selected_features)} features using {primary_method.upper()}")
    
    results['final_features'] = selected_features
    results['n_features_selected'] = len(selected_features)
    results['primary_method'] = primary_method
    
    return selected_features, results


def prepare_train_val_test_split(df: pd.DataFrame,
                                 target_col: str = 'Accident_Severity',
                                 selected_features: List[str] = None,
                                 test_frac: float = 0.2,
                                 val_frac: float = 0.5) -> Tuple:
    """
    Split data into train/validation/test sets, apply SMOTE, and scale.
    
    Args:
        df: Preprocessed dataframe with target
        target_col: Name of target column
        selected_features: Features to keep (None = use all numeric)
        test_frac: Fraction for test split (default 0.2)
        val_frac: Fraction of remaining for validation (default 0.5 → 40% train, 40% val)
    
    Returns:
        (X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names)
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 6: TRAIN/VALIDATION/TEST SPLIT & BALANCING")
    logger.info("=" * 70 + "\n")
    
    # Separate features and target
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found")
    
    y = df[target_col]
    
    # Select features
    if selected_features:
        X = df[selected_features].copy()
        logger.info(f"Using {len(selected_features)} selected features")
    else:
        # Use all numeric columns except target
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if target_col in numeric_cols:
            numeric_cols.remove(target_col)
        X = df[numeric_cols].copy()
        logger.info(f"Using all {len(numeric_cols)} numeric features")
    
    logger.info(f"\nClass distribution (before any splitting):\n{y.value_counts()}\n")
    
    # SMOTE cannot handle NaNs
    # Fill missing values in selected features before splitting
    logger.info("✓ Imputing NaNs in selected features (required for SMOTE)...")
    for col in X.columns:
        if X[col].isnull().any():
            null_count = X[col].isnull().sum()
            if X[col].dtype in ['float64', 'int64']:
                X[col] = X[col].fillna(X[col].median())
            else:
                X[col] = X[col].fillna(X[col].mode()[0] if len(X[col].mode()) > 0 else 'Unknown')
            logger.info(f"  - {col}: Imputed {null_count} NaN values")
    
    # Step 1: Train/test split (80/20)
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y,
        test_size=test_frac,
        random_state=CONFIG['random_state'],
        stratify=y
    )
    
    # Step 2: Train/validation split of remaining 80% (50/50 → 40% each)
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val,
        test_size=val_frac,
        random_state=CONFIG['random_state'],
        stratify=y_train_val
    )
    
    logger.info("Train/Val/Test split:")
    logger.info(f"  Training:   {X_train.shape[0]:7d} samples ({X_train.shape[0]/len(y)*100:5.1f}%)")
    logger.info(f"  Validation: {X_val.shape[0]:7d} samples ({X_val.shape[0]/len(y)*100:5.1f}%)")
    logger.info(f"  Test:       {X_test.shape[0]:7d} samples ({X_test.shape[0]/len(y)*100:5.1f}%)\n")
    
    logger.info("Class distribution BEFORE SMOTE:\n")
    logger.info("Training set:")
    logger.info(y_train.value_counts().to_string())
    
    # Apply SMOTE to training data ONLY
    if CONFIG['apply_smote']:
        logger.info(f"\n✓ Applying SMOTE to training split only...")
        X_train, y_train = apply_smote(
            X_train, y_train,
            sampling_strategy=CONFIG['smote_sampling_strategy'],
            random_state=CONFIG['random_state']
        )
    
    # Scale features (fit on training data, apply to all)
    logger.info(f"\n✓ Fitting scaler on training data...")
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_val_scaled = scaler.transform(X_val)
    X_test_scaled = scaler.transform(X_test)
    
    # Convert back to DataFrame
    X_train_scaled = pd.DataFrame(X_train_scaled, columns=X_train.columns, index=X_train.index)
    X_val_scaled = pd.DataFrame(X_val_scaled, columns=X_val.columns, index=X_val.index)
    X_test_scaled = pd.DataFrame(X_test_scaled, columns=X_test.columns, index=X_test.index)
    
    logger.info(f"  Feature scaling complete (StandardScaler)\n")
    
    # Fill any remaining NaNs with 0 (from scaling operations)
    X_train_scaled = X_train_scaled.fillna(0)
    X_val_scaled = X_val_scaled.fillna(0)
    X_test_scaled = X_test_scaled.fillna(0)
    
    return X_train_scaled, X_val_scaled, X_test_scaled, y_train, y_val, y_test, scaler, X_train.columns.tolist()



def save_artifacts(X_train: pd.DataFrame,
                  X_val: pd.DataFrame,
                  X_test: pd.DataFrame,
                  y_train: pd.Series,
                  y_val: pd.Series,
                  y_test: pd.Series,
                  scaler: StandardScaler,
                  feature_names: List[str],
                  selection_results: dict,
                  preprocessing_metadata: dict) -> None:
    """
    Save all preprocessed data and metadata.
    
    Args:
        X_train, X_val, X_test: Feature matrices
        y_train, y_val, y_test: Target arrays
        scaler: Fitted scaler object
        feature_names: List of feature column names
        selection_results: Feature selection results dict
        preprocessing_metadata: Preprocessing metadata
    """
    logger.info("\n" + "=" * 70)
    logger.info("STEP 7: SAVING ARTIFACTS")
    logger.info("=" * 70 + "\n")
    
    # Save training split
    with open(PROCESSED_DIR / 'X_train.pkl', 'wb') as f:
        pickle.dump(X_train, f)
    logger.info(f"✓ Saved X_train.pkl ({X_train.shape})")
    
    with open(PROCESSED_DIR / 'y_train.pkl', 'wb') as f:
        pickle.dump(y_train, f)
    logger.info(f"✓ Saved y_train.pkl ({len(y_train)} samples)")
    
    # Save validation split
    with open(PROCESSED_DIR / 'X_val.pkl', 'wb') as f:
        pickle.dump(X_val, f)
    logger.info(f"✓ Saved X_val.pkl ({X_val.shape})")
    
    with open(PROCESSED_DIR / 'y_val.pkl', 'wb') as f:
        pickle.dump(y_val, f)
    logger.info(f"✓ Saved y_val.pkl ({len(y_val)} samples)")
    
    # Save test split
    with open(PROCESSED_DIR / 'X_test.pkl', 'wb') as f:
        pickle.dump(X_test, f)
    logger.info(f"✓ Saved X_test.pkl ({X_test.shape})")
    
    with open(PROCESSED_DIR / 'y_test.pkl', 'wb') as f:
        pickle.dump(y_test, f)
    logger.info(f"✓ Saved y_test.pkl ({len(y_test)} samples)")
    
    # Save scaler and feature names
    with open(PROCESSED_DIR / 'scaler.pkl', 'wb') as f:
        pickle.dump(scaler, f)
    logger.info(f"✓ Saved scaler.pkl")
    
    with open(PROCESSED_DIR / 'feature_names.pkl', 'wb') as f:
        pickle.dump(feature_names, f)
    logger.info(f"✓ Saved feature_names.pkl ({len(feature_names)} features)")
    
    # Save metadata as JSON
    metadata = {
        'preprocessing_timestamp': datetime.now().isoformat(),
        'config': CONFIG,
        'data_shapes': {
            'X_train': X_train.shape,
            'X_val': X_val.shape,
            'X_test': X_test.shape,
            'y_train': y_train.shape,
            'y_val': y_val.shape,
            'y_test': y_test.shape,
        },
        'class_distribution': {
            'train': y_train.value_counts().to_dict(),
            'val': y_val.value_counts().to_dict(),
            'test': y_test.value_counts().to_dict(),
        },
        'feature_names': feature_names,
        'feature_selection_results': selection_results,
        'preprocessing_metadata': preprocessing_metadata,
    }
    
    with open(PROCESSED_DIR / 'preprocessing_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2, default=str)
    logger.info(f"✓ Saved preprocessing_metadata.json")
    
    logger.info(f"\n✓ All artifacts saved to: {PROCESSED_DIR}\n")


def generate_report(X_train: pd.DataFrame,
                   X_val: pd.DataFrame,
                   X_test: pd.DataFrame,
                   y_train: pd.Series,
                   y_val: pd.Series,
                   y_test: pd.Series,
                   feature_names: List[str],
                   selection_results: dict = None) -> str:
    """
    Generate a text summary report of preprocessing.
    
    Returns:
        Report text
    """
    report = []
    report.append("=" * 70)
    report.append("DATA PREPROCESSING REPORT")
    report.append("=" * 70)
    report.append(f"\nGenerated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    if selection_results and CONFIG.get('compare_feature_selection', False):
        report.append(f"\nFEATURE SELECTION COMPARISON MODE:")
        if 'rfecv' in selection_results:
            report.append(f"  - RFECV selected: {selection_results['rfecv']['n_features']} features")
        if 'model_based' in selection_results:
            report.append(f"  - Random Forest selected: {selection_results['model_based']['n_features']} features")
        report.append(f"  - Primary method used: {selection_results.get('primary_method', 'rfecv').upper()}")
        report.append(f"\nBoth feature sets are saved for later model comparison")
    
    report.append("\n" + "-" * 70)
    report.append("DATA SHAPES")
    report.append("-" * 70)
    report.append(f"X_train: {X_train.shape}")
    report.append(f"X_val:   {X_val.shape}")
    report.append(f"X_test:  {X_test.shape}")
    report.append(f"y_train: {y_train.shape}")
    report.append(f"y_val:   {y_val.shape}")
    report.append(f"y_test:  {y_test.shape}")
    
    report.append("\n" + "-" * 70)
    report.append("CLASS DISTRIBUTION")
    report.append("-" * 70)
    
    report.append("\nTraining set:")
    report.append(y_train.value_counts().to_string())
    report.append(f"\n  Percentages:")
    report.append((y_train.value_counts(normalize=True) * 100).to_string())
    
    report.append("\n\nValidation set:")
    report.append(y_val.value_counts().to_string())
    report.append(f"\n  Percentages:")
    report.append((y_val.value_counts(normalize=True) * 100).to_string())
    
    report.append("\n\nTest set:")
    report.append(y_test.value_counts().to_string())
    report.append(f"\n  Percentages:")
    report.append((y_test.value_counts(normalize=True) * 100).to_string())
    
    report.append("\n" + "-" * 70)
    report.append("SELECTED FEATURES")
    report.append("-" * 70)
    report.append(f"Total features: {len(feature_names)}\n")
    for i, fname in enumerate(feature_names, 1):
        report.append(f"  {i:2d}. {fname}")
    
    report.append("\n" + "=" * 70)
    
    return "\n".join(report)


def main():
    """Main preprocessing pipeline."""
    try:
        logger.info("\n" + "="*70)
        logger.info("ACCIDENT SEVERITY PREDICTION — DATA PREPROCESSING PIPELINE")
        logger.info("="*70)
        
        # 1. Load raw merged data
        df = load_data()
        logger.info(f"After load: shape={df.shape}")
        logger.info(f"  Raw Accident_Severity value counts:\n{df['Accident_Severity'].value_counts()}")
        logger.info(f"  Unique classes: {df['Accident_Severity'].unique()}")
        
        # 2. Clean data (per Phase 2 validation report)
        df = clean_data(df)
        logger.info(f"After clean: shape={df.shape}")
        logger.info(f"  Raw Accident_Severity value counts:\n{df['Accident_Severity'].value_counts()}")
        logger.info(f"  Unique classes: {df['Accident_Severity'].unique()}")
        
        # 2.5. Engineer features with target encoding (BEFORE target variable encoding)
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3: FEATURE ENGINEERING (Target Encoding BEFORE Target Encoding)")
        logger.info("=" * 70 + "\n")
        df = engineer_features(df)
        logger.info(f"After engineer: shape={df.shape}, Accident_Severity classes={df['Accident_Severity'].unique()}")
        
        # 2.6. Encode target variable to numeric (AFTER target encoding)
        logger.info("\n" + "=" * 70)
        logger.info("STEP 3.5: TARGET VARIABLE ENCODING (After Target Encoding)")
        logger.info("=" * 70 + "\n")
        df = encode_target_variable(df)
        logger.info(f"After encode_target: shape={df.shape}, Accident_Severity dtype={df['Accident_Severity'].dtype}")
        logger.info(f"  Accident_Severity value counts:\n{df['Accident_Severity'].value_counts().sort_index()}")
        logger.info(f"After engineer: shape={df.shape}, Accident_Severity classes={df['Accident_Severity'].unique()}")
        
        # 4. Preprocess features
        df, preprocessing_metadata = preprocess_features(df)
        logger.info(f"After preprocess: shape={df.shape}, Accident_Severity classes={df['Accident_Severity'].unique()}")
        
        # 5. Select features
        selected_features, selection_results = select_features(
            df.drop(columns=['Accident_Severity'], errors='ignore'),
            df['Accident_Severity'] if 'Accident_Severity' in df.columns else None
        )
        
        # 6. Train/Val/Test split + SMOTE + Scaling
        X_train, X_val, X_test, y_train, y_val, y_test, scaler, feature_names = prepare_train_val_test_split(
            df,
            target_col='Accident_Severity',
            selected_features=selected_features
        )
        
        # 7. Save artifacts
        save_artifacts(
            X_train, X_val, X_test,
            y_train, y_val, y_test,
            scaler,
            feature_names,
            selection_results,
            preprocessing_metadata
        )
        
        # 8. Generate and save report
        report_text = generate_report(
            X_train, X_val, X_test, 
            y_train, y_val, y_test, 
            feature_names,
            selection_results=selection_results
        )
        
        with open(REPORTS_DIR / 'preprocessing_report.txt', 'w') as f:
            f.write(report_text)
        
        logger.info("✓ Report saved to: reports/preprocessing_report.txt")
        
        # 9. If comparison mode, save alternative feature sets for later testing
        if CONFIG.get('compare_feature_selection', False):
            logger.info("\n" + "="*70)
            logger.info("SAVING ALTERNATIVE FEATURE SETS FOR COMPARISON")
            logger.info("="*70 + "\n")
            
            if 'rfecv' in selection_results and 'model_based' in selection_results:
                comparison_data = {
                    'rfecv_features': selection_results['rfecv']['selected_features'],
                    'model_based_features': selection_results['model_based']['selected_features'],
                    'rfecv_n_features': selection_results['rfecv']['n_features'],
                    'model_based_n_features': selection_results['model_based']['n_features'],
                    'primary_method': selection_results['primary_method'],
                }
                
                with open(PROCESSED_DIR / 'feature_selection_comparison.pkl', 'wb') as f:
                    pickle.dump(comparison_data, f)
                
                logger.info(f"✓ Saved feature selection comparison data")
                logger.info(f"  - RFECV: {selection_results['rfecv']['n_features']} features")
                logger.info(f"  - Random Forest: {selection_results['model_based']['n_features']} features")
                logger.info(f"  - File: data/processed/feature_selection_comparison.pkl\n")
        
        logger.info("="*70)
        logger.info("PREPROCESSING PIPELINE COMPLETE ✓")
        logger.info("="*70)
        logger.info("\nNext steps: Use the preprocessed data for model training")
        logger.info(f"  - Training data: {PROCESSED_DIR / 'X_train.pkl'}")
        logger.info(f"  - Validation data: {PROCESSED_DIR / 'X_val.pkl'}")
        logger.info(f"  - Test data: {PROCESSED_DIR / 'X_test.pkl'}")
        
        if CONFIG.get('compare_feature_selection', False):
            logger.info(f"\n  - Feature comparison: {PROCESSED_DIR / 'feature_selection_comparison.pkl'}")
            logger.info(f"    Use this to test both feature sets during model evaluation")
        
        logger.info("\n")
        
    except Exception as e:
        logger.error(f"\n✗ PREPROCESSING FAILED: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
