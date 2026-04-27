"""
Feature Engineering Module — UK Road Accidents Project

This module provides functions for creating, transforming, and selecting features
for the accident severity classification task.

Helper Functions:
    - create_temporal_features()            : Extract hour, day, weekend flags
    - create_lighting_features()            : Encode darkness conditions
    - create_weather_features()             : Encode adverse weather conditions
    - create_road_risk_features()           : Composite risk score from speed and road type
    - create_weather_composite_features()   : Combine weather with other risk factors
    - create_vehicle_features()             : Aggregate vehicle-level attributes
    - encode_categorical_features()         : One-hot encode categorical variables
    - detect_and_handle_outliers()          : IQR-based outlier detection and handling
    - handle_missing_values()               : Drop high-missing columns, impute others
    - select_features_statistical()         : Correlation-based feature selection
    - select_features_model_based()         : Tree-based feature importance
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import spearmanr
from typing import Tuple, List
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def create_temporal_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract temporal features from Date/Time columns.
    
    Creates:
        - hour_of_day: 0-23 (rushes hours, night driving patterns)
        - day_of_week: 0-6 (weekday vs weekend behavior)
        - is_weekend: Binary (1 if Sat/Sun)
        - month: 1-12 (seasonal patterns)
        - season: {Spring, Summer, Autumn, Winter}
    
    Args:
        df: DataFrame with 'Date' column (datetime or string)
    
    Returns:
        DataFrame with temporal features added
    """
    df = df.copy()
    
    # Ensure Date is datetime
    if not pd.api.types.is_datetime64_any_dtype(df['Date']):
        df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
    
    # Extract components
    df['hour_of_day'] = df['Date'].dt.hour
    df['day_of_week'] = df['Date'].dt.dayofweek  # 0=Monday, 6=Sunday
    df['is_weekend'] = ((df['day_of_week'] == 5) | (df['day_of_week'] == 6)).astype(int)
    df['month'] = df['Date'].dt.month
    
    # Map months to seasons
    season_map = {12: 'Winter', 1: 'Winter', 2: 'Winter',
                  3: 'Spring', 4: 'Spring', 5: 'Spring',
                  6: 'Summer', 7: 'Summer', 8: 'Summer',
                  9: 'Autumn', 10: 'Autumn', 11: 'Autumn'}
    df['season'] = df['month'].map(season_map)
    
    logger.info("✓ Temporal features created (hour_of_day, is_weekend, season)")
    return df


def create_lighting_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create binary lighting condition flag.
    
    Creates:
        - is_dark: 1 if any darkness condition, 0 for daylight
    
    Args:
        df: DataFrame with 'Light_Conditions' column
    
    Returns:
        DataFrame with lighting features added
    """
    df = df.copy()
    
    if 'Light_Conditions' in df.columns:
        # Assume values like 'Daylight', 'Darkness - lights lit', 'Darkness - no lights', etc.
        dark_keywords = ['Darkness', 'darkness']
        df['is_dark'] = df['Light_Conditions'].fillna('').apply(
            lambda x: 1 if any(kw in str(x) for kw in dark_keywords) else 0
        )
    else:
        df['is_dark'] = 0
        logger.warning("Light_Conditions column not found, is_dark set to 0")
    
    logger.info("✓ Lighting features created (is_dark)")
    return df


def create_weather_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create adverse weather conditions flag.
    
    Creates:
        - is_adverse_weather: 1 if precipitation > 0 OR snow_depth > 0 OR cloud_cover >= 6
    
    Uses columns: prcp (precipitation), snwd (snow depth), cldc (cloud cover in oktas)
    
    Args:
        df: DataFrame with weather columns
    
    Returns:
        DataFrame with weather features added
    """
    df = df.copy()
    
    # Initialize to 0
    df['is_adverse_weather'] = 0
    
    # Check precipitation
    if 'prcp' in df.columns:
        df['is_adverse_weather'] = df['is_adverse_weather'] | (df['prcp'].fillna(0) > 0).astype(int)
    
    # Check snow depth
    if 'snwd' in df.columns:
        df['is_adverse_weather'] = df['is_adverse_weather'] | (df['snwd'].fillna(0) > 0).astype(int)
    
    # Check cloud cover (mostly overcast = 6+ oktas)
    if 'cldc' in df.columns:
        df['is_adverse_weather'] = df['is_adverse_weather'] | (df['cldc'].fillna(0) >= 6).astype(int)
    
    logger.info("✓ Weather features created (is_adverse_weather)")
    return df


def create_weather_composite_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite features combining weather with other risk factors.
    
    Creates:
        - adverse_dark: 1 if adverse weather AND dark conditions (high-risk combination)
        - precip_single_carriageway: 1 if precipitation > threshold AND single carriageway
        - temp_road_risk: Temperature-adjusted road risk score
        - seasonal_risk: Seasonal risk indicator based on temperature and month
    
    Args:
        df: DataFrame with weather and road features
    
    Returns:
        DataFrame with composite weather features added
    """
    df = df.copy()
    
    # 1. Adverse weather + darkness (high-risk combination)
    if 'is_adverse_weather' in df.columns and 'is_dark' in df.columns:
        df['adverse_dark'] = (df['is_adverse_weather'] * df['is_dark']).astype(int)
    else:
        df['adverse_dark'] = 0
        logger.warning("is_adverse_weather or is_dark not found, adverse_dark set to 0")
    
    # 2. Precipitation on single carriageway (slippery conditions on high-speed roads)
    if 'prcp' in df.columns and 'Road_Type' in df.columns:
        df['precip_single_carriageway'] = (
            (df['prcp'].fillna(0) > 1.0) &  # >1mm precipitation (light rain threshold)
            (df['Road_Type'].fillna('') == 'Single carriageway')
        ).astype(int)
    else:
        df['precip_single_carriageway'] = 0
        logger.warning("prcp or Road_Type not found, precip_single_carriageway set to 0")
    
    # 3. Temperature-adjusted road risk (cold temps increase risk on high-speed roads)
    if 'temp' in df.columns and 'road_risk_score' in df.columns:
        # Cold (<5°C) increases risk on high road_risk_score
        df['temp_road_risk'] = df['road_risk_score'] * (1 + (df['temp'].fillna(15) < 5).astype(int) * 0.5)
    else:
        df['temp_road_risk'] = df.get('road_risk_score', 0)
        logger.warning("temp or road_risk_score not found, temp_road_risk set to road_risk_score")
    
    # 4. Seasonal risk (winter months + cold temps)
    if 'temp' in df.columns and 'month' in df.columns:
        # Winter months (Dec, Jan, Feb) with cold temps (<5°C)
        df['seasonal_risk'] = (
            (df['month'].isin([12, 1, 2])) & 
            (df['temp'].fillna(15) < 5)
        ).astype(int)
    else:
        df['seasonal_risk'] = 0
        logger.warning("temp or month not found, seasonal_risk set to 0")
    
    logger.info("✓ Composite weather features created (adverse_dark, precip_single_carriageway, temp_road_risk, seasonal_risk)")
    return df


def create_road_risk_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create composite road risk score.
    
    Creates:
        - road_risk_score: Ordinal combining Speed_limit and Road_Type
            Higher score = higher risk (rural, single carriageway, high speed)
    
    Args:
        df: DataFrame with 'Speed_limit' and 'Road_Type' columns
    
    Returns:
        DataFrame with road risk features added
    """
    df = df.copy()
    
    # Skip if Speed_limit not available (some datasets may not have it)
    if 'Speed_limit' not in df.columns:
        logger.info("⚠ Speed_limit column not found - skipping road_risk_score")
        return df
    
    # Base score from speed limit
    speed_risk = {20: 1, 30: 2, 40: 3, 50: 4, 60: 5, 70: 6}
    df['speed_risk'] = df['Speed_limit'].fillna(30).map(speed_risk).fillna(3)
    
    # Road type multiplier
    road_type_mult = {
        'Single carriageway': 1.5,
        'Dual carriageway': 1.0,
        'Roundabout': 0.8,
        'One-way street': 0.9,
        'Slip road': 0.7
    }
    df['road_mult'] = df['Road_Type'].fillna('Single carriageway').map(road_type_mult).fillna(1.0)
    
    # Composite score
    df['road_risk_score'] = (df['speed_risk'] * df['road_mult']).astype(int)
    df = df.drop(['speed_risk', 'road_mult'], axis=1)
    
    logger.info("✓ Road risk features created (road_risk_score)")
    return df


def create_vehicle_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate vehicle-level features to accident level.
    
    Creates:
        - max_driver_age: Max age across all drivers in the accident
        - min_driver_age: Min age across all drivers
        - num_motorcycles: Count of motorcycles involved
        - involves_pedestrian: 1 if any pedestrian-type vehicle involved
        - avg_vehicle_age: Average vehicle age
    
    Args:
        df: DataFrame with vehicle-level rows (may have duplicates per Accident_Index)
    
    Returns:
        DataFrame grouped to accident level with aggregated features
    """
    df = df.copy()
    
    # For vehicle features, we assume the dataframe is already merged at accident level
    # or we need to aggregate. Here we create the features from existing aggregates.
    
    # If Age_of_Driver exists, ensure it's numeric
    if 'Age_of_Driver' in df.columns:
        df['Age_of_Driver'] = pd.to_numeric(df['Age_of_Driver'], errors='coerce')
        
        # Keep max age (already aggregated ideally, but ensure)
        if 'max_driver_age' not in df.columns:
            # If multiple rows per accident, group and aggregate
            if df.groupby('Accident_Index', as_index=False).size()['size'].max() > 1:
                df['max_driver_age'] = df.groupby('Accident_Index')['Age_of_Driver'].transform('max')
                df['min_driver_age'] = df.groupby('Accident_Index')['Age_of_Driver'].transform('min')
            else:
                df['max_driver_age'] = df['Age_of_Driver']
                df['min_driver_age'] = df['Age_of_Driver']
    
    # Motorcycle count
    if 'Vehicle_Type' in df.columns:
        def count_motorcycles(vtype):
            if pd.isna(vtype):
                return 0
            return 1 if 'Motorcycle' in str(vtype) else 0
        
        if 'num_motorcycles' not in df.columns:
            if df.groupby('Accident_Index', as_index=False).size()['size'].max() > 1:
                df['num_motorcycles'] = df.groupby('Accident_Index')['Vehicle_Type'].transform(
                    lambda x: sum(count_motorcycles(v) for v in x)
                )
            else:
                df['num_motorcycles'] = df['Vehicle_Type'].apply(count_motorcycles)
    
    # Pedestrian involvement (assume already aggregated or check casualty types)
    if 'involves_pedestrian' not in df.columns:
        df['involves_pedestrian'] = 0
    
    logger.info("✓ Vehicle features created (max_driver_age, num_motorcycles, etc.)")
    return df


def detect_and_handle_outliers(df: pd.DataFrame, 
                               method: str = 'iqr',
                               iqr_multiplier: float = 1.5,
                               action: str = 'clip') -> Tuple[pd.DataFrame, dict]:
    """
    Detect and handle outliers in numerical columns using IQR method.
    
    Rationale: 
    - IQR method is robust to extreme outliers (resistant to skewness)
    - Suitable for accident severity data which may have extreme cases
    - Clipping preserves data structure (better than removal for imbalanced data)
    
    Args:
        df: DataFrame to process
        method: 'iqr' (Interquartile Range) - standard statistical method
        iqr_multiplier: Typically 1.5 (standard), use 3.0 for conservative approach
        action: 'clip' (cap values) or 'remove' (drop rows with outliers)
    
    Returns:
        (df_processed, outlier_stats) - processed dataframe and statistics dict
    """
    df = df.copy()
    outlier_stats = {'method': method, 'action': action, 'columns_affected': []}
    
    # Identify numerical columns (excluding identifiers)
    num_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['Accident_Index', 'Vehicle_Index', 'Casualty_Index']
    num_cols = [col for col in num_cols if col not in exclude_cols]
    
    for col in num_cols:
        if df[col].isna().sum() > 0:
            continue  # Skip columns with missing values (handle separately)
        
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - (iqr_multiplier * IQR)
        upper_bound = Q3 + (iqr_multiplier * IQR)
        
        outlier_mask = (df[col] < lower_bound) | (df[col] > upper_bound)
        n_outliers = outlier_mask.sum()
        
        if n_outliers > 0:
            outlier_stats['columns_affected'].append({
                'column': col,
                'n_outliers': int(n_outliers),
                'pct_outliers': round(100 * n_outliers / len(df), 2),
                'lower_bound': round(lower_bound, 3),
                'upper_bound': round(upper_bound, 3)
            })
            
            if action == 'clip':
                df[col] = df[col].clip(lower_bound, upper_bound)
                logger.debug(f"  ⚠ Clipped {n_outliers} outliers in {col}")
            elif action == 'remove':
                df = df[~outlier_mask]
                logger.debug(f"  ⚠ Removed {n_outliers} outlier rows from {col}")
    
    if outlier_stats['columns_affected']:
        logger.info(f"✓ Outlier detection complete ({method}={iqr_multiplier}, action={action})")
        logger.info(f"  Affected columns: {len(outlier_stats['columns_affected'])}")
    else:
        logger.info(f"✓ No outliers detected in numerical columns")
    
    return df, outlier_stats


def handle_missing_values(df: pd.DataFrame, missing_threshold: float = 50.0) -> pd.DataFrame:
    """
    Drop columns with high missingness and impute remaining missing values.
    
    Weather columns (temp, prcp, wspd, etc.) are retained even with high missingness
    and imputed using median, as they provide valuable predictive information.
    
    Args:
        df: DataFrame to process
        missing_threshold: % threshold above which to drop column (default: 50%)
    
    Returns:
        DataFrame with missing values handled
    """
    df = df.copy()
    
    # Calculate missing %
    missing_pct = (df.isna().sum() / len(df)) * 100
    
    # Weather columns to retain regardless of missingness
    weather_cols = ['time', 'temp', 'tmin', 'tmax', 'rhum', 'prcp', 'snwd', 'wspd', 'wpgt', 'pres', 'tsun', 'cldc']
    
    # Drop high-missing columns (per Phase 2 validation), but keep weather columns
    cols_to_drop = missing_pct[missing_pct > missing_threshold].index.tolist()
    cols_to_drop = [col for col in cols_to_drop if col not in weather_cols]
    
    if cols_to_drop:
        logger.info(f"Dropping columns with >{missing_threshold}% missing: {cols_to_drop}")
    else:
        logger.info(f"No non-weather columns with >{missing_threshold}% missing")
    
    df = df.drop(columns=cols_to_drop, errors='ignore')
    
    # Impute remaining missing values
    # Numerical: forward fill then median
    num_cols = df.select_dtypes(include=[np.number]).columns
    for col in num_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())
    
    # Categorical: forward fill then 'Unknown'
    cat_cols = df.select_dtypes(include=['object']).columns
    for col in cat_cols:
        if df[col].isna().any():
            df[col] = df[col].fillna('Unknown')
    
    logger.info(f"✓ Missing values handled. Remaining: {df.isna().sum().sum()}")
    return df


def encode_categorical_features(df: pd.DataFrame, 
                               categorical_cols: List[str] = None,
                               method: str = 'onehot',
                               max_categories: int = 10) -> Tuple[pd.DataFrame, dict]:
    """
    Encode categorical features (one-hot or label encoding).
    
    Args:
        df: DataFrame with categorical columns
        categorical_cols: List of columns to encode (auto-detect if None)
        method: 'onehot' or 'label'
        max_categories: Max unique values for one-hot encoding
    
    Returns:
        (encoded_df, encoders_dict) - encoded dataframe and encoder objects for inverse transform
    """
    df = df.copy()
    encoders = {}
    
    if categorical_cols is None:
        categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    for col in categorical_cols:
        if col not in df.columns:
            continue
        
        n_unique = df[col].nunique()
        
        if method == 'onehot' and n_unique <= max_categories:
            # One-hot encode
            dummies = pd.get_dummies(df[col], prefix=col, drop_first=False)
            df = pd.concat([df, dummies], axis=1)
            df = df.drop(columns=[col])
            encoders[col] = ('onehot', dummies.columns.tolist())
        else:
            # Label encode
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
            encoders[col] = ('label', le)
    
    logger.info(f"✓ Encoded {len(encoders)} categorical features")
    return df, encoders


def scale_numerical_features(df: pd.DataFrame, 
                            numerical_cols: List[str] = None,
                            scaler: StandardScaler = None) -> Tuple[pd.DataFrame, StandardScaler]:
    """
    Standardize numerical features (zero mean, unit variance).
    
    Args:
        df: DataFrame with numerical columns
        numerical_cols: List of columns to scale (auto-detect if None)
        scaler: Pre-fitted scaler (for test data); if None, fit on this data
    
    Returns:
        (scaled_df, scaler_object)
    """
    df = df.copy()
    
    if numerical_cols is None:
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if scaler is None:
        scaler = StandardScaler()
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
        logger.info(f"✓ Fitted scaler on {len(numerical_cols)} numerical features")
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
        logger.info(f"✓ Applied pre-fitted scaler to {len(numerical_cols)} features")
    
    return df, scaler


def select_features_statistical(X: pd.DataFrame, 
                               y: pd.Series,
                               method: str = 'spearman',
                               top_k: int = 10) -> Tuple[List[str], pd.DataFrame]:
    """
    Select top-k features using statistical correlation methods.
    
    Args:
        X: Feature matrix (numerical only)
        y: Target variable
        method: 'spearman', 'pearson', or 'mutual_info'
        top_k: Number of features to select
    
    Returns:
        (selected_feature_names, correlation_scores_df)
    """
    X_numeric = X.select_dtypes(include=[np.number])
    
    if method == 'spearman':
        # Use Spearman (better for non-normal distributions)
        scores = []
        for col in X_numeric.columns:
            corr, p_value = spearmanr(X_numeric[col].fillna(X_numeric[col].median()), y)
            scores.append({'feature': col, 'score': abs(corr), 'p_value': p_value})
        
    elif method == 'mutual_info':
        # Mutual information
        mi_scores = mutual_info_classif(X_numeric.fillna(X_numeric.median()), y, random_state=42)
        scores = [{'feature': col, 'score': score, 'p_value': None} 
                 for col, score in zip(X_numeric.columns, mi_scores)]
    
    else:  # pearson
        scores = []
        for col in X_numeric.columns:
            corr = X_numeric[col].corr(y)
            scores.append({'feature': col, 'score': abs(corr), 'p_value': None})
    
    scores_df = pd.DataFrame(scores).sort_values('score', ascending=False)
    selected_features = scores_df.head(top_k)['feature'].tolist()
    
    logger.info(f"✓ Selected {len(selected_features)} features using {method}:\n{scores_df.head(top_k)}")
    return selected_features, scores_df


def select_features_model_based(X: pd.DataFrame, 
                               y: pd.Series,
                               top_k: int = 10,
                               random_state: int = 42) -> Tuple[List[str], pd.DataFrame]:
    """
    Select top-k features using Random Forest feature importance.
    
    LEAKAGE PREVENTION:
    Post-accident variables (NOT available at prediction time) are automatically excluded:
    - Number_of_Casualties: Directly caused by accident severity
    - Did_Police_Officer_Attend_Scene_of_Accident: Recorded post-incident
    
    Safe pre-accident features to keep:
    - Speed_limit: Known before accident
    - Road_Type: Known infrastructure
    - Light_Conditions: Environmental condition
    - is_adverse_weather: Environmental condition
    - Vehicle_Type: Vehicle attribute
    - Engine_Capacity_.CC.: Vehicle attribute
    - hour_of_day: Time of accident
    
    Args:
        X: Feature matrix (preprocessed)
        y: Target variable
        top_k: Number of features to select
        random_state: Random seed
    
    Returns:
        (selected_feature_names, importance_scores_df)
    """
    # Train quick RF model
    rf = RandomForestClassifier(n_estimators=50, max_depth=10, n_jobs=-1, random_state=random_state)
    rf.fit(X.fillna(0), y)
    
    # Get importance
    importance_scores = pd.DataFrame({
        'feature': X.columns,
        'importance': rf.feature_importances_
    }).sort_values('importance', ascending=False)
    
    # LEAKAGE PREVENTION: Exclude features that leak target information
    # These are post-accident variables NOT available at prediction time:
    leakage_features = [
        'Number_of_Casualties',                        # Directly caused by severity (can't predict before accident)
        'Did_Police_Officer_Attend_Scene_of_Accident'  # Result of severity, recorded post-incident
    ]
    importance_scores = importance_scores[~importance_scores['feature'].isin(leakage_features)]
    
    # SYSTEMATIC CORRELATION FILTER: Remove one feature from each highly correlated pair
    # Calculate correlation matrix for top-k features and remove lower-importance features
    # from pairs with correlation > 0.8
    selected_temp = importance_scores.head(top_k)['feature'].tolist()
    if len(selected_temp) > 1:
        # Get correlation matrix for selected features
        corr_matrix = X[selected_temp].corr()
        
        # Find highly correlated pairs (correlation > 0.8)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feat1, feat2, corr_val))
        
        # Remove lower-importance feature from each pair
        features_to_remove = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            imp1 = importance_scores[importance_scores['feature'] == feat1]['importance'].values[0]
            imp2 = importance_scores[importance_scores['feature'] == feat2]['importance'].values[0]
            
            if imp1 > imp2:
                features_to_remove.add(feat2)
                logger.info(f"  ⚠ Removed {feat2} due to high correlation with {feat1} (r={corr_val:.3f}, kept higher importance)")
            else:
                features_to_remove.add(feat1)
                logger.info(f"  ⚠ Removed {feat1} due to high correlation with {feat2} (r={corr_val:.3f}, kept higher importance)")
        
        if features_to_remove:
            importance_scores = importance_scores[~importance_scores['feature'].isin(features_to_remove)]
    
    selected_features = importance_scores.head(top_k)['feature'].tolist()
    
    logger.info(f"✓ Selected {len(selected_features)} features using Random Forest importance (excluding leakage features {leakage_features}):\n{importance_scores.head(top_k)}")
    return selected_features, importance_scores


def apply_smote(X_train: pd.DataFrame, 
               y_train: pd.Series,
               sampling_strategy: str = 'not majority',
               random_state: int = 42) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Apply SMOTE to balance training data (synthetic oversampling).
    
    Args:
        X_train: Training features
        y_train: Training target
        sampling_strategy: SMOTE strategy (see imbalanced-learn docs)
        random_state: Random seed
    
    Returns:
        (X_train_balanced, y_train_balanced)
    """
    smote = SMOTE(sampling_strategy=sampling_strategy, random_state=random_state)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train)
    
    logger.info(f"✓ SMOTE applied. Original shape: {X_train.shape} → Balanced: {X_train_balanced.shape}")
    logger.info(f"  Class distribution after SMOTE:\n{pd.Series(y_train_balanced).value_counts()}")
    
    return pd.DataFrame(X_train_balanced, columns=X_train.columns), pd.Series(y_train_balanced)
