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
    - select_features_rfecv()               : RFECV-based recursive feature elimination
"""

import logging
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder
from sklearn.feature_selection import mutual_info_classif, RFECV
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.linear_model import LogisticRegression
from scipy.stats import spearmanr
from typing import Tuple, List
from imblearn.over_sampling import SMOTE

logger = logging.getLogger(__name__)


def encode_target_variable(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode target variable (Accident_Severity) to numeric for modeling.
    
    Maps:
        - Slight: 0
        - Serious: 1
        - Fatal: 2
    
    Args:
        df: DataFrame with 'Accident_Severity' column
    
    Returns:
        DataFrame with Accident_Severity encoded as numeric (int64)
    """
    df = df.copy()
    
    if 'Accident_Severity' in df.columns:
        # Check if already numeric
        if not pd.api.types.is_numeric_dtype(df['Accident_Severity']):
            severity_map = {'Slight': 0, 'Serious': 1, 'Fatal': 2}
            df['Accident_Severity'] = df['Accident_Severity'].map(severity_map)
            # Ensure numeric dtype
            df['Accident_Severity'] = df['Accident_Severity'].astype('int64')
            logger.info("✓ Target variable (Accident_Severity) encoded: Slight=0, Serious=1, Fatal=2")
        else:
            logger.info("✓ Target variable (Accident_Severity) already numeric")
    
    return df


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
        DataFrame with lighting features added (Light_Conditions dropped)
    """
    df = df.copy()
    
    if 'Light_Conditions' in df.columns:
        # Assume values like 'Daylight', 'Darkness - lights lit', 'Darkness - no lights', etc.
        dark_keywords = ['Darkness', 'darkness']
        df['is_dark'] = df['Light_Conditions'].fillna('').apply(
            lambda x: 1 if any(kw in str(x) for kw in dark_keywords) else 0
        )
        df = df.drop(columns=['Light_Conditions'])
    else:
        df['is_dark'] = 0
        logger.warning("Light_Conditions column not found, is_dark set to 0")
    
    logger.info("✓ Lighting features created (is_dark)")
    return df


def encode_road_type_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode Road_Type feature to numeric values.
    
    Creates:
        - Road_Type: Encoded road type (0-2)
    
    Args:
        df: DataFrame with 'Road_Type' column
    
    Returns:
        DataFrame with encoded Road_Type
    """
    df = df.copy()
    
    if 'Road_Type' in df.columns:
        road_type_encode = {
            "Single carriageway": 1,
            "Dual carriageway": 2,
            "Roundabout": 0,
            "One way street": 1,
            "Slip road": 2,
            "Unknown": 1
        }
        df['Road_Type'] = df['Road_Type'].map(road_type_encode).fillna(1).astype(int)
        logger.debug("✓ Road_Type encoded")
    
    logger.info("✓ Road_Type features encoded")
    return df


def encode_road_surface_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode road surface condition to is_wet_road flag.
    
    Creates:
        - is_wet_road: 1 if "Not Dry", 0 if "Dry"
    
    Args:
        df: DataFrame with 'Road_Surface_Conditions' column
    
    Returns:
        DataFrame with is_wet_road feature (Road_Surface_Conditions dropped)
    """
    df = df.copy()
    
    if 'Road_Surface_Conditions' in df.columns:
        # Convert non-Dry to 1, Dry to 0
        df['is_wet_road'] = (df['Road_Surface_Conditions'] != 'Dry').astype(int)
        df = df.drop(columns=['Road_Surface_Conditions'])
        logger.debug("✓ is_wet_road created")
    
    logger.info("✓ Road surface features encoded (is_wet_road)")
    return df


def encode_weather_condition_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode weather conditions to is_adverse_weather flag.
    
    Creates:
        - is_adverse_weather: 1 if weather != "Fine no high winds", 0 otherwise
    
    Args:
        df: DataFrame with 'Weather_Conditions' column
    
    Returns:
        DataFrame with is_adverse_weather feature (Weather_Conditions dropped)
    """
    df = df.copy()
    
    if 'Weather_Conditions' in df.columns:
        # Convert non-"Fine no high winds" to 1, "Fine no high winds" to 0
        df['is_adverse_weather'] = (df['Weather_Conditions'] != 'Fine no high winds').astype(int)
        df = df.drop(columns=['Weather_Conditions'])
        logger.debug("✓ is_adverse_weather created")
    
    logger.info("✓ Weather condition features encoded (is_adverse_weather)")
    return df


def encode_urban_rural_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode urban/rural area to is_urban flag.
    
    Creates:
        - is_urban: 1 if 'Urban' or 'Built-up area', 0 otherwise
    
    Args:
        df: DataFrame with 'Urban_or_Rural_Area' column
    
    Returns:
        DataFrame with is_urban feature (Urban_or_Rural_Area dropped)
    """
    df = df.copy()
    
    if 'Urban_or_Rural_Area' in df.columns:
        # Map to binary: Urban/Built-up = 1, Rural = 0
        df['is_urban'] = (df['Urban_or_Rural_Area'].str.contains('Urban|Built', case=False, na=False)).astype(int)
        df = df.drop(columns=['Urban_or_Rural_Area'])
        logger.debug("✓ is_urban created")
    
    logger.info("✓ Urban/rural features encoded (is_urban)")
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


def encode_vehicle_attributes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode vehicle-specific attributes from raw data.
    
    Creates:
        - Vehicle_Type: Encoded vehicle type (0-5)
        - X1st_Point_of_Impact: Encoded impact point (-1 to 4)
        - is_petrol: Binary petrol engine flag
        - is_towing: Binary towing flag
        - engine_size_category: Binned engine capacity (-1 to 3)
    
    Args:
        df: DataFrame with raw vehicle columns
    
    Returns:
        DataFrame with encoded vehicle attributes
    """
    df = df.copy()
    
    # Vehicle Type encoding
    if 'Vehicle_Type' in df.columns:
        vehicle_type_map = {
            "Car": 0, "Taxi/Private hire car": 0,
            "Motorcycle over 500cc": 5, "Motorcycle 125cc and under": 5,
            "Motorcycle 50cc and under": 5, "Motorcycle over 125cc and up to 500cc": 5,
            "Motorcycle - unknown cc": 5, "Electric motorcycle": 5,
            "Van / Goods 3.5 tonnes mgw or under": 2, "Goods over 3.5t. and under 7.5t": 2,
            "Goods 7.5 tonnes mgw and over": 3, "Goods vehicle - unknown weight": 3,
            "Bus or coach (17 or more pass seats)": 1, "Minibus (8 - 16 passenger seats)": 1,
            "Agricultural vehicle": 4, "Other vehicle": 4,
        }
        df['Vehicle_Type'] = df['Vehicle_Type'].map(vehicle_type_map).fillna(4).astype(int)
        logger.debug("✓ Vehicle_Type encoded")
    
    # Point of Impact encoding
    if 'X1st_Point_of_Impact' in df.columns:
        point_impact_map = {
            "Data missing or out of range": -1, "Did not impact": 0,
            "Back": 1, "Front": 2, "Nearside": 3, "Offside": 4
        }
        df['X1st_Point_of_Impact'] = df['X1st_Point_of_Impact'].map(point_impact_map).fillna(0).astype(int)
        logger.debug("✓ X1st_Point_of_Impact encoded")
    
    # Petrol engine flag
    if 'Propulsion_Code' in df.columns:
        df['is_petrol'] = (df['Propulsion_Code'] == 'Petrol').astype(int)
        df = df.drop(columns=['Propulsion_Code'])
        logger.debug("✓ is_petrol created")
    
    # Towing/Articulation flag
    if 'Towing_and_Articulation' in df.columns:
        towing_map = {
            "No tow/articulation": 0, "Caravan": 1, "Other tow": 1,
            "Single trailer": 1, "Double or multiple trailer": 1,
            "Articulated vehicle": 1, "Data missing or out of range": 0,
        }
        df['is_towing'] = df['Towing_and_Articulation'].map(towing_map).fillna(0).astype(int)
        df = df.drop(columns=['Towing_and_Articulation'])
        logger.debug("✓ is_towing created")
    
    # Engine size category (binned)
    if 'Engine_Capacity_.CC.' in df.columns:
        df['engine_size_category'] = pd.cut(
            df['Engine_Capacity_.CC.'],
            bins=[0, 1400, 2000, 4000, float('inf')],
            labels=[0, 1, 2, 3]
        )
        df['engine_size_category'] = pd.to_numeric(df['engine_size_category'], errors='coerce').fillna(-1).astype(int)
        logger.debug("✓ engine_size_category created")
    
    logger.info("✓ Vehicle attributes encoded (Vehicle_Type, X1st_Point_of_Impact, is_petrol, is_towing, engine_size_category)")
    return df


def encode_driver_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode driver-specific attributes.
    
    Creates:
        - age: Driver age from midpoints of age bands
        - is_male: Binary gender encoding (1=Male, 0=Female)
        - driver_area_encoded: Driver home area type (-1 to 2)
    
    Args:
        df: DataFrame with driver columns
    
    Returns:
        DataFrame with encoded driver features
    """
    df = df.copy()
    
    # Driver age from bands
    if 'Age_Band_of_Driver' in df.columns:
        age_midpoint_map = {
            "0 - 5": 2.5, "6 - 10": 8, "11 - 15": 13, "16 - 20": 18,
            "21 - 25": 23, "26 - 35": 30, "36 - 45": 40, "46 - 55": 50,
            "56 - 65": 60, "66 - 75": 70, "Over 75": 80
        }
        df['age'] = df['Age_Band_of_Driver'].map(age_midpoint_map).fillna(30)
        df = df.drop(columns=['Age_Band_of_Driver'])
        logger.debug("✓ age created from Age_Band_of_Driver")
    
    # Driver gender
    if 'Sex_of_Driver' in df.columns:
        gender_map = {"Male": 1, "Female": 0, "Not known": 1, "Data missing or out of range": 1}
        df['is_male'] = df['Sex_of_Driver'].map(gender_map).fillna(1).astype(int)
        df = df.drop(columns=['Sex_of_Driver'])
        logger.debug("✓ is_male created")
    
    # Driver home area
    if 'Driver_Home_Area_Type' in df.columns:
        driver_area_map = {
            "Urban area": 0, "Small town": 1, "Rural": 2, "Data missing or out of range": -1
        }
        df['driver_area_encoded'] = df['Driver_Home_Area_Type'].map(driver_area_map).fillna(-1).astype(int)
        df = df.drop(columns=['Driver_Home_Area_Type'])
        logger.debug("✓ driver_area_encoded created")
    
    logger.info("✓ Driver features encoded (age, is_male, driver_area_encoded)")
    return df


def encode_manoeuvre_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode vehicle manoeuvre types.
    
    Creates:
        - manoeuvre_encoded: Grouped manoeuvre types (0-5, -1 for unknown)
    
    Args:
        df: DataFrame with Vehicle_Manoeuvre column
    
    Returns:
        DataFrame with encoded manoeuvre
    """
    df = df.copy()
    
    if 'Vehicle_Manoeuvre' in df.columns:
        manoeuvre_map = {
            "Going ahead other": "going_ahead", "Going ahead right-hand bend": "going_ahead",
            "Going ahead left-hand bend": "going_ahead", "Turning right": "turning",
            "Turning left": "turning", "U-turn": "turning",
            "Waiting to go - held up": "stationary", "Waiting to turn right": "stationary",
            "Waiting to turn left": "stationary", "Parked": "stationary",
            "Slowing or stopping": "stationary", "Overtaking moving vehicle - offside": "overtaking",
            "Overtaking static vehicle - offside": "overtaking", "Overtaking - nearside": "overtaking",
            "Changing lane to right": "lane_change", "Changing lane to left": "lane_change",
            "Moving off": "other", "Reversing": "other", "Data missing or out of range": "unknown",
        }
        
        manoeuvre_encode = {
            "going_ahead": 0, "stationary": 1, "turning": 2,
            "lane_change": 3, "other": 4, "overtaking": 5, "unknown": -1,
        }
        
        df['manoeuvre_grouped'] = df['Vehicle_Manoeuvre'].map(manoeuvre_map)
        df['manoeuvre_encoded'] = df['manoeuvre_grouped'].map(manoeuvre_encode).fillna(-1).astype(int)
        df = df.drop(columns=['Vehicle_Manoeuvre', 'manoeuvre_grouped'])
        logger.debug("✓ manoeuvre_encoded created")
    
    logger.info("✓ Manoeuvre features encoded (manoeuvre_encoded)")
    return df


def encode_junction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode junction-related attributes.
    
    Creates:
        - Junction_Location: Encoded junction location (0-5, -1 for unknown)
        - junction_control_encoded: Encoded junction control type (0-4, -1 for unknown)
        - junction_detail_encoded: Encoded junction detail (0-6, -1 for unknown)
    
    Args:
        df: DataFrame with junction columns
    
    Returns:
        DataFrame with encoded junction features
    """
    df = df.copy()
    
    # Junction Location
    if 'Junction_Location' in df.columns:
        junction_location_map = {
            "Not at or within 20 metres of junction": 0,
            "Cleared junction or waiting/parked at junction exit": 1,
            "Entering from slip road": 2, "Leaving roundabout": 2,
            "Entering roundabout": 3, "Leaving main road": 3, "Entering main road": 3,
            "Mid Junction - on roundabout or on main road": 4,
            "Approaching junction or waiting/parked at junction approach": 5,
            "Data missing or out of range": -1,
        }
        df['Junction_Location'] = df['Junction_Location'].map(junction_location_map).fillna(-1).astype(int)
        logger.debug("✓ Junction_Location encoded")
    
    # Junction Control
    if 'Junction_Control' in df.columns:
        junction_control_map = {
            "Not at junction or within 20 metres": 0, "Authorised person": 1,
            "Stop sign": 2, "Auto traffic signal": 3, "Give way or uncontrolled": 4,
            "Data missing or out of range": -1,
        }
        df['junction_control_encoded'] = df['Junction_Control'].map(junction_control_map).fillna(-1).astype(int)
        df = df.drop(columns=['Junction_Control'])
        logger.debug("✓ junction_control_encoded created")
    
    # Junction Detail
    if 'Junction_Detail' in df.columns:
        junction_detail_map = {
            "Not at junction or within 20 metres": 0, "Slip road": 1,
            "Mini-roundabout": 2, "Roundabout": 2, "Private drive or entrance": 3,
            "T or staggered junction": 4, "Other junction": 4, "Crossroads": 5,
            "More than 4 arms (not roundabout)": 6, "Data missing or out of range": -1,
        }
        df['junction_detail_encoded'] = df['Junction_Detail'].map(junction_detail_map).fillna(-1).astype(int)
        df = df.drop(columns=['Junction_Detail'])
        logger.debug("✓ junction_detail_encoded created")
    
    logger.info("✓ Junction features encoded (Junction_Location, junction_control_encoded, junction_detail_encoded)")
    return df


def encode_journey_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode journey-related attributes.
    
    Creates:
        - journey_purpose_encoded: Encoded journey purpose (-1 to 2)
    
    Args:
        df: DataFrame with Journey_Purpose_of_Driver column
    
    Returns:
        DataFrame with encoded journey features
    """
    df = df.copy()
    
    if 'Journey_Purpose_of_Driver' in df.columns:
        journey_map = {
            "Commuting to/from work": 0, "Taking pupil to/from school": 0,
            "Pupil riding to/from school": 0, "Other": 1, "Not known": 1,
            "Other/Not known (2005-10)": 1, "Journey as part of work": 2,
            "Data missing or out of range": -1,
        }
        df['journey_purpose_encoded'] = df['Journey_Purpose_of_Driver'].map(journey_map).fillna(-1).astype(int)
        df = df.drop(columns=['Journey_Purpose_of_Driver'])
        logger.debug("✓ journey_purpose_encoded created")
    
    logger.info("✓ Journey features encoded (journey_purpose_encoded)")
    return df


def encode_administrative_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode administrative and location-based aggregated features.
    
    Creates:
        - make: Vehicle make encoded by mean accident severity
        - district_severity_rate: Mean severity by district
        - district_accident_volume: Number of accidents by district
        - highway_severity_rate: Mean severity by highway
        - highway_accident_volume: Number of accidents by highway
        - 1st_Road_Class: Encoded road class
    
    Args:
        df: DataFrame with administrative columns (Accident_Severity already numeric)
    
    Returns:
        DataFrame with encoded administrative features
    """
    df = df.copy()
    
    # Vehicle make encoded by severity (target encoding)
    # Use string severity values for proper target encoding
    if 'make' in df.columns and 'Accident_Severity' in df.columns:
        # If target is numeric, map back to strings for proper encoding
        if pd.api.types.is_numeric_dtype(df['Accident_Severity']):
            severity_reverse_map = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
            df_temp = df.copy()
            df_temp['Accident_Severity_str'] = df_temp['Accident_Severity'].map(severity_reverse_map)
            # Create severity weights for proper encoding
            severity_weights = {'Slight': 1, 'Serious': 2, 'Fatal': 3}
            make_encoding = df_temp.groupby('make')['Accident_Severity_str'].apply(lambda x: x.map(severity_weights).mean())
        else:
            # Use string values directly
            severity_weights = {'Slight': 1, 'Serious': 2, 'Fatal': 3}
            make_encoding = df.groupby('make')['Accident_Severity'].apply(lambda x: x.map(severity_weights).mean())
        
        df['make'] = df['make'].map(make_encoding).fillna(1.0)  # Default to Slight weight
        logger.debug("✓ make encoded by severity")
    
    # District-level features
    if 'Local_Authority_(District)' in df.columns and 'Accident_Severity' in df.columns:
        # Use proper severity weights for target encoding
        if pd.api.types.is_numeric_dtype(df['Accident_Severity']):
            severity_reverse_map = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
            df_temp = df.copy()
            df_temp['Accident_Severity_str'] = df_temp['Accident_Severity'].map(severity_reverse_map)
            severity_weights = {'Slight': 1, 'Serious': 2, 'Fatal': 3}
            district_severity = df_temp.groupby('Local_Authority_(District)')['Accident_Severity_str'].apply(lambda x: x.map(severity_weights).mean())
        else:
            severity_weights = {'Slight': 1, 'Serious': 2, 'Fatal': 3}
            district_severity = df.groupby('Local_Authority_(District)')['Accident_Severity'].apply(lambda x: x.map(severity_weights).mean())
        
        district_volume = df.groupby('Local_Authority_(District)').size()
        df['district_severity_rate'] = df['Local_Authority_(District)'].map(district_severity).fillna(1.0)  # Default to Slight weight
        df['district_accident_volume'] = df['Local_Authority_(District)'].map(district_volume).fillna(df.groupby('Local_Authority_(District)').size().median())
        df = df.drop(columns=['Local_Authority_(District)'])
        logger.debug("✓ District features created")
    
    # Highway-level features
    if 'Local_Authority_(Highway)' in df.columns and 'Accident_Severity' in df.columns:
        # Use proper severity weights for target encoding
        if pd.api.types.is_numeric_dtype(df['Accident_Severity']):
            severity_reverse_map = {0: 'Slight', 1: 'Serious', 2: 'Fatal'}
            df_temp = df.copy()
            df_temp['Accident_Severity_str'] = df_temp['Accident_Severity'].map(severity_reverse_map)
            severity_weights = {'Slight': 1, 'Serious': 2, 'Fatal': 3}
            highway_severity = df_temp.groupby('Local_Authority_(Highway)')['Accident_Severity_str'].apply(lambda x: x.map(severity_weights).mean())
        else:
            severity_weights = {'Slight': 1, 'Serious': 2, 'Fatal': 3}
            highway_severity = df.groupby('Local_Authority_(Highway)')['Accident_Severity'].apply(lambda x: x.map(severity_weights).mean())
        
        highway_volume = df.groupby('Local_Authority_(Highway)').size()
        df['highway_severity_rate'] = df['Local_Authority_(Highway)'].map(highway_severity).fillna(1.0)  # Default to Slight weight
        df['highway_accident_volume'] = df['Local_Authority_(Highway)'].map(highway_volume).fillna(df.groupby('Local_Authority_(Highway)').size().median())
        df = df.drop(columns=['Local_Authority_(Highway)'])
        logger.debug("✓ Highway features created")
    
    # Road class encoding
    if '1st_Road_Class' in df.columns:
        road_class_map = {"M": 0, "A": 1, "D": 2, "B": 3, "C": 4, "Unclassified": 5}
        df['1st_Road_Class'] = df['1st_Road_Class'].map(road_class_map).fillna(5).astype(int)
        logger.debug("✓ 1st_Road_Class encoded")
    
    logger.info("✓ Administrative features encoded (make, district_*, highway_*, 1st_Road_Class)")
    return df


def create_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create interaction features combining multiple attributes.
    
    Creates:
        - speed_x_road_risk: Speed limit × Road Type interaction
        - wet_road_speed: Wet road + high speed (≥60 km/h) interaction
        - young_driver_night: Young driver (≤25) + darkness interaction
    
    Args:
        df: DataFrame with base features
    
    Returns:
        DataFrame with interaction features added
    """
    df = df.copy()
    
    # Speed × Road Risk interaction
    if 'Speed_limit' in df.columns and 'Road_Type' in df.columns:
        df['speed_x_road_risk'] = (df['Speed_limit'] * df['Road_Type']).astype(int)
        logger.debug("✓ speed_x_road_risk created")
    
    # Wet road + high speed interaction
    if 'is_wet_road' in df.columns and 'Speed_limit' in df.columns:
        df['wet_road_speed'] = ((df['is_wet_road'] == 1) & (df['Speed_limit'] >= 60)).astype(int)
        logger.debug("✓ wet_road_speed created")
    
    # Young driver + darkness interaction
    if 'age' in df.columns and 'is_dark' in df.columns:
        df['young_driver_night'] = ((df['age'] <= 25) & (df['is_dark'] == 1)).astype(int)
        logger.debug("✓ young_driver_night created")
    
    logger.info("✓ Interaction features created (speed_x_road_risk, wet_road_speed, young_driver_night)")
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
    
    # Identify numerical columns (excluding identifiers and target variable)
    num_cols = df.select_dtypes(include=[np.number]).columns
    exclude_cols = ['Accident_Index', 'Vehicle_Index', 'Casualty_Index', 'Accident_Severity']
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


def select_features_rfecv(X: pd.DataFrame, 
                         y: pd.Series,
                         min_features_to_select: int = 15,
                         random_state: int = 42,
                         cv_folds: int = 5) -> Tuple[List[str], pd.DataFrame]:
    """
    Select features using RFECV (Recursive Feature Elimination with Cross-Validation).
    
    This method iteratively removes features and ranks them by importance across multiple
    cross-validation folds, automatically determining the optimal number of features.
    
    RATIONALE (from Phase 3 notebook):
    - LogisticRegression used as base estimator for linear relationships
    - StratifiedKFold(5) ensures balanced class distribution across folds
    - Automatically identifies optimal feature set through cross-validation
    - More robust than single-shot methods (better generalizes to unseen data)
    
    LEAKAGE PREVENTION:
    Post-accident variables (NOT available at prediction time) are automatically excluded:
    - Number_of_Casualties: Directly caused by accident severity
    - Did_Police_Officer_Attend_Scene_of_Accident: Recorded post-incident
    
    Args:
        X: Feature matrix (numeric only)
        y: Target variable
        min_features_to_select: Minimum features to retain (default: 15, requirement: ≥7)
        random_state: Random seed for reproducibility
        cv_folds: Number of cross-validation folds (default: 5)
    
    Returns:
        (selected_feature_names, ranking_scores_df) 
        - selected_feature_names: List of feature names selected by RFECV
        - ranking_scores_df: DataFrame with features ranked by selection order
    """
    # Filter to numeric columns only
    X_numeric = X.select_dtypes(include=[np.number]).copy()
    
    if len(X_numeric.columns) == 0:
        raise ValueError("No numeric columns found for RFECV")
    
    logger.info(f"Starting RFECV with {len(X_numeric.columns)} numeric features")
    logger.info(f"  Base estimator: LogisticRegression")
    logger.info(f"  Cross-validation: StratifiedKFold({cv_folds})")
    logger.info(f"  Minimum features to select: {min_features_to_select}\n")
    
    # ─────────────────────────────────────────────────────────────────────────
    # LEAKAGE PREVENTION: Exclude post-accident variables
    # ─────────────────────────────────────────────────────────────────────────
    # These features are recorded AFTER the accident, not available at prediction time:
    leakage_features = [
        'Number_of_Casualties',                        # Directly caused by severity (can't predict before accident)
        'Did_Police_Officer_Attend_Scene_of_Accident'  # Result of severity, recorded post-incident
    ]
    
    cols_to_drop_leakage = [col for col in leakage_features if col in X_numeric.columns]
    if cols_to_drop_leakage:
        logger.info(f"LEAKAGE PREVENTION: Excluding {len(cols_to_drop_leakage)} post-accident variables:")
        for col in cols_to_drop_leakage:
            logger.info(f"    - {col}")
        X_numeric = X_numeric.drop(columns=cols_to_drop_leakage)
        logger.info(f"  Features after leakage prevention: {len(X_numeric.columns)}\n")
    
    # Drop features with >20% missing values
    logger.info("Dropping features with >20% missing values...")
    missing_pct = (X_numeric.isnull().sum() / len(X_numeric)) * 100
    cols_to_drop = missing_pct[missing_pct > 20].index.tolist()
    
    if cols_to_drop:
        logger.info(f"  Dropping {len(cols_to_drop)} features with >20% missing:")
        for col in cols_to_drop:
            logger.info(f"    - {col}: {missing_pct[col]:.1f}% missing")
        X_numeric = X_numeric.drop(columns=cols_to_drop)
    else:
        logger.info("  No features with >20% missing")
    
    logger.info(f"  Remaining features for RFECV: {len(X_numeric.columns)}\n")
    
    # Impute remaining missing values using mean (matches notebook approach)
    # But handle columns with all NaN by forward/backward fill, then drop if still all NaN
    logger.info("Imputing remaining missing values with forward fill, backward fill, then median...")
    
    # Forward fill then backward fill to handle sequences of NaN
    X_numeric = X_numeric.ffill().bfill()
    
    # For remaining NaN, use median (not mean, as mean of all-NaN is NaN)
    for col in X_numeric.columns:
        if X_numeric[col].isnull().any():
            median_val = X_numeric[col].median()
            if pd.notna(median_val):
                X_numeric[col].fillna(median_val, inplace=True)
            else:
                # If entire column is NaN, drop it
                logger.warning(f"  ⚠ Column '{col}' is entirely NaN, dropping from RFECV")
                X_numeric = X_numeric.drop(columns=[col])
    
    # Verify no NaNs remain
    if X_numeric.isnull().any().any():
        # Final fallback: drop any remaining columns with NaN
        cols_with_nan = X_numeric.columns[X_numeric.isnull().any()].tolist()
        logger.warning(f"  ⚠ Dropping columns still containing NaN: {cols_with_nan}")
        X_numeric = X_numeric.dropna(axis=1)
    
    # Initialize RFECV 
    cv = StratifiedKFold(cv_folds)
    # Use class_weight='balanced' to handle imbalanced data without resampling
    # This gives higher weight to minority classes automatically
    estimator = LogisticRegression(class_weight='balanced', max_iter=100)
    
    logger.info(f"Feature matrix shape: {X_numeric.shape}")
    logger.info(f"Target variable shape: {y.shape}")
    logger.info(f"Target variable type: {type(y)}")
    
    # Check class distribution before fitting
    logger.info("\nTarget variable class distribution:")
    class_dist = pd.Series(y).value_counts().sort_index()
    logger.info(f"Unique classes: {class_dist.index.tolist()}")
    for class_val, count in class_dist.items():
        pct = 100 * count / len(y)
        logger.info(f"  Class {class_val}: {count:,} samples ({pct:.2f}%)")
    
    # Verify we have at least 2 classes
    if len(class_dist) < 2:
        logger.error(f"❌ RFECV requires at least 2 classes, but found {len(class_dist)}")
        logger.error(f"   Unique class values: {class_dist.index.tolist()}")
        logger.error(f"   Class counts:\n{class_dist}")
        raise ValueError(f"Cannot fit RFECV with only {len(class_dist)} class(es). Need at least 2 classes.")
    
    logger.info(f"\n✓ Class distribution verified ({len(class_dist)} classes present)")
    logger.info(f"✓ Using class_weight='balanced' to handle imbalance\n")
    
    # Use stratified k-fold with min_samples_leaf to avoid single-class folds
    # For severely imbalanced data, we may need to reduce n_splits
    try:
        rfecv = RFECV(
            estimator=estimator,
            step=3,  # Remove 3 features at a time for efficiency
            cv=cv,
            scoring='accuracy',
            min_features_to_select=min_features_to_select,
            n_jobs=2,
        )
        
        # Fit RFECV
        logger.info("Fitting RFECV on all data...")
        rfecv.fit(X_numeric, y)
        
    except ValueError as e:
        if "only one class" in str(e):
            logger.warning(f"  ⚠ StratifiedKFold(5) created single-class folds with this imbalanced data")
            logger.warning(f"  Retrying with StratifiedKFold(3)...")
            
            # Retry with 3 folds instead of 5
            cv = StratifiedKFold(3)
            rfecv = RFECV(
                estimator=estimator,
                step=3,
                cv=cv,
                scoring='accuracy',
                min_features_to_select=min_features_to_select,
                n_jobs=2,
            )
            rfecv.fit(X_numeric, y)
        else:
            raise
    
    # Get selected features
    selected_features = X_numeric.columns[rfecv.support_].tolist()
    
    # Create ranking dataframe (needed for correlation filter)
    ranking_df = pd.DataFrame({
        'feature': X_numeric.columns,
        'ranking': rfecv.ranking_,
        'support': rfecv.support_,
    }).sort_values('ranking')
    
    # SYSTEMATIC CORRELATION FILTER: Remove one feature from each highly correlated pair
    # This prevents redundant information (e.g., speed_limit and speed_x_road_risk)
    if len(selected_features) > 1:
        # Get correlation matrix for selected features only
        corr_matrix = X_numeric[selected_features].corr()
        
        # Find highly correlated pairs (correlation > 0.8)
        high_corr_pairs = []
        for i in range(len(corr_matrix.columns)):
            for j in range(i+1, len(corr_matrix.columns)):
                corr_val = corr_matrix.iloc[i, j]
                if abs(corr_val) > 0.8:
                    feat1 = corr_matrix.columns[i]
                    feat2 = corr_matrix.columns[j]
                    high_corr_pairs.append((feat1, feat2, corr_val))
        
        # Remove lower-ranked feature from each pair (lower ranking = removed earlier by RFECV = less important)
        features_to_remove = set()
        for feat1, feat2, corr_val in high_corr_pairs:
            rank1 = ranking_df[ranking_df['feature'] == feat1]['ranking'].values[0]
            rank2 = ranking_df[ranking_df['feature'] == feat2]['ranking'].values[0]
            
            if rank1 < rank2:
                features_to_remove.add(feat2)
                logger.info(f"  ⚠ Removing {feat2} due to high correlation with {feat1} (r={corr_val:.3f}, kept lower-ranked)")
            else:
                features_to_remove.add(feat1)
                logger.info(f"  ⚠ Removing {feat1} due to high correlation with {feat2} (r={corr_val:.3f}, kept lower-ranked)")
        
        if features_to_remove:
            selected_features = [f for f in selected_features if f not in features_to_remove]
            logger.info(f"  Features after correlation filter: {len(selected_features)}\n")
    
    logger.info(f"✓ RFECV Complete")
    logger.info(f"  Optimal features selected: {len(selected_features)}")
    logger.info(f"  CV accuracy score (mean): {rfecv.cv_results_['mean_test_score'].max():.4f}\n")
    logger.info(f"Selected features:\n")
    
    for i, feat in enumerate(selected_features, 1):
        logger.info(f"  {i:2d}. {feat}")
    
    return selected_features, ranking_df

