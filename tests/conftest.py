"""
Pytest configuration and shared fixtures for the Accident Severity Prediction test suite.

This module provides common test data, fixtures, and utilities used across
all test modules to ensure consistent testing and reduce code duplication.
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
import tempfile
import shutil
from typing import Dict, Tuple
import sys
from datetime import datetime

# Add paths for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))  # Add src directory first
sys.path.insert(0, str(PROJECT_ROOT))  # Add root directory for models module


@pytest.fixture
def sample_accident_data() -> pd.DataFrame:
    """
    Create sample accident data for testing.
    
    Returns:
        pd.DataFrame: Sample dataset with typical accident data structure
    """
    np.random.seed(42)
    n_samples = 100
    
    dates = pd.date_range('2023-01-01', periods=n_samples, freq='D')
    data = {
        'Accident_Index': [f'2023{str(i).zfill(6)}' for i in range(n_samples)],
        'Date': dates,
        'Day_of_Week': dates.day_name().tolist(),
        'Accident_Severity': np.random.choice(['Slight', 'Serious', 'Fatal'], n_samples, p=[0.7, 0.25, 0.05]),
        'Latitude': np.random.uniform(50.0, 55.0, n_samples),
        'Longitude': np.random.uniform(-3.0, 1.0, n_samples),
        'Speed_limit': np.random.choice([20, 30, 40, 50, 60, 70], n_samples),
        'Road_Type': np.random.choice(['Single carriageway', 'Dual carriageway', 'Roundabout'], n_samples),
        'Road_Surface_Conditions': np.random.choice(['Dry', 'Wet/Damp', 'Snow'], n_samples),
        'Weather_Conditions': np.random.choice(['Fine no high winds', 'Raining', 'Snowing'], n_samples),
        'Light_Conditions': np.random.choice(['Daylight', 'Darkness - lights lit', 'Darkness - no lights'], n_samples),
        'Urban_or_Rural_Area': np.random.choice(['Urban', 'Rural'], n_samples),
        'Number_of_Vehicles': np.random.randint(1, 5, n_samples),
        'Number_of_Casualties': np.random.randint(1, 8, n_samples),
        'Local_Authority_(District)': np.random.choice([1, 2, 3, 4, 5], n_samples),
        'Local_Authority_(Highway)': np.random.choice([1, 2, 3, 4, 5], n_samples),
        '1st_Road_Class': np.random.choice(['A', 'B', 'C', 'M', 'Unclassified'], n_samples),
    }
    
    # Add some missing values for testing
    for col in ['Speed_limit', 'Road_Type', 'Weather_Conditions']:
        missing_idx = np.random.choice(n_samples, size=int(0.05 * n_samples), replace=False)
        data[col] = [None if i in missing_idx else data[col][i] for i in range(n_samples)]
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_vehicle_data() -> pd.DataFrame:
    """
    Create sample vehicle data for testing.
    
    Returns:
        pd.DataFrame: Sample dataset with vehicle-level information
    """
    np.random.seed(42)
    n_samples = 50
    
    data = {
        'Accident_Index': [f'2023{str(i).zfill(6)}' for i in range(n_samples)],
        'Vehicle_Type': np.random.choice(['Car', 'Motorcycle', 'Van', 'Bus'], n_samples),
        'Age_of_Driver': np.random.randint(18, 80, n_samples),
        'Age_Band_of_Driver': np.random.choice(['21-25', '26-35', '36-45', '46-55'], n_samples),
        'Sex_of_Driver': np.random.choice(['Male', 'Female'], n_samples),
        'Engine_Capacity_.CC.': np.random.choice([1000, 1500, 2000, 2500], n_samples),
        'Propulsion_Code': np.random.choice(['Petrol', 'Diesel', 'Electric'], n_samples),
        'Towing_and_Articulation': np.random.choice(['No tow/articulation', 'Caravan'], n_samples),
        'Vehicle_Manoeuvre': np.random.choice(['Going ahead other', 'Turning right', 'Turning left'], n_samples),
        'X1st_Point_of_Impact': np.random.choice(['Front', 'Back', 'Nearside', 'Offside'], n_samples),
        'Journey_Purpose_of_Driver': np.random.choice(['Commuting to/from work', 'Other'], n_samples),
        'Driver_Home_Area_Type': np.random.choice(['Urban area', 'Small town', 'Rural'], n_samples),
    }
    
    # Add some missing values
    for col in ['Age_of_Driver', 'Engine_Capacity_.CC.']:
        missing_idx = np.random.choice(n_samples, size=int(0.1 * n_samples), replace=False)
        data[col] = [None if i in missing_idx else data[col][i] for i in range(n_samples)]
    
    return pd.DataFrame(data)


@pytest.fixture
def sample_weather_data() -> pd.DataFrame:
    """
    Create sample weather data for testing.
    
    Returns:
        pd.DataFrame: Sample dataset with weather information
    """
    np.random.seed(42)
    n_samples = 100
    
    data = {
        'Date': pd.date_range('2023-01-01', periods=n_samples, freq='D'),
        'temp': np.random.normal(15, 8, n_samples),  # Temperature in Celsius
        'prcp': np.random.exponential(2, n_samples),  # Precipitation in mm
        'snwd': np.random.exponential(0.5, n_samples),  # Snow depth in mm
        'cldc': np.random.randint(0, 9, n_samples),  # Cloud cover in oktas (0-8)
    }
    
    return pd.DataFrame(data)


@pytest.fixture
def merged_sample_data(sample_accident_data, sample_vehicle_data) -> pd.DataFrame:
    """
    Create merged sample data combining accidents and vehicles.
    
    Returns:
        pd.DataFrame: Merged dataset for comprehensive testing
    """
    # Merge accident and vehicle data on Accident_Index
    merged = pd.merge(
        sample_accident_data,
        sample_vehicle_data,
        on='Accident_Index',
        how='inner'
    )
    
    return merged


@pytest.fixture
def temp_data_dir():
    """
    Create a temporary directory for test data files.
    
    Returns:
        Path: Temporary directory path
    """
    temp_dir = tempfile.mkdtemp()
    yield Path(temp_dir)
    shutil.rmtree(temp_dir)


@pytest.fixture
def sample_config() -> Dict:
    """
    Provide sample configuration for testing.
    
    Returns:
        Dict: Configuration parameters
    """
    return {
        'test_size': 0.2,
        'random_state': 42,
        'missing_threshold': 50.0,
        'feature_selection_method': 'model_based',
        'rfecv_min_features': 15,
        'rfecv_cv_folds': 5,
        'compare_feature_selection': False,
        'apply_smote': True,
        'smote_sampling_strategy': 'not majority',
    }


@pytest.fixture
def mock_model():
    """
    Create a mock model for testing model-related functionality.
    
    Returns:
        object: Mock model with predict and predict_proba methods
    """
    from unittest.mock import MagicMock
    
    # Create a basic MagicMock
    mock = MagicMock()
    mock.classes_ = np.array(['Fatal', 'Serious', 'Slight'])
    
    # Set default behaviors using side_effect
    def default_predict(X):
        if isinstance(X, pd.DataFrame):
            n_samples = len(X)
        else:
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        return np.random.choice(mock.classes_, n_samples)
    
    def default_predict_proba(X):
        if isinstance(X, pd.DataFrame):
            n_samples = len(X)
        else:
            n_samples = X.shape[0] if hasattr(X, 'shape') else len(X)
        # Random probabilities that sum to 1
        probs = np.random.dirichlet(np.ones(3), size=n_samples)
        return probs
    
    # Set up methods with side_effect for default behavior
    mock.predict.side_effect = default_predict
    mock.predict_proba.side_effect = default_predict_proba
    mock.fit.return_value = mock
    
    return mock


@pytest.fixture(scope="session")
def test_data_summary():
    """
    Provide summary information about test data for debugging.
    
    Returns:
        Dict: Summary of test data characteristics
    """
    return {
        'accident_samples': 100,
        'vehicle_samples': 50,
        'weather_samples': 100,
        'severity_classes': ['Fatal', 'Serious', 'Slight'],
        'road_types': ['Single carriageway', 'Dual carriageway', 'Roundabout'],
        'weather_conditions': ['Fine no high winds', 'Raining', 'Snowing'],
    }


# Custom pytest markers for categorizing tests
pytest_plugins = []

def pytest_configure(config):
    """
    Configure custom markers for test categorization.
    """
    config.addinivalue_line(
        "markers", "unit: Mark test as a unit test"
    )
    config.addinivalue_line(
        "markers", "integration: Mark test as an integration test"
    )
    config.addinivalue_line(
        "markers", "slow: Mark test as slow running"
    )
    config.addinivalue_line(
        "markers", "data_quality: Mark test as data quality validation"
    )
    config.addinivalue_line(
        "markers", "model_validation: Mark test as model validation"
    )


# Helper functions for test utilities
def create_test_dataframe_with_nulls(rows: int = 50, null_percentage: float = 0.2) -> pd.DataFrame:
    """
    Create a test DataFrame with controlled null values.
    
    Args:
        rows: Number of rows to generate
        null_percentage: Percentage of null values to include
    
    Returns:
        pd.DataFrame: Test DataFrame with null values
    """
    np.random.seed(42)
    
    data = {
        'numeric_col': np.random.randn(rows),
        'categorical_col': np.random.choice(['A', 'B', 'C'], rows),
        'datetime_col': pd.date_range('2023-01-01', periods=rows, freq='D'),
    }
    
    df = pd.DataFrame(data)
    
    # Add null values
    for col in df.columns:
        null_idx = np.random.choice(
            rows, 
            size=int(null_percentage * rows), 
            replace=False
        )
        df.loc[null_idx, col] = None
    
    return df


def assert_dataframe_equal(df1: pd.DataFrame, df2: pd.DataFrame, check_dtype: bool = True):
    """
    Assert two DataFrames are equal with better error messages.
    
    Args:
        df1: First DataFrame
        df2: Second DataFrame
        check_dtype: Whether to check dtypes
    """
    try:
        pd.testing.assert_frame_equal(df1, df2, check_dtype=check_dtype)
    except AssertionError as e:
        # Provide more detailed error information
        print(f"DataFrame shapes: {df1.shape} vs {df2.shape}")
        print(f"DataFrame columns: {list(df1.columns)} vs {list(df2.columns)}")
        if df1.shape == df2.shape:
            print(f"Columns with differences: {(df1 != df2).any().sum()}")
        raise e
