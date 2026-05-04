"""
Unit tests for data preprocessing module.

Test Coverage:
- Data cleaning functionality
- Missing value handling
- Outlier detection and treatment
- Data transformation
- Preprocessing pipeline integration
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys
import pickle
from unittest.mock import patch, MagicMock

# Add paths for imports BEFORE trying to import modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.preprocess import DataCleaner, clean_data, engineer_features, preprocess_features
except ImportError as e:
    pytest.skip(f"preprocess module not found: {e}", allow_module_level=True)


class TestDataCleaner:
    """Test cases for the DataCleaner class."""
    
    @pytest.mark.unit
    def test_cleaner_initialization(self):
        """Test DataCleaner initialization."""
        cleaner = DataCleaner()
        
        assert hasattr(cleaner, 'cleaning_log')
        assert hasattr(cleaner, 'HIGH_MISSING_COLS')
        assert hasattr(cleaner, 'VALID_SPEED_LIMITS')
        assert hasattr(cleaner, 'REQUIRED_COLS')
        assert cleaner.initial_shape is None
        assert cleaner.final_shape is None
    
    @pytest.mark.unit
    def test_drop_high_missingness_columns(self, sample_accident_data):
        """Test dropping high-missingness columns."""
        cleaner = DataCleaner()
        
        # Add high-missingness columns
        data = sample_accident_data.copy()
        data['high_missing_col'] = [None] * len(data)
        data['moderate_missing_col'] = [1] * len(data)
        
        result = cleaner.drop_high_missingness_columns(data)
        
        # Should drop high-missingness columns
        assert 'high_missing_col' not in result.columns
        assert 'moderate_missing_col' in result.columns
    
    @pytest.mark.unit
    def test_remove_invalid_speed_limits(self, sample_accident_data):
        """Test removal of invalid speed limits."""
        cleaner = DataCleaner()
        
        # Add invalid speed limits
        data = sample_accident_data.copy()
        data.loc[0, 'Speed_limit'] = 999  # Invalid
        data.loc[1, 'Speed_limit'] = 15   # Invalid
        
        initial_len = len(data)
        result = cleaner.remove_invalid_Speed_limits(data)
        final_len = len(result)
        
        assert final_len < initial_len
        # All remaining speed limits should be valid
        valid_speeds = set(result['Speed_limit'].dropna())
        assert valid_speeds.issubset(cleaner.VALID_SPEED_LIMITS)
    
    @pytest.mark.unit
    def test_correct_day_of_week(self, sample_accident_data):
        """Test correction of day of week inconsistencies."""
        cleaner = DataCleaner()
        
        # Create inconsistent data
        data = sample_accident_data.copy()
        data.loc[0, 'Date'] = '2023-01-02'  # Monday
        data.loc[0, 'Day_of_Week'] = 'Tuesday'  # Inconsistent
        
        result = cleaner.correct_day_of_week(data)
        
        # Day_of_Week column should be removed to prevent string conversion issues
        assert 'Day_of_Week' not in result.columns
    
    @pytest.mark.unit
    def test_handle_invalid_age_bands(self, sample_accident_data):
        """Test handling of invalid age bands."""
        cleaner = DataCleaner()
        
        # Add invalid age bands to vehicle data
        data = sample_accident_data.copy()
        data['Age_Band_of_Driver'] = data.get('Age_Band_of_Driver', ['21-25'] * len(data))
        data.loc[0, 'Age_Band_of_Driver'] = 'Invalid'
        data.loc[1, 'Age_Band_of_Driver'] = '999'
        
        result = cleaner.handle_invalid_age_bands(data)
        
        # Invalid values should be marked as NaN
        assert pd.isna(result.loc[0, 'Age_Band_of_Driver'])
        assert pd.isna(result.loc[1, 'Age_Band_of_Driver'])
    
    @pytest.mark.unit
    def test_handle_sex_of_driver_codes(self, sample_accident_data):
        """Test handling of invalid sex of driver codes."""
        cleaner = DataCleaner()
        
        # Add invalid sex codes
        data = sample_accident_data.copy()
        data['Sex_of_Driver'] = data.get('Sex_of_Driver', ['Male'] * len(data))
        data.loc[0, 'Sex_of_Driver'] = 'X'
        data.loc[1, 'Sex_of_Driver'] = '999'
        
        result = cleaner.handle_sex_of_driver_codes(data)
        
        # Invalid codes should be standardized to 'Not known'
        assert result.loc[0, 'Sex_of_Driver'] == 'Not known'
        assert result.loc[1, 'Sex_of_Driver'] == 'Not known'
    
    @pytest.mark.unit
    def test_remove_rows_with_required_nulls(self, sample_accident_data):
        """Test removal of rows with null values in required columns."""
        cleaner = DataCleaner()
        
        # Add null values to required columns
        data = sample_accident_data.copy()
        data.loc[0, 'Latitude'] = None
        data.loc[1, 'Longitude'] = None
        
        initial_len = len(data)
        result = cleaner.remove_rows_with_required_nulls(data)
        final_len = len(result)
        
        assert final_len < initial_len
        # No nulls should remain in required columns
        for col in cleaner.REQUIRED_COLS:
            if col in result.columns:
                assert result[col].notna().all()
    
    @pytest.mark.unit
    def test_check_and_remove_duplicates(self, sample_accident_data):
        """Test duplicate detection and removal."""
        cleaner = DataCleaner()
        
        # Add duplicate rows
        data = pd.concat([sample_accident_data, sample_accident_data.iloc[[0]]], ignore_index=True)
        
        initial_len = len(data)
        result = cleaner.check_and_remove_duplicates(data)
        final_len = len(result)
        
        assert final_len < initial_len
    
    @pytest.mark.unit
    def test_clean_pipeline_complete(self, sample_accident_data):
        """Test complete cleaning pipeline."""
        cleaner = DataCleaner()
        
        # Add various data quality issues
        data = sample_accident_data.copy()
        data.loc[0, 'Speed_limit'] = 999  # Invalid speed
        data.loc[1, 'Date'] = '2023-01-02'
        data.loc[1, 'Day_of_Week'] = 'Tuesday'  # Inconsistent
        data.loc[2, 'Latitude'] = None  # Required null
        
        result = cleaner.clean(data)
        
        # Should track initial and final shapes
        assert cleaner.initial_shape is not None
        assert cleaner.final_shape is not None
        assert cleaner.initial_shape[0] >= cleaner.final_shape[0]
        
        # Should have cleaning log
        assert len(cleaner.cleaning_log) > 0


class TestCleanData:
    """Test cases for the clean_data function."""
    
    @pytest.mark.unit
    def test_clean_data_function(self, sample_accident_data):
        """Test clean_data standalone function."""
        # Add some data quality issues
        data = sample_accident_data.copy()
        data.loc[0, 'Speed_limit'] = 999
        
        result = clean_data(data)
        
        # Should return cleaned DataFrame
        assert isinstance(result, pd.DataFrame)
        assert len(result) <= len(data)
        
        # Should not have invalid speed limits
        if 'Speed_limit' in result.columns:
            valid_speeds = set(result['Speed_limit'].dropna())
            assert valid_speeds.issubset({20, 30, 40, 50, 60, 70})


class TestEngineerFeatures:
    """Test cases for feature engineering functions."""
    
    @pytest.mark.unit
    def test_engineer_features_basic(self, sample_accident_data):
        """Test basic feature engineering."""
        result = engineer_features(sample_accident_data)
        
        # Should return DataFrame with more columns
        assert isinstance(result, pd.DataFrame)
        assert result.shape[1] >= sample_accident_data.shape[1]
        
        # Should have engineered features
        expected_features = [
            'hour_of_day', 'day_of_week', 'is_weekend', 'month', 'season',
            'is_dark', 'is_adverse_weather', 'road_risk_score'
        ]
        
        for feature in expected_features:
            if feature in result.columns:
                assert result[feature].notna().sum() > 0
    
    @pytest.mark.unit
    def test_temporal_features_creation(self, sample_accident_data):
        """Test temporal feature creation."""
        from src.features.build_features import create_temporal_features
        
        result = create_temporal_features(sample_accident_data)
        
        # Check temporal features
        assert 'hour_of_day' in result.columns
        assert 'day_of_week' in result.columns
        assert 'is_weekend' in result.columns
        assert 'month' in result.columns
        assert 'season' in result.columns
        
        # Check value ranges
        assert result['hour_of_day'].between(0, 23).all()
        assert result['day_of_week'].between(0, 6).all()
        assert result['is_weekend'].isin([0, 1]).all()
        assert result['month'].between(1, 12).all()
        assert result['season'].isin(['Spring', 'Summer', 'Autumn', 'Winter']).all()
    
    @pytest.mark.unit
    def test_lighting_features_creation(self, sample_accident_data):
        """Test lighting feature creation."""
        from src.features.build_features import create_lighting_features
        
        result = create_lighting_features(sample_accident_data)
        
        assert 'is_dark' in result.columns
        assert result['is_dark'].isin([0, 1]).all()
        assert 'Light_Conditions' not in result.columns  # Should be dropped
    
    @pytest.mark.unit
    def test_road_risk_features_creation(self, sample_accident_data):
        """Test road risk feature creation."""
        from src.features.build_features import create_road_risk_features
        
        result = create_road_risk_features(sample_accident_data)
        
        if 'Speed_limit' in sample_accident_data.columns:
            assert 'road_risk_score' in result.columns
            assert result['road_risk_score'].min() >= 0
    
    @pytest.mark.unit
    def test_vehicle_features_creation(self, sample_accident_data):
        """Test vehicle feature creation."""
        from src.features.build_features import create_vehicle_features
        
        result = create_vehicle_features(sample_accident_data)
        
        # Should not crash even without vehicle-specific columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_accident_data)


class TestPreprocessFeatures:
    """Test cases for feature preprocessing functions."""
    
    @pytest.mark.unit
    def test_preprocess_features_complete(self, sample_accident_data):
        """Test complete feature preprocessing pipeline."""
        # Engineer features first
        engineered_data = engineer_features(sample_accident_data)
        
        result, metadata = preprocess_features(engineered_data)
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(metadata, dict)
        assert len(result) > 0
        assert 'outlier_stats' in metadata
        assert 'encoders' in metadata
    
    @pytest.mark.unit
    def test_missing_value_handling(self, sample_accident_data):
        """Test missing value handling."""
        from src.features.build_features import handle_missing_values
        
        # Add missing values
        data = sample_accident_data.copy()
        data.loc[0:5, 'Speed_limit'] = None
        
        result = handle_missing_values(data, missing_threshold=50.0)
        
        # Should handle missing values appropriately
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    @pytest.mark.unit
    def test_outlier_detection(self, sample_accident_data):
        """Test outlier detection and handling."""
        from src.features.build_features import detect_and_handle_outliers
        
        # Add outliers
        data = sample_accident_data.copy()
        if 'Speed_limit' in data.columns:
            data.loc[0, 'Speed_limit'] = 999  # Outlier
        
        result, stats = detect_and_handle_outliers(
            data, method='iqr', iqr_multiplier=1.5, action='clip'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(stats, dict)
        assert len(result) == len(data)
    
    @pytest.mark.unit
    def test_categorical_encoding(self, sample_accident_data):
        """Test categorical feature encoding."""
        from src.features.build_features import encode_categorical_features
        
        # Identify categorical columns
        categorical_cols = sample_accident_data.select_dtypes(include=['object']).columns.tolist()
        categorical_cols = [c for c in categorical_cols if c not in ['Accident_Severity', 'Accident_Index', 'Date']]
        
        if categorical_cols:
            result, encoders = encode_categorical_features(
                sample_accident_data, 
                categorical_cols=categorical_cols,
                method='label',
                max_categories=20
            )
            
            assert isinstance(result, pd.DataFrame)
            assert isinstance(encoders, dict)


class TestPreprocessingIntegration:
    """Integration tests for the complete preprocessing pipeline."""
    
    @pytest.mark.integration
    def test_complete_preprocessing_pipeline(self, sample_accident_data):
        """Test the complete preprocessing pipeline from raw to ready."""
        # Step 1: Clean data
        cleaned_data = clean_data(sample_accident_data)
        
        # Step 2: Engineer features
        engineered_data = engineer_features(cleaned_data)
        
        # Step 3: Preprocess features
        preprocessed_data, metadata = preprocess_features(engineered_data)
        
        # Verify pipeline results
        assert isinstance(preprocessed_data, pd.DataFrame)
        assert len(preprocessed_data) > 0
        assert isinstance(metadata, dict)
        
        # Should have target variable
        assert 'Accident_Severity' in preprocessed_data.columns
        
        # Should have mostly numeric features after preprocessing
        numeric_cols = preprocessed_data.select_dtypes(include=[np.number]).columns
        assert len(numeric_cols) > 0
    
    @pytest.mark.integration
    def test_preprocessing_with_missing_data(self):
        """Test preprocessing with significant missing data."""
        # Create data with many missing values
        n_rows = 100
        problematic_data = pd.DataFrame({
            'Accident_Severity': np.random.choice(['Slight', 'Serious', 'Fatal'], n_rows),
            'Speed_limit': [None] * n_rows,  # All missing
            'Latitude': np.random.uniform(50, 55, n_rows),
            'Longitude': np.random.uniform(-3, 1, n_rows),
            'Date': pd.date_range('2023-01-01', periods=n_rows),
        })
        
        # Should handle missing data gracefully
        cleaned_data = clean_data(problematic_data)
        assert isinstance(cleaned_data, pd.DataFrame)
        
        # May drop rows with missing required fields
        if len(cleaned_data) > 0:
            engineered_data = engineer_features(cleaned_data)
            assert isinstance(engineered_data, pd.DataFrame)


class TestPreprocessingPerformance:
    """Performance tests for preprocessing operations."""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_preprocessing_performance(self):
        """Test preprocessing performance with larger dataset."""
        # Create larger test dataset
        n_rows = 5000
        large_data = pd.DataFrame({
            'Accident_Severity': np.random.choice(['Slight', 'Serious', 'Fatal'], n_rows),
            'Speed_limit': np.random.choice([20, 30, 40, 50, 60, 70], n_rows),
            'Latitude': np.random.uniform(50, 55, n_rows),
            'Longitude': np.random.uniform(-3, 1, n_rows),
            'Date': pd.date_range('2020-01-01', periods=n_rows, freq='h'),
            'Road_Type': np.random.choice(['Single carriageway', 'Dual carriageway'], n_rows),
        })
        
        import time
        
        # Test cleaning performance
        start_time = time.time()
        cleaned_data = clean_data(large_data)
        cleaning_time = time.time() - start_time
        
        # Test feature engineering performance
        start_time = time.time()
        engineered_data = engineer_features(cleaned_data)
        engineering_time = time.time() - start_time
        
        # Performance assertions
        assert cleaning_time < 30.0, f"Cleaning too slow: {cleaning_time:.2f}s"
        assert engineering_time < 30.0, f"Feature engineering too slow: {engineering_time:.2f}s"
        
        assert len(engineered_data) > 0


class TestPreprocessingEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_preprocessing_empty_dataframe(self):
        """Test preprocessing with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        # Should handle empty data gracefully
        with pytest.raises((ValueError, IndexError)):
            clean_data(empty_df)
    
    @pytest.mark.unit
    def test_preprocessing_single_record(self):
        """Test preprocessing with single record."""
        single_record = pd.DataFrame({
            'Accident_Severity': ['Slight'],
            'Speed_limit': [30],
            'Latitude': [51.5],
            'Longitude': [-0.1],
            'Date': ['2023-01-01'],
        })
        
        cleaned_data = clean_data(single_record)
        assert len(cleaned_data) == 1
        
        engineered_data = engineer_features(cleaned_data)
        assert len(engineered_data) == 1
    
    @pytest.mark.unit
    def test_preprocessing_all_same_values(self):
        """Test preprocessing with columns that have all same values."""
        n_rows = 100
        uniform_data = pd.DataFrame({
            'Accident_Index': [f'2023{str(i).zfill(6)}' for i in range(n_rows)],
            'Accident_Severity': ['Slight'] * n_rows,
            'Speed_limit': [30] * n_rows,
            'Latitude': [51.5] * n_rows,
            'Longitude': [-0.1] * n_rows,
            'Date': ['2023-01-01'] * n_rows,
            'Day_of_Week': ['Sunday'] * n_rows,  # Jan 1, 2023 is Sunday
        })
        
        cleaned_data = clean_data(uniform_data)
        engineered_data = engineer_features(cleaned_data)
        
        assert len(engineered_data) == n_rows
        assert engineered_data['Speed_limit'].nunique() == 1
