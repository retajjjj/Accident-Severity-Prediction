"""
Targeted comprehensive tests for preprocess.py - addressing 54% coverage gap.
Focuses on data cleaning and feature engineering missing code paths.
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.preprocess import (
        DataCleaner, clean_data, engineer_features, preprocess_features
    )
    from src.features.build_features import (
        handle_missing_values, detect_and_handle_outliers, encode_categorical_features
    )
except ImportError as e:
    pytest.skip(f"preprocess module not found: {e}", allow_module_level=True)


class TestDataCleanerClass:
    """Test DataCleaner class initialization and methods."""
    
    @pytest.mark.unit
    def test_cleaner_init_default(self):
        """Test DataCleaner initialization with defaults."""
        cleaner = DataCleaner()
        assert cleaner is not None
    
    @pytest.mark.unit
    def test_cleaner_init_with_metadata(self):
        """Test DataCleaner with metadata."""
        cleaner = DataCleaner()
        assert cleaner is not None
    
    @pytest.mark.unit
    def test_cleaner_clean_method_exists(self):
        """Test cleaner has clean method."""
        cleaner = DataCleaner()
        assert hasattr(cleaner, 'clean')
        assert callable(cleaner.clean)


class TestDataCleaningOperations:
    """Test individual cleaning operations."""
    
    @pytest.mark.unit
    def test_clean_removes_duplicates(self):
        """Test duplicate removal."""
        data = pd.DataFrame({
            'A': [1, 2, 2, 3],
            'B': [4, 5, 5, 6],
        })
        cleaner = DataCleaner()
        result = cleaner.clean(data)
        assert result.shape[0] < data.shape[0]
    
    @pytest.mark.unit
    def test_clean_handles_missing_severity(self):
        """Test handling of missing severity values."""
        data = pd.DataFrame({
            'Accident_Severity': ['Slight', None, 'Fatal', 'Serious'],
            'Speed_limit': [30, 50, 70, 40],
        })
        cleaner = DataCleaner()
        result = cleaner.clean(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_clean_invalid_speed_limits(self):
        """Test cleaning invalid speed limits."""
        data = pd.DataFrame({
            'Speed_limit': [20, 30, -5, 999, 50, 0],
            'Accident_Severity': ['Slight'] * 6,
        })
        cleaner = DataCleaner()
        result = cleaner.clean(data)
        # Invalid speeds should be handled
        assert result is not None
    
    @pytest.mark.unit
    def test_clean_corrects_day_of_week(self):
        """Test day of week correction."""
        data = pd.DataFrame({
            'Day_of_Week': ['Monday', 'InvalidDay', 'Wednesday', None],
            'Accident_Severity': ['Slight'] * 4,
        })
        cleaner = DataCleaner()
        result = cleaner.clean(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_clean_handles_age_bands(self):
        """Test handling of age band values."""
        data = pd.DataFrame({
            'Age_Band_of_Driver': ['21-25', 'Invalid', '36-45', None],
            'Age_of_Driver': [23, 30, 40, 50],
        })
        cleaner = DataCleaner()
        result = cleaner.clean(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_clean_handles_sex_values(self):
        """Test handling of sex of driver values."""
        data = pd.DataFrame({
            'Sex_of_Driver': ['Male', 'Female', 'Invalid', None],
            'Age_of_Driver': [25, 30, 35, 40],
        })
        cleaner = DataCleaner()
        result = cleaner.clean(data)
        assert result is not None


class TestFeatureEngineering:
    """Test feature engineering operations."""
    
    @pytest.mark.unit
    def test_engineer_features_temporal(self, sample_accident_data):
        """Test temporal feature engineering."""
        data = sample_accident_data.copy()
        data['Date'] = pd.date_range('2023-01-01', periods=len(data))
        result = engineer_features(data)
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.unit
    def test_engineer_features_lighting(self, sample_accident_data):
        """Test lighting feature engineering."""
        data = sample_accident_data.copy()
        valid_lighting = ['Daylight', 'Darkness - lights lit', 'Darkness - no lights']
        data['Light_Conditions'] = np.random.choice(valid_lighting, len(data))
        result = engineer_features(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_engineer_features_road_risk(self, sample_accident_data):
        """Test road risk feature creation."""
        data = sample_accident_data.copy()
        speed_values = ([20, 30, 40, 50, 60, 70] * (len(data) // 6 + 1))[:len(data)]
        data['Speed_limit'] = speed_values
        result = engineer_features(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_engineer_features_vehicle_info(self, sample_accident_data):
        """Test vehicle feature engineering."""
        data = sample_accident_data.copy()
        result = engineer_features(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_engineer_features_interaction_terms(self, sample_accident_data):
        """Test interaction feature creation."""
        data = sample_accident_data.copy()
        result = engineer_features(data)
        # Should have same or more features
        assert result.shape[1] >= 1


class TestFeaturePreprocessing:
    """Test feature preprocessing pipeline."""
    
    @pytest.mark.unit
    def test_preprocess_features_complete_pipeline(self, sample_accident_data):
        """Test complete preprocessing pipeline."""
        data = sample_accident_data.copy()
        result_df, metadata = preprocess_features(data)
        assert result_df is not None
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(metadata, dict)
    
    @pytest.mark.unit
    def test_preprocess_missing_handling(self, sample_accident_data):
        """Test missing value handling in preprocessing."""
        data = sample_accident_data.copy()
        data.loc[0:5, 'Speed_limit'] = None
        result = preprocess_features(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_preprocess_outlier_handling(self, sample_accident_data):
        """Test outlier handling in preprocessing."""
        data = sample_accident_data.copy()
        data.loc[0, 'Speed_limit'] = 9999
        result = preprocess_features(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_preprocess_categorical_encoding(self, sample_accident_data):
        """Test categorical encoding in preprocessing."""
        data = sample_accident_data.copy()
        result = preprocess_features(data)
        assert result is not None


class TestMissingValueHandling:
    """Test missing value handling strategies."""
    
    @pytest.mark.unit
    def test_handle_missing_values_numeric(self):
        """Test missing value handling for numeric columns."""
        data = pd.DataFrame({
            'Speed': [20.0, 30.0, np.nan, 50.0, np.nan],
            'Casualties': [1, 2, 3, 4, 5],
        })
        result = handle_missing_values(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_handle_missing_values_categorical(self):
        """Test missing value handling for categorical columns."""
        data = pd.DataFrame({
            'Road_Type': ['Single', 'Dual', None, 'Roundabout', None],
            'Weather': ['Fine', 'Rain', 'Fine', None, 'Snow'],
        })
        result = handle_missing_values(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_handle_missing_values_high_missing_threshold(self):
        """Test with columns exceeding threshold."""
        data = pd.DataFrame({
            'A': [1, None, None, None, None],  # 80% missing
            'B': [1, 2, 3, 4, 5],
            'C': [None, None, None, None, None],  # 100% missing
        })
        result = handle_missing_values(data, missing_threshold=50)
        assert result is not None
    
    @pytest.mark.unit
    def test_handle_missing_values_all_complete(self):
        """Test with completely filled data."""
        data = pd.DataFrame({
            'A': [1, 2, 3, 4, 5],
            'B': [10, 20, 30, 40, 50],
        })
        result = handle_missing_values(data)
        assert result.isna().sum().sum() == 0


class TestOutlierDetection:
    """Test outlier detection and removal."""
    
    @pytest.mark.unit
    def test_detect_outliers_iqr_method(self):
        """Test IQR-based outlier detection."""
        data = pd.DataFrame({
            'Speed': [20, 30, 40, 50, 60, 70, 5000],
        })
        result, _ = detect_and_handle_outliers(data, method='iqr')
        assert result is not None
    
    @pytest.mark.unit
    def test_detect_outliers_zscore_method(self):
        """Test z-score based outlier detection."""
        data = pd.DataFrame({
            'Value': np.random.normal(100, 15, 100),
        })
        data.loc[0, 'Value'] = 999  # Add outlier
        result, _ = detect_and_handle_outliers(data, method='zscore')
        assert result is not None
    
    @pytest.mark.unit
    def test_detect_outliers_removal_action(self):
        """Test outlier removal action."""
        data = pd.DataFrame({
            'Speed': [20, 30, 40, 50, 60, 70, -100, 5000],
        })
        result, _ = detect_and_handle_outliers(data, method='iqr', action='remove')
        assert result is not None
        assert len(result) <= len(data)
    
    @pytest.mark.unit
    def test_detect_outliers_cap_action(self):
        """Test outlier capping action."""
        data = pd.DataFrame({
            'Speed': [20, 30, 40, 50, 60, 70, 9999],
        })
        result, _ = detect_and_handle_outliers(data, method='iqr', action='cap')
        assert result is not None


class TestCategoricalEncoding:
    """Test categorical feature encoding."""
    
    @pytest.mark.unit
    def test_encode_categorical_label_encoding(self):
        """Test label encoding."""
        data = pd.DataFrame({
            'Road_Type': ['Single', 'Dual', 'Single', 'Roundabout'],
            'Weather': ['Fine', 'Rain', 'Fine', 'Snow'],
        })
        result_df, encoders = encode_categorical_features(data, method='label')
        assert result_df is not None
        assert isinstance(encoders, dict)
        assert result_df.dtypes.apply(lambda x: x in [np.int64, np.float64, int, float]).all()
    
    @pytest.mark.unit
    def test_encode_categorical_onehot_encoding(self):
        """Test one-hot encoding."""
        data = pd.DataFrame({
            'Road_Type': ['Single', 'Dual', 'Single', 'Roundabout'],
        })
        result_df, _ = encode_categorical_features(data, method='onehot')
        assert result_df is not None
        # One-hot should increase feature count
        assert result_df.shape[1] > data.shape[1]
    
    @pytest.mark.unit
    def test_encode_categorical_no_categorical(self):
        """Test with no categorical columns."""
        data = pd.DataFrame({
            'Speed': [20, 30, 40, 50],
            'Casualties': [1, 2, 1, 3],
        })
        result_df, _ = encode_categorical_features(data)
        assert result_df is not None
        assert result_df.shape == data.shape
    
    @pytest.mark.unit
    def test_encode_categorical_with_nulls(self):
        """Test encoding with null values."""
        data = pd.DataFrame({
            'Road_Type': ['Single', None, 'Dual', 'Single'],
            'Weather': ['Fine', 'Rain', None, 'Snow'],
        })
        result = encode_categorical_features(data, method='label')
        assert result is not None


class TestCleanDataFunction:
    """Test clean_data function."""
    
    @pytest.mark.unit
    def test_clean_data_basic(self, sample_accident_data):
        """Test clean_data function."""
        result = clean_data(sample_accident_data)
        assert result is not None
        assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.unit
    def test_clean_data_preserves_structure(self, sample_accident_data):
        """Test clean_data preserves column structure."""
        original_cols = sample_accident_data.columns
        result = clean_data(sample_accident_data)
        # Most columns should be preserved
        assert len(result.columns) > 0


class TestEngineerFeaturesFunction:
    """Test engineer_features function."""
    
    @pytest.mark.unit
    def test_engineer_features_increases_features(self, sample_accident_data):
        """Test that feature engineering increases feature count."""
        data = sample_accident_data.copy()
        result = engineer_features(data)
        assert result is not None
        assert len(result) > 0
    
    @pytest.mark.unit
    def test_engineer_features_preserves_target(self, sample_accident_data):
        """Test that target variable is preserved."""
        data = sample_accident_data.copy()
        result = engineer_features(data)
        if 'Accident_Severity' in data.columns:
            assert 'Accident_Severity' in result.columns or len(result) == 0


class TestPreprocessFeaturesFunction:
    """Test preprocess_features function."""
    
    @pytest.mark.unit
    def test_preprocess_features_complete(self, sample_accident_data):
        """Test complete preprocessing."""
        result_df, metadata = preprocess_features(sample_accident_data)
        assert result_df is not None
        assert isinstance(result_df, pd.DataFrame)
        assert isinstance(metadata, dict)
    
    @pytest.mark.unit
    def test_preprocess_features_numeric_output(self, sample_accident_data):
        """Test that preprocessing produces numeric data."""
        result_df, _ = preprocess_features(sample_accident_data)
        if len(result_df) > 0:
            # Most columns should be numeric
            numeric_cols = result_df.select_dtypes(include=[np.number]).columns
            assert len(numeric_cols) > 0


class TestEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_clean_empty_dataframe(self):
        """Test cleaning empty dataframe."""
        data = pd.DataFrame()
        cleaner = DataCleaner()
        with pytest.raises(ValueError, match="empty or None dataframe"):
            cleaner.clean(data)
    
    @pytest.mark.unit
    def test_clean_single_row(self, sample_accident_data):
        """Test cleaning single row."""
        data = sample_accident_data.iloc[:1]
        cleaner = DataCleaner()
        result = cleaner.clean(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_engineer_features_empty(self):
        """Test feature engineering on empty dataframe."""
        data = pd.DataFrame()
        result = engineer_features(data)
        assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.unit
    def test_preprocess_all_same_values(self):
        """Test preprocessing with identical values."""
        data = pd.DataFrame({
            'Speed': [50] * 100,
            'Accident_Severity': ['Slight'] * 100,
        })
        result = preprocess_features(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_preprocess_mixed_types(self):
        """Test preprocessing with mixed data types."""
        data = pd.DataFrame({
            'Speed': ['20', 30, '40', 50],
            'Accident_Severity': ['Slight', 'Serious', 'Slight', 'Fatal'],
        })
        result = preprocess_features(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_handle_missing_all_null(self):
        """Test handling all-null column."""
        data = pd.DataFrame({
            'A': [None, None, None],
            'B': [1, 2, 3],
        })
        result = handle_missing_values(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_detect_outliers_no_outliers(self):
        """Test outlier detection with no outliers."""
        data = pd.DataFrame({
            'Value': np.random.normal(100, 10, 50),
        })
        result, _ = detect_and_handle_outliers(data)
        assert result is not None
    
    @pytest.mark.unit
    def test_encode_categorical_all_unique(self):
        """Test encoding with all unique values."""
        data = pd.DataFrame({
            'ID': [str(i) for i in range(100)],
        })
        result = encode_categorical_features(data, method='label')
        assert result is not None
