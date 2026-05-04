"""
Unit tests for feature engineering module.

Test Coverage:
- Temporal feature creation
- Weather and road condition encoding
- Vehicle and driver feature engineering
- Feature selection and transformation
- Interaction features creation
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
import sys

# Add paths for imports BEFORE trying to import modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.features.build_features import (
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
        apply_smote_tomek,
    )
except ImportError as e:
    pytest.skip(f"features module not found: {e}", allow_module_level=True)


class TestTargetEncoding:
    """Test cases for target variable encoding."""
    
    @pytest.mark.unit
    def test_encode_target_variable_string(self, sample_accident_data):
        """Test encoding string target variable."""
        result = encode_target_variable(sample_accident_data)
        
        assert 'Accident_Severity' in result.columns
        assert result['Accident_Severity'].dtype in ['int64', 'int32']
        assert set(result['Accident_Severity'].unique()).issubset({0, 1, 2})
    
    @pytest.mark.unit
    def test_encode_target_variable_already_numeric(self, sample_accident_data):
        """Test encoding already numeric target variable."""
        data = sample_accident_data.copy()
        # Convert to numeric first
        severity_map = {'Slight': 0, 'Serious': 1, 'Fatal': 2}
        data['Accident_Severity'] = data['Accident_Severity'].map(severity_map)
        
        result = encode_target_variable(data)
        
        # Should remain unchanged
        assert result['Accident_Severity'].dtype in ['int64', 'int32']
        assert set(result['Accident_Severity'].unique()).issubset({0, 1, 2})
    
    @pytest.mark.unit
    def test_encode_target_variable_no_target(self, sample_accident_data):
        """Test encoding when target column is missing."""
        data = sample_accident_data.drop('Accident_Severity', axis=1)
        
        result = encode_target_variable(data)
        
        # Should not crash and may or may not add the column
        assert isinstance(result, pd.DataFrame)


class TestTemporalFeatures:
    """Test cases for temporal feature creation."""
    
    @pytest.mark.unit
    def test_create_temporal_features_basic(self, sample_accident_data):
        """Test basic temporal feature creation."""
        result = create_temporal_features(sample_accident_data)
        
        # Check new temporal features
        expected_features = ['hour_of_day', 'day_of_week', 'is_weekend', 'month', 'season']
        for feature in expected_features:
            assert feature in result.columns
        
        # Check value ranges
        assert result['hour_of_day'].between(0, 23).all()
        assert result['day_of_week'].between(0, 6).all()
        assert result['is_weekend'].isin([0, 1]).all()
        assert result['month'].between(1, 12).all()
        assert result['season'].isin(['Spring', 'Summer', 'Autumn', 'Winter']).all()
    
    @pytest.mark.unit
    def test_create_temporal_features_string_dates(self):
        """Test temporal features with string dates."""
        data = pd.DataFrame({
            'Date': ['2023-01-15', '2023-06-15', '2023-12-15'],
            'Accident_Severity': ['Slight', 'Serious', 'Fatal']
        })
        
        result = create_temporal_features(data)
        
        assert 'hour_of_day' in result.columns
        assert 'season' in result.columns
        assert result.loc[0, 'season'] == 'Winter'
        assert result.loc[1, 'season'] == 'Summer'
    
    @pytest.mark.unit
    def test_create_temporal_features_invalid_dates(self):
        """Test temporal features with invalid dates."""
        data = pd.DataFrame({
            'Date': ['invalid_date', '2023-01-01', None],
            'Accident_Severity': ['Slight', 'Serious', 'Fatal']
        })
        
        result = create_temporal_features(data)
        
        # Should handle invalid dates gracefully
        assert 'hour_of_day' in result.columns
        assert len(result) == 3
    
    @pytest.mark.unit
    def test_create_temporal_features_no_date_column(self):
        """Test temporal features without Date column."""
        data = pd.DataFrame({
            'Accident_Severity': ['Slight', 'Serious', 'Fatal'],
            'Speed_limit': [30, 40, 50]
        })
        
        result = create_temporal_features(data)
        
        # Should not crash
        assert isinstance(result, pd.DataFrame)


class TestLightingFeatures:
    """Test cases for lighting condition features."""
    
    @pytest.mark.unit
    def test_create_lighting_features_basic(self, sample_accident_data):
        """Test basic lighting feature creation."""
        result = create_lighting_features(sample_accident_data)
        
        assert 'is_dark' in result.columns
        assert result['is_dark'].isin([0, 1]).all()
        assert 'Light_Conditions' not in result.columns  # Should be dropped
    
    @pytest.mark.unit
    def test_create_lighting_features_various_conditions(self):
        """Test lighting features with various conditions."""
        data = pd.DataFrame({
            'Light_Conditions': ['Daylight', 'Darkness - lights lit', 'Darkness - no lights'],
            'Accident_Severity': ['Slight', 'Serious', 'Fatal']
        })
        
        result = create_lighting_features(data)
        
        # Daylight should be 0, darkness should be 1
        assert result.loc[0, 'is_dark'] == 0
        assert result.loc[1, 'is_dark'] == 1
        assert result.loc[2, 'is_dark'] == 1
    
    @pytest.mark.unit
    def test_create_lighting_features_no_column(self):
        """Test lighting features without Light_Conditions column."""
        data = pd.DataFrame({
            'Accident_Severity': ['Slight', 'Serious'],
            'Speed_limit': [30, 40]
        })
        
        result = create_lighting_features(data)
        
        assert 'is_dark' in result.columns
        assert (result['is_dark'] == 0).all()  # Should default to 0


class TestRoadEncoding:
    """Test cases for road-related encoding."""
    
    @pytest.mark.unit
    def test_encode_road_type_features(self, sample_accident_data):
        """Test road type encoding."""
        result = encode_road_type_features(sample_accident_data)
        
        if 'Road_Type' in sample_accident_data.columns:
            assert 'Road_Type' in result.columns
            assert result['Road_Type'].dtype in ['int64', 'int32']
    
    @pytest.mark.unit
    def test_encode_road_surface_features(self, sample_accident_data):
        """Test road surface encoding."""
        result = encode_road_surface_features(sample_accident_data)
        
        if 'Road_Surface_Conditions' in sample_accident_data.columns:
            assert 'is_wet_road' in result.columns
            assert result['is_wet_road'].isin([0, 1]).all()
            assert 'Road_Surface_Conditions' not in result.columns
    
    @pytest.mark.unit
    def test_encode_weather_condition_features(self, sample_accident_data):
        """Test weather condition encoding."""
        result = encode_weather_condition_features(sample_accident_data)
        
        if 'Weather_Conditions' in sample_accident_data.columns:
            assert 'is_adverse_weather' in result.columns
            assert result['is_adverse_weather'].isin([0, 1]).all()
            assert 'Weather_Conditions' not in result.columns
    
    @pytest.mark.unit
    def test_encode_urban_rural_features(self, sample_accident_data):
        """Test urban/rural encoding."""
        result = encode_urban_rural_features(sample_accident_data)
        
        if 'Urban_or_Rural_Area' in sample_accident_data.columns:
            assert 'is_urban' in result.columns
            assert result['is_urban'].isin([0, 1]).all()
            assert 'Urban_or_Rural_Area' not in result.columns


class TestWeatherFeatures:
    """Test cases for weather-related features."""
    
    @pytest.mark.unit
    def test_create_weather_features_basic(self, sample_weather_data):
        """Test basic weather feature creation."""
        result = create_weather_features(sample_weather_data)
        
        assert 'is_adverse_weather' in result.columns
        assert result['is_adverse_weather'].isin([0, 1]).all()
    
    @pytest.mark.unit
    def test_create_weather_features_with_precipitation(self):
        """Test weather features with precipitation data."""
        data = pd.DataFrame({
            'prcp': [0.0, 2.5, 0.0, 5.0],  # Precipitation in mm
            'temp': [15.0, 20.0, 10.0, 5.0],
        })
        
        result = create_weather_features(data)
        
        # Should detect precipitation
        assert result.loc[0, 'is_adverse_weather'] == 0  # No precipitation
        assert result.loc[1, 'is_adverse_weather'] == 1  # Some precipitation
        assert result.loc[2, 'is_adverse_weather'] == 0  # No precipitation
        assert result.loc[3, 'is_adverse_weather'] == 1  # High precipitation
    
    @pytest.mark.unit
    def test_create_weather_features_with_snow(self):
        """Test weather features with snow data."""
        data = pd.DataFrame({
            'snwd': [0.0, 0.0, 5.0, 10.0],  # Snow depth in mm
            'temp': [15.0, 20.0, -2.0, -5.0],
        })
        
        result = create_weather_features(data)
        
        # Should detect snow
        assert result.loc[0, 'is_adverse_weather'] == 0  # No snow
        assert result.loc[1, 'is_adverse_weather'] == 0  # No snow
        assert result.loc[2, 'is_adverse_weather'] == 1  # Some snow
        assert result.loc[3, 'is_adverse_weather'] == 1  # High snow
    
    @pytest.mark.unit
    def test_create_weather_composite_features(self, sample_accident_data):
        """Test composite weather features."""
        # Add required columns for composite features
        data = sample_accident_data.copy()
        data['is_adverse_weather'] = ([0, 1, 0, 1] * (len(data) // 4 + 1))[:len(data)]
        data['is_dark'] = ([0, 0, 1, 1] * (len(data) // 4 + 1))[:len(data)]
        
        result = create_weather_composite_features(data)
        
        assert 'adverse_dark' in result.columns
        assert result['adverse_dark'].isin([0, 1]).all()


class TestRoadRiskFeatures:
    """Test cases for road risk features."""
    
    @pytest.mark.unit
    def test_create_road_risk_features_basic(self, sample_accident_data):
        """Test basic road risk feature creation."""
        result = create_road_risk_features(sample_accident_data)
        
        if 'Speed_limit' in sample_accident_data.columns:
            assert 'road_risk_score' in result.columns
            assert result['road_risk_score'].min() >= 0
        else:
            # Should not crash without Speed_limit
            assert isinstance(result, pd.DataFrame)
    
    @pytest.mark.unit
    def test_create_road_risk_features_various_speeds(self):
        """Test road risk features with various speed limits."""
        data = pd.DataFrame({
            'Speed_limit': [20, 30, 40, 50, 60, 70],
            'Road_Type': ['Single carriageway'] * 6,
        })
        
        result = create_road_risk_features(data)
        
        assert 'road_risk_score' in result.columns
        # Higher speed limits should generally have higher risk scores
        assert result.loc[5, 'road_risk_score'] >= result.loc[0, 'road_risk_score']


class TestVehicleFeatures:
    """Test cases for vehicle-related features."""
    
    @pytest.mark.unit
    def test_create_vehicle_features_basic(self, sample_vehicle_data):
        """Test basic vehicle feature creation."""
        result = create_vehicle_features(sample_vehicle_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_vehicle_data)
    
    @pytest.mark.unit
    def test_encode_vehicle_attributes(self, sample_vehicle_data):
        """Test vehicle attribute encoding."""
        result = encode_vehicle_attributes(sample_vehicle_data)
        
        if 'Vehicle_Type' in sample_vehicle_data.columns:
            assert 'Vehicle_Type' in result.columns
            assert result['Vehicle_Type'].dtype in ['int64', 'int32']
        
        if 'Engine_Capacity_.CC.' in sample_vehicle_data.columns:
            assert 'engine_size_category' in result.columns
    
    @pytest.mark.unit
    def test_encode_driver_features(self, sample_vehicle_data):
        """Test driver feature encoding."""
        result = encode_driver_features(sample_vehicle_data)
        
        if 'Age_Band_of_Driver' in sample_vehicle_data.columns:
            assert 'age' in result.columns
            assert result['age'].notna().sum() > 0
        
        if 'Sex_of_Driver' in sample_vehicle_data.columns:
            assert 'is_male' in result.columns
            assert result['is_male'].isin([0, 1]).all()


class TestJunctionFeatures:
    """Test cases for junction-related features."""
    
    @pytest.mark.unit
    def test_encode_manoeuvre_features(self, sample_vehicle_data):
        """Test manoeuvre encoding."""
        result = encode_manoeuvre_features(sample_vehicle_data)
        
        if 'Vehicle_Manoeuvre' in sample_vehicle_data.columns:
            assert 'manoeuvre_encoded' in result.columns
            assert result['manoeuvre_encoded'].dtype in ['int64', 'int32']
    
    @pytest.mark.unit
    def test_encode_junction_features(self, sample_vehicle_data):
        """Test junction feature encoding."""
        result = encode_junction_features(sample_vehicle_data)
        
        # Should not crash even without junction columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_vehicle_data)
    
    @pytest.mark.unit
    def test_encode_journey_features(self, sample_vehicle_data):
        """Test journey feature encoding."""
        result = encode_journey_features(sample_vehicle_data)
        
        if 'Journey_Purpose_of_Driver' in sample_vehicle_data.columns:
            assert 'journey_purpose_encoded' in result.columns


class TestAdministrativeFeatures:
    """Test cases for administrative features."""
    
    @pytest.mark.unit
    def test_encode_administrative_features(self, sample_accident_data):
        """Test administrative feature encoding."""
        result = encode_administrative_features(sample_accident_data)
        
        # Should not crash even without administrative columns
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_accident_data)
    
    @pytest.mark.unit
    def test_encode_administrative_with_district(self, sample_accident_data):
        """Test administrative features with district data."""
        # Add district column for testing
        data = sample_accident_data.copy()
        data['Local_Authority_(District)'] = ([1, 2, 3, 4, 5] * (len(data) // 5 + 1))[:len(data)]
        
        result = encode_administrative_features(data)
        
        if 'Local_Authority_(District)' in data.columns:
            assert 'district_severity_rate' in result.columns
            assert 'district_accident_volume' in result.columns


class TestInteractionFeatures:
    """Test cases for interaction features."""
    
    @pytest.mark.unit
    def test_create_interaction_features_basic(self, sample_accident_data):
        """Test basic interaction feature creation."""
        result = create_interaction_features(sample_accident_data)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) == len(sample_accident_data)
    
    @pytest.mark.unit
    def test_create_interaction_features_with_required_cols(self):
        """Test interaction features with required columns."""
        data = pd.DataFrame({
            'Speed_limit': [30, 40, 50, 60],
            'Road_Type': ['Single carriageway', 'Dual carriageway', 'Single carriageway', 'Dual carriageway'],
            'is_wet_road': [0, 1, 0, 1],
            'age': [25, 35, 45, 55],
        })
        
        result = create_interaction_features(data)
        
        # Should create interaction features
        assert 'speed_x_road_risk' in result.columns
        assert 'wet_road_speed' in result.columns


class TestMissingValueHandling:
    """Test cases for missing value handling."""
    
    @pytest.mark.unit
    def test_handle_missing_values_basic(self, sample_accident_data):
        """Test basic missing value handling."""
        # Add missing values
        data = sample_accident_data.copy()
        data.loc[0:5, 'Speed_limit'] = None
        
        result = handle_missing_values(data, missing_threshold=50.0)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
    
    @pytest.mark.unit
    def test_handle_missing_values_high_threshold(self, sample_accident_data):
        """Test missing value handling with high threshold."""
        # Add column with high missingness
        data = sample_accident_data.copy()
        data['high_missing_col'] = [None] * len(data)
        
        result = handle_missing_values(data, missing_threshold=20.0)
        
        # Should drop high-missingness column
        assert 'high_missing_col' not in result.columns
    
    @pytest.mark.unit
    def test_handle_missing_values_no_missing(self, sample_accident_data):
        """Test missing value handling with no missing values."""
        result = handle_missing_values(sample_accident_data, missing_threshold=50.0)
        
        assert len(result) == len(sample_accident_data)


class TestOutlierDetection:
    """Test cases for outlier detection and handling."""
    
    @pytest.mark.unit
    def test_detect_and_handle_outliers_iqr(self, sample_accident_data):
        """Test IQR-based outlier detection."""
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
    def test_detect_and_handle_outliers_remove(self, sample_accident_data):
        """Test outlier removal."""
        result, stats = detect_and_handle_outliers(
            sample_accident_data, method='iqr', action='remove'
        )
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(stats, dict)
        # May have fewer rows if outliers were removed
        assert len(result) <= len(sample_accident_data)


class TestCategoricalEncoding:
    """Test cases for categorical encoding."""
    
    @pytest.mark.unit
    def test_encode_categorical_features_label(self, sample_accident_data):
        """Test label encoding of categorical features."""
        categorical_cols = ['Road_Type', 'Weather_Conditions']
        available_cols = [c for c in categorical_cols if c in sample_accident_data.columns]
        
        if available_cols:
            result, encoders = encode_categorical_features(
                sample_accident_data, 
                categorical_cols=available_cols,
                method='label'
            )
            
            assert isinstance(result, pd.DataFrame)
            assert isinstance(encoders, dict)
            
            for col in available_cols:
                if col in result.columns:
                    assert result[col].dtype in ['int64', 'int32', 'float64']
    
    @pytest.mark.unit
    def test_encode_categorical_features_onehot(self, sample_accident_data):
        """Test one-hot encoding of categorical features."""
        categorical_cols = ['Road_Type']
        available_cols = [c for c in categorical_cols if c in sample_accident_data.columns]
        
        if available_cols:
            result, encoders = encode_categorical_features(
                sample_accident_data, 
                categorical_cols=available_cols,
                method='onehot'
            )
            
            assert isinstance(result, pd.DataFrame)
            assert isinstance(encoders, dict)
    
    @pytest.mark.unit
    def test_encode_categorical_features_no_categorical(self):
        """Test encoding with no categorical columns."""
        data = pd.DataFrame({
            'numeric1': [1, 2, 3],
            'numeric2': [4.5, 5.5, 6.5],
        })
        
        result, encoders = encode_categorical_features(data, categorical_cols=[])
        
        assert isinstance(result, pd.DataFrame)
        assert isinstance(encoders, dict)
        assert len(encoders) == 0


class TestFeatureSelection:
    """Test cases for feature selection methods."""
    
    @pytest.mark.unit
    def test_select_features_model_based(self, sample_accident_data):
        """Test model-based feature selection."""
        # Prepare data for feature selection
        data = sample_accident_data.copy()
        
        # Create numeric features
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            # Create target
            y = pd.Series([0, 1, 0, 1, 0] * (len(data) // 5 + 1))[:len(data)]
            
            selected_features, importance = select_features_model_based(
                numeric_data, y, top_k=5
            )
            
            assert isinstance(selected_features, list)
            assert isinstance(importance, pd.DataFrame)
            assert len(selected_features) <= 5
    
    @pytest.mark.unit
    def test_select_features_rfecv(self, sample_accident_data):
        """Test RFECV feature selection."""
        # Prepare data for feature selection
        data = sample_accident_data.copy()
        
        # Create numeric features
        numeric_data = data.select_dtypes(include=[np.number])
        if len(numeric_data.columns) > 1:
            # Create target
            y = pd.Series([0, 1, 0, 1, 0] * (len(data) // 5 + 1))[:len(data)]
            
            selected_features, ranking = select_features_rfecv(
                numeric_data, y, min_features_to_select=2, cv_folds=3
            )
            
            assert isinstance(selected_features, list)
            assert isinstance(ranking, pd.DataFrame)
            assert len(selected_features) >= 2


class TestSMOTEApplication:
    """Test cases for SMOTE application."""
    
    @pytest.mark.unit
    def test_apply_smote_basic(self):
        """Test basic SMOTE application."""
        # Create imbalanced dataset
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        })
        y = pd.Series([0] * 80 + [1] * 20)  # Imbalanced
        
        result_X, result_y = apply_smote(X, y, sampling_strategy='auto')
        
        assert isinstance(result_X, pd.DataFrame)
        assert isinstance(result_y, pd.Series)
        assert len(result_X) > len(X)  # Should have more samples
        assert len(result_y) > len(y)
    
    @pytest.mark.unit
    def test_apply_smote_tomek(self):
        """Test SMOTE-Tomek application."""
        # Create imbalanced dataset
        X = pd.DataFrame({
            'feature1': np.random.randn(100),
            'feature2': np.random.randn(100),
        })
        y = pd.Series([0] * 80 + [1] * 20)  # Imbalanced
        
        result_X, result_y = apply_smote_tomek(X, y, sampling_strategy='auto')
        
        assert isinstance(result_X, pd.DataFrame)
        assert isinstance(result_y, pd.Series)
        assert len(result_X) >= len(X)  # Should have at least original samples


class TestFeatureEngineeringIntegration:
    """Integration tests for feature engineering pipeline."""
    
    @pytest.mark.integration
    def test_complete_feature_engineering_pipeline(self, sample_accident_data):
        """Test complete feature engineering pipeline."""
        # Step 1: Encode target
        data = encode_target_variable(sample_accident_data)
        
        # Step 2: Create temporal features
        data = create_temporal_features(data)
        
        # Step 3: Create lighting features
        data = create_lighting_features(data)
        
        # Step 4: Encode road features
        data = encode_road_type_features(data)
        data = encode_road_surface_features(data)
        data = encode_weather_condition_features(data)
        data = encode_urban_rural_features(data)
        
        # Step 5: Create weather features
        data = create_weather_features(data)
        
        # Step 6: Create road risk features
        data = create_road_risk_features(data)
        
        # Step 7: Create composite features
        data = create_weather_composite_features(data)
        
        # Step 8: Create interaction features
        data = create_interaction_features(data)
        
        # Verify pipeline results
        assert isinstance(data, pd.DataFrame)
        assert len(data) > 0
        assert 'Accident_Severity' in data.columns
        
        # Should have engineered features
        engineered_features = ['hour_of_day', 'is_weekend', 'is_dark', 'road_risk_score']
        for feature in engineered_features:
            if feature in data.columns:
                assert data[feature].notna().sum() > 0
    
    @pytest.mark.integration
    def test_feature_engineering_with_missing_data(self):
        """Test feature engineering with missing data."""
        # Create data with missing values
        n_rows = 50
        data = pd.DataFrame({
            'Accident_Severity': np.random.choice(['Slight', 'Serious', 'Fatal'], n_rows),
            'Date': pd.date_range('2023-01-01', periods=n_rows),
            'Speed_limit': [None] * 10 + [30] * (n_rows - 10),  # Some missing
            'Road_Type': ['Single carriageway'] * n_rows,
        })
        
        # Should handle missing data gracefully
        result = encode_target_variable(data)
        result = create_temporal_features(result)
        result = create_lighting_features(result)
        result = encode_road_type_features(result)
        result = create_road_risk_features(result)
        
        assert isinstance(result, pd.DataFrame)
        assert len(result) > 0
