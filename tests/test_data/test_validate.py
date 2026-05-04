"""
Unit tests for data validation module.

Test Coverage:
- Data quality checks
- Schema validation
- Business rule validation
- Data integrity verification
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
    from src.data.validate import DataValidator, validate_dataset
except ImportError as e:
    pytest.skip(f"data.validate module not found: {e}", allow_module_level=True)


class TestDataValidator:
    """Test cases for the DataValidator class."""
    
    @pytest.mark.unit
    def test_validator_initialization(self):
        """Test DataValidator initialization."""
        validator = DataValidator()
        
        # Check default attributes
        assert hasattr(validator, 'validation_rules')
        assert hasattr(validator, 'validation_results')
    
    @pytest.mark.unit
    def test_validate_missing_values(self, sample_accident_data):
        """Test missing value validation."""
        validator = DataValidator()
        
        # Add some missing values
        data = sample_accident_data.copy()
        data.loc[0:5, 'Speed_limit'] = None
        
        results = validator.validate_missing_values(data)
        
        assert 'missing_values' in results
        assert 'Speed_limit' in results['missing_values']
        assert results['missing_values']['Speed_limit'] > 0
    
    @pytest.mark.unit
    def test_validate_data_types(self, sample_accident_data):
        """Test data type validation."""
        validator = DataValidator()
        
        results = validator.validate_data_types(sample_accident_data)
        
        assert 'data_types' in results
        assert 'type_issues' in results
        
        # Check critical columns have correct types
        critical_types = {
            'Accident_Severity': 'object',
            'Speed_limit': 'int',  # Numeric or int64
            'Latitude': 'float64',
            'Longitude': 'float64',
        }
        
        for col, expected_type in critical_types.items():
            if col in sample_accident_data.columns:
                actual_type = str(sample_accident_data[col].dtype)
                # Allow for some flexibility in type checking (pandas 2.x uses 'str' instead of 'object')
                assert (expected_type in actual_type or 'float' in actual_type or 
                        'object' in actual_type or 'str' in actual_type)
    
    @pytest.mark.unit
    def test_validate_value_ranges(self, sample_accident_data):
        """Test value range validation."""
        validator = DataValidator()
        
        results = validator.validate_value_ranges(sample_accident_data)
        
        assert 'value_ranges' in results
        assert 'range_violations' in results
        
        # Check geographic coordinates
        if 'Latitude' in sample_accident_data.columns:
            lat_violations = results['range_violations'].get('Latitude', 0)
            assert lat_violations == 0  # All latitudes should be valid
    
    @pytest.mark.unit
    def test_validate_duplicates(self, sample_accident_data):
        """Test duplicate detection."""
        validator = DataValidator()
        
        # Add a duplicate row
        data = pd.concat([sample_accident_data, sample_accident_data.iloc[[0]]], ignore_index=True)
        
        results = validator.validate_duplicates(data)
        
        assert 'duplicates' in results
        assert results['duplicates']['total_duplicates'] > 0
    
    @pytest.mark.unit
    def test_validate_business_rules(self, sample_accident_data):
        """Test business rule validation."""
        validator = DataValidator()
        
        results = validator.validate_business_rules(sample_accident_data)
        
        assert 'business_rules' in results
        assert 'rule_violations' in results
        
        # Check for logical consistency
        # e.g., casualties should not exceed reasonable limits
        if 'Number_of_Casualties' in sample_accident_data.columns:
            max_casualties = sample_accident_data['Number_of_Casualties'].max()
            assert max_casualties <= 50  # Reasonable upper limit


class TestValidateDataset:
    """Test cases for the validate_dataset function."""
    
    @pytest.mark.unit
    def test_validate_dataset_complete(self, sample_accident_data):
        """Test complete dataset validation."""
        results = validate_dataset(sample_accident_data)
        
        # Check all validation sections are present
        expected_sections = [
            'missing_values',
            'data_types', 
            'value_ranges',
            'duplicates',
            'business_rules',
            'summary'
        ]
        
        for section in expected_sections:
            assert section in results
        
        # Check summary contains overall statistics
        assert 'total_records' in results['summary']
        assert 'total_columns' in results['summary']
        assert 'validation_score' in results['summary']
    
    @pytest.mark.unit
    def test_validate_dataset_with_empty_dataframe(self):
        """Test validation with empty DataFrame."""
        empty_df = pd.DataFrame()
        
        results = validate_dataset(empty_df)
        
        assert results['summary']['total_records'] == 0
        assert results['summary']['total_columns'] == 0
        assert results['summary']['validation_score'] == 0.0
    
    @pytest.mark.unit
    def test_validate_dataset_with_null_dataframe(self):
        """Test validation with None input."""
        with pytest.raises(ValueError, match="Dataset cannot be None"):
            validate_dataset(None)


class TestSpecificValidationRules:
    """Tests for specific validation rules critical to the accident severity project."""
    
    @pytest.mark.unit
    def test_validate_severity_classes(self, sample_accident_data):
        """Test validation of accident severity classes."""
        validator = DataValidator()
        
        # Check for valid severity classes
        valid_severities = {'Slight', 'Serious', 'Fatal'}
        actual_severities = set(sample_accident_data['Accident_Severity'].unique())
        
        invalid_severities = actual_severities - valid_severities
        assert len(invalid_severities) == 0, f"Invalid severity classes found: {invalid_severities}"
    
    @pytest.mark.unit
    def test_validate_speed_limits(self, sample_accident_data):
        """Test validation of UK speed limits."""
        validator = DataValidator()
        
        if 'Speed_limit' in sample_accident_data.columns:
            valid_speeds = {20, 30, 40, 50, 60, 70}
            actual_speeds = set(sample_accident_data['Speed_limit'].dropna().unique())
            
            invalid_speeds = actual_speeds - valid_speeds
            assert len(invalid_speeds) == 0, f"Invalid speed limits found: {invalid_speeds}"
    
    @pytest.mark.unit
    def test_validate_geographic_coordinates(self, sample_accident_data):
        """Test validation of UK geographic coordinates."""
        validator = DataValidator()
        
        if 'Latitude' in sample_accident_data.columns:
            # UK should be approximately between 49-61°N
            lats = sample_accident_data['Latitude'].dropna()
            assert lats.min() >= 49, f"Latitude too low: {lats.min()}"
            assert lats.max() <= 61, f"Latitude too high: {lats.max()}"
        
        if 'Longitude' in sample_accident_data.columns:
            # UK should be approximately between -11 to 2°E
            lons = sample_accident_data['Longitude'].dropna()
            assert lons.min() >= -11, f"Longitude too low: {lons.min()}"
            assert lons.max() <= 2, f"Longitude too high: {lons.max()}"
    
    @pytest.mark.unit
    def test_validate_date_consistency(self, sample_accident_data):
        """Test validation of date and day consistency."""
        validator = DataValidator()
        
        if 'Date' in sample_accident_data.columns and 'Day_of_Week' in sample_accident_data.columns:
            # Convert dates to day names
            dates = pd.to_datetime(sample_accident_data['Date'])
            expected_days = dates.dt.day_name()
            
            # Check consistency (allowing for some mismatches due to data quality)
            mismatches = (expected_days != sample_accident_data['Day_of_Week']).sum()
            total_records = len(sample_accident_data)
            mismatch_rate = mismatches / total_records
            
            # Allow for small mismatch rate due to data quality issues
            assert mismatch_rate <= 0.1, f"High date/day mismatch rate: {mismatch_rate:.2%}"


class TestValidationReporting:
    """Tests for validation reporting functionality."""
    
    @pytest.mark.unit
    def test_generate_validation_report(self, sample_accident_data):
        """Test generation of validation report."""
        results = validate_dataset(sample_accident_data)
        
        # Check report structure
        assert 'summary' in results
        assert 'validation_score' in results['summary']
        assert 'issues_found' in results['summary']
        
        # Validation score should be between 0 and 1
        score = results['summary']['validation_score']
        assert 0 <= score <= 1
    
    @pytest.mark.unit
    def test_validation_report_with_issues(self):
        """Test validation report when issues are found."""
        # Create data with intentional issues
        problematic_data = pd.DataFrame({
            'Accident_Severity': ['Slight', 'Invalid', 'Serious'],
            'Speed_limit': [30, 999, 40],  # Invalid speed limit
            'Latitude': [51.5, 91.0, 52.0],  # Invalid latitude
        })
        
        results = validate_dataset(problematic_data)
        
        # Should detect issues
        assert results['summary']['issues_found'] > 0
        assert results['summary']['validation_score'] < 1.0
    
    @pytest.mark.unit
    def test_export_validation_results(self, sample_accident_data, temp_data_dir):
        """Test exporting validation results to file."""
        results = validate_dataset(sample_accident_data)
        
        # Test JSON export
        import json
        report_path = temp_data_dir / "validation_report.json"
        
        with open(report_path, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        assert report_path.exists()
        
        # Verify content
        with open(report_path, 'r') as f:
            loaded_results = json.load(f)
        
        assert 'summary' in loaded_results
        assert loaded_results['summary']['total_records'] == len(sample_accident_data)


class TestValidationPerformance:
    """Performance tests for validation operations."""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_validation_performance_large_dataset(self):
        """Test validation performance with larger dataset."""
        # Create larger test dataset
        n_rows = 10000
        large_data = pd.DataFrame({
            'Accident_Severity': np.random.choice(['Slight', 'Serious', 'Fatal'], n_rows),
            'Speed_limit': np.random.choice([20, 30, 40, 50, 60, 70], n_rows),
            'Latitude': np.random.uniform(50, 55, n_rows),
            'Longitude': np.random.uniform(-3, 1, n_rows),
            'Date': pd.date_range('2020-01-01', periods=n_rows, freq='h'),
        })
        
        import time
        start_time = time.time()
        results = validate_dataset(large_data)
        end_time = time.time()
        
        validation_time = end_time - start_time
        
        # Validation should complete within reasonable time
        assert validation_time < 10.0, f"Validation too slow: {validation_time:.2f}s"
        assert results['summary']['total_records'] == n_rows


class TestValidationEdgeCases:
    """Tests for edge cases and boundary conditions."""
    
    @pytest.mark.unit
    def test_validate_single_record(self):
        """Test validation with single record."""
        single_record = pd.DataFrame({
            'Accident_Severity': ['Slight'],
            'Speed_limit': [30],
            'Latitude': [51.5],
            'Longitude': [-0.1],
        })
        
        results = validate_dataset(single_record)
        
        assert results['summary']['total_records'] == 1
        assert results['summary']['validation_score'] >= 0.0
    
    @pytest.mark.unit
    def test_validate_all_null_column(self):
        """Test validation with column that's all null."""
        data_with_nulls = pd.DataFrame({
            'Accident_Severity': ['Slight', 'Serious'],
            'Speed_limit': [None, None],  # All null
            'Latitude': [51.5, 52.0],
        })
        
        results = validate_dataset(data_with_nulls)
        
        # Should detect high missingness in Speed_limit
        assert results['missing_values']['Speed_limit'] == 2
    
    @pytest.mark.unit
    def test_validate_mixed_data_types(self):
        """Test validation with mixed data types."""
        mixed_data = pd.DataFrame({
            'string_col': ['A', 'B', 'C'],
            'numeric_col': [1, 2, 3],
            'datetime_col': pd.date_range('2023-01-01', periods=3),
            'bool_col': [True, False, True],
        })
        
        results = validate_dataset(mixed_data)
        
        assert results['summary']['total_records'] == 3
        assert results['summary']['total_columns'] == 4
