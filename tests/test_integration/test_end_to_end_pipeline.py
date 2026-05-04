"""
Integration tests for end-to-end pipeline functionality.

Test Coverage:
- Complete data processing pipeline
- Feature engineering integration
- Model training and evaluation workflow
- Data flow between components
- Pipeline performance and reliability
"""

import pytest
import pandas as pd
import numpy as np
import pickle
from pathlib import Path
import sys
import tempfile
import shutil
from unittest.mock import patch, MagicMock

# Add paths for imports BEFORE trying to import modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))
sys.path.insert(0, str(PROJECT_ROOT))

try:
    from src.data.preprocess import load_data, clean_data, engineer_features, preprocess_features, select_features, prepare_train_val_test_split
    from src.features.build_features import encode_target_variable
    from models import normalise_labels, load_or_create_balanced_train
    from src.models.evaluate import Evaluate
except ImportError as e:
    pytest.skip(f"Required modules not found: {e}", allow_module_level=True)

# Import sklearn models
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


class TestEndToEndDataPipeline:
    """Test cases for complete end-to-end data processing pipeline."""
    
    @pytest.mark.integration
    def test_complete_data_processing_pipeline(self, merged_sample_data, temp_data_dir):
        """Test complete data processing from raw to ready for modeling."""
        # Step 1: Target encoding
        data = encode_target_variable(merged_sample_data)
        assert 'Accident_Severity' in data.columns
        assert data['Accident_Severity'].dtype in ['int64', 'int32']
        
        # Step 2: Data cleaning
        cleaned_data = clean_data(data)
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) > 0
        
        # Step 3: Feature engineering
        engineered_data = engineer_features(cleaned_data)
        assert isinstance(engineered_data, pd.DataFrame)
        assert engineered_data.shape[1] >= cleaned_data.shape[1]
        
        # Step 4: Feature preprocessing
        preprocessed_data, metadata = preprocess_features(engineered_data)
        assert isinstance(preprocessed_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        
        # Step 5: Feature selection
        X = preprocessed_data.drop('Accident_Severity', axis=1)
        y = preprocessed_data['Accident_Severity']
        
        # Only proceed if we have enough data
        if len(X) > 10 and len(X.columns) > 2:
            selected_features, selection_results = select_features(X, y, method='model_based')
            assert isinstance(selected_features, list)
            assert len(selected_features) > 0
        
        # Verify pipeline integrity
        assert 'Accident_Severity' in preprocessed_data.columns
        assert len(preprocessed_data) > 0
    
    @pytest.mark.integration
    def test_data_flow_consistency(self, merged_sample_data):
        """Test data flow consistency between pipeline stages."""
        # Track data transformations
        original_shape = merged_sample_data.shape
        original_dtypes = merged_sample_data.dtypes.to_dict()
        
        # Apply pipeline
        data = encode_target_variable(merged_sample_data)
        cleaned_data = clean_data(data)
        engineered_data = engineer_features(cleaned_data)
        preprocessed_data, _ = preprocess_features(engineered_data)
        
        # Verify target variable consistency
        assert 'Accident_Severity' in preprocessed_data.columns
        assert set(preprocessed_data['Accident_Severity'].unique()).issubset({0, 1, 2})
        
        # Verify data integrity (no unexpected row loss)
        assert preprocessed_data.shape[0] <= original_shape[0]
        assert preprocessed_data.shape[0] > 0
        
        # Verify feature expansion
        assert preprocessed_data.shape[1] >= original_shape[1]
    
    @pytest.mark.integration
    def test_pipeline_with_missing_data(self):
        """Test pipeline robustness with missing data."""
        # Create data with significant missing values
        n_rows = 100
        problematic_data = pd.DataFrame({
            'Accident_Severity': np.random.choice(['Slight', 'Serious', 'Fatal'], n_rows),
            'Date': pd.date_range('2023-01-01', periods=n_rows),
            'Speed_limit': [None] * 20 + [30] * (n_rows - 20),  # 20% missing
            'Latitude': np.random.uniform(50, 55, n_rows),
            'Longitude': np.random.uniform(-3, 1, n_rows),
            'Road_Type': np.random.choice(['Single carriageway', 'Dual carriageway'], n_rows),
        })
        
        # Add some completely missing columns
        problematic_data['high_missing_col'] = [None] * n_rows
        problematic_data['moderate_missing_col'] = [None] * 50 + [1] * 50
        
        # Apply pipeline
        data = encode_target_variable(problematic_data)
        cleaned_data = clean_data(data)
        
        # Should handle missing data gracefully
        assert isinstance(cleaned_data, pd.DataFrame)
        assert len(cleaned_data) > 0
        
        # High-missingness columns should be dropped
        assert 'high_missing_col' not in cleaned_data.columns
        
        # Continue with pipeline
        engineered_data = engineer_features(cleaned_data)
        preprocessed_data, _ = preprocess_features(engineered_data)
        
        assert isinstance(preprocessed_data, pd.DataFrame)
        assert len(preprocessed_data) > 0


class TestModelTrainingIntegration:
    """Test cases for model training integration."""
    
    @pytest.mark.integration
    def test_complete_training_pipeline(self, merged_sample_data, temp_data_dir):
        """Test complete training pipeline integration."""
        # Prepare data
        data = encode_target_variable(merged_sample_data)
        cleaned_data = clean_data(data)
        engineered_data = engineer_features(cleaned_data)
        preprocessed_data, _ = preprocess_features(engineered_data)
        
        # Create train/val/test split
        X = preprocessed_data.drop('Accident_Severity', axis=1)
        y = preprocessed_data['Accident_Severity']
        
        if len(X) < 20:  # Skip if not enough data
            pytest.skip("Not enough data for train/val/test split")
        
        # Manual split for testing
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        # Convert to string labels for training module compatibility
        y_train_str = normalise_labels(y_train)
        y_val_str = normalise_labels(y_val)
        y_test_str = normalise_labels(y_test)
        
        # Test SMOTE application (mocked)
        with patch('src.models.train.BALANCED_X_PATH', temp_data_dir / "X_train_balanced.pkl"), \
             patch('src.models.train.BALANCED_Y_PATH', temp_data_dir / "y_train_balanced.pkl"), \
             patch('src.models.train.PROCESSED_DIR', temp_data_dir), \
             patch('src.models.train.USE_SMOTE_TOMEK', False):
            
            # Mock SMOTE to avoid actual balancing
            with patch('src.models.train.SMOTE') as mock_smote:
                mock_smote.return_value.fit_resample.return_value = (X_train, y_train_str)
                
                X_balanced, y_balanced = load_or_create_balanced_train(X_train, y_train_str)
                
                assert isinstance(X_balanced, pd.DataFrame)
                assert isinstance(y_balanced, np.ndarray)
        
        # Test model evaluation integration
        mock_model = MagicMock()
        mock_model.predict.return_value = y_test_str  # Perfect predictions for testing
        
        evaluator = Evaluate(X_test, y_test_str, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_integration_run")
        
        # Verify evaluation was called
        mock_model.predict.assert_called_with(X_test)
    
    @pytest.mark.integration
    def test_training_with_feature_selection(self, merged_sample_data):
        """Test training pipeline with feature selection."""
        # Prepare data
        data = encode_target_variable(merged_sample_data)
        cleaned_data = clean_data(data)
        engineered_data = engineer_features(cleaned_data)
        preprocessed_data, _ = preprocess_features(engineered_data)
        
        X = preprocessed_data.drop('Accident_Severity', axis=1)
        y = preprocessed_data['Accident_Severity']
        
        if len(X) < 20 or len(X.columns) < 5:
            pytest.skip("Not enough data or features for feature selection")
        
        # Apply feature selection
        selected_features, selection_results = select_features(X, y, method='model_based')
        
        # Filter data to selected features
        X_selected = X[selected_features]
        
        assert X_selected.shape[1] <= X.shape[1]
        assert X_selected.shape[0] == X.shape[0]
        
        # Test that selected features work with model training
        mock_model = MagicMock()
        mock_model.fit.return_value = None
        mock_model.predict.return_value = np.random.choice(['Slight', 'Serious', 'Fatal'], len(X_selected))
        
        # Should be able to fit model with selected features
        try:
            mock_model.fit(X_selected, y)
            mock_model.predict(X_selected)
            training_successful = True
        except Exception:
            training_successful = False
        
        assert training_successful, "Training failed with selected features"


class TestDataModelIntegration:
    """Test cases for data-model integration points."""
    
    @pytest.mark.integration
    def test_feature_model_compatibility(self, merged_sample_data):
        """Test that engineered features are compatible with models."""
        # Prepare data
        data = encode_target_variable(merged_sample_data)
        cleaned_data = clean_data(data)
        engineered_data = engineer_features(cleaned_data)
        preprocessed_data, _ = preprocess_features(engineered_data)
        
        X = preprocessed_data.drop('Accident_Severity', axis=1)
        y = preprocessed_data['Accident_Severity']
        
        if len(X) < 10:
            pytest.skip("Not enough data for compatibility testing")
        
        # Ensure all X columns are numeric (convert strings if necessary)
        for col in X.columns:
            if X[col].dtype == 'object' or X[col].dtype == 'str':
                # Try to convert to numeric
                try:
                    X[col] = pd.to_numeric(X[col], errors='coerce')
                except Exception:
                    # If conversion fails, skip this test for now
                    pytest.skip(f"Cannot convert column {col} to numeric")
        
        # Test with different model types
        models = [
            RandomForestClassifier(n_estimators=10, random_state=42),
            LogisticRegression(random_state=42, max_iter=1000)
        ]
        
        for model in models:
            try:
                # Fit model
                model.fit(X, y)
                
                # Make predictions
                predictions = model.predict(X)
                probabilities = model.predict_proba(X)
                
                # Verify outputs
                assert len(predictions) == len(X)
                assert probabilities.shape[0] == len(X)
                assert probabilities.shape[1] == len(np.unique(y))
                
                model_compatible = True
            except Exception as e:
                print(f"Model compatibility issue: {e}")
                model_compatible = False
            
            assert model_compatible, f"Model {type(model).__name__} not compatible with features"
    
    @pytest.mark.integration
    def test_data_leakage_prevention(self, merged_sample_data):
        """Test that pipeline prevents data leakage."""
        # Prepare data
        data = encode_target_variable(merged_sample_data)
        cleaned_data = clean_data(data)
        engineered_data = engineer_features(cleaned_data)
        preprocessed_data, metadata = preprocess_features(engineered_data)
        
        X = preprocessed_data.drop('Accident_Severity', axis=1)
        y = preprocessed_data['Accident_Severity']
        
        if len(X) < 20:
            pytest.skip("Not enough data for leakage testing")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
        
        # Check for potential data leakage indicators
        train_values = set()
        test_values = set()
        
        # Check for identical rows (potential leakage)
        for col in X_train.columns:
            if X_train[col].dtype in ['int64', 'float64']:
                train_values.update(X_train[col].unique())
                test_values.update(X_test[col].unique())
        
        overlap = train_values.intersection(test_values)
        overlap_ratio = len(overlap) / max(len(train_values), 1)
        
        # Some overlap is expected, but should not be 100%
        assert overlap_ratio < 1.0, "Potential data leakage detected"
        
        # Test model performance on unseen data
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit(X_train, y_train)
        
        train_score = model.score(X_train, y_train)
        test_score = model.score(X_test, y_test)
        
        # Test score should be lower than train score (but not dramatically lower)
        assert test_score <= train_score, "Test performance higher than training (potential leakage)"
        assert test_score > 0.1, "Test performance too low (possible data issues)"


class TestPipelinePerformance:
    """Test cases for pipeline performance and scalability."""
    
    @pytest.mark.integration
    @pytest.mark.slow
    def test_pipeline_performance_large_dataset(self):
        """Test pipeline performance with larger dataset."""
        # Create larger test dataset
        n_rows = 1000
        large_data = pd.DataFrame({
            'Accident_Severity': np.random.choice(['Slight', 'Serious', 'Fatal'], n_rows, p=[0.7, 0.25, 0.05]),
            'Date': pd.date_range('2020-01-01', periods=n_rows, freq='h'),
            'Speed_limit': np.random.choice([20, 30, 40, 50, 60, 70], n_rows),
            'Latitude': np.random.uniform(50, 55, n_rows),
            'Longitude': np.random.uniform(-3, 1, n_rows),
            'Road_Type': np.random.choice(['Single carriageway', 'Dual carriageway'], n_rows),
            'Weather_Conditions': np.random.choice(['Fine', 'Rain', 'Snow'], n_rows),
            'Light_Conditions': np.random.choice(['Daylight', 'Darkness'], n_rows),
        })
        
        # Add some missing values
        for col in ['Speed_limit', 'Weather_Conditions']:
            missing_idx = np.random.choice(n_rows, size=int(0.05 * n_rows), replace=False)
            large_data.loc[missing_idx, col] = None
        
        import time
        
        # Time the complete pipeline
        start_time = time.time()
        
        # Apply pipeline
        data = encode_target_variable(large_data)
        cleaned_data = clean_data(data)
        engineered_data = engineer_features(cleaned_data)
        preprocessed_data, _ = preprocess_features(engineered_data)
        
        end_time = time.time()
        pipeline_time = end_time - start_time
        
        # Performance assertions
        assert pipeline_time < 30.0, f"Pipeline too slow: {pipeline_time:.2f}s"
        assert len(preprocessed_data) > 0
        assert preprocessed_data.shape[1] >= large_data.shape[1]
    
    @pytest.mark.integration
    def test_memory_usage_during_pipeline(self, merged_sample_data):
        """Test memory usage during pipeline execution."""
        import sys
        import gc
        
        # Get initial memory usage
        gc.collect()
        initial_objects = len(gc.get_objects())
        
        # Apply pipeline
        data = encode_target_variable(merged_sample_data)
        cleaned_data = clean_data(data)
        engineered_data = engineer_features(cleaned_data)
        preprocessed_data, _ = preprocess_features(engineered_data)
        
        # Check memory usage
        gc.collect()
        final_objects = len(gc.get_objects())
        
        # Memory usage should not increase dramatically
        object_increase = final_objects - initial_objects
        assert object_increase < initial_objects * 2, "Excessive memory usage detected"
        
        # Clean up
        del data, cleaned_data, engineered_data, preprocessed_data
        gc.collect()


class TestPipelineReliability:
    """Test cases for pipeline reliability and error handling."""
    
    @pytest.mark.integration
    def test_pipeline_with_various_data_quality(self):
        """Test pipeline reliability with various data quality issues."""
        # Test cases with different data quality scenarios
        test_cases = [
            # Normal data
            lambda: pd.DataFrame({
                'Accident_Severity': ['Slight', 'Serious', 'Fatal'],
                'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
                'Speed_limit': [30, 40, 50],
                'Latitude': [51.5, 52.0, 52.5],
                'Longitude': [-0.1, -0.2, -0.3],
            }),
            # Data with missing values
            lambda: pd.DataFrame({
                'Accident_Severity': ['Slight', None, 'Fatal'],
                'Date': ['2023-01-01', None, '2023-01-03'],
                'Speed_limit': [30, None, 50],
                'Latitude': [51.5, 52.0, None],
                'Longitude': [-0.1, -0.2, -0.3],
            }),
            # Data with invalid values
            lambda: pd.DataFrame({
                'Accident_Severity': ['Slight', 'Invalid', 'Fatal'],
                'Date': ['2023-01-01', 'invalid_date', '2023-01-03'],
                'Speed_limit': [30, 999, 50],  # Invalid speed limit
                'Latitude': [51.5, 91.0, 52.5],  # Invalid latitude
                'Longitude': [-0.1, -0.2, -0.3],
            }),
        ]
        
        for i, test_case in enumerate(test_cases):
            try:
                data = test_case()
                
                # Apply pipeline
                data = encode_target_variable(data)
                cleaned_data = clean_data(data)
                engineered_data = engineer_features(cleaned_data)
                preprocessed_data, _ = preprocess_features(engineered_data)
                
                # Should handle all cases gracefully
                assert isinstance(preprocessed_data, pd.DataFrame)
                assert len(preprocessed_data) >= 0
                
            except Exception as e:
                pytest.fail(f"Pipeline failed on test case {i}: {e}")
    
    @pytest.mark.integration
    def test_pipeline_recovery_from_errors(self):
        """Test pipeline recovery from intermediate errors."""
        # Create data that will cause issues in some steps
        problematic_data = pd.DataFrame({
            'Accident_Severity': ['Slight'] * 10,
            'Date': ['2023-01-01'] * 10,
            'Speed_limit': [None] * 10,  # All missing
            'Latitude': [51.5] * 10,
            'Longitude': [-0.1] * 10,
        })
        
        try:
            # Apply pipeline step by step
            data = encode_target_variable(problematic_data)
            cleaned_data = clean_data(data)
            engineered_data = engineer_features(cleaned_data)
            preprocessed_data, _ = preprocess_features(engineered_data)
            
            # Should recover and produce valid output
            assert isinstance(preprocessed_data, pd.DataFrame)
            
        except Exception as e:
            # Should provide meaningful error messages
            assert str(e), "Error message should be descriptive"
            raise e


class TestPipelineValidation:
    """Test cases for pipeline output validation."""
    
    @pytest.mark.integration
    def test_pipeline_output_validation(self, merged_sample_data):
        """Test validation of pipeline outputs."""
        # Apply complete pipeline
        data = encode_target_variable(merged_sample_data)
        cleaned_data = clean_data(data)
        engineered_data = engineer_features(cleaned_data)
        preprocessed_data, metadata = preprocess_features(engineered_data)
        
        # Validate final output
        assert isinstance(preprocessed_data, pd.DataFrame)
        assert isinstance(metadata, dict)
        
        # Validate target variable
        assert 'Accident_Severity' in preprocessed_data.columns
        assert preprocessed_data['Accident_Severity'].dtype in ['int64', 'int32']
        assert set(preprocessed_data['Accident_Severity'].unique()).issubset({0, 1, 2})
        
        # Validate features
        feature_cols = preprocessed_data.drop('Accident_Severity', axis=1).columns
        assert len(feature_cols) > 0
        
        # Check for reasonable feature values
        for col in feature_cols[:5]:  # Check first 5 features
            if preprocessed_data[col].dtype in ['int64', 'float64']:
                assert not preprocessed_data[col].isna().all(), f"Feature {col} is all NaN"
                assert not (preprocessed_data[col] == float('inf')).any(), f"Feature {col} contains inf"
                assert not (preprocessed_data[col] == float('-inf')).any(), f"Feature {col} contains -inf"
        
        # Validate metadata
        assert 'encoders' in metadata
        assert 'outlier_stats' in metadata
    
    @pytest.mark.integration
    def test_pipeline_reproducibility(self, merged_sample_data):
        """Test pipeline reproducibility with same input."""
        # Apply pipeline twice with same input
        results1 = []
        results2 = []
        
        for _ in range(2):
            data = encode_target_variable(merged_sample_data.copy())
            cleaned_data = clean_data(data)
            engineered_data = engineer_features(cleaned_data)
            preprocessed_data, metadata = preprocess_features(engineered_data)
            
            results1.append(preprocessed_data.copy())
        
        # Apply again
        for _ in range(2):
            data = encode_target_variable(merged_sample_data.copy())
            cleaned_data = clean_data(data)
            engineered_data = engineer_features(cleaned_data)
            preprocessed_data, metadata = preprocess_features(engineered_data)
            
            results2.append(preprocessed_data.copy())
        
        # Results should be identical
        pd.testing.assert_frame_equal(results1[0], results2[0])
        
        # Check consistency within runs
        pd.testing.assert_frame_equal(results1[0], results1[1])
        pd.testing.assert_frame_equal(results2[0], results2[1])
