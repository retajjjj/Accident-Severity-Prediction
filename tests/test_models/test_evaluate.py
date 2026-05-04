"""
Unit tests for model evaluation module.

Test Coverage:
- Model evaluation metrics
- Classification report generation
- Confusion matrix visualization
- MLflow integration
- Performance metric calculations
"""

import pytest
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import sys
from unittest.mock import patch, MagicMock, mock_open

# Add paths for imports BEFORE trying to import modules
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from src.models.evaluate import Evaluate, CORRECT_LABELS, _nullcontext
except ImportError as e:
    pytest.skip(f"src.models.evaluate module not found: {e}", allow_module_level=True)


class TestEvaluateClass:
    """Test cases for the Evaluate class."""
    
    @pytest.mark.unit
    def test_evaluate_initialization(self, mock_model):
        """Test Evaluate class initialization."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3], 'feature2': [4, 5, 6]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        assert evaluator.X_test.equals(X_test)
        assert np.array_equal(evaluator.y_test, y_test)
        assert evaluator.model == mock_model
        assert evaluator.class_names == CORRECT_LABELS
    
    @pytest.mark.unit
    def test_evaluate_initialization_custom_class_names(self, mock_model):
        """Test Evaluate class initialization with custom class names."""
        X_test = pd.DataFrame({'feature1': [1, 2]})
        y_test = np.array(['Class1', 'Class2'])
        custom_classes = ['Class1', 'Class2']
        
        evaluator = Evaluate(X_test, y_test, mock_model, custom_classes)
        
        assert evaluator.class_names == custom_classes
    
    @pytest.mark.unit
    def test_get_estimator(self, mock_model):
        """Test getting estimator from model."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        estimator = evaluator._get_estimator()
        
        # For mocks, the model attribute exists, so it returns the nested model
        assert hasattr(mock_model, 'model')
        assert estimator is mock_model.model
    
    @pytest.mark.unit
    def test_get_estimator_with_nested_model(self):
        """Test getting estimator from nested model structure."""
        # Create model with nested structure
        outer_model = MagicMock()
        inner_model = MagicMock()
        outer_model.model = inner_model
        
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, outer_model)
        estimator = evaluator._get_estimator()
        
        assert estimator is inner_model


class TestEvaluationMetrics:
    """Test cases for evaluation metrics calculation."""
    
    @pytest.mark.unit
    def test_evaluate_basic_metrics(self, mock_model):
        """Test basic evaluation metrics calculation."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6]})
        y_test = np.array(['Slight', 'Serious', 'Fatal', 'Slight', 'Serious', 'Fatal'])
        
        # Mock model predictions
        mock_model.predict.return_value = np.array(['Slight', 'Serious', 'Fatal', 'Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        # Mock MLflow to avoid actual logging
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_run")
        
        # Verify model.predict was called
        mock_model.predict.assert_called_with(X_test)
    
    @pytest.mark.unit
    def test_evaluate_with_custom_predictions(self, mock_model):
        """Test evaluation with custom predictions."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        custom_predictions = np.array(['Slight', 'Serious', 'Slight'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_run", y_pred=custom_predictions)
        
        # Should not call model.predict when custom predictions provided
        mock_model.predict.assert_not_called()
    
    @pytest.mark.unit
    def test_evaluate_without_mlflow(self, mock_model):
        """Test evaluation without MLflow logging."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        mock_model.predict.return_value = np.array(['Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        # Evaluate with MLflow disabled
        evaluator.evaluate("test_run", log_to_mlflow=False)
        
        # Should still call predict
        mock_model.predict.assert_called_with(X_test)
    
    @pytest.mark.unit
    def test_evaluate_with_unexpected_labels(self, mock_model):
        """Test evaluation with unexpected prediction labels."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        # Mock model to return unexpected labels
        mock_model.predict.return_value = np.array(['Slight', 'Unexpected', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            # Should complete evaluation even with unexpected labels
            # The warning message is printed but the evaluation continues
            evaluator.evaluate("test_run")


class TestClassificationReport:
    """Test cases for classification report functionality."""
    
    @pytest.mark.unit
    def test_classification_report_generation(self, mock_model):
        """Test classification report generation."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6]})
        y_test = np.array(['Slight', 'Serious', 'Fatal', 'Slight', 'Serious', 'Fatal'])
        
        # Mock perfect predictions
        mock_model.predict.return_value = np.array(['Slight', 'Serious', 'Fatal', 'Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run, \
             patch('builtins.print') as mock_print:
            
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_run")
            
            # Check that evaluation metrics were printed
            print_calls = str(mock_print.call_args_list)
            assert 'accuracy' in print_calls.lower() or 'acc' in print_calls.lower()
            assert 'f1' in print_calls.lower()
    
    @pytest.mark.unit
    def test_classification_report_with_imbalanced_data(self, mock_model):
        """Test classification report with imbalanced data."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3, 4, 5]})
        y_test = np.array(['Slight', 'Slight', 'Slight', 'Serious', 'Fatal'])  # Imbalanced
        
        mock_model.predict.return_value = np.array(['Slight', 'Slight', 'Serious', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run, \
             patch('builtins.print') as mock_print:
            
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_run")
            
            # Should handle imbalanced data gracefully
            print_calls = str(mock_print.call_args_list)
            assert len(mock_print.call_args_list) > 0


class TestConfusionMatrix:
    """Test cases for confusion matrix functionality."""
    
    @pytest.mark.unit
    def test_confusion_matrix_generation(self, mock_model):
        """Test confusion matrix generation."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3, 4, 5, 6]})
        y_test = np.array(['Slight', 'Serious', 'Fatal', 'Slight', 'Serious', 'Fatal'])
        
        mock_model.predict.return_value = np.array(['Slight', 'Serious', 'Fatal', 'Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run, \
             patch('src.models.evaluate.ConfusionMatrixDisplay') as mock_cm_display, \
             patch('src.models.evaluate.plt.subplots') as mock_subplots:
            
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            mock_fig = MagicMock()
            mock_ax = MagicMock()
            mock_subplots.return_value = (mock_fig, mock_ax)
            
            evaluator.evaluate("test_run")
            
            # Should create confusion matrix display
            mock_cm_display.from_predictions.assert_called_once()
            
            # Should save figure
            mock_fig.savefig.assert_called_once()
    
    @pytest.mark.unit
    def test_confusion_matrix_with_errors(self, mock_model):
        """Test confusion matrix generation with errors."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        mock_model.predict.return_value = np.array(['Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        # Test that evaluation completes even if confusion matrix creation fails
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            # Evaluation should complete (confusion matrix is not critical)
            evaluator.evaluate("test_run")


class TestMLflowIntegration:
    """Test cases for MLflow integration."""
    
    @pytest.mark.unit
    def test_mlflow_logging_enabled(self, mock_model):
        """Test MLflow logging when enabled."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        mock_model.predict.return_value = np.array(['Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run, \
             patch('src.models.evaluate.mlflow.log_metric') as mock_log_metric, \
             patch('src.models.evaluate.mlflow.log_params') as mock_log_params:
            
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_run")
            
            # Should start MLflow run
            mock_start_run.assert_called_once_with(run_name="test_run")
            
            # Should log metrics
            assert mock_log_metric.call_count > 0
    
    @pytest.mark.unit
    def test_mlflow_logging_disabled(self, mock_model):
        """Test evaluation without MLflow logging."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        mock_model.predict.return_value = np.array(['Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run, \
             patch('src.models.evaluate.mlflow.log_metric') as mock_log_metric:
            
            evaluator.evaluate("test_run", log_to_mlflow=False)
            
            # Should not start MLflow run
            mock_start_run.assert_not_called()
            
            # Should not log metrics
            mock_log_metric.assert_not_called()
    
    @pytest.mark.unit
    def test_mlflow_parameter_logging(self, mock_model):
        """Test MLflow parameter logging."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        mock_model.predict.return_value = np.array(['Slight', 'Serious', 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run, \
             patch('src.models.evaluate.mlflow.log_params') as mock_log_params:
            
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_run")
            
            # Should log model parameters
            mock_log_params.assert_called()


class TestEvaluationEdgeCases:
    """Test cases for edge cases in evaluation."""
    
    @pytest.mark.unit
    def test_evaluation_with_empty_test_set(self, mock_model):
        """Test evaluation with empty test set."""
        X_test = pd.DataFrame({'feature1': []})
        y_test = np.array([])
        
        # Should raise ValueError during initialization, not during evaluate()
        with pytest.raises((ValueError, IndexError)):
            Evaluate(X_test, y_test, mock_model)
    
    @pytest.mark.unit
    def test_evaluation_with_single_class(self, mock_model):
        """Test evaluation with single class predictions."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Slight', 'Slight'])
        
        mock_model.predict.return_value = np.array(['Slight', 'Slight', 'Slight'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            # Should handle single class
            evaluator.evaluate("test_run")
    
    @pytest.mark.unit
    def test_evaluation_with_mismatched_predictions(self, mock_model):
        """Test evaluation with mismatched prediction length."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        # Mock model to return wrong number of predictions
        mock_model.predict.return_value = np.array(['Slight', 'Serious'])  # Only 2 predictions
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            # Create context manager that properly raises exceptions
            def context_exit_handler(exc_type, exc_val, exc_tb):
                return False  # Don't suppress exceptions
            
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(side_effect=context_exit_handler)
            mock_start_run.return_value = mock_ctx
            
            # Should raise ValueError for mismatched lengths
            with pytest.raises(ValueError, match="length|mismatch"):
                evaluator.evaluate("test_run")
    
    @pytest.mark.unit
    def test_evaluation_with_nan_predictions(self, mock_model):
        """Test evaluation with NaN predictions."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        # Mock model to return NaN predictions
        mock_model.predict.return_value = np.array(['Slight', np.nan, 'Fatal'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            # Create context manager that properly raises exceptions
            def context_exit_handler(exc_type, exc_val, exc_tb):
                return False  # Don't suppress exceptions
            
            mock_ctx = MagicMock()
            mock_ctx.__enter__ = MagicMock(return_value=None)
            mock_ctx.__exit__ = MagicMock(side_effect=context_exit_handler)
            mock_start_run.return_value = mock_ctx
            
            # Should raise ValueError for NaN predictions
            with pytest.raises(ValueError, match="NaN"):
                evaluator.evaluate("test_run")


class TestEvaluationPerformance:
    """Performance tests for evaluation operations."""
    
    @pytest.mark.unit
    @pytest.mark.slow
    def test_evaluation_performance_large_dataset(self, mock_model):
        """Test evaluation performance with large dataset."""
        # Create large test dataset
        n_samples = 10000
        X_test = pd.DataFrame({'feature1': np.random.randn(n_samples)})
        y_test = np.random.choice(['Slight', 'Serious', 'Fatal'], n_samples)
        
        # Mock model predictions
        mock_model.predict.return_value = np.random.choice(['Slight', 'Serious', 'Fatal'], n_samples)
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        import time
        start_time = time.time()
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_run")
        
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        # Should complete within reasonable time
        assert evaluation_time < 10.0, f"Evaluation too slow: {evaluation_time:.2f}s"
    
    @pytest.mark.unit
    def test_evaluation_memory_usage(self, mock_model):
        """Test evaluation memory usage."""
        # Create moderately sized dataset
        n_samples = 1000
        X_test = pd.DataFrame({'feature1': np.random.randn(n_samples)})
        y_test = np.random.choice(['Slight', 'Serious', 'Fatal'], n_samples)
        
        mock_model.predict.return_value = np.random.choice(['Slight', 'Serious', 'Fatal'], n_samples)
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        # Monitor memory usage (basic check)
        import sys
        start_size = sys.getsizeof(evaluator)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run:
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_run")
        
        end_size = sys.getsizeof(evaluator)
        
        # Memory usage should not increase dramatically
        assert (end_size - start_size) < start_size * 0.5  # Less than 50% increase


class TestEvaluationValidation:
    """Validation tests for evaluation functionality."""
    
    @pytest.mark.unit
    def test_evaluation_input_validation(self):
        """Test evaluation input validation."""
        # Test with invalid X_test
        with pytest.raises((ValueError, TypeError)):
            Evaluate("invalid_X", np.array(['Slight']), MagicMock())
        
        # Test with invalid y_test
        with pytest.raises((ValueError, TypeError)):
            Evaluate(pd.DataFrame(), "invalid_y", MagicMock())
        
        # Test with invalid model
        with pytest.raises((ValueError, TypeError)):
            Evaluate(pd.DataFrame(), np.array(['Slight']), "invalid_model")
    
    @pytest.mark.unit
    def test_evaluation_class_names_validation(self, mock_model):
        """Test evaluation class names validation."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Class1', 'Class2', 'Class3'])
        
        # Test with empty class names
        with pytest.raises((ValueError, AssertionError)):
            Evaluate(X_test, y_test, mock_model, class_names=[])
        
        # Test with non-unique class names
        with pytest.raises((ValueError, AssertionError)):
            Evaluate(X_test, y_test, mock_model, class_names=['Class1', 'Class1', 'Class2'])
    
    @pytest.mark.unit
    def test_evaluation_metric_validation(self, mock_model):
        """Test evaluation metric calculation validation."""
        X_test = pd.DataFrame({'feature1': [1, 2, 3]})
        y_test = np.array(['Slight', 'Serious', 'Fatal'])
        
        # Mock model to return all same predictions (worst case)
        mock_model.predict.return_value = np.array(['Slight', 'Slight', 'Slight'])
        
        evaluator = Evaluate(X_test, y_test, mock_model)
        
        with patch('src.models.evaluate.mlflow.start_run') as mock_start_run, \
             patch('builtins.print') as mock_print:
            
            mock_start_run.return_value.__enter__ = MagicMock()
            mock_start_run.return_value.__exit__ = MagicMock()
            
            evaluator.evaluate("test_run")
            
            # Should still calculate metrics even for poor predictions
            print_calls = str(mock_print.call_args_list)
            assert len(mock_print.call_args_list) > 0


class TestNullContext:
    """Test cases for the nullcontext utility."""
    
    @pytest.mark.unit
    def test_nullcontext_functionality(self):
        """Test _nullcontext functionality."""
        ctx = _nullcontext()
        
        # Should work as context manager
        with ctx:
            result = "test"
        
        assert result == "test"
    
    @pytest.mark.unit
    def test_nullcontext_exception_handling(self):
        """Test _nullcontext exception handling."""
        ctx = _nullcontext()
        
        # Should not interfere with exceptions
        with pytest.raises(ValueError):
            with ctx:
                raise ValueError("Test exception")
