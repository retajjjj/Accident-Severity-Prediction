# Accident Severity Prediction

<a target="_blank" href="https://cookiecutter-data-science.drivendata.org/">
    <img src="https://img.shields.io/badge/CCDS-Project%20template-328F97?logo=cookiecutter" />
</a>

A comprehensive machine learning project for predicting accident severity using data science best practices. This project implements a complete data pipeline from raw data acquisition to model training and evaluation, with extensive testing and validation.

## Live API Deployment Attempt

**Public Endpoint:** [Deployed on Railway](https://accident-severity-prediction.up.railway.app)

### API Usage

**Health Check:**
```bash
curl https://accident-severity-prediction.up.railway.app/health
```

**Predict Accident Severity:**
```bash
curl -X POST https://accident-severity-prediction.up.railway.app/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Speed_limit": 30,
    "Road_Type": "Single carriageway",
    "Weather_Conditions": "Fine",
    "Light_Conditions": "Daylight",
    "Number_of_Vehicles": 1,
    "Number_of_Casualties": 1,
    "Day_of_Week": "Friday",
    "Urban_or_Rural_Area": "Urban"
  }'
```

**Response:**
```json
{
  "prediction": "Slight",
  "probabilities": {
    "Fatal": 0.02,
    "Serious": 0.15,
    "Slight": 0.83
  },
  "confidence": 0.83
}
```

## Project Status

### Production Ready
**Test Results (Latest):**
- **Tests:** 254 tests passing (100% pass rate)
- **Coverage:** 56.9% code coverage
- **Status:** Production ready with comprehensive validation
- **Execution:** 27.01 seconds full test suite

**Key Metrics:**
- **254 Tests** - Comprehensive test coverage
- **100% Pass Rate** - All tests passing
- **56.9% Coverage** - Strong code coverage
- **Data Validation** - 82/82 validation tests complete
- **Integration Tested** - End-to-end pipeline validated

## Testing Framework

The project uses a comprehensive testing framework with:

- **254 tests** covering all critical functionality
- **100% pass rate** with zero failures
- **Data validation** with 82 specialized tests
- **Integration testing** for end-to-end workflows
- **Performance testing** for scalability validation

### Test Categories
- **Unit Tests:** 232 tests (91.3%)
- **Integration Tests:** 22 tests (8.7%)
- **Data Validation:** 82 tests
- **Feature Engineering:** 42 tests
- **Model Operations:** 32 tests

## Quick Start

### Prerequisites
- Python 3.11+
- Poetry (dependency management)

```

### Testing
```bash
# Run all tests with coverage
make test-all

# Run specific test categories
make test-unit
make test-integration
make test-performance

# Generate coverage report
make coverage
```

## Test Coverage

| Module | Coverage | Status |
|--------|----------|--------|
| `src/data/validate.py` | 60% | Good |
| `src/data/preprocess.py` | 54% | Medium |
| `src/features/build_features.py` | 82% | Excellent |
| `src/models/train.py` | 71% | Good |
| `src/models/evaluate.py` | 84% | Excellent |
| **Overall** | **56.9%** | **Production Ready** |




## Project Organization as learnt in lectures - cookiecutter architecture

```
├── LICENSE             <- Open-source license if one is chosen
├── Makefile            <- Makefile with convenience commands like `make data` or `make train`
├── README.md           <- The top-level README for developers using this project.
├── .gitignore          <- Git ignore patterns
├── requirements-test.txt <- Test dependencies
├── poetry.lock         <- Poetry lock file
├── pyproject.toml     <- Poetry project configuration
├── pytest.ini         <- Pytest configuration
├── docker         <- Pytest configuration
│
├── catboost_info      <- CatBoost model information
│   ├── learn          <- Training data
│   └── test           <- Test data
│
├── data
│   ├── external       <- Data from third party sources.
│   ├── interim        <- Intermediate data that has been transformed.
│   ├── processed      <- The final, canonical data sets for modeling.
│   └── raw            <- The original, immutable data dump.
│
│
├── docs               <- A default mkdocs project for each module
│
├── mlruns             <- MLflow tracking data
│
├── models             <- Trained and serialized models, model predictions, or model summaries
│
├── notebooks          <- Jupyter notebooks.
├── references         <- Data dictionaries, manuals, and all other explanatory materials.
│
├── pyproject.toml     <- Project configuration file with package metadata for
│                         accident_severity_predictor and configuration for tools like black
│
├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
│   ├── coverage       <- Data from third party sources.
│   ├── figures        <- Intermediate data that has been transformed.
│   ├── mlflow_artifacts      <- The final, canonical data sets for modeling.
│   ├── test_results        <- Test results from model evaluation
│   └── validation          <- Validation results from model evaluation
|
├── scripts             <- Helper scripts for data processing, model training, and running tests
├── src                  <- Source code for use in this project.
│   ├── api              <- API endpoints
│   ├── data             <- Data processing scripts
│   ├── features         <- Feature engineering scripts
│   └── models           <- Trained and serialized models
│                         generated with `pip freeze > requirements.txt`
│
└── tests               <- Test files for the project
  ├── test_data         <- Test data
  ├── test_features     <- Test features
  ├── test_integration  <- Integration tests
  ├── test_models       <- Model tests
  └── conftest.py       <- Configuration for pytest

```

---
