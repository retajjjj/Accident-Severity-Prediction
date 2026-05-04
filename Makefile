PROJECT_NAME = src
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

## Install Python dependencies
.PHONY: requirements
requirements:
	poetry install
	
## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


## environment
.PHONY: create_environment
create_environment:
	poetry env use $(PYTHON_VERSION)
	@echo ">>> Poetry environment created. Activate with: "
	@echo '$$(poetry env activate)'
	@echo ">>> Or run commands with:\npoetry run <command>"


## Download and validate data
.PHONY: data
data:
	poetry run python src/data/acquire.py
	poetry run python src/data/validate.py

## Preprocess data: engineer features, select, balance (SMOTE), scale
.PHONY: preprocess
preprocess:
	poetry run python src/data/preprocess.py

## preprocessing
.PHONY: pipeline
pipeline: data 

## unit test
.PHONY: test
test:
	poetry run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

## Train all models and log to MLflow
.PHONY: train
train:
	poetry run python src/models/train.py

## EDA 
.PHONY: eda
eda:
	poetry run python src/notebooks/5-EDA.py
	poetry run python src/dashboard.py

## Full pipeline: preprocess → train → evaluate
.PHONY: full
full: preprocess train eda



## Lint using ruff (use `make format` to do formatting)
.PHONY: lint
lint:
	ruff format --check
	ruff check

## Format source code with ruff
.PHONY: format
format:
	ruff check --fix
	ruff format


