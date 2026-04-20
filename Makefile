#################################################################################
# GLOBALS                                                                       #
#################################################################################

PROJECT_NAME = accident_severity_predictor
PYTHON_VERSION = 3.13
PYTHON_INTERPRETER = python

#################################################################################
# COMMANDS                                                                      #
#################################################################################


## Install Python dependencies
.PHONY: requirements
requirements:
	poetry install
	



## Delete all compiled Python files
.PHONY: clean
clean:
	find . -type f -name "*.py[co]" -delete
	find . -type d -name "__pycache__" -delete


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





## Set up Python interpreter environment
.PHONY: create_environment
create_environment:
	poetry env use $(PYTHON_VERSION)
	@echo ">>> Poetry environment created. Activate with: "
	@echo '$$(poetry env activate)'
	@echo ">>> Or run commands with:\npoetry run <command>"




#################################################################################
# PROJECT RULES                                                                 #
#################################################################################

## Download and validate data
.PHONY: data
data:
	poetry run python src/data/acquire.py
	poetry run python src/data/validate.py

## Preprocess data: engineer features, select, balance (SMOTE), scale
.PHONY: preprocess
preprocess:
	poetry run python src/data/preprocess.py

## Run all preprocessing steps (data + preprocess)
.PHONY: pipeline
pipeline: data preprocess

## Run unit tests with coverage
.PHONY: test
test:
	poetry run pytest tests/ -v --cov=src --cov-report=html --cov-report=term

## Train all models and log to MLflow
.PHONY: train
train:
	poetry run python src/models/train.py

## Generate EDA and evaluation report
.PHONY: eda
eda:
	poetry run python src/reports/eda.py

## Full pipeline: preprocess → train → evaluate
.PHONY: full
full: preprocess train eda

#################################################################################
# Self Documenting Commands                                                     #
#################################################################################

.DEFAULT_GOAL := help

define PRINT_HELP_PYSCRIPT
import re, sys; \
lines = '\n'.join([line for line in sys.stdin]); \
matches = re.findall(r'\n## (.*)\n[\s\S]+?\n([a-zA-Z_-]+):', lines); \
print('Available rules:\n'); \
print('\n'.join(['{:25}{}'.format(*reversed(match)) for match in matches]))
endef
export PRINT_HELP_PYSCRIPT

help:
	@$(PYTHON_INTERPRETER) -c "${PRINT_HELP_PYSCRIPT}" < $(MAKEFILE_LIST)
