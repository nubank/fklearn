help:  ## This help
	@awk 'BEGIN {FS = ":.*?## "} /^[a-zA-Z_-]+:.*?## / {printf "\033[36m%-20s\033[0m %s\n", $$1, $$2}' $(MAKEFILE_LIST) | sort

clean: ## Clean local environment
	@find . -name "*.pyc" | xargs rm -rf
	@find . -name "*.pyo" | xargs rm -rf
	@find . -name "__pycache__" -type d | xargs rm -rf
	@rm -f .coverage
	@rm -rf htmlcov/
	@rm -f coverage.xml
	@rm -f *.log

test: clean ## Run tests
	@python -m pytest tests/

requirements-dev:
	@pip install -qe .[test_deps]

requirements:
	@pip install -U -r requirements.txt

requirements-catboots:
	@pip install -U -r requirements_catboots.txt

requirements-demos:
	@pip install -U -r requirements_demos.txt

requirements-lgbm:
	@pip install -U -r requirements_lgbm.txt

requirements-test:
	@pip install -U -r requirements_test.txt

requirements-tools:
	@pip install -U -r requirements_tools.txt

requirements-xgboost:
	@pip install -U -r requirements_xgboost.txt
