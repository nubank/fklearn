# fklearn: Functional Machine Learning

![PyPI](https://img.shields.io/pypi/v/fklearn.svg?style=flat-square)
[![Documentation Status](https://readthedocs.org/projects/fklearn/badge/?version=latest)](https://fklearn.readthedocs.io/en/latest/?badge=latest)
[![Gitter](https://badges.gitter.im/fklearn-python/community.svg)](https://gitter.im/fklearn-python/community?utm_source=badge&utm_medium=badge&utm_campaign=pr-badge)
![Tests](https://github.com/nubank/fklearn/actions/workflows/push.yaml/badge.svg?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](https://opensource.org/licenses/Apache-2.0)

**fklearn** uses functional programming principles to make it easier to solve real problems with Machine Learning.

The name is a reference to the widely known [scikit-learn](https://scikit-learn.org/stable/) library.

**fklearn Principles**

1. Validation should reflect real-life situations.
2. Production models should match validated models.
3. Models should be production-ready with few extra steps.
4. Reproducibility and in-depth analysis of model results should be easy to achieve.


[Documentation](https://fklearn.readthedocs.io/en/latest/) |
[Getting Started](https://fklearn.readthedocs.io/en/latest/getting_started.html) |
[API Docs](https://fklearn.readthedocs.io/en/latest/api/modules.html) |
[Contributing](https://fklearn.readthedocs.io/en/latest/contributing.html) |


## Installation

To install via pip:

```
pip install fklearn
```

To install with optional dependencies:

```
pip install fklearn[lgbm]       # LightGBM support
pip install fklearn[xgboost]    # XGBoost support
pip install fklearn[catboost]   # CatBoost support
pip install fklearn[all_models] # All model backends
pip install fklearn[all]        # All models + tools
```

## Development with UV

fklearn uses [uv](https://docs.astral.sh/uv/) for dependency management. `uv sync`
creates a virtual environment, installs all locked dependencies, and installs
fklearn itself in **editable mode** (the default for uv projects) so changes
under `src/` are picked up without reinstalling.

### Setup
```bash
uv sync                 # core deps + dev group
uv sync --all-extras    # also installs lgbm / xgboost / catboost / tools / demos / docs
```

The `dev` dependency group (pytest, ruff, mypy, hypothesis) is included by
default via `tool.uv.default-groups`, so `uv sync` alone is enough for most
development workflows.

### Running Tests
```bash
uv run pytest --cov=src/
```

### Linting
```bash
uv run ruff check src/ tests/
uv run ruff format src/ tests/
```

### Adding Dependencies
```bash
uv add <package-name>          # runtime dependency
uv add --dev <package-name>    # dev dependency
```

### Note for Nubank contributors

Regenerate the lockfile with `--default-index https://pypi.org/simple/`:

```bash
uv lock --default-index https://pypi.org/simple/
```

## License

[Apache License 2.0](LICENSE)
