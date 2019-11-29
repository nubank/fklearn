# Changelog

## [1.17.0] - 2019-11-29
- **Bug Fix**
    - Adapt classification learners to >=0.31.0 shap versions

## [1.16.0] - 2019-10-07
- **Enhancement**
    - Improve split evaluator to avoid unexpected errors
- **New**
    - Now users can install only the set of requirements they need
    - Add Target encoding learner
    - Add PR AUC and rename AUC evaluator to ROC AUC
- **Bug Fix**
    - Fix bug with space_time_split_dataset fn
- **Documentation**
    - Update space time split DOCSTRING to match the actual behaviour
    - Add more tutorials(Pydata)


## [1.15.1] - 2019-08-16
- **Enhancement**
    - Now learners that have a model exposes it in the logs as `object` key

## [1.15.0] - 2019-08-12
- **Enhancement**
    - Make `custom_transformer` a pure function
    - Remove unused requirements
- **New**
    - Now features created by one hot enconding can be used in the next steps of pipeline
    - Shap multiclass support
    - Custom model pipeline
- **Bug Fix**
    - Fix the way one hot encoding handle nans
- **Documentation**
    - Minor fix flake8 documentation to make it work in other shells
    - Fix fbeta_score_evaluator docstring
    - Fix typo on onehot_categorizer
    - New tutorial from meetup presentation

## [1.14.0] - 2019-04-30
- **Enhancement**
    - Validator accepts predict_oof as argument
- **New**
    - Add CatBoosting regressor
    - Data corruption(Macacaos)
- **Documentation**
    - Multiple fixes in the documentation
    - Add Contribution guide

## [1.13.4] - 2019-04-25
- **Enhancement**
    - Add predict_oof as argument to validator

## [1.13.3] - 2019-04-24
- **Bug Fix**
    - Fix warning in `placeholder_imputer`

## [1.13.2] - 2019-04-10
- **Bug Fix**
    - Fixing missing warner when there is no row with missing values

## [1.13.1] - 2019-04-10
- **New**
    - Add missing warner transformation

## [1.13.0] - 2019-04-01
- **New**
    - Add public version code
