# Changelog

## [4.0.1] - 2025-02-26
- **Bug Fix**
  - Fix fn _has_one_unfilled_arg in build_pipeline to correctly check the default value of the parameters
  - Restricts the maximum version of lightgbm to 4.5

## [4.0.0] - 2024-08-12
- **Enhancement**
  - Add support for python 3.11 and 3.10

## [3.0.0] - 2023-11-08
- **Enhancement**
  - Remove support for python 3.6 and 3.7.
  - Bumps in joblib, numpy, pandas, scikit-learn, statsmodels, toolz, catboost, lightgbm, shap, xgboost 
    and test auxiliary packages.

## [2.3.1] - 2023-04-11
- **Bugfix**
  - Remove incorrect `lightgbm` import from common paths

## [2.3.0] - 2023-03-28
- **Enhacement**
  - Bump maximum allowed `scikit-learn`
  - Move from CircleCI to Github Actions
  - Add optional `weight_column` argument for evaluators
  - Change default of `min_df` from 20 to 1 on `TfidfVectorizer`
  - Include new optional LGBM parameters to `lgbm_classification_learner`

## [2.2.1] - 2022-09-06
- **Bug Fix**
  - Including a necessary init file to allow the import of the causal cate learners.
  - Fix a docstring issue where the description of causal learners were not showing all parameters.

## [2.2.0] - 2022-08-25
- **Enhancement**
  - Including Classification S-Learner and T-Learner models to the causal cate learning library.
- **Bug Fix**
  - Fix validator behavior when receiving data containing gaps and a time based split function that
    could generate empty 
    training and testing folds and then break. 
    The argument `drop_empty_folds` can be set to `True` to drop invalid folds from validation and 
    store them in the 
    log.
- **Documentation**
  - Including Classification S-Learner and T-Learner documentation, also changing validator documentation to 
    reflect changes.

## [2.1.0] - 2022-07-25
- **Enhancement**
    - Add optional parameter `return_eval_logs_on_train` to the `validator` function,
    enabling it to return the evaluation logs for all training folds instead of just
    the first one
- **Bug Fix**
    - Fix import in `pd_extractors.py` for Python 3.10 compatibility
    - Set a minimal version of Python (3.6.2) for Fklearn
- **Documentation**
    - Fixing some typos, broken links and general improvement on the documentation

## [2.0.0] - 2021-12-28
- **Possible breaking changes**
    - Allow greater versions of `catboost`, `lightgbm`, `xgboost`, `shap`, `swifter`
    (mostly due to deprecation of support to Python 3.5 and older versions). Libraries depending on
    `fklearn` can still restrict the versions of the aforementioned libraries, keeping the previous
    behavior.

## [1.24.0] - 2021-12-06
- **New**
    - Add causal curves summary
- **Bug fix**
    - Set correct learner name for learners with column_duplicatable decorator

## [1.23.0] - 2021-10-29
- **New**
    - Add common causal evaluation techniques
    - Add methods to debias a dataframe with a treatment T and confounders X

## [1.22.2] - 2021-09-01
- **Bug fix**
    - Remove `cloudpickle` from requirements

## [1.22.1] - 2021-09-01
- **Bug fix**
    - Remove cloudpickle from parallel_validator

## [1.22.0] -  2021-02-09
- **Enhancement**
    - Add verbose method to `validator` and `parallel_validator`
    - Add column_duplicator decorator to value_mapper
- *Bug Fix*
    - Fix Spatial LC check
    - Fix circleci

## [1.21.0] - 2020-10-02
- **Enhancement**
    - Now transformers can create a new column instead of replace the input
- **Bug Fix**
    - Make requirements more flexible to cover the latest releases
    - split_evaluator_extractor now supports eval_name parameter
    - Fixed `drop_first_column` behaviour in onehot categorizer
- **New**
    - Add learner to calibrate predictions based on a fairness metric
- **Documentation**
    - Fixed docstrings for `reverse_time_learning_curve_splitter` and `feature_importance_backward_selection`

## [1.20.0] - 2020-07-13
- **Enhancement**
    - Now Catboost learner is pickable

## [1.19.0] - 2020-06-17
- **Enhancement**
    - Improve `space_time_split_dataset` performance

## [1.18.0] - 2020-05-08
- **Enhancement**
    - Allow users to inform a Placeholder value in imputer learner
- **New**
    - Add Normalized Discount Cumulative Gain evaluator
- **Bug Fix**
    - Fix some sklearn related warnings
    - Fix get_recovery logic in make_confounded_data method
- **Documentation**
    - Add target_categorizer documentation

## [1.17.0] - 2020-02-28
- **Enhancement**
    - Allow users to set a gap between training and holdout in time splitters
    - Raise Errors instead of use asserts
- **New**
    - Support pipelines with duplicated learners
    - Add stratified split method
- **Bug Fix**
    - Fix space_time_split holdout
    - Fix compatibility with newer shap version

## [1.16.1] - 2020-01-02
- **Enhancement**
    - Increasing isotonic calibration regression by adding upper and lower bounds.

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
