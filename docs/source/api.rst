===
API
===

This is a list with all the **fklearn** functions. Docstrings must provide enough information
in order to understand any individual function.

Datasets
--------

.. currentmodule:: fklearn.data.datasets

.. autosummary::
   fklearn.data.datasets.make_confounded_data
   fklearn.data.datasets.make_tutorial_data

Pd_extractors
-------------

.. currentmodule:: fklearn.metrics.pd_extractors

.. autosummary::

Rebalancing
-----------

.. currentmodule:: fklearn.preprocessing.rebalancing

.. autosummary::
   fklearn.preprocessing.rebalancing.rebalance_by_categorical
   fklearn.preprocessing.rebalancing.rebalance_by_continuous

Splitting
---------

.. currentmodule:: fklearn.preprocessing.splitting

.. autosummary::
   fklearn.preprocessing.splitting.space_time_split_dataset
   fklearn.preprocessing.splitting.time_split_dataset

Calibration
-----------

.. currentmodule:: fklearn.training.calibration

.. autosummary::
   fklearn.training.calibration.isotonic_calibration_learner

Classification
--------------

.. currentmodule:: fklearn.training.classification

.. autosummary::
   fklearn.training.classification.lgbm_classification_learner
   fklearn.training.classification.logistic_classification_learner
   fklearn.training.classification.nlp_logistic_classification_learner
   fklearn.training.classification.xgb_classification_learner

Ensemble
--------

.. currentmodule:: fklearn.training.ensemble

.. autosummary::
   fklearn.training.ensemble.xgb_octopus_classification_learner

Imputation
----------

.. currentmodule:: fklearn.training.imputation

.. autosummary::
   fklearn.training.imputation.imputer
   fklearn.training.imputation.placeholder_imputer


Pipeline
--------

.. currentmodule:: fklearn.training.pipeline

.. autosummary::
   fklearn.training.pipeline.build_pipeline

Regression
----------

.. currentmodule:: fklearn.training.regression

.. autosummary::
   fklearn.training.regression.gp_regression_learner
   fklearn.training.regression.lgbm_regression_learner
   fklearn.training.regression.linear_regression_learner
   fklearn.training.regression.xgb_regression_learner

Transformation
--------------

.. currentmodule:: fklearn.training.transformation

.. autosummary::
   fklearn.training.transformation.apply_replacements
   fklearn.training.transformation.capper
   fklearn.training.transformation.count_categorizer
   fklearn.training.transformation.custom_transformer
   fklearn.training.transformation.discrete_ecdfer
   fklearn.training.transformation.ecdfer
   fklearn.training.transformation.floorer
   fklearn.training.transformation.label_categorizer
   fklearn.training.transformation.missing_warner
   fklearn.training.transformation.null_injector
   fklearn.training.transformation.onehot_categorizer
   fklearn.training.transformation.prediction_ranger
   fklearn.training.transformation.quantile_biner
   fklearn.training.transformation.rank_categorical
   fklearn.training.transformation.selector
   fklearn.training.transformation.standard_scaler
   fklearn.training.transformation.truncate_categorical
   fklearn.training.transformation.value_mapper

Unsupervised
------------

.. currentmodule:: fklearn.training.unsupervised

.. autosummary::
   fklearn.training.unsupervised.isolation_forest_learner

Utils
-----

.. currentmodule:: fklearn.training.utils
.. autosummary::

Model_agnostic_fc
-----------------

.. currentmodule:: fklearn.tuning.model_agnostic_fc

.. autosummary::
   fklearn.tuning.model_agnostic_fc.correlation_feature_selection
   fklearn.tuning.model_agnostic_fc.variance_feature_selection

Parameter_tuners
----------------

.. currentmodule:: fklearn.tuning.parameter_tuners

.. autosummary::
   fklearn.tuning.parameter_tuners.grid_search_cv
   fklearn.tuning.parameter_tuners.random_search_tuner
   fklearn.tuning.parameter_tuners.seed

Samplers
--------

.. currentmodule:: fklearn.tuning.samplers

.. autosummary::
   fklearn.tuning.samplers.remove_by_feature_importance
   fklearn.tuning.samplers.remove_by_feature_shuffling
   fklearn.tuning.samplers.remove_features_subsets

Selectors
---------

.. currentmodule:: fklearn.tuning.selectors

.. autosummary::
   fklearn.tuning.selectors.backward_subset_feature_selection
   fklearn.tuning.selectors.feature_importance_backward_selection
   fklearn.tuning.selectors.poor_man_boruta_selection

Stoppers
--------

.. currentmodule:: fklearn.tuning.stoppers

.. autosummary::
   fklearn.tuning.stoppers.aggregate_stop_funcs
   fklearn.tuning.stoppers.stop_by_iter_num
   fklearn.tuning.stoppers.stop_by_no_improvement
   fklearn.tuning.stoppers.stop_by_no_improvement_parallel
   fklearn.tuning.stoppers.stop_by_num_features
   fklearn.tuning.stoppers.stop_by_num_features_parallel

Utils
-----

.. currentmodule:: fklearn.tuning.utils
.. autosummary::

Types
-----

.. currentmodule:: fklearn.types.types
.. autosummary::

Evaluators
----------

.. currentmodule:: fklearn.validation.evaluators

.. autosummary::
   fklearn.validation.evaluators.auc_evaluator
   fklearn.validation.evaluators.brier_score_evaluator
   fklearn.validation.evaluators.combined_evaluators
   fklearn.validation.evaluators.correlation_evaluator
   fklearn.validation.evaluators.expected_calibration_error_evaluator
   fklearn.validation.evaluators.fbeta_score_evaluator
   fklearn.validation.evaluators.generic_sklearn_evaluator
   fklearn.validation.evaluators.hash_evaluator
   fklearn.validation.evaluators.logloss_evaluator
   fklearn.validation.evaluators.mean_prediction_evaluator
   fklearn.validation.evaluators.mse_evaluator
   fklearn.validation.evaluators.permutation_evaluator
   fklearn.validation.evaluators.precision_evaluator
   fklearn.validation.evaluators.r2_evaluator
   fklearn.validation.evaluators.recall_evaluator
   fklearn.validation.evaluators.spearman_evaluator
   fklearn.validation.evaluators.split_evaluator
   fklearn.validation.evaluators.temporal_split_evaluator

Splitters
---------

.. currentmodule:: fklearn.validation.splitters

.. autosummary::
   fklearn.validation.splitters.forward_stability_curve_time_splitter
   fklearn.validation.splitters.k_fold_splitter
   fklearn.validation.splitters.out_of_time_and_space_splitter
   fklearn.validation.splitters.reverse_time_learning_curve_splitter
   fklearn.validation.splitters.spatial_learning_curve_splitter
   fklearn.validation.splitters.stability_curve_time_in_space_splitter
   fklearn.validation.splitters.stability_curve_time_space_splitter
   fklearn.validation.splitters.stability_curve_time_splitter
   fklearn.validation.splitters.time_and_space_learning_curve_splitter
   fklearn.validation.splitters.time_learning_curve_splitter

Validator
---------

.. currentmodule:: fklearn.validation.validator

.. autosummary::
   fklearn.validation.validator.parallel_validator
   fklearn.validation.validator.validator
   fklearn.validation.validator.validator_iteration

Definitions
-----------

.. automodule:: fklearn.data.datasets
   :members:

.. automodule:: fklearn.metrics.pd_extractors
   :members:

.. automodule:: fklearn.preprocessing.rebalancing
   :members:

.. automodule:: fklearn.preprocessing.splitting
   :members:

.. automodule:: fklearn.training.calibration
   :members:

.. automodule:: fklearn.training.classification
   :members:

.. automodule:: fklearn.training.ensemble
   :members:

.. automodule:: fklearn.training.imputation
   :members:

.. automodule:: fklearn.training.pipeline
   :members:

.. automodule:: fklearn.training.regression
   :members:

.. automodule:: fklearn.training.transformation
   :members:

.. automodule:: fklearn.training.unsupervised
   :members:

.. automodule:: fklearn.training.utils
   :members:

.. automodule:: fklearn.tuning.model_agnostic_fc
   :members:

.. automodule:: fklearn.tuning.parameter_tuners
   :members:

.. automodule:: fklearn.tuning.samplers
   :members:

.. automodule:: fklearn.tuning.selectors
   :members:

.. automodule:: fklearn.tuning.stoppers
   :members:

.. automodule:: fklearn.tuning.utils
   :members:

.. automodule:: fklearn.types.types
   :members:

.. automodule:: fklearn.validation.evaluators
   :members:

.. automodule:: fklearn.validation.splitters
   :members:

.. automodule:: fklearn.validation.validator
   :members:
