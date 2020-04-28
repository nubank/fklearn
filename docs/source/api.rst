===
API
===

This is a list with all relevant **fklearn** functions. Docstrings should provide enough information
in order to understand any individual function.


Preprocessing
-------------

Rebalancing (fklearn.preprocessing.rebalancing)
###############################################

.. currentmodule:: fklearn.preprocessing.rebalancing

.. autosummary::
   rebalance_by_categorical
   rebalance_by_continuous

Splitting (fklearn.preprocessing.splitting)
###########################################

.. currentmodule:: fklearn.preprocessing.splitting

.. autosummary::
   space_time_split_dataset
   time_split_dataset


Training
--------

Calibration (fklearn.training.calibration)
##########################################

.. currentmodule:: fklearn.training.calibration

.. autosummary::
   isotonic_calibration_learner

Classification (fklearn.training.classification)
################################################

.. currentmodule:: fklearn.training.classification

.. autosummary::
   lgbm_classification_learner
   logistic_classification_learner
   nlp_logistic_classification_learner
   xgb_classification_learner

Ensemble (fklearn.training.ensemble)
####################################

.. currentmodule:: fklearn.training.ensemble

.. autosummary::
   xgb_octopus_classification_learner

Imputation (fklearn.training.imputation)
########################################

.. currentmodule:: fklearn.training.imputation

.. autosummary::
   imputer
   placeholder_imputer


Pipeline (fklearn.training.pipeline)
####################################

.. currentmodule:: fklearn.training.pipeline

.. autosummary::
   build_pipeline

Regression (fklearn.training.regression)
########################################

.. currentmodule:: fklearn.training.regression

.. autosummary::
   gp_regression_learner
   lgbm_regression_learner
   linear_regression_learner
   xgb_regression_learner

Transformation (fklearn.training.transformation)
################################################

.. currentmodule:: fklearn.training.transformation

.. autosummary::
   apply_replacements
   capper
   count_categorizer
   custom_transformer
   discrete_ecdfer
   ecdfer
   floorer
   label_categorizer
   missing_warner
   null_injector
   onehot_categorizer
   prediction_ranger
   quantile_biner
   rank_categorical
   selector
   standard_scaler
   truncate_categorical
   value_mapper

Unsupervised (fklearn.training.unsupervised)
############################################

.. currentmodule:: fklearn.training.unsupervised

.. autosummary::
   isolation_forest_learner


Tuning
------

Model Agnostic Feature Choice (fklearn.tuning.model_agnostic_fc)
################################################################

.. currentmodule:: fklearn.tuning.model_agnostic_fc

.. autosummary::
   correlation_feature_selection
   variance_feature_selection

Parameter Tuning (fklearn.tuning.parameter_tuners)
##################################################

.. currentmodule:: fklearn.tuning.parameter_tuners

.. autosummary::
   grid_search_cv
   random_search_tuner
   seed

Samplers (fklearn.tuning.samplers)
##################################

.. currentmodule:: fklearn.tuning.samplers

.. autosummary::
   remove_by_feature_importance
   remove_by_feature_shuffling
   remove_features_subsets

Selectors (fklearn.tuning.selectors)
####################################

.. currentmodule:: fklearn.tuning.selectors

.. autosummary::
   backward_subset_feature_selection
   feature_importance_backward_selection
   poor_man_boruta_selection

Stoppers (fklearn.tuning.stoppers)
##################################

.. currentmodule:: fklearn.tuning.stoppers

.. autosummary::
   aggregate_stop_funcs
   stop_by_iter_num
   stop_by_no_improvement
   stop_by_no_improvement_parallel
   stop_by_num_features
   stop_by_num_features_parallel

Validation
----------

Evaluators (fklearn.validation.evaluators)
##########################################

.. currentmodule:: fklearn.validation.evaluators

.. autosummary::
   roc_auc_evaluator
   pr_auc_evaluator
   brier_score_evaluator
   combined_evaluators
   correlation_evaluator
   expected_calibration_error_evaluator
   fbeta_score_evaluator
   generic_sklearn_evaluator
   hash_evaluator
   logloss_evaluator
   mean_prediction_evaluator
   mse_evaluator
   permutation_evaluator
   precision_evaluator
   r2_evaluator
   recall_evaluator
   spearman_evaluator
   ndcg_evaluator
   split_evaluator
   temporal_split_evaluator

Splitters (fklearn.validation.splitters)
########################################

.. currentmodule:: fklearn.validation.splitters

.. autosummary::
   forward_stability_curve_time_splitter
   k_fold_splitter
   out_of_time_and_space_splitter
   reverse_time_learning_curve_splitter
   spatial_learning_curve_splitter
   stability_curve_time_in_space_splitter
   stability_curve_time_space_splitter
   stability_curve_time_splitter
   time_and_space_learning_curve_splitter
   time_learning_curve_splitter

Validator (fklearn.validation.validator)
########################################

.. currentmodule:: fklearn.validation.validator

.. autosummary::
   parallel_validator
   validator
   validator_iteration

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
