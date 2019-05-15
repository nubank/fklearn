from typing import Any, Dict, List, TypeVar

import pandas as pd
from toolz import curry, assoc, compose

from fklearn.training.classification import xgb_classification_learner
from fklearn.common_docstrings import learner_pred_fn_docstring, learner_return_docstring
from fklearn.types import LearnerReturnType
from fklearn.training.utils import log_learner_time

T = TypeVar('T')


@curry
@log_learner_time(learner_name='xgb_octopus_classification_learner')
def xgb_octopus_classification_learner(train_set: pd.DataFrame,
                                       learning_rate_by_bin: Dict[T, float],
                                       num_estimators_by_bin: Dict[T, int],
                                       extra_params_by_bin: Dict[T, Dict[str, Any]],
                                       features_by_bin: Dict[T, List[str]],
                                       train_split_col: str,
                                       train_split_bins: List,
                                       nthread: int,
                                       target_column: str,
                                       prediction_column: str = "prediction") -> LearnerReturnType:

    """
    Octopus ensemble allows you to inject domain specific knowledge to force a split in an initial feature, instead of
    assuming the tree model will do that intelligent split on its own. It works by first defining a split on your
    dataset and then training one individual model in each separated dataset.

    Parameters
    ----------
    train_set: pd.DataFrame
        A Pandas' DataFrame with features, target columns and a splitting column that must be categorical.

    learning_rate_by_bin: dict
        A dictionary of learning rate in the XGBoost model to use in each model split. Ex: if you want to
        split your training by tenure and you have a tenure column with integer values [1,2,3,...,12], you have to
        specify a list of learning rates for each split::

            {
                1: 0.08,
                2: 0.08,
                ...
                12: 0.1
            }

    num_estimators_by_bin: dict
        A dictionary of number of tree estimators in the XGBoost model to use in each model split. Ex: if you want to
        split your training by tenure and you have a tenure column with integer values [1,2,3,...,12], you have to
        specify a list of estimators for each split::

            {
                1: 300,
                2: 250,
                ...
                12: 300
            }

    extra_params_by_bin: dict
        A dictionary of extra parameters dictionaries in the XGBoost model to use in each model split. Ex: if you want
        to split your training by tenure and you have a tenure column with integer values [1,2,3,...,12], you have to
        specify a list of extra parameters for each split::

            {
                1: {
                    'reg_alpha': 0.0,
                    'colsample_bytree': 0.4,
                    ...
                    'colsample_bylevel': 0.8
                    }
                2: {
                    'reg_alpha': 0.1,
                    'colsample_bytree': 0.6,
                    ...
                    'colsample_bylevel': 0.4
                    }
                ...
                12: {
                    'reg_alpha': 0.0,
                    'colsample_bytree': 0.7,
                    ...
                    'colsample_bylevel': 1.0
                    }
            }

    features_by_bin: dict
        A dictionary of features to use in each model split. Ex: if you want to split your training by tenure and you
        have a tenure column with integer values [1,2,3,...,12], you have to specify a list of features for each split::

            {
                1: [feature-1, feature-2, feature-3, ...],
                2: [feature-1, feature-3, feature-5, ...],
                ...
                12: [feature-2, feature-4, feature-8, ...]
            }

    train_split_col: str
        The name of the categorical column where the model will make the splits. Ex: if you want to split your training
        by tenure, you can have a categorical column called "tenure".

    train_split_bins: list
        A list with the actual values of the categories from the `train_split_col`. Ex: if you want to split your
        training by tenure and you have a tenure column with integer values [1,2,3,...,12] you can pass this list and
        you will split your training into 12 different models.

    nthread: int
        Number of threads for the XGBoost learners.

    target_column: str
        The name of the target column.

    prediction_column: str
        The name of the column with the predictions from the model.
    """

    train_fns = {b: xgb_classification_learner(features=features_by_bin[b],
                                               learning_rate=learning_rate_by_bin[b],
                                               num_estimators=num_estimators_by_bin[b],
                                               target=target_column,
                                               extra_params=assoc(extra_params_by_bin[b], 'nthread', nthread),
                                               prediction_column=prediction_column + "_bin_" + str(b))
                 for b in train_split_bins}

    train_sets = {b: train_set[train_set[train_split_col] == b]
                  for b in train_split_bins}

    train_results = {b: train_fns[b](train_sets[b])
                     for b in train_split_bins}

    # train_results is a 3-tuple (prediction functions, predicted train dataset, train logs)
    pred_fns = {b: train_results[b][0] for b in train_split_bins}
    train_logs = {b: train_results[b][2] for b in train_split_bins}

    def p(df: pd.DataFrame) -> pd.DataFrame:
        pred_fn = compose(*pred_fns.values())

        return (pred_fn(df)
                .assign(pred_bin=prediction_column + "_bin_" + df[train_split_col].astype(str))
                .assign(prediction=lambda d: d.lookup(d.index.values,
                                                      d.pred_bin.values.squeeze()))
                .rename(index=str, columns={"prediction": prediction_column})
                .drop("pred_bin", axis=1))

    p.__doc__ = learner_pred_fn_docstring("xgb_octopus_classification_learner")

    log = {
        'xgb_octopus_classification_learner': {
            'features': features_by_bin,
            'target': target_column,
            'prediction_column': prediction_column,
            'package': "xgboost",
            'train_logs': train_logs,
            'parameters': extra_params_by_bin,
            'training_samples': len(train_set)
        }
    }

    return p, p(train_set), log


xgb_octopus_classification_learner.__doc__ += learner_return_docstring("Octopus XGB Classifier")
