import pandas as pd
import pytest
from tests import LOGS, PARALLEL_LOGS
from toolz.curried import first

from fklearn.metrics.pd_extractors import evaluator_extractor
from fklearn.training.classification import logistic_classification_learner
from fklearn.tuning.samplers import (remove_by_feature_importance,
                                     remove_by_feature_shuffling,
                                     remove_features_subsets)
from fklearn.validation.evaluators import roc_auc_evaluator


@pytest.fixture()
def logs():
    return LOGS


@pytest.fixture()
def parallel_logs():
    return PARALLEL_LOGS


@pytest.fixture()
def base_extractor():
    return evaluator_extractor(evaluator_name='roc_auc_evaluator__target')


@pytest.fixture()
def metric_name():
    return 'roc_auc_evaluator__target'


@pytest.fixture()
def train_df():
    df_train_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        'x3': [13.0, 10.0, 13.0, 10.0],
        "x4": [1, 1, 0, 1],
        'x5': [13.0, 10.0, 13.0, 10.0],
        "x6": [1, 1, 0, 1],
        "w": [2, 1, 2, 0.5],
        'target': [0, 1, 0, 1]
    })

    df_train_binary2 = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        'x3': [13.0, 10.0, 13.0, 10.0],
        "x4": [1, 1, 0, 1],
        'x5': [13.0, 10.0, 13.0, 10.0],
        "x6": [1, 1, 0, 1],
        "w": [2, 1, 2, 0.5],
        'target': [0, 1, 0, 1]
    })

    df_train_binary3 = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        'x3': [13.0, 10.0, 13.0, 10.0],
        "x4": [1, 1, 0, 1],
        'x5': [13.0, 10.0, 13.0, 10.0],
        "x6": [1, 1, 0, 1],
        "w": [2, 1, 2, 0.5],
        'target': [0, 1, 0, 1]
    })

    return pd.concat([df_train_binary, df_train_binary2, df_train_binary3])


@pytest.fixture()
def holdout_df():
    return pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [13.0, 10.0, 13.0, 10.0],
        "x2": [1, 1, 0, 1],
        'x3': [13.0, 10.0, 13.0, 10.0],
        "x4": [1, 1, 0, 1],
        'x5': [13.2, 10.5, 13.7, 11.0],
        "x6": [1.4, 3.2, 0, 4.6],
        "w": [1, 2, 0, 0.5],
        'target': [1, 0, 0, 1]
    })


@pytest.fixture()
def train_fn():
    return logistic_classification_learner(target="target",
                                           prediction_column="prediction",
                                           weight_column="w",
                                           params={"random_state": 52})


@pytest.fixture()
def eval_fn():
    return roc_auc_evaluator


def test_remove_by_feature_importance(logs):
    log = first(logs)
    next_features = remove_by_feature_importance(log, num_removed_by_step=2)
    assert next_features == ["x1", "x3", "x5"]


def test_remove_features_subsets(logs, base_extractor, metric_name):
    next_subsets = remove_features_subsets(logs, base_extractor, metric_name, num_removed_by_step=1)
    assert sorted(next_subsets) == [('first',), ('second',)]


def test_remove_by_shuffling(train_df, holdout_df, train_fn, eval_fn, base_extractor, metric_name, logs):
    features = ["x1", "x2", "x3", "x4", "x5", "x6"]
    predict_fn, _, train_logs = train_fn(train_df, features)
    next_features = remove_by_feature_shuffling(logs[0], predict_fn, eval_fn, holdout_df, base_extractor, metric_name,
                                                max_removed_by_step=3, threshold=0.5, speed_up_by_importance=True)

    assert sorted(next_features) == sorted(['x2', 'x5', 'x6'])

    # tests speed_up_by_importance=False
    next_features = remove_by_feature_shuffling(logs[0], predict_fn, eval_fn, holdout_df, base_extractor, metric_name,
                                                max_removed_by_step=3, threshold=0.5,
                                                speed_up_by_importance=False, parallel=True, nthread=2)

    assert sorted(next_features) == sorted(['x1', 'x2', 'x4'])
