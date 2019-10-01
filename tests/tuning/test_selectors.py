import pandas as pd
import pytest

from toolz.curried import first

from fklearn.metrics.pd_extractors import evaluator_extractor
from fklearn.training.classification import logistic_classification_learner
from fklearn.tuning.utils import get_used_features
from fklearn.tuning.selectors import \
    feature_importance_backward_selection, poor_man_boruta_selection, backward_subset_feature_selection
from fklearn.validation.evaluators import roc_auc_evaluator
from fklearn.validation.splitters import k_fold_splitter

from tests import LOGS, PARALLEL_LOGS


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


@pytest.fixture()
def split_fn():
    return k_fold_splitter(n_splits=2, random_state=30)


def test_feature_importance_backward_selection(train_df, train_fn, eval_fn, split_fn, base_extractor, metric_name):
    features = ["x1", "x2", "x3", "x4", "x5", "x6"]
    logs = feature_importance_backward_selection(train_df, train_fn, features, split_fn, eval_fn,
                                                 base_extractor, metric_name,
                                                 num_removed_by_step=1, threshold=0,
                                                 early_stop=10, iter_limit=50, min_remaining_features=5)
    assert len(get_used_features(first(logs))) <= 5  # Assert stop by remaining features

    logs = feature_importance_backward_selection(train_df, train_fn, features, split_fn, eval_fn,
                                                 base_extractor, metric_name,
                                                 num_removed_by_step=1, threshold=0,
                                                 early_stop=10, iter_limit=1, min_remaining_features=3)
    assert len(logs) == 1  # Assert stop by iter limit

    logs = feature_importance_backward_selection(train_df, train_fn, features, split_fn, eval_fn, base_extractor,
                                                 metric_name,
                                                 num_removed_by_step=1, threshold=1, early_stop=2, iter_limit=50,
                                                 min_remaining_features=1)
    assert len(logs) == 2  # Assert stop by early_stop


def test_poor_man_boruta_selection(train_df, holdout_df, train_fn, eval_fn, base_extractor, metric_name):
    features = ["x1", "x2", "x3", "x4", "x5", "x6"]
    logs = poor_man_boruta_selection(train_df, holdout_df, train_fn,
                                     features,
                                     eval_fn, base_extractor, metric_name,
                                     max_removed_by_step=1, threshold=0,
                                     early_stop=10, iter_limit=50,
                                     min_remaining_features=5)

    assert len(get_used_features(first(logs))) <= 6  # Assert stop by remaining features

    logs = poor_man_boruta_selection(train_df, holdout_df,
                                     train_fn, features,
                                     eval_fn, base_extractor, metric_name,
                                     max_removed_by_step=1, threshold=0,
                                     early_stop=10, iter_limit=1,
                                     min_remaining_features=3)
    assert len(logs) == 1  # Assert stop by iter limit

    logs = poor_man_boruta_selection(train_df, holdout_df,
                                     train_fn, features,
                                     eval_fn, base_extractor, metric_name,
                                     max_removed_by_step=1, threshold=1,
                                     early_stop=2, iter_limit=50,
                                     min_remaining_features=1)
    assert len(logs) == 2  # Assert stop by early_stop


def test_backward_subset_feature_selection(train_df, train_fn, eval_fn, split_fn, base_extractor, metric_name):
    features_sets = {"first": ["x1", "x2"], "second": ["x4", "x5"], "third": ["x3", "x6"]}

    logs = backward_subset_feature_selection(train_df, train_fn, features_sets, split_fn, eval_fn, base_extractor,
                                             metric_name,
                                             num_removed_by_step=1, threshold=-1, early_stop=10, iter_limit=50,
                                             min_remaining_features=5)
    assert len(get_used_features(first(logs)[0])) <= 5  # Assert stop by remaining features

    logs = backward_subset_feature_selection(train_df, train_fn, features_sets, split_fn, eval_fn, base_extractor,
                                             metric_name,
                                             num_removed_by_step=1, threshold=0, early_stop=10, iter_limit=1,
                                             min_remaining_features=3)

    assert len(logs) == 1  # Assert stop by iter limit

    logs = backward_subset_feature_selection(train_df, train_fn, features_sets, split_fn, eval_fn, base_extractor,
                                             metric_name,
                                             num_removed_by_step=1, threshold=1, early_stop=2, iter_limit=50,
                                             min_remaining_features=1)

    assert len(logs) == 2  # Assert stop by early_stop
