import pandas as pd
import pytest

from fklearn.training.classification import logistic_classification_learner
from fklearn.metrics.pd_extractors import evaluator_extractor
from fklearn.tuning.utils import \
    get_avg_metric_from_extractor, get_best_performing_log, get_used_features, gen_key_avgs_from_dicts, \
    gen_key_avgs_from_logs, order_feature_importance_avg_from_logs, gen_key_avgs_from_iteration, gen_dict_extract


@pytest.fixture()
def logs():
    return [
        {'train_log':
            {
                'xgb_classification_learner': {'features': ['x1', 'x2', 'x4', 'x5', 'x3', 'x6'], 'target': 'target',
                                               'prediction_column': 'prediction', 'package': 'xgboost',
                                               'package_version': '0.6',
                                               'parameters': {'objective': 'binary:logistic', 'max_depth': 3,
                                                              'min_child_weight': 0, 'lambda': 0, 'eta': 1},
                                               'feature_importance': {'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1},
                                               'training_samples': 8, 'running_time': '0.019 s'}
            },
         'validator_log': [
             {'fold_num': 0, 'eval_results': [{'roc_auc_evaluator__target': 0.8}],
              'split_log': {'train_size': 8, 'test_size': 8}},
             {'fold_num': 1, 'eval_results': [{'roc_auc_evaluator__target': 0.8}],
              'split_log': {'train_size': 8, 'test_size': 8}}],
         'used_subsets': ['first', 'second']},
        {'train_log':
            {
                'xgb_classification_learner': {'features': ['x1', 'x2', 'x4', 'x5', 'x3', 'x6'], 'target': 'target',
                                               'prediction_column': 'prediction', 'package': 'xgboost',
                                               'package_version': '0.6',
                                               'parameters': {'objective': 'binary:logistic', 'max_depth': 3,
                                                              'min_child_weight': 0, 'lambda': 0, 'eta': 1},
                                               'feature_importance': {'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1},
                                               'training_samples': 8, 'running_time': '0.019 s'}
            },
         'validator_log': [
             {'fold_num': 0, 'eval_results': [{'roc_auc_evaluator__target': 0.6}],
              'split_log': {'train_size': 8, 'test_size': 8}},
             {'fold_num': 1, 'eval_results': [{'roc_auc_evaluator__target': 0.6}],
              'split_log': {'train_size': 8, 'test_size': 8}}],
         'used_subsets': ['first', 'third']}
    ]


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


def test_get_avg_metric_from_extractor(logs, base_extractor, metric_name):
    result = get_avg_metric_from_extractor(logs[0], base_extractor, metric_name)
    assert result == 0.8


def test_get_best_performing_log(logs, base_extractor, metric_name):
    result = get_best_performing_log(logs, base_extractor, metric_name)
    assert result == logs[0]


def test_get_used_features(logs):
    result = get_used_features(logs[0])
    assert result == ['x1', 'x2', 'x4', 'x5', 'x3', 'x6']


def test_order_feature_importance_avg_from_logs(logs):
    result = order_feature_importance_avg_from_logs(logs[0])
    assert result == ['x1', 'x3', 'x5', 'x6', 'x2']


def test_gen_key_avgs_from_logs(logs):
    result = gen_key_avgs_from_logs("feature_importance", logs)
    assert result == {'x1': 8.0, 'x5': 2.0, 'x3': 3.0, 'x6': 1.0, 'x2': 1.0}


def test_gen_key_avgs_from_iteration(logs):
    result = gen_key_avgs_from_iteration("feature_importance", logs[0])
    assert result == {'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1}


def test_gen_key_avgs_from_dicts():
    result = gen_key_avgs_from_dicts(
        [{'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1}, {'x1': 9, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1}]
    )
    assert result == {'x1': 8.5, 'x5': 2.0, 'x3': 3.0, 'x6': 1.0, 'x2': 1.0}


def test_gen_dict_extract(logs):
    result = list(gen_dict_extract("feature_importance", logs[0]))
    assert result == [{'x1': 8, 'x5': 2, 'x3': 3, 'x6': 1, 'x2': 1}]
