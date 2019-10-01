import pytest

from fklearn.metrics.pd_extractors import evaluator_extractor
from fklearn.tuning.stoppers import \
    stop_by_iter_num, stop_by_no_improvement, stop_by_num_features, \
    stop_by_no_improvement_parallel, stop_by_num_features_parallel

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


def test_stop_by_iter_num(logs):
    assert stop_by_iter_num(logs, iter_limit=1)
    assert not stop_by_iter_num(logs, iter_limit=10)


def test_stop_by_no_improvement(logs, base_extractor, metric_name):
    assert stop_by_no_improvement(logs, base_extractor, metric_name, early_stop=2, threshold=1)
    assert not stop_by_no_improvement(logs, base_extractor, metric_name, early_stop=2, threshold=0.000001)

    assert stop_by_no_improvement(logs, base_extractor, metric_name, early_stop=2, threshold=0.4)
    assert not stop_by_no_improvement(logs, base_extractor, metric_name, early_stop=5, threshold=0.4)


def test_stop_by_num_features(logs):
    assert stop_by_num_features(logs, min_num_features=10)
    assert not stop_by_num_features(logs, min_num_features=2)


def test_stop_by_no_improvement_parallel(parallel_logs, base_extractor, metric_name):
    assert stop_by_no_improvement_parallel(parallel_logs, base_extractor, metric_name, early_stop=2, threshold=1)
    assert not stop_by_no_improvement_parallel(parallel_logs, base_extractor, metric_name, early_stop=2,
                                               threshold=0.000001)

    assert stop_by_no_improvement_parallel(parallel_logs, base_extractor, metric_name, early_stop=2, threshold=0.4)
    assert not stop_by_no_improvement_parallel(parallel_logs, base_extractor, metric_name, early_stop=5, threshold=0.4)


def test_stop_by_num_features_parallel(parallel_logs, base_extractor, metric_name):
    assert stop_by_num_features_parallel(parallel_logs, base_extractor, metric_name, min_num_features=10)
    assert not stop_by_num_features_parallel(parallel_logs, base_extractor, metric_name, min_num_features=2)
