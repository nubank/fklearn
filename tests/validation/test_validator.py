import warnings

import pandas as pd
from toolz.functoolz import identity

from fklearn.validation.validator import validator_iteration, validator, parallel_validator
from fklearn.validation.perturbators import perturbator, nullify


def train_fn(df):
    def p(new_df):
        return new_df.assign(prediction=0)

    log = {'xgb_classification_learner': {
        'features': ['f1'],
        'target': 'target',
        'prediction_column': "prediction",
        'package': "xgboost",
        'package_version': "3",
        'parameters': {"a": 3},
        'feature_importance': {"f1": 1},
        'running_time': "3 ms",
        'training_samples': len(df)
    }}

    return p, p(df), log


def eval_fn(test_data):
    return {'some_score': 1.2}


def split_fn(df):
    return [([0, 1], [[2, 3], [2], [3]]),
            ([2, 3], [[0, 1]])], [{"fold": 1}, {"fold": 2}]


perturb_fn_train = identity
perturb_fn_test = perturbator(cols=['rows'], corruption_fn=nullify(perc=0.25))


data = pd.DataFrame({
    'rows': ['row1', 'row2', 'row3', 'row4']
})


def test_validator_iteration():
    train_index = [0, 1]
    test_indexes = [[2, 3]]

    result = validator_iteration(data, train_index, test_indexes, 1, train_fn, eval_fn)

    assert result['fold_num'] == 1
    assert result['train_log']['xgb_classification_learner']['features'] == ['f1']
    assert result['eval_results'][0]['some_score'] == 1.2

    # test empty dataset warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validator_iteration(data, [], test_indexes, 1, train_fn, eval_fn)
        assert len(w) == 1
        assert "train_data.shape is (0, 1)" in str(w[-1].message)


def test_validator():
    result = validator(data, split_fn, train_fn, eval_fn, perturb_fn_train, perturb_fn_test)

    validator_log = result["validator_log"]

    assert len(validator_log) == 2
    assert validator_log[0]['fold_num'] == 0
    assert result['train_log']['xgb_classification_learner']['features'] == ['f1']

    assert len(validator_log[0]['eval_results']) == 3

    assert validator_log[1]['fold_num'] == 1
    assert len(validator_log[1]['eval_results']) == 1

    perturbator_log = result["perturbator_log"]

    assert perturbator_log['perturbated_train'] == []
    assert perturbator_log['perturbated_test'] == ['rows']


def test_parallel_validator():
    result = parallel_validator(data, split_fn, train_fn, eval_fn, n_jobs=2)

    validator_log = result["validator_log"]

    assert len(validator_log) == 2
    assert validator_log[0]['fold_num'] == 0
    assert result['train_log'][0]['xgb_classification_learner']['features'] == ['f1']

    assert len(validator_log[0]['eval_results']) == 3

    assert validator_log[1]['fold_num'] == 1
    assert len(validator_log[1]['eval_results']) == 1
