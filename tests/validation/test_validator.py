import warnings
from datetime import datetime, timedelta

import pandas as pd
from toolz.functoolz import identity

from fklearn.training.classification import lgbm_classification_learner
from fklearn.validation import splitters, evaluators
from fklearn.validation.validator import (
    validator_iteration,
    validator,
    parallel_validator,
)
from fklearn.validation.perturbators import perturbator, nullify
import pytest


def train_fn(df):
    def p(new_df):
        return new_df.assign(prediction=0)

    log = {
        "xgb_classification_learner": {
            "features": ["f1"],
            "target": "target",
            "prediction_column": "prediction",
            "package": "xgboost",
            "package_version": "3",
            "parameters": {"a": 3},
            "feature_importance": {"f1": 1},
            "running_time": "3 ms",
            "training_samples": len(df),
        }
    }

    return p, p(df), log


def eval_fn(test_data):
    return {"some_score": 1.2}


def split_fn(df):
    return [([0, 1], [[2, 3], [2], [3]]), ([2, 3], [[0, 1]])], [
        {"fold": 1},
        {"fold": 2},
    ]


perturb_fn_train = identity
perturb_fn_test = perturbator(cols=["rows"], corruption_fn=nullify(perc=0.25))


@pytest.fixture
def data():
    return pd.DataFrame({"rows": ["row1", "row2", "row3", "row4"]})


def test_validator_iteration(data):
    train_index = [0, 1]
    test_indexes = [[2, 3]]

    result = validator_iteration(data, train_index, test_indexes, 1, train_fn, eval_fn)

    assert result["fold_num"] == 1
    assert result["train_log"]["xgb_classification_learner"]["features"] == ["f1"]
    assert result["eval_results"][0]["some_score"] == 1.2

    # test return_eval_fn_on_train=True
    result = validator_iteration(
        data, train_index, test_indexes, 1, train_fn, eval_fn, False, True
    )
    assert result["train_log"]["eval_results"]["some_score"] == 1.2

    # test empty dataset warning
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        validator_iteration(data, [], test_indexes, 1, train_fn, eval_fn)
        assert len(w) == 1
        assert "train_data.shape is (0, 1)" in str(w[-1].message)


def test_validator():
    model = lgbm_classification_learner(
        target="target", features=["feat1", "feat2"], extra_params={"verbose": -1}
    )

    df_no_gap = pd.DataFrame(
        {
            "feat1": [1, 2, 1, 5, 2, 1, 5, 6, 8, 1, 7, 5, 1, 2],
            "feat2": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            "target": [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
            "date": [
                datetime.strptime(d, "%Y-%m-%d")
                for d in [
                    "2021-01-01",
                    "2021-01-09",
                    "2021-02-08",
                    "2021-02-10",
                    "2021-03-20",
                    "2021-03-11",
                    "2021-04-01",
                    "2021-04-02",
                    "2021-05-15",
                    "2021-05-15",
                    "2021-06-01",
                    "2021-06-02",
                    "2021-07-15",
                    "2021-07-15",
                ]
            ],
        }
    )

    split_fn_no_gap = splitters.forward_stability_curve_time_splitter(
        training_time_start=df_no_gap["date"].min(),
        training_time_end=df_no_gap["date"].min() + timedelta(60),
        time_column="date",
        step=timedelta(30),
        holdout_size=timedelta(30),
    )

    eval_fn_no_gap = evaluators.roc_auc_evaluator(
        prediction_column="prediction", target_column="target"
    )

    perturb_fn_train_no_gap = identity
    perturb_fn_test_no_gap = perturbator(
        cols=["feat1"], corruption_fn=nullify(perc=0.25)
    )

    result_no_gap = validator(
        df_no_gap,
        split_fn_no_gap,
        model,
        eval_fn_no_gap,
        perturb_fn_train_no_gap,
        perturb_fn_test_no_gap,
    )

    validator_log = result_no_gap["validator_log"]

    assert len(validator_log[1]) == 4
    assert validator_log[0]["fold_num"] == 0
    assert result_no_gap["train_log"]["lgbm_classification_learner"]["features"] == [
        "feat1",
        "feat2",
    ]

    assert len(validator_log[0]["eval_results"]) == 1

    assert validator_log[1]["fold_num"] == 1
    assert len(validator_log[1]["eval_results"]) == 1

    perturbator_log = result_no_gap["perturbator_log"]

    assert perturbator_log["perturbated_train"] == []
    assert perturbator_log["perturbated_test"] == ["feat1"]


def test_validator_with_gap():
    model = lgbm_classification_learner(
        target="target", features=["feat1", "feat2"], extra_params={"verbose": -1}
    )

    df_gap = pd.DataFrame(
        {
            "feat1": [1, 2, 1, 5, 2, 1, 5, 6, 8, 1, 7, 5, 1, 2],
            "feat2": [1, 0, 1, 0, 0, 1, 0, 1, 0, 0, 0, 0, 1, 0],
            "target": [1, 0, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 1, 0],
            "date": [
                datetime.strptime(d, "%Y-%m-%d")
                for d in [
                    "2021-01-01",
                    "2021-01-09",
                    "2021-02-08",
                    "2021-02-10",
                    "2021-03-20",
                    "2021-03-11",
                    "2021-07-01",
                    "2021-07-02",
                    "2021-09-15",
                    "2021-09-15",
                    "2021-10-01",
                    "2021-10-02",
                    "2021-11-15",
                    "2021-11-15",
                ]
            ],
        }
    )

    split_fn_gap = splitters.forward_stability_curve_time_splitter(
        training_time_start=df_gap["date"].min(),
        training_time_end=df_gap["date"].min() + timedelta(60),
        time_column="date",
        step=timedelta(30),
        holdout_size=timedelta(30),
    )

    eval_fn_gap = evaluators.roc_auc_evaluator(
        prediction_column="prediction", target_column="target"
    )

    with pytest.raises(Exception):
        validator(df_gap, split_fn_gap, model, eval_fn_gap, drop_empty_folds=False)

    validator_log = validator(
        df_gap, split_fn_gap, model, eval_fn_gap, drop_empty_folds=True
    )["validator_log"]

    assert len(validator_log[0]) == 3
    assert len(validator_log[0]["eval_results"]) == 1


def test_parallel_validator(data):
    result = parallel_validator(data, split_fn, train_fn, eval_fn, n_jobs=2)

    validator_log = result["validator_log"]

    assert len(validator_log) == 2
    assert validator_log[0]["fold_num"] == 0
    assert result["train_log"][0]["xgb_classification_learner"]["features"] == ["f1"]

    assert len(validator_log[0]["eval_results"]) == 3

    assert validator_log[1]["fold_num"] == 1
    assert len(validator_log[1]["eval_results"]) == 1
