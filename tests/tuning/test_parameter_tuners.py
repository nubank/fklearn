import numpy as np
import pandas as pd
from toolz import curry

from fklearn.training.classification import xgb_classification_learner
from fklearn.tuning.parameter_tuners import random_search_tuner, grid_search_cv
from fklearn.validation.evaluators import auc_evaluator
from fklearn.validation.splitters import out_of_time_and_space_splitter


def test_random_search_tuner(tmpdir):
    train_set = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id3"],
        'date': pd.to_datetime(["2016-01-01", "2016-02-01", "2016-03-01", "2016-04-01"]),
        'x': [.2, .9, .3, .3],
        'target': [0, 1, 0, 1]
    })

    eval_fn = auc_evaluator(target_column="target")

    space = {
        'learning_rate': lambda: np.random.choice([1e-3, 1e-2, 1e-1, 1, 10]),
        'num_estimators': lambda: np.random.choice([1, 2, 3])
    }

    @curry
    def param_train_fn(space, train_set):
        return xgb_classification_learner(features=["x"],
                                          target="target",
                                          learning_rate=space["learning_rate"],
                                          num_estimators=space["num_estimators"])(train_set)

    split_fn = out_of_time_and_space_splitter(n_splits=2, in_time_limit="2016-05-01",
                                              space_column="id", time_column="date")

    tuning_log = random_search_tuner(space=space,
                                     train_set=train_set,
                                     param_train_fn=param_train_fn,
                                     split_fn=split_fn,
                                     eval_fn=eval_fn,
                                     iterations=5,
                                     random_seed=42)
    assert len(tuning_log) == 5


def test_grid_search_tuner(tmpdir):
    train_set = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id3"],
        'date': pd.to_datetime(["2016-01-01", "2016-02-01", "2016-03-01", "2016-04-01"]),
        'x': [.2, .9, .3, .3],
        'target': [0, 1, 0, 1]
    })

    eval_fn = auc_evaluator(target_column="target")

    space = {
        'learning_rate': lambda: [1e-3, 1e-2, 1e-1],
        'num_estimators': lambda: [1, 2],
        'silent': lambda: [True]
    }

    @curry
    def param_train_fn(space, train_set):
        return xgb_classification_learner(features=["x"],
                                          target="target",
                                          learning_rate=space["learning_rate"],
                                          num_estimators=space["num_estimators"])(train_set)

    split_fn = out_of_time_and_space_splitter(n_splits=2, in_time_limit="2016-05-01",
                                              space_column="id", time_column="date")

    tuning_log = grid_search_cv(space=space,
                                train_set=train_set,
                                param_train_fn=param_train_fn,
                                split_fn=split_fn,
                                eval_fn=eval_fn)

    assert len(tuning_log) == 3 * 2

    space = {
        'learning_rate': lambda: [1e-3, 1e-2, 1e-1, 1],
        'num_estimators': lambda: [1, 2],
        'silent': lambda: [True]
    }

    tuning_log = grid_search_cv(space=space,
                                train_set=train_set,
                                param_train_fn=param_train_fn,
                                split_fn=split_fn,
                                eval_fn=eval_fn)

    assert len(tuning_log) == 4 * 2
