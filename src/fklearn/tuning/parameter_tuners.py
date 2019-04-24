from collections import OrderedDict
from itertools import product
from typing import Callable, List

from numpy.random import seed
import pandas as pd
from toolz import curry, partial

from fklearn.validation.validator import parallel_validator, validator
from fklearn.types import EvalFnType, LearnerFnType, LogType, SplitterFnType, ValidatorReturnType

SaveIntermediaryFnType = Callable[[ValidatorReturnType], None]


@curry
def random_search_tuner(space: LogType,
                        train_set: pd.DataFrame,
                        param_train_fn: Callable[[LogType], LearnerFnType],
                        split_fn: SplitterFnType,
                        eval_fn: EvalFnType,
                        iterations: int,
                        random_seed: int = 1,
                        save_intermediary_fn: SaveIntermediaryFnType = None,
                        n_jobs: int = 1) -> List[ValidatorReturnType]:
    """
    Runs several training functions with each run taken from the parameter space

    Parameters
    ----------
    space : dict
        A dictionary with keys as parameter for the model and values as callable that return a parameter.
        Callable must take no parameters and can return always a constant value.
        Example::

            space = {
                'learning_rate': lambda: np.random.choice([1e-3, 1e-2, 1e-1, 1, 10]),
                'num_estimators': lambda: np.random.choice([20, 100, 150])
                }

    train_set : pd.DataFrame
        The training set

    param_train_fn : function(space, train_set) ->  p, new_df, train_log
        A curried training function that os only function of the parameters for the model and the training set.
        Example::

            @curry
            def param_train_fn(space, train_set):
                return xgb_classification_learner(features=["x"],
                                                  target="target",
                                                  learning_rate=space["learning_rate"],
                                                  num_estimators=space["num_estimators"])(train_set)

    split_fn : function(dataset) -> list of folds
        Partially defined split function that takes a dataset and returns
        a list of folds. Each fold is a Tuple of arrays. The fist array in
        each tuple contains training indexes while the second array
        contains validation indexes.
        Examples::

            out_of_time_and_space_splitter(n_splits=n_splits,
                                           in_time_limit=in_time_limit,
                                           space_column=space_column,
                                           time_column=time_column)

    eval_fn : function(dataset) -> eval_log
        A base evaluation function that returns a simple evaluation log. Can't be a spited or the extractor won't work.
        Example: auc_evaluator(target_column="target")

    iterations : int
        The number of iterations to run the parameter tuner

    random_seed : int
        Random seed

    save_intermediary_fn : function(log) -> save to file
        Partially defined saver function that receives a log result from a
        tuning step and appends it into a file
        Example: save_intermediary_result(save_path='tuning.pkl')

    n_jobs : int
        Number of parallel processes to spawn when evaluating a training function

    Returns
    ----------
    tuning_log : list of dict
        A list of tuning log, each containing a training log and a validation log.
    """
    validation_fn = partial(parallel_validator, n_jobs=n_jobs) if n_jobs > 1 else validator

    def tune_iteration() -> ValidatorReturnType:
        iter_space = {k: space[k]() for k in space}
        train_fn = param_train_fn(iter_space)
        validator_log = validation_fn(train_data=train_set, split_fn=split_fn, train_fn=train_fn, eval_fn=eval_fn)

        if save_intermediary_fn is not None:
            save_intermediary_fn(validator_log)

        return validator_log

    seed(random_seed)

    return [tune_iteration() for _ in range(iterations)]


@curry
def grid_search_cv(space: LogType,
                   train_set: pd.DataFrame,
                   param_train_fn: Callable[[LogType], LearnerFnType],
                   split_fn: SplitterFnType,
                   eval_fn: EvalFnType,
                   save_intermediary_fn: SaveIntermediaryFnType = None,
                   load_intermediary_fn: Callable[[str], List[ValidatorReturnType]] = None,
                   warm_start_file: str = None,
                   n_jobs: int = 1) -> List[ValidatorReturnType]:
    """
    Runs several training functions with each run taken from the parameter space

    Parameters
    ----------
    space : dict
        A dictionary with keys as parameter for the model and values as callable that return a parameter.
        Callable must take no parameters and can return always a constant value.
        Example::

            space = {
                'learning_rate': lambda: [1e-3, 1e-2, 1e-1, 1, 10],
                'num_estimators': lambda: [20, 100, 150]
                }

    train_set : pd.DataFrame
        The training set

    param_train_fn : function(space, train_set) ->  p, new_df, train_log
        A curried training function that os only function of the parameters for the model and the training set.
        Example::

            @curry
            def param_train_fn(space, train_set):
                return xgb_classification_learner(features=["x"],
                                                  target="target",
                                                  learning_rate=space["learning_rate"],
                                                  num_estimators=space["num_estimators"])(train_set)

    split_fn : function(dataset) -> list of folds
        Partially defined split function that takes a dataset and returns
        a list of folds. Each fold is a Tuple of arrays. The fist array in
        each tuple contains training indexes while the second array
        contains validation indexes.
        Examples::

            out_of_time_and_space_splitter(n_splits=n_splits,
                                           in_time_limit=in_time_limit,
                                           space_column=space_column,
                                           time_column=time_column)

    eval_fn : function(dataset) -> eval_log
        A base evaluation function that returns a simple evaluation log. Can't be a spited or the extractor won't work.
        Example: auc_evaluator(target_column="target")

    save_intermediary_fn : function(log) -> save to file
        Partially defined saver function that receives a log result from a
        tuning step and saves it into a file
        Example: save_intermediary_result(save_path='tuning.pkl')

    load_intermediary_fn : function(path) -> save to file
        Partially defined load function that receives a path and loads previous logs
        from this file
        Example: load_intermediary_result('tuning.pkl')

    warm_start_file: str
        File containing intermediary results for grid search. If this file
        is present, we will perform grid search from the last combination of
        parameters.

    n_jobs : int
        Number of parallel processes to spawn when evaluating a training function


    Returns
    ----------
    tuning_log : list of dict
        A list of tuning log, each containing a training log and a validation log.
    """

    validation_fn = partial(parallel_validator, n_jobs=n_jobs) if n_jobs > 1 else validator

    def tune_iteration(iter_space: LogType) -> ValidatorReturnType:
        train_fn = param_train_fn(iter_space)
        validator_log = validation_fn(train_data=train_set, split_fn=split_fn, train_fn=train_fn, eval_fn=eval_fn)
        validator_log['iter_space'] = OrderedDict(sorted(iter_space.items()))

        if save_intermediary_fn is not None:
            save_intermediary_fn(validator_log)

        return validator_log

    sorted_space_keys = sorted(space.keys())
    params = (space[k]() for k in sorted_space_keys)
    combinations = set(product(*params))

    if warm_start_file is not None and load_intermediary_fn is not None:
        results = load_intermediary_fn(warm_start_file)
        computed_combs = set([tuple(log['iter_space'].values()) for log in results])  # type: ignore
        combinations = combinations.difference(computed_combs)

    return [tune_iteration({k_v[0]: k_v[1] for k_v in zip(sorted_space_keys, comb)}) for comb in combinations]
