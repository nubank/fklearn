import gc
from typing import Dict, Tuple
import warnings

import cloudpickle
from joblib import Parallel, delayed
import pandas as pd
from toolz.curried import assoc, curry, dissoc, first, map, partial, pipe

from fklearn.types import EvalFnType, LearnerFnType, LogType, SplitterFnType, ValidatorReturnType


def validator_iteration(data: pd.DataFrame,
                        train_index: pd.Index,
                        test_indexes: pd.Index,
                        fold_num: int,
                        train_fn: LearnerFnType,
                        eval_fn: EvalFnType,
                        predict_oof: bool = False) -> LogType:
    """
    Perform an iteration of train test split, training and evaluation.

    Parameters
    ----------
    data : pandas.DataFrame
        A Pandas' DataFrame with training and testing subsets

    train_index : numpy.Array
        The index of the training subset of `data`.

    test_indexes : list of numpy.Array
        A list of indexes of the testing subsets of `data`.

    fold_num : int
        The number of the fold in the current iteration

    train_fn : function pandas.DataFrame -> prediction_function, predictions_dataset, logs
        A partially defined learning function that takes a training set and
        returns a predict function, a dataset with training predictions and training
        logs.

    eval_fn : function pandas.DataFrame -> dict
        A partially defined evaluation function that takes a dataset with prediction and
        returns the evaluation logs.

    predict_oof : bool
        Whether to return out of fold predictions on the logs

    Returns
    ----------
    A log-like dictionary evaluations.
    """

    train_data = data.iloc[train_index]

    empty_set_warn = "Splitter on validator_iteration in generating an empty training dataset. train_data.shape is %s" \
                     % str(train_data.shape)
    warnings.warn(empty_set_warn) if train_data.shape[0] == 0 else None  # type: ignore

    predict_fn, train_out, train_log = train_fn(train_data)

    eval_results = []
    oof_predictions = []
    for test_index in test_indexes:
        test_predictions = predict_fn(data.iloc[test_index])
        eval_results.append(eval_fn(test_predictions))
        if predict_oof:
            oof_predictions.append(test_predictions)

    logs = {'fold_num': fold_num,
            'train_log': train_log,
            'eval_results': eval_results}

    return assoc(logs, "oof_predictions", oof_predictions) if predict_oof else logs


@curry
def validator(train_data: pd.DataFrame,
              split_fn: SplitterFnType,
              train_fn: LearnerFnType,
              eval_fn: EvalFnType) -> ValidatorReturnType:
    """
    Splits the training data into folds given by the split function and
    performs a train-evaluation sequence on each fold by calling
    ``validator_iteration``.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame with training data

    split_fn : function pandas.DataFrame ->  list of tuple
        Partially defined split function that takes a dataset and returns
        a list of folds. Each fold is a Tuple of arrays. The fist array in
        each tuple contains training indexes while the second array
        contains validation indexes.

    train_fn : function pandas.DataFrame -> prediction_function, predictions_dataset, logs
        A partially defined learning function that takes a training set and
        returns a predict function, a dataset with training predictions and training
        logs.

    eval_fn : function pandas.DataFrame -> dict
        A partially defined evaluation function that takes a dataset with prediction and
        returns the evaluation logs.

    predict_oof : bool
        Whether to return out of fold predictions on the logs

    Returns
    ----------
    A list of log-like dictionary evaluations.
    """

    folds, logs = split_fn(train_data)

    def fold_iter(fold: Tuple[int, Tuple[pd.Index, pd.Index]]) -> LogType:
        (fold_num, (train_index, test_indexes)) = fold
        return validator_iteration(train_data, train_index, test_indexes, fold_num, train_fn, eval_fn)

    zipped_logs = pipe(folds,
                       enumerate,
                       map(fold_iter),
                       partial(zip, logs))

    def _join_split_log(log_tuple: Tuple[LogType, LogType]) -> Tuple[LogType, LogType]:
        train_log = {}
        split_log, validator_log = log_tuple
        train_log["train_log"] = validator_log["train_log"]
        return train_log, assoc(dissoc(validator_log, "train_log"), "split_log", split_log)

    train_logs, validator_logs = zip(*map(_join_split_log, zipped_logs))
    first_train_log = first(train_logs)
    return assoc(first_train_log, "validator_log", list(validator_logs))


def parallel_validator_iteration(train_data: pd.DataFrame,
                                 fold: Tuple[int, Tuple[pd.Index, pd.Index]],
                                 train_fn: LearnerFnType,
                                 eval_fn: EvalFnType,
                                 predict_oof: bool) -> LogType:
    (fold_num, (train_index, test_indexes)) = fold
    train_fn = cloudpickle.loads(train_fn)
    eval_fn = cloudpickle.loads(eval_fn)
    return validator_iteration(train_data, train_index, test_indexes, fold_num, train_fn, eval_fn, predict_oof)


@curry
def parallel_validator(train_data: pd.DataFrame,
                       split_fn: SplitterFnType,
                       train_fn: LearnerFnType,
                       eval_fn: EvalFnType,
                       n_jobs: int = 1,
                       predict_oof: bool = False) -> ValidatorReturnType:
    """
    Splits the training data into folds given by the split function and
    performs a train-evaluation sequence on each fold. Tries to run each
    fold in parallel using up to n_jobs processes.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame with training data

    split_fn : function pandas.DataFrame ->  list of tuple
        Partially defined split function that takes a dataset and returns
        a list of folds. Each fold is a Tuple of arrays. The fist array in
        each tuple contains training indexes while the second array
        contains validation indexes.

    train_fn : function pandas.DataFrame -> prediction_function, predictions_dataset, logs
        A partially defined learning function that takes a training set and
        returns a predict function, a dataset with training predictions and training
        logs.

    eval_fn : function pandas.DataFrame -> dict
        A partially defined evaluation function that takes a dataset with prediction and
        returns the evaluation logs.

    n_jobs : int
        Number of parallel processes to spawn.

    predict_oof : bool
        Whether to return out of fold predictions on the logs

    Returns
    ----------
    A list log-like dictionary evaluations.
    """
    folds, logs = split_fn(train_data)

    dumped_train_fn = cloudpickle.dumps(train_fn)
    dumped_eval_fn = cloudpickle.dumps(eval_fn)

    result = Parallel(n_jobs=n_jobs, backend="threading")(
        delayed(parallel_validator_iteration)(train_data, x, dumped_train_fn, dumped_eval_fn, predict_oof)
        for x in enumerate(folds))
    gc.collect()

    train_log = {"train_log": [fold_result["train_log"] for fold_result in result]}

    @curry
    def kwdissoc(d: Dict, key: str) -> Dict:
        return dissoc(d, key)

    validator_logs = pipe(result,
                          partial(zip, logs),
                          map(lambda log_tuple: assoc(log_tuple[1], "split_log", log_tuple[0])),
                          map(kwdissoc(key="train_log")),
                          list)

    return assoc(train_log, "validator_log", validator_logs)
