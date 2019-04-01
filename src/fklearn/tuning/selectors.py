from typing import Callable, Dict, List

from toolz.curried import pipe, first, mapcat
import pandas as pd

from fklearn.tuning.samplers import remove_features_subsets, remove_by_feature_importance, remove_by_feature_shuffling
from fklearn.tuning.stoppers import stop_by_num_features, stop_by_num_features_parallel, stop_by_iter_num, \
    stop_by_no_improvement, stop_by_no_improvement_parallel, aggregate_stop_funcs
from fklearn.validation.validator import parallel_validator
from fklearn.types import EvalFnType, ExtractorFnType, LearnerReturnType, ListLogListType, LogListType, SplitterFnType,\
    ValidatorReturnType, LogType

SaveIntermediaryFnType = Callable[[List[ValidatorReturnType]], None]
TuningLearnerFnType = Callable[[pd.DataFrame, List[str]], LearnerReturnType]


def feature_importance_backward_selection(train_data: pd.DataFrame,
                                          param_train_fn: TuningLearnerFnType,
                                          features: List[str],
                                          split_fn: SplitterFnType,
                                          eval_fn: EvalFnType,
                                          extractor: ExtractorFnType,
                                          metric_name: str,
                                          num_removed_by_step: int = 5,
                                          threshold: float = 0.005,
                                          early_stop: int = 2,
                                          iter_limit: int = 50,
                                          min_remaining_features: int = 50,
                                          save_intermediary_fn: SaveIntermediaryFnType = None,
                                          n_jobs: int = 1) -> ListLogListType:
    """
        Performs train-evaluation iterations while subsampling the used features
        to compute statistics about feature relevance

        Parameters
        ----------
        train_data : pandas.DataFrame
            A Pandas' DataFrame with training data

        auxiliary_columns: list of str
            List of columns from the dataset that are not used as features but are
            used for evaluation or cross validation. (id, date, etc)

        param_train_fn : function (DataFrame, List of Strings) -> prediction_function, predictions_dataset, logs
            A partially defined learning function that takes a training set and a feature list and
            returns a predict function, a dataset with training predictions and training
            logs.

        features: list of str
            Elements must be columns of the train_data

        split_fn : function pandas.DataFrame ->  list of tuple
            Partially defined split function that takes a dataset and returns
            a list of folds. Each fold is a Tuple of arrays. The fist array in
            each tuple contains training indexes while the second array
            contains validation indexes.

        eval_fn : function pandas.DataFrame -> dict
            A partially defined evaluation function that takes a dataset with prediction and
            returns the evaluation logs.

        extractor: function str -> float
            A extractor that take a string and returns the value of that string on a dict

        metric_name: str
            String with the name of the column that refers to the metric column to be extracted

        num_removed_by_step: int (default 5)
            Number of features removed at each iteration

        threshold: float (default 0.005)
            Threshold for model performance comparison

        early_stop: int (default 2)
            Number of rounds without improvement before stopping process

        iter_limit: int (default 50)
            Maximum number of iterations before stopping

        min_remaining_features: int (default 50)
            Minimum number of features that should remain in the model,
            combining num_removed_by_step and iter_limit accomplishes the same
            functionality as this parameter.

        save_intermediary_fn : function(log) -> save to file
            Partially defined saver function that receives a log result from a
            tuning step and appends it into a file
            Example: save_intermediary_result(save_path='tuning.pkl')

        n_jobs : int
            Number of parallel processes to spawn.

        Returns
        ----------
        Logs: list of list of dict
            A list log-like lists of dictionaries evaluations. Each element of the
            list is validation step of the algorithm.

    """

    selector_fn = remove_by_feature_importance(num_removed_by_step=num_removed_by_step)

    stop_fn = aggregate_stop_funcs(
        stop_by_no_improvement(extractor=extractor, metric_name=metric_name, early_stop=early_stop,
                               threshold=threshold),
        stop_by_iter_num(iter_limit=iter_limit),
        stop_by_num_features(min_num_features=min_remaining_features))

    train_fn = lambda df: param_train_fn(df, features)
    first_logs = parallel_validator(train_data, split_fn, train_fn, eval_fn, n_jobs=n_jobs)

    logs = [first_logs]
    while not stop_fn(logs):
        curr_log = first(logs)

        new_features = selector_fn(curr_log)
        new_train_fn = lambda df: param_train_fn(df, new_features)
        next_log = parallel_validator(train_data, split_fn, new_train_fn, eval_fn, n_jobs=n_jobs)

        if save_intermediary_fn is not None:
            save_intermediary_fn(next_log)

        logs = [next_log] + logs

    return logs


def poor_man_boruta_selection(train_data: pd.DataFrame,
                              test_data: pd.DataFrame,
                              param_train_fn: TuningLearnerFnType,
                              features: List[str],
                              eval_fn: EvalFnType,
                              extractor: ExtractorFnType,
                              metric_name: str,
                              max_removed_by_step: int = 5,
                              threshold: float = 0.005,
                              early_stop: int = 2,
                              iter_limit: int = 50,
                              min_remaining_features: int = 50,
                              save_intermediary_fn: Callable[[LogType], None] = None,
                              speed_up_by_importance: bool = False,
                              parallel: bool = False,
                              nthread: int = 1,
                              seed: int = 7) -> LogListType:
    """
        Performs train-evaluation iterations while shuffiling the used features
        to compute statistics about feature relevance

        Parameters
        ----------
        train_data : pandas.DataFrame
            A Pandas' DataFrame with training data

        test_data : pandas.DataFrame
            A Pandas' DataFrame with test data

        param_train_fn : function (pandas.DataFrame, list of str) -> prediction_function, predictions_dataset, logs
            A partially defined AND curried learning function that takes a training set and a feature list and
            returns a predict function, a dataset with training predictions and training
            logs.

        features: list of str
            Elements must be columns of the train_data

        eval_fn : function pandas.DataFrame -> dict
            A partially defined evaluation function that takes a dataset with prediction and
            returns the evaluation logs.

        extractor: function str -> float
            A extractor that take a string and returns the value of that string on a dict

        metric_name: str
            String with the name of the column that refers to the metric column to be extracted

        max_removed_by_step: int (default 5)
            The maximum number of features to remove. It will only consider the least max_removed_by_step in terms of
            feature importance. If speed_up_by_importance=True it will first filter the least relevant feature an
            shuffle only those. If speed_up_by_importance=False it will shuffle all features and drop the last
            max_removed_by_step in terms of PIMP. In both cases, the features will only be removed if drop in
            performance is up to the defined threshold.

        threshold: float (default 0.005)
            Threshold for model performance comparison

        early_stop: int (default 2)
            Number of rounds without improvement before stopping process

        iter_limit: int (default 50)
            Maximum number of iterations before stopping

        min_remaining_features: int (default 50)
            Minimum number of features that should remain in the model,
            combining num_removed_by_step and iter_limit accomplishes the same
            functionality as this parameter.

        save_intermediary_fn: function(log) -> save to file
            Partially defined saver function that receives a log result from a
            tuning step and appends it into a file
            Example: save_intermediary_result(save_path='tuning.pkl')

        speed_up_by_importance: bool (default True)
            If it should narrow search looking at feature importance first before getting PIMP importance. If True,
            will only shuffle the top num_removed_by_step in terms of feature importance.

        max_removed_by_step: int (default 50)
            If speed_up_by_importance=False, this will limit the number of features dropped by iteration. It will only
            drop the max_removed_by_step features that decrease the metric by the least when dropped.

        parallel: bool (default False)
            Run shuffling and prediction in parallel. Only applies if speed_up_by_importance=False

        nthread: int (default 1)
            Number of threads to run predictions. ONly applied if speed_up_by_importance=False

        seed: int (default 7)
            random state for consistency.


        Returns
        ----------
        logs: list of list of dict
            A list log-like lists of dictionaries evaluations. Each element of the
            list is validation step of the algorithm.

    """

    selector_fn = remove_by_feature_shuffling(eval_fn=eval_fn,
                                              eval_data=test_data,
                                              extractor=extractor,
                                              metric_name=metric_name,
                                              max_removed_by_step=max_removed_by_step,
                                              threshold=threshold,
                                              speed_up_by_importance=speed_up_by_importance,
                                              parallel=parallel,
                                              nthread=nthread,
                                              seed=seed)

    stop_fn = aggregate_stop_funcs(
        stop_by_no_improvement(extractor=extractor, metric_name=metric_name, early_stop=early_stop,
                               threshold=threshold),
        stop_by_iter_num(iter_limit=iter_limit),
        stop_by_num_features(min_num_features=min_remaining_features)
    )

    predict_fn_first, _, train_logs = param_train_fn(train_data, features)
    eval_logs = eval_fn(predict_fn_first(test_data))

    first_logs = {
        'train_log': train_logs,
        'validator_log': [
            {
                'fold_num': 0,
                'split_log': {
                    'train_size': train_data.shape[0],
                    'test_size': test_data.shape[0]
                },
                'eval_results': [eval_logs]
            }
        ]
    }

    logs = [first_logs]
    predict_fn = predict_fn_first

    while not stop_fn(logs):  # type: ignore
        next_features = pipe(logs, first, selector_fn(predict_fn=predict_fn))

        if len(next_features) == 0:
            break

        next_predict_fn, _, next_train_logs = param_train_fn(train_data, next_features)

        eval_logs = pipe(test_data, next_predict_fn, eval_fn)
        next_log = {'train_log': next_train_logs, 'validator_log': [
            {'fold_num': 0, 'split_log': {'train_size': train_data.shape[0], 'test_size': test_data.shape[0]},
             'eval_results': [eval_logs]}]}

        logs = [next_log] + logs

        if save_intermediary_fn is not None:
            save_intermediary_fn(next_log)

        predict_fn = next_predict_fn

    return logs


def backward_subset_feature_selection(train_data: pd.DataFrame,
                                      param_train_fn: TuningLearnerFnType,
                                      features_sets: Dict[str, List[str]],
                                      split_fn: SplitterFnType,
                                      eval_fn: EvalFnType,
                                      extractor: ExtractorFnType,
                                      metric_name: str,
                                      threshold: float = 0.005,
                                      num_removed_by_step: int = 3,
                                      early_stop: int = 2,
                                      iter_limit: int = 50,
                                      min_remaining_features: int = 50,
                                      save_intermediary_fn: SaveIntermediaryFnType = None,
                                      n_jobs: int = 1) -> ListLogListType:
    """
        Performs train-evaluation iterations while testing the subsets of features
        to compute statistics about the importance of each feature category

        Parameters
        ----------
        train_data : pandas.DataFrame
            A Pandas' DataFrame with training data

        param_train_fn : function (pandas.DataFrame, list of str) -> prediction_function, predictions_dataset, logs
            A partially defined learning function that takes a training set and a feature list and
            returns a predict function, a dataset with training predictions and training
            logs.

        features_sets: dict of string -> list
            Each String Key on the dict is a subset of columns from the dataset, the function will
            analyse the influence of each group of features on the model performance

        split_fn : function pandas.DataFrame ->  list of tuple
            Partially defined split function that takes a dataset and returns
            a list of folds. Each fold is a Tuple of arrays. The fist array in
            each tuple contains training indexes while the second array
            contains validation indexes.

        eval_fn : function pandas.DataFrame -> dict
            A partially defined evaluation function that takes a dataset with prediction and
            returns the evaluation logs.

        extractor: function str -> float
            A extractor that take a string and returns the value of that string on a dict

        metric_name: str
            String with the name of the column that refers to the metric column to be extracted

        num_removed_by_step: int (default 3)
            Number of features removed at each iteration

        threshold: float (default 0.005)
            Threshold for model performance comparison

        early_stop: int (default 2)
            Number of rounds without improvement before stopping process

        iter_limit: int (default 50)
            Maximum number of iterations before stopping

        min_remaining_features: int (default 50)
            Minimum number of features that should remain in the model,
            combining num_removed_by_step and iter_limit accomplishes the same
            functionality as this parameter.

        save_intermediary_fn : function(log) -> save to file
            Partially defined saver function that receives a log result from a
            tuning step and appends it into a file
            Example: save_intermediary_result(save_path='tuning.pkl')

        n_jobs : int
            Number of parallel processes to spawn.

        Returns
        ----------
        logs: list of list of dict
            A list log-like lists of dictionaries evaluations. Each element of the
            list is validation step of the algorithm.

    """

    selector_fn = remove_features_subsets(extractor=extractor,
                                          metric_name=metric_name,
                                          num_removed_by_step=num_removed_by_step)

    stop_fn = aggregate_stop_funcs(
        stop_by_no_improvement_parallel(extractor=extractor, metric_name=metric_name, early_stop=early_stop,
                                        threshold=threshold),
        stop_by_iter_num(iter_limit=iter_limit),
        stop_by_num_features_parallel(extractor=extractor, metric_name=metric_name,
                                      min_num_features=min_remaining_features)
    )

    used_subsets = [features_sets.keys()]

    used_features = [list(mapcat(lambda key: features_sets[key], subset)) for subset in used_subsets]

    trainers = [lambda df: param_train_fn(df, feat) for feat in used_features]

    first_val_logs = [parallel_validator(train_data, split_fn, train_func, eval_fn, n_jobs) for train_func in trainers]
    logs = [[dict(log, **{"used_subsets": list(subset)}) for log, subset in zip(first_val_logs, used_subsets)]]

    while not stop_fn(logs):
        curr_log = first(logs)

        new_subsets = selector_fn(curr_log)
        new_features = [list(mapcat(lambda key: features_sets[key], subset)) for subset in new_subsets]

        trainers = [lambda df: param_train_fn(df, feat) for feat in new_features]

        val_logs = [parallel_validator(train_data, split_fn, train_func, eval_fn, n_jobs) for train_func in trainers]

        new_logs = [dict(log, **{"used_subsets": subset}) for log, subset in zip(val_logs, new_subsets)]

        if save_intermediary_fn is not None:
            save_intermediary_fn(new_logs)

        logs = [new_logs] + logs

    return logs
