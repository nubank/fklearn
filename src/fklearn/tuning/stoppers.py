from typing import Callable

from toolz.curried import curry, take, first

from fklearn.tuning.utils import get_best_performing_log, get_avg_metric_from_extractor, get_used_features
from fklearn.types import ExtractorFnType, ListLogListType


StopFnType = Callable[[ListLogListType], bool]


def aggregate_stop_funcs(*stop_funcs: StopFnType) -> StopFnType:
    """
    Aggregate stop functions

    Parameters
    ----------
    stop_funcs: list of function list of dict -> bool

    Returns
    -------
    l: function logs -> bool
        Function that performs the Or logic of all stop_fn applied to the
        logs
    """

    def p(logs: ListLogListType) -> bool:
        return any([stop_fn(logs) for stop_fn in stop_funcs])

    return p


@curry
def stop_by_iter_num(logs: ListLogListType,
                     iter_limit: int = 50) -> bool:
    """
        Checks for logs to see if feature selection should stop

        Parameters
        ----------
        logs : list of list of dict
            A list of log-like lists of dictionaries evaluations.

        iter_limit: int (default 50)
            Limit of Iterations

        Returns
        ----------
        stop: bool
            A boolean whether to stop recursion or not
    """

    return len(logs) >= iter_limit


@curry
def stop_by_no_improvement(logs: ListLogListType,
                           extractor: ExtractorFnType,
                           metric_name: str,
                           early_stop: int = 3,
                           threshold: float = 0.001) -> bool:
    """
    Checks for logs to see if feature selection should stop

    Parameters
    ----------
    logs : list of list of dict
        A list of log-like lists of dictionaries evaluations.

    extractor: function str -> float
        A extractor that take a string and returns the value of that string on a dict

    metric_name: str
        String with the name of the column that refers to the metric column to be extracted

    early_stop: int (default 3)
        Number of iteration without improval before stopping

    threshold: float (default 0.001)
        Threshold for model performance comparison

    Returns
    ----------
    stop: bool
        A boolean whether to stop recursion or not
    """

    if len(logs) < early_stop:
        return False

    limited_logs = list(take(early_stop, logs))
    curr_auc = get_avg_metric_from_extractor(limited_logs[-1], extractor, metric_name)

    return all(
        [(curr_auc - get_avg_metric_from_extractor(log, extractor, metric_name)) <= threshold
         for log in limited_logs[:-1]]
    )


@curry
def stop_by_no_improvement_parallel(logs: ListLogListType,
                                    extractor: ExtractorFnType,
                                    metric_name: str,
                                    early_stop: int = 3,
                                    threshold: float = 0.001) -> bool:
    """
    Checks for logs to see if feature selection should stop

    Parameters
    ----------
    logs : list of list of dict
        A list of log-like lists of dictionaries evaluations.

    extractor: function str -> float
        A extractor that take a string and returns the value of that string on a dict

    metric_name: str
        String with the name of the column that refers to the metric column to be extracted

    early_stop: int (default 3)
        Number of iterations without improvements before stopping

    threshold: float (default 0.001)
        Threshold for model performance comparison

    Returns
    ----------
    stop: bool
        A boolean whether to stop recursion or not
    """

    if len(logs) < early_stop:
        return False

    log_list = [get_best_performing_log(log, extractor, metric_name) for log in logs]

    limited_logs = list(take(early_stop, log_list))
    curr_auc = get_avg_metric_from_extractor(limited_logs[-1], extractor, metric_name)

    return all(
        [(curr_auc - get_avg_metric_from_extractor(log, extractor, metric_name)) <= threshold
         for log in limited_logs[:-1]])


@curry
def stop_by_num_features(logs: ListLogListType,
                         min_num_features: int = 50) -> bool:
    """
    Checks for logs to see if feature selection should stop

    Parameters
    ----------
    logs : list of list of dict
        A list of log-like lists of dictionaries evaluations.

    min_num_features: int (default 50)
        The minimun number of features the model can have before stopping

    Returns
    -------
    stop: bool
        A boolean whether to stop recursion or not
    """

    return len(get_used_features(first(logs))) <= min_num_features


@curry
def stop_by_num_features_parallel(logs: ListLogListType,
                                  extractor: ExtractorFnType,
                                  metric_name: str,
                                  min_num_features: int = 50) -> bool:
    """
    Selects the best log out of a list to see if feature selection should stop

    Parameters
    ----------
    logs : list of list of list of dict
        A list of log-like lists of dictionaries evaluations.

    extractor: function str -> float
        A extractor that take a string and returns the value of that string on a dict

    metric_name: str
        String with the name of the column that refers to the metric column to be extracted

    min_num_features: int (default 50)
        The minimun number of features the model can have before stopping

    Returns
    ----------
    stop: bool
        A boolean whether to stop recursion or not
    """

    best_log = get_best_performing_log(first(logs), extractor, metric_name)

    return stop_by_num_features([best_log], min_num_features)
