import gc
from itertools import combinations
from typing import List, Tuple

import pandas as pd
from joblib import Parallel, delayed
from numpy import random
from toolz.curried import curry, first, compose, valfilter, sorted, pipe, take

from fklearn.tuning.utils import order_feature_importance_avg_from_logs, get_best_performing_log, gen_dict_extract, \
    get_avg_metric_from_extractor, get_used_features, gen_validator_log
from fklearn.types import EvalFnType, ExtractorFnType, LogListType, LogType, PredictFnType


@curry
def remove_by_feature_importance(log: LogType,
                                 num_removed_by_step: int = 5) -> List[str]:
    """
        Performs feature selection based on feature importance

        Parameters
        ----------
        log : dict
            Dictionaries evaluations.

        num_removed_by_step: int (default 5)
            The number of features to remove

        Returns
        ----------
        features: list of str
            The remaining features after removing based on feature importance

    """
    return order_feature_importance_avg_from_logs(log)[:-num_removed_by_step]


@curry
def remove_features_subsets(log_list: LogListType,
                            extractor: ExtractorFnType,
                            metric_name: str,
                            num_removed_by_step: int = 1) -> List[Tuple[str, ...]]:
    """
        Performs feature selection based on the best performing model out of
        several trained models

        Parameters
        ----------
        log_list : list of dict
            A list of log-like lists of dictionaries evaluations.

        extractor: function string -> float
            A extractor that take a string and returns the value of that string on a dict

        metric_name: str
            String with the name of the column that refers to the metric column to be extracted

        num_removed_by_step: int (default 1)
            The number of features to remove

        Returns
        ----------
        keys: list of str
            The remaining keys of feature sets after choosing the current best subset

    """

    best_log = get_best_performing_log(log_list, extractor, metric_name)
    best_subset: List[str] = first(gen_dict_extract('used_subsets', best_log))

    return list(combinations(best_subset, len(best_subset) - num_removed_by_step))


@curry
def remove_by_feature_shuffling(log: LogType,
                                predict_fn: PredictFnType,
                                eval_fn: EvalFnType,
                                eval_data: pd.DataFrame,
                                extractor: ExtractorFnType,
                                metric_name: str,
                                max_removed_by_step: int = 50,
                                threshold: float = 0.005,
                                speed_up_by_importance: bool = False,
                                parallel: bool = False,
                                nthread: int = 1,
                                seed: int = 7) -> List[str]:

    """
        Performs feature selection based on the evaluation of the test vs the
        evaluation of the test with randomly shuffled features

        Parameters
        ----------
        log : LogType
            Dictionaries evaluations.

        predict_fn: function pandas.DataFrame -> pandas.DataFrame
            A partially defined predictor that takes a DataFrame and returns the
            predicted score for this dataframe

        eval_fn : function DataFrame -> log dict
            A partially defined evaluation function that takes a dataset with prediction and
            returns the evaluation logs.

        eval_data: pandas.DataFrame
            Data used to evaluate the model after shuffling

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

        speed_up_by_importance: bool (default True)
            If it should narrow search looking at feature importance first before getting PIMP importance. If True,
            will only shuffle the top num_removed_by_step in terms of feature importance.

        parallel: bool (default False)

        nthread: int (default 1)

        seed: int (default 7)
            Random seed

        Returns
        ----------
        features: list of str
            The remaining features after removing based on feature importance

    """
    random.seed(seed)

    curr_metric = get_avg_metric_from_extractor(log, extractor, metric_name)
    eval_size = eval_data.shape[0]

    features_to_shuffle = order_feature_importance_avg_from_logs(log)[-max_removed_by_step:] \
        if speed_up_by_importance else get_used_features(log)

    def shuffle(feature: str) -> pd.DataFrame:
        return eval_data.assign(**{feature: eval_data[feature].sample(frac=1.0)})

    feature_to_delta_metric = compose(lambda m: curr_metric - m,
                                      get_avg_metric_from_extractor(extractor=extractor, metric_name=metric_name),
                                      gen_validator_log(fold_num=0, test_size=eval_size), eval_fn, predict_fn, shuffle)

    if parallel:
        metrics = Parallel(n_jobs=nthread, backend="threading")(
            delayed(feature_to_delta_metric)(feature) for feature in features_to_shuffle)
        feature_to_delta_metric = dict(zip(features_to_shuffle, metrics))
        gc.collect()

    else:
        feature_to_delta_metric = {feature: feature_to_delta_metric(feature) for feature in features_to_shuffle}

    return pipe(feature_to_delta_metric,
                valfilter(lambda delta_metric: delta_metric < threshold),
                sorted(key=lambda f: feature_to_delta_metric.get(f)),
                take(max_removed_by_step),
                list)
