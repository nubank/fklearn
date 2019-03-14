from itertools import combinations
from typing import List, Tuple

import pandas as pd
from toolz.curried import curry, first

from fklearn.types import EvalFnType, ExtractorFnType, LogListType, LogType, PredictFnType
from fklearn.tuning.utils import order_feature_importance_avg_from_logs, get_best_performing_log, gen_dict_extract, \
    get_avg_metric_from_extractor


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
                                num_removed_by_step: int = 5,
                                threshold: float = 0.005,
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

        num_removed_by_step: int (default 5)
            The number of features to remove

        threshold: float (default 0.005)
            Threshold for model performance comparison

        seed: int (default 7)
            Random seed

        Returns
        ----------
        features: list of str
            The remaining features after removing based on feature importance

    """
    features_to_shuffle = order_feature_importance_avg_from_logs(log)[-num_removed_by_step:]

    df_shadow = eval_data.assign(**{feature: eval_data[feature].sample(frac=1.0, random_state=seed)
                                    for feature in features_to_shuffle})

    eval_log = eval_fn(predict_fn(df_shadow))
    shuffled_logs = {
        'validator_log': [{'fold_num': 0, 'split_log': {'test_size': df_shadow.shape[0]}, 'eval_results': [eval_log]}]}

    curr_auc = get_avg_metric_from_extractor(log, extractor, metric_name)

    shuffled_auc = get_avg_metric_from_extractor(shuffled_logs, extractor, metric_name)

    next_features = order_feature_importance_avg_from_logs(
        log) if curr_auc - shuffled_auc <= threshold else order_feature_importance_avg_from_logs(
            log)[:-num_removed_by_step]

    return next_features
