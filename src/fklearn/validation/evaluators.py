from typing import Any, Callable, Iterable, List

import toolz as fp
from toolz import curry
import pandas as pd
import numpy as np
from pandas.util import hash_pandas_object
from sklearn.metrics import roc_auc_score, r2_score, mean_squared_error, log_loss, precision_score, recall_score, \
    fbeta_score, brier_score_loss, mean_absolute_error

from fklearn.types import EvalFnType, EvalReturnType, PredictFnType, UncurriedEvalFnType


def generic_sklearn_evaluator(name_prefix: str, sklearn_metric: Callable[..., float]) -> UncurriedEvalFnType:
    """
    Returns an evaluator build from a metric from sklearn.metrics

    Parameters
    ----------
    name_prefix: str
        The default name of the evaluator will be name_prefix + target_column.

    sklearn_metric: Callable
        Metric function from sklearn.metrics. It should take as parameters y_true, y_score, kwargs.

    Returns
    ----------
    eval_fn: Callable
       An evaluator function that uses the provided metric
    """

    def p(test_data: pd.DataFrame,
          prediction_column: str = "prediction",
          target_column: str = "target",
          eval_name: str = None,
          **kwargs: Any) -> EvalReturnType:
        try:
            score = sklearn_metric(test_data[target_column], test_data[prediction_column], **kwargs)
        except ValueError:
            # this might happen if there's only one class in the fold
            score = np.nan

        if eval_name is None:
            eval_name = name_prefix + target_column

        return {eval_name: score}

    return p


@curry
def auc_evaluator(test_data: pd.DataFrame,
                  prediction_column: str = "prediction",
                  target_column: str = "target",
                  eval_name: str = None) -> EvalReturnType:
    """
    Computes the ROC AUC score, given true label and prediction scores.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the ROC AUC Score
    """

    eval_fn = generic_sklearn_evaluator("auc_evaluator__", roc_auc_score)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})

    return eval_fn(eval_data, prediction_column, target_column, eval_name)


@curry
def precision_evaluator(test_data: pd.DataFrame,
                        threshold: float = 0.5,
                        prediction_column: str = "prediction",
                        target_column: str = "target",
                        eval_name: str = None) -> EvalReturnType:
    """
    Computes the precision score, given true label and prediction scores.

    Parameters
    ----------
    test_data : pandas.DataFrame
        A Pandas' DataFrame with with target and prediction scores.

    threshold : float
        A threshold for the prediction column above which samples
         will be classified as 1

    prediction_column : str
        The name of the column in `test_data` with the prediction scores.

    target_column : str
        The name of the column in `test_data` with the binary target.

    eval_name : str, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Precision Score
    """
    eval_fn = generic_sklearn_evaluator("precision_evaluator__", precision_score)
    eval_data = test_data.assign(**{prediction_column: (test_data[prediction_column] > threshold).astype(int)})

    return eval_fn(eval_data, prediction_column, target_column, eval_name)


@curry
def recall_evaluator(test_data: pd.DataFrame,
                     threshold: float = 0.5,
                     prediction_column: str = "prediction",
                     target_column: str = "target",
                     eval_name: str = None) -> EvalReturnType:
    """
    Computes the recall score, given true label and prediction scores.

    Parameters
    ----------

    test_data : pandas.DataFrame
        A Pandas' DataFrame with with target and prediction scores.

    threshold : float
        A threshold for the prediction column above which samples
         will be classified as 1

    prediction_column : str
        The name of the column in `test_data` with the prediction scores.

    target_column : str
        The name of the column in `test_data` with the binary target.

    eval_name : str, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Precision Score
    """

    eval_data = test_data.assign(**{prediction_column: (test_data[prediction_column] > threshold).astype(int)})
    eval_fn = generic_sklearn_evaluator("recall_evaluator__", recall_score)

    return eval_fn(eval_data, prediction_column, target_column, eval_name)


@curry
def fbeta_score_evaluator(test_data: pd.DataFrame,
                          threshold: float = 0.5,
                          beta: float = 1.0,
                          prediction_column: str = "prediction",
                          target_column: str = "target",
                          eval_name: str = None) -> EvalReturnType:
    """
    Computes the recall score, given true label and prediction scores.

    Parameters
    ----------

    test_data : pandas.DataFrame
        A Pandas' DataFrame with with target and prediction scores.

    threshold : float
        A threshold for the prediction column above which samples
         will be classified as 1

    beta : float
        The beta parameter determines the weight of precision in the combined score.
        beta < 1 lends more weight to precision, while beta > 1 favors recall
        (beta -> 0 considers only precision, beta -> inf only recall).

    prediction_column : str
        The name of the column in `test_data` with the prediction scores.

    target_column : str
        The name of the column in `test_data` with the binary target.

    eval_name : str, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Precision Score
    """

    eval_data = test_data.assign(**{prediction_column: (test_data[prediction_column] > threshold).astype(int)})
    eval_fn = generic_sklearn_evaluator("fbeta_evaluator__", fbeta_score)

    return eval_fn(eval_data, prediction_column, target_column, eval_name, beta=beta)


@curry
def logloss_evaluator(test_data: pd.DataFrame,
                      prediction_column: str = "prediction",
                      target_column: str = "target",
                      eval_name: str = None) -> EvalReturnType:
    """
    Computes the logloss score, given true label and prediction scores.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the logloss score.
    """

    eval_fn = generic_sklearn_evaluator("logloss_evaluator__", log_loss)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})

    return eval_fn(eval_data, prediction_column, target_column, eval_name)


@curry
def brier_score_evaluator(test_data: pd.DataFrame,
                          prediction_column: str = "prediction",
                          target_column: str = "target",
                          eval_name: str = None) -> EvalReturnType:
    """
    Computes the Brier score, given true label and prediction scores.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    eval_name : String, optional (default=None)
        The name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Brier score.
    """

    eval_fn = generic_sklearn_evaluator("brier_score_evaluator__", brier_score_loss)
    eval_data = test_data.assign(**{target_column: lambda df: df[target_column].astype(int)})

    return eval_fn(eval_data, prediction_column, target_column, eval_name)


@curry
def expected_calibration_error_evaluator(test_data: pd.DataFrame,
                                         prediction_column: str = "prediction",
                                         target_column: str = "target",
                                         eval_name: str = None,
                                         n_bins: int = 100,
                                         bin_choice: str = "count") -> EvalReturnType:
    """
    Computes the expected calibration error (ECE), given true label and prediction scores.
    See "On Calibration of Modern Neural Networks"(https://arxiv.org/abs/1706.04599) for more information.

    The ECE is the distance between the actuals observed frequency and the predicted probabilities,
    for a given choice of bins.

    Perfect calibration results in a score of 0.

    For example, if for the bin [0, 0.1] we have the three data points:
      1. prediction: 0.1, actual: 0
      2. prediction: 0.05, actual: 1
      3. prediction: 0.0, actual 0

    Then the predicted average is (0.1 + 0.05 + 0.00)/3 = 0.05, and the empirical frequency is (0 + 1 + 0)/3 = 1/3.
    Therefore, the distance for this bin is::

        |1/3 - 0.05| ~= 0.28.

    Graphical intuition::

        Actuals (empirical frequency between 0 and 1)
        |     *
        |   *
        | *
         ______ Predictions (probabilties between 0 and 1)

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and prediction scores.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction scores.

    target_column : String
        The name of the column in `test_data` with the binary target.

    eval_name : String, optional (default=None)
        The name of the evaluator as it will appear in the logs.

    n_bins: Int (default=100)
        The number of bins.
        This is a trade-off between the number of points in each bin and the probability range they span.
        You want a small enough range that still contains a significant number of points for the distance to work.

    bin_choice: String (default="count")
        Two possibilities:
        "count" for equally populated bins (e.g. uses `pandas.qcut` for the bins)
        "prob" for equally spaced probabilities (e.g. uses `pandas.cut` for the bins),
        with distance weighed by the number of samples in each bin.

    Returns
    -------
    log: dict
       A log-like dictionary with the expected calibration error.
    """

    if eval_name is None:
        eval_name = "expected_calibration_error_evaluator__" + target_column

    if bin_choice == "count":
        bins = pd.qcut(test_data[prediction_column], q=n_bins)
    elif bin_choice == "prob":
        bins = pd.cut(test_data[prediction_column], bins=n_bins)
    else:
        raise AttributeError("Invalid bin_choice")

    metric_df = pd.DataFrame({"bins": bins,
                              "predictions": test_data[prediction_column],
                              "actuals": test_data[target_column]})

    agg_df = metric_df.groupby("bins").agg({"bins": "count", "predictions": "mean", "actuals": "mean"})

    sample_weight = None
    if bin_choice == "prob":
        sample_weight = agg_df["bins"].values

    distance = mean_absolute_error(agg_df["actuals"].values, agg_df["predictions"].values, sample_weight=sample_weight)

    return {eval_name: distance}


@curry
def r2_evaluator(test_data: pd.DataFrame,
                 prediction_column: str = "prediction",
                 target_column: str = "target",
                 eval_name: str = None) -> EvalReturnType:
    """
    Computes the R2 score, given true label and predictions.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and prediction.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction.

    target_column : String
        The name of the column in `test_data` with the continuous target.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the R2 Score
    """

    eval_fn = generic_sklearn_evaluator("r2_evaluator__", r2_score)

    return eval_fn(test_data, prediction_column, target_column, eval_name)


@curry
def mse_evaluator(test_data: pd.DataFrame,
                  prediction_column: str = "prediction",
                  target_column: str = "target",
                  eval_name: str = None) -> EvalReturnType:
    """
    Computes the Mean Squared Error, given true label and predictions.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and predictions.

    prediction_column : Strings
        The name of the column in `test_data` with the predictions.

    target_column : String
        The name of the column in `test_data` with the continuous target.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the MSE Score
    """
    eval_fn = generic_sklearn_evaluator("mse_evaluator__", mean_squared_error)

    return eval_fn(test_data, prediction_column, target_column, eval_name)


@curry
def mean_prediction_evaluator(test_data: pd.DataFrame,
                              prediction_column: str = "prediction",
                              eval_name: str = None) -> EvalReturnType:
    """
    Computes mean for the specified column.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with a column to compute the mean

    prediction_column : Strings
        The name of the column in `test_data` to compute the mean.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the column mean
    """

    if eval_name is None:
        eval_name = 'mean_evaluator__' + prediction_column

    return {eval_name: test_data[prediction_column].mean()}


@curry
def correlation_evaluator(test_data: pd.DataFrame,
                          prediction_column: str = "prediction",
                          target_column: str = "target",
                          eval_name: str = None) -> EvalReturnType:
    """
    Computes the Pearson correlation between prediction and target.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and prediction.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction.

    target_column : String
        The name of the column in `test_data` with the continuous target.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Pearson correlation
    """

    if eval_name is None:
        eval_name = "correlation_evaluator__" + target_column

    score = test_data[[prediction_column, target_column]].corr(method="pearson").iloc[0, 1]
    return {eval_name: score}


@curry
def spearman_evaluator(test_data: pd.DataFrame,
                       prediction_column: str = "prediction",
                       target_column: str = "target",
                       eval_name: str = None) -> EvalReturnType:
    """
    Computes the Spearman correlation between prediction and target.
    The Spearman correlation evaluates the rank order between two variables:
    https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and prediction.

    prediction_column : Strings
        The name of the column in `test_data` with the prediction.

    target_column : String
        The name of the column in `test_data` with the continuous target.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with the Spearman correlation
    """

    if eval_name is None:
        eval_name = "spearman_evaluator__" + target_column

    score = test_data[[prediction_column, target_column]].corr(method="spearman").iloc[0, 1]
    return {eval_name: score}


@curry
def combined_evaluators(test_data: pd.DataFrame,
                        evaluators: List[EvalFnType]) -> EvalReturnType:
    """
    Combine partially applies evaluation functions.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame to apply the evaluators on

    evaluators: List
        List of evaluator functions

    Returns
    ----------
    log: dict
        A log-like dictionary with the column mean
    """
    return fp.merge(e(test_data) for e in evaluators)


@curry
def split_evaluator(test_data: pd.DataFrame,
                    eval_fn: EvalFnType,
                    split_col: str,
                    split_values: Iterable = None,
                    eval_name: str = None) -> EvalReturnType:
    """
    Splits the dataset into the categories in `split_col` and evaluate
    model performance in each split. Useful when you belive the model
    performs differs in a sub population defined by `split_col`.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and predictions.

    eval_fn : function DataFrame -> Log Dict
        A partially applied evaluation function.

    split_col : String
        The name of the column in `test_data` to split by.

    split_values : Array, optional (default=None)
        An Array to split by. If not provided, `test_data[split_col].unique()`
        will be used.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    ----------
    log: dict
        A log-like dictionary with evaluation results by split.
    """
    if split_values is None:
        split_values = test_data[split_col].unique()

    if eval_name is None:
        eval_name = 'split_evaluator__' + split_col

    return {eval_name + "_" + str(value): eval_fn(test_data.loc[lambda df: df[split_col] == value])
            for value in split_values}


@curry
def temporal_split_evaluator(test_data: pd.DataFrame,
                             eval_fn: EvalFnType,
                             time_col: str,
                             time_format: str = "%Y-%m",
                             split_values: Iterable[str] = None,
                             eval_name: str = None) -> EvalReturnType:
    """
    Splits the dataset into the temporal categories by `time_col` and evaluate
    model performance in each split.

    The splits are implicitly defined by the `time_format`.
    For example, for the default time format ("%Y-%m"), we will split by year and month.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target and predictions.

    eval_fn : function DataFrame -> Log Dict
        A partially applied evaluation function.

    time_col : string
        The name of the column in `test_data` to split by.

    time_format : string
        The way to format the `time_col` into temporal categories.

    split_values : Array of string, optional (default=None)
        An array of date formatted strings to split the evaluation by.
        If not provided, all unique formatted dates will be used.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    -------
    log: dict
        A log-like dictionary with evaluation results by split.
    """

    formatted_time_col = test_data[time_col].dt.strftime(time_format)
    unique_values = formatted_time_col.unique()

    if eval_name is None:
        eval_name = 'split_evaluator__' + time_col

    if split_values is None:
        split_values = unique_values
    else:
        assert all(sv in unique_values for sv in split_values), (
            "All split values must be present in the column (after date formatting it)")

    return {eval_name + "_" + str(value): eval_fn(test_data.loc[lambda df: formatted_time_col == value])
            for value in split_values}


@curry
def permutation_evaluator(test_data: pd.DataFrame,
                          predict_fn: PredictFnType,
                          eval_fn: EvalFnType,
                          baseline: bool = True,
                          features: List[str] = None,
                          shuffle_all_at_once: bool = False,
                          random_state: int = None) -> EvalReturnType:
    """
    Permutation importance evaluator.
    It works by shuffling one or more features on test_data dataframe,
    getting the preditions with predict_fn, and evaluating the results with eval_fn.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with with target, predictions and features.

    predict_fn : function DataFrame -> DataFrame
        Function that receives the input dataframe and returns a dataframe with the pipeline predictions.

    eval_fn : function DataFrame -> Log Dict
        A partially applied evaluation function.

    baseline: bool
        Also evaluates the predict_fn on an unshuffled baseline.

    features : List of strings
        The features to shuffle and then evaluate eval_fn on the shuffled results.
        The default case shuffles all dataframe columns.

    shuffle_all_at_once: bool
        Shuffle all features at once instead of one per turn.

    random_state: int
        Seed to be used by the random number generator.

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    Returns
    -------
    log: dict
        A log-like dictionary with evaluation results by feature shuffle.
        Use the permutation_extractor for better visualization of the results.
    """

    if features is None:
        features = list(test_data.columns)

    def col_shuffler(f: str) -> np.ndarray:
        return test_data[f].sample(frac=1.0, random_state=random_state).values

    def permutation_eval(features_to_shuffle: List[str]) -> EvalReturnType:
        shuffled_cols = {f: col_shuffler(f) for f in features_to_shuffle}
        return eval_fn(predict_fn(test_data.assign(**shuffled_cols)))

    if shuffle_all_at_once:
        permutation_results = {'-'.join(features): permutation_eval(features)}
    else:
        permutation_results = {f: permutation_eval([f]) for f in features}

    feature_importance = {'permutation_importance': permutation_results}

    if baseline:
        baseline_results = {'permutation_importance_baseline': eval_fn(predict_fn(test_data))}
    else:
        baseline_results = {}

    return fp.merge(feature_importance, baseline_results)


@curry
def hash_evaluator(test_data: pd.DataFrame,
                   hash_columns: List[str] = None,
                   eval_name: str = None,
                   consider_index: bool = False) -> EvalReturnType:
    """
    Computes the hash of a pandas dataframe, filtered by hash columns. The
    purpose is to uniquely identify a dataframe, to be able to check if two
    dataframes are equal or not.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame to be hashed.

    hash_columns : List[str], optional (default=None)
        A list of column names to filter the dataframe before hashing. If None,
        it will hash the dataframe with all the columns

    eval_name : String, optional (default=None)
        the name of the evaluator as it will appear in the logs.

    consider_index: bool, optional (default=False)
        If true, will consider the index of the dataframe to calculate the hash.
        The default behaviour will ignore the index and just hash the content of
        the features.

    Returns
    -------
    log: dict
        A log-like dictionary with the hash of the dataframe
    """
    if hash_columns is None:
        hash_columns = test_data.columns

    def calculate_dataframe_hash(df: pd.DataFrame, eval_name: str) -> EvalReturnType:
        # Get the hashes per row, them sum all of them in a single value
        return {eval_name: hash_pandas_object(df).sum()}

    if eval_name is None:
        eval_name = "hash_evaluator__" + "_".join(sorted(hash_columns))
    eval_data = test_data[hash_columns]

    if not consider_index:  # set 0 for all indexes
        return calculate_dataframe_hash(eval_data.set_index(np.zeros(len(eval_data), dtype="int")), eval_name)

    return calculate_dataframe_hash(eval_data, eval_name)
