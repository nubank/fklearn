from typing import List

import numpy as np
import pandas as pd
from toolz import curry, partial

from fklearn.types import EffectFnType, PartitionFnType
from fklearn.causal.effects import linear_effect
from fklearn.causal.partitioners import quantile_partitioner


@curry
def effect_by_segment(df: pd.DataFrame,
                      treatment: str,
                      outcome: str,
                      prediction: str,
                      segments: int = 10,
                      effect_fn: EffectFnType = linear_effect,
                      partition_fn: PartitionFnType = quantile_partitioner) -> pd.Series:
    """
    Segments the dataset using a partition function and estimates the treatment effect by segment.
    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.
    treatment : Strings
        The name of the treatment column in `df`.
    outcome : Strings
        The name of the outcome column in `df`.
    prediction : Strings
        The name of the prediction column in `df`.
    segments : Integer
        The number of the segments to create. Uses Pandas' qcut under the hood.
    effect_fn : function (df: pandas.DataFrame, treatment: str, outcome: str) -> int or Array of int
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.
    partition_fn : function (series: pandas.Series, segments: int) -> Array
        A function that returns an array of bins to be used to partition the prediction column.


    Returns
    ----------
    effect by band : Pandas' Series
        The effect stored in a Pandas' series were the indexes are the segments
    """

    effect_fn_partial = partial(effect_fn, treatment_column=treatment, outcome_column=outcome)

    bins = partition_fn(df[prediction], segments=segments)

    return (df
            .assign(**{f"{prediction}_band": pd.cut(df[prediction], bins=bins, include_lowest=True)})
            .groupby(f"{prediction}_band")
            .apply(effect_fn_partial))


@curry
def cumulative_effect_curve(df: pd.DataFrame,
                            treatment: str,
                            outcome: str,
                            prediction: str,
                            min_rows: int = 30,
                            steps: int = 100,
                            effect_fn: EffectFnType = linear_effect) -> np.ndarray:
    """
    Orders the dataset by prediction and computes the cumulative effect curve according to that ordering

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : Strings
        The name of the treatment column in `df`.

    outcome : Strings
        The name of the outcome column in `df`.

    prediction : Strings
        The name of the prediction column in `df`.

    min_rows : Integer
        Minimum number of observations needed to have a valid result.

    steps : Integer
        The number of cumulative steps to iterate when accumulating the effect

    effect_fn : function (df: pandas.DataFrame, treatment: str, outcome: str) -> int or Array of int
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.


    Returns
    ----------
    cumulative effect curve: Numpy's Array
        The cumulative treatment effect according to the predictions ordering.
    """

    size = df.shape[0]
    ordered_df = df.sort_values(prediction, ascending=False).reset_index(drop=True)
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    return np.array([effect_fn(ordered_df.head(rows), treatment, outcome) for rows in n_rows])


@curry
def cumulative_gain_curve(df: pd.DataFrame,
                          treatment: str,
                          outcome: str,
                          prediction: str,
                          min_rows: int = 30,
                          steps: int = 100,
                          effect_fn: EffectFnType = linear_effect) -> np.ndarray:
    """
    Orders the dataset by prediction and computes the cumulative gain (effect * proportional sample size) curve
     according to that ordering.

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : Strings
        The name of the treatment column in `df`.

    outcome : Strings
        The name of the outcome column in `df`.

    prediction : Strings
        The name of the prediction column in `df`.

    min_rows : Integer
        Minimum number of observations needed to have a valid result.

    steps : Integer
        The number of cumulative steps to iterate when accumulating the effect

    effect_fn : function (df: pandas.DataFrame, treatment: str, outcome: str) -> int or Array of int
        A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
        of the outcome column.


    Returns
    ----------
    cumulative gain curve: float
        The cumulative gain according to the predictions ordering.
    """

    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]

    cum_effect = cumulative_effect_curve(df=df, treatment=treatment, outcome=outcome, prediction=prediction,
                                         min_rows=min_rows, steps=steps, effect_fn=effect_fn)

    return np.array([effect * (rows / size) for rows, effect in zip(n_rows, cum_effect)])


@curry
def relative_cumulative_gain_curve(df: pd.DataFrame,
                                   treatment: str,
                                   outcome: str,
                                   prediction: str,
                                   min_rows: int = 30,
                                   steps: int = 100,
                                   effect_fn: EffectFnType = linear_effect) -> np.ndarray:
    """
     Orders the dataset by prediction and computes the relative cumulative gain curve curve according to that ordering.
     The relative gain is simply the cumulative effect minus the Average Treatment Effect (ATE) times the relative
     sample size.

     Parameters
     ----------
     df : Pandas' DataFrame
         A Pandas' DataFrame with target and prediction scores.

     treatment : Strings
         The name of the treatment column in `df`.

     outcome : Strings
         The name of the outcome column in `df`.

     prediction : Strings
         The name of the prediction column in `df`.

     min_rows : Integer
         Minimum number of observations needed to have a valid result.

     steps : Integer
         The number of cumulative steps to iterate when accumulating the effect

     effect_fn : function (df: pandas.DataFrame, treatment: str, outcome: str) -> int or Array of int
         A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
         of the outcome column.


     Returns
     ----------
     relative cumulative gain curve: float
         The relative cumulative gain according to the predictions ordering.
     """

    ate = effect_fn(df, treatment, outcome)
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]

    cum_effect = cumulative_effect_curve(df=df, treatment=treatment, outcome=outcome, prediction=prediction,
                                         min_rows=min_rows, steps=steps, effect_fn=effect_fn)

    return np.array([(effect - ate) * (rows / size) for rows, effect in zip(n_rows, cum_effect)])


@curry
def effect_curves(
    df: pd.DataFrame,
    treatment: str,
    outcome: str,
    prediction: str,
    min_rows: int = 30,
    steps: int = 100,
    effect_fn: EffectFnType = linear_effect,
) -> pd.DataFrame:
    """
     Creates a dataset summarizing the effect curves: cumulative effect, cumulative gain and
     relative cumulative gain. The dataset also contains two columns referencing the data
     used to compute the curves at each step: number of samples and fraction of samples used.
     Moreover one column indicating the cumulative gain for a corresponding random model is
     also included as a benchmark.

     Parameters
     ----------
     df : Pandas' DataFrame
         A Pandas' DataFrame with target and prediction scores.

     treatment : Strings
         The name of the treatment column in `df`.

     outcome : Strings
         The name of the outcome column in `df`.

     prediction : Strings
         The name of the prediction column in `df`.

     min_rows : Integer
         Minimum number of observations needed to have a valid result.

     steps : Integer
         The number of cumulative steps to iterate when accumulating the effect

     effect_fn : function (df: pandas.DataFrame, treatment: str, outcome: str) -> int or Array of int
         A function that computes the treatment effect given a dataframe, the name of the treatment column and the name
         of the outcome column.


     Returns
     ----------
     summary curves dataset: pd.DataFrame
         The dataset with the results for multiple validation causal curves according to the predictions ordering.
    """

    size: int = df.shape[0]
    n_rows: List[int] = list(range(min_rows, size, size // steps)) + [size]

    cum_effect: np.ndarray = cumulative_effect_curve(
        df=df,
        treatment=treatment,
        outcome=outcome,
        prediction=prediction,
        min_rows=min_rows,
        steps=steps,
        effect_fn=effect_fn,
    )
    ate: float = cum_effect[-1]

    return pd.DataFrame({"samples_count": n_rows, "cumulative_effect_curve": cum_effect}).assign(
        samples_fraction=lambda x: x["samples_count"] / size,
        cumulative_gain_curve=lambda x: x["samples_fraction"] * x["cumulative_effect_curve"],
        random_model_cumulative_gain_curve=lambda x: x["samples_fraction"] * ate,
        relative_cumulative_gain_curve=lambda x: (
            x["samples_fraction"] * x["cumulative_effect_curve"] - x["random_model_cumulative_gain_curve"]
        ),
    )
