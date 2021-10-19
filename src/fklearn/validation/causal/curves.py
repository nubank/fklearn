import numpy as np
import pandas as pd
from toolz import curry, partial

from fklearn.types import EffectFnType


@curry
def linear_effect(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """
    Computes the linear coefficient from regressing the outcome on the treatment: cov(outcome, treatment)/var(treatment)

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : Strings
        The name of the treatment column in `df`.

    outcome : Strings
        The name of the outcome column in `df`.


    Returns
    ----------
    effect: float
        The linear coefficient from regressing the outcome on the treatment: cov(outcome, treatment)/var(treatment)
    """
    cov_mat = df[[treatment, outcome]].cov()
    return cov_mat.iloc[0, 1] / cov_mat.iloc[0, 0]


@curry
def linear_effect_ci(df: pd.DataFrame, treatment: str, outcome: str, z: int = 1.96) -> np.ndarray:
    """
    Computes the confidence interval of the linear coefficient from regressing the outcome on the treatment.

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : Strings
        The name of the treatment column in `df`.

    outcome : Strings
        The name of the outcome column in `df`.

    z : Float (default=1.96)
        The z score (number of standard errors away from the mean) for the confidence level.

    Returns
    ----------
    effect: Numpy's Array
        The confidence interval of the linear coefficient from regressing the outcome on the treatment.
    """

    n = df.shape[0]
    t_bar = df[treatment].mean()
    beta1 = linear_effect(df, outcome, treatment)
    beta0 = df[outcome].mean() - beta1 * t_bar
    e = df[outcome] - (beta0 + beta1 * df[treatment])
    se = np.sqrt(((1 / (n - 2)) * np.sum(e ** 2)) / np.sum((df[treatment] - t_bar) ** 2))
    return np.array([beta1 - z * se, beta1 + z * se])


@curry
def spearman_effect(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """
    Computes the Spearman correlation between the treatment and the outcome

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : Strings
        The name of the treatment column in `df`.

    outcome : Strings
        The name of the outcome column in `df`.


    Returns
    ----------
    effect: float
        The Spearman correlation between the treatment and the outcome
    """

    return df[[treatment, outcome]].corr(method="spearman").iloc[0, 1]


@curry
def pearson_effect(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """
    Computes the Pearson correlation between the treatment and the outcome

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment : Strings
        The name of the treatment column in `df`.

    outcome : Strings
        The name of the outcome column in `df`.


    Returns
    ----------
    effect: float
        The Pearson correlation between the treatment and the outcome
    """

    return df[[treatment, outcome]].corr(method="pearson").iloc[0, 1]


@curry
def effect_by_segment(df,
                      treatment: str,
                      outcome: str,
                      prediction: str,
                      segments: int = 10,
                      effect_fn: EffectFnType = linear_effect) -> pd.Series:
    """
    Segments the dataset by a prediction's quantile and estimates the treatment effect by segment.

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

    Returns
    ----------
    effect by band : Pandas' Series
        The effect stored in a Pandas' series were the indexes are the segments
    """

    effect_fn_partial = partial(effect_fn, treatment=treatment, outcome=outcome)
    return (df
            .assign(**{f"{prediction}_band": pd.qcut(df[prediction], q=segments)})
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
    return np.array([effect_fn(df=ordered_df.head(rows), outcome=outcome, treatment=treatment) for rows in n_rows])


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

    ate = effect_fn(df=df, treatment=treatment, outcome=outcome)
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]

    cum_effect = cumulative_effect_curve(df=df, treatment=treatment, outcome=outcome, prediction=prediction,
                                         min_rows=min_rows, steps=steps, effect_fn=effect_fn)

    return np.array([(effect - ate) * (rows / size) for rows, effect in zip(n_rows, cum_effect)])


@curry
def area_under_the_cumulative_gain_curve(df: pd.DataFrame,
                                         treatment: str,
                                         outcome: str,
                                         prediction: str,
                                         min_rows: int = 30,
                                         steps: int = 100,
                                         effect_fn: EffectFnType = linear_effect) -> float:
    """
     Orders the dataset by prediction and computes the area under the cumulative gain curve, according to that ordering.

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
     area under the cumulative gain curve: float
         The area under the cumulative gain curve according to the predictions ordering.
     """

    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]

    cum_effect = cumulative_effect_curve(df=df, treatment=treatment, outcome=outcome, prediction=prediction,
                                         min_rows=min_rows, steps=steps, effect_fn=effect_fn)

    return sum([effect * (rows ** 2 / size) for rows, effect in zip(n_rows, cum_effect)])


@curry
def area_under_the_relative_cumulative_gain_curve(df: pd.DataFrame,
                                                  treatment: str,
                                                  outcome: str,
                                                  prediction: str,
                                                  min_rows: int = 30,
                                                  steps: int = 100,
                                                  effect_fn: EffectFnType = linear_effect) -> float:
    """
     Orders the dataset by prediction and computes the area under the relative cumulative gain curve, according to that
      ordering.

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
     area under the relative cumulative gain curve: float
         The area under the relative cumulative gain curve according to the predictions ordering.
     """

    ate = effect_fn(df=df, treatment=treatment, outcome=outcome)
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]

    cum_effect = cumulative_effect_curve(df=df, treatment=treatment, outcome=outcome, prediction=prediction,
                                         min_rows=min_rows, steps=steps, effect_fn=effect_fn)

    return sum([(effect - ate) * (rows**2 / size) for rows, effect in zip(n_rows, cum_effect)])
