import pandas as pd
from toolz import curry

from fklearn.types import EffectFnType
from fklearn.causal.validation.curves import cumulative_effect_curve
from fklearn.causal.effects import linear_effect


@curry
def area_under_the_cumulative_effect_curve(df: pd.DataFrame,
                                           treatment: str,
                                           outcome: str,
                                           prediction: str,
                                           min_rows: int = 30,
                                           steps: int = 100,
                                           effect_fn: EffectFnType = linear_effect) -> float:
    """
     Orders the dataset by prediction and computes the area under the cumulative effect curve, according to that
     ordering.

     Parameters
     ----------
     df : Pandas' DataFrame
         A Pandas' DataFrame with target and prediction scores.

     treatment : str
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
     area_under_the_cumulative_gain_curve: float
         The area under the cumulative gain curve according to the predictions ordering.
     """

    ate = effect_fn(df, treatment, outcome)
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    step_sizes = [min_rows] + [t - s for s, t in zip(n_rows, n_rows[1:])]

    cum_effect = cumulative_effect_curve(df=df, treatment=treatment, outcome=outcome, prediction=prediction,
                                         min_rows=min_rows, steps=steps, effect_fn=effect_fn)

    return abs(sum([(effect - ate) * (step_size / size) for effect, step_size in zip(cum_effect, step_sizes)]))


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
     area_under_the_cumulative_gain_curve: float
         The area under the cumulative gain curve according to the predictions ordering.
     """

    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    step_sizes = [min_rows] + [t - s for s, t in zip(n_rows, n_rows[1:])]

    cum_effect = cumulative_effect_curve(df=df, treatment=treatment, outcome=outcome, prediction=prediction,
                                         min_rows=min_rows, steps=steps, effect_fn=effect_fn)

    return abs(sum([effect * (rows / size) * (step_size / size)
                    for rows, effect, step_size in zip(n_rows, cum_effect, step_sizes)]))


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

    ate = effect_fn(df, treatment, outcome)
    size = df.shape[0]
    n_rows = list(range(min_rows, size, size // steps)) + [size]
    step_sizes = [min_rows] + [t - s for s, t in zip(n_rows, n_rows[1:])]

    cum_effect = cumulative_effect_curve(df=df, treatment=treatment, outcome=outcome, prediction=prediction,
                                         min_rows=min_rows, steps=steps, effect_fn=effect_fn)

    return abs(sum([(effect - ate) * (rows / size) * (step_size / size)
                    for rows, effect, step_size in zip(n_rows, cum_effect, step_sizes)]))
