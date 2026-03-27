import pandas as pd
from toolz import curry
from typing import Dict, Callable

from fklearn.validation.evaluators import (
    spearman_evaluator,
    correlation_evaluator,
    linear_coefficient_evaluator,
    exponential_coefficient_evaluator,
    logistic_coefficient_evaluator,
)


def _apply_effect(
    evaluator: Callable[..., Dict[str, float]], df: pd.DataFrame, treatment_column: str, outcome_column: str
) -> float:
    return evaluator(df, treatment_column, outcome_column, eval_name="effect")["effect"]


@curry
def linear_effect(df: pd.DataFrame, treatment_column: str, outcome_column: str) -> float:
    """
    Computes the linear coefficient from regressing the outcome on the treatment: cov(outcome, treatment)/var(treatment)

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment_column : str
        The name of the treatment column in `df`.

    outcome_column : str
        The name of the outcome column in `df`.


    Returns
    ----------
    effect: float
        The linear coefficient from regressing the outcome on the treatment: cov(outcome, treatment)/var(treatment)
    """
    return _apply_effect(linear_coefficient_evaluator, df, treatment_column, outcome_column)


@curry
def spearman_effect(df: pd.DataFrame, treatment_column: str, outcome_column: str) -> float:
    """
    Computes the Spearman correlation between the treatment and the outcome

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment_column : str
        The name of the treatment column in `df`.

    outcome_column : str
        The name of the outcome column in `df`.


    Returns
    ----------
    effect: float
        The Spearman correlation between the treatment and the outcome
    """

    return _apply_effect(spearman_evaluator, df, treatment_column, outcome_column)


@curry
def pearson_effect(df: pd.DataFrame, treatment_column: str, outcome_column: str) -> float:
    """
    Computes the Pearson correlation between the treatment and the outcome

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment_column : str
        The name of the treatment column in `df`.

    outcome_column : str
        The name of the outcome column in `df`.


    Returns
    ----------
    effect: float
        The Pearson correlation between the treatment and the outcome
    """
    return _apply_effect(correlation_evaluator, df, treatment_column, outcome_column)


@curry
def exponential_coefficient_effect(df: pd.DataFrame, treatment_column: str, outcome_column: str) -> float:
    """
    Computes the exponential coefficient between the treatment and the outcome. Finds a1 in the following equation
    outcome = exp(a0 + a1 treatment) + error


    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment_column : str
        The name of the treatment column in `df`.

    outcome_column : str
        The name of the outcome column in `df`.


    Returns
    ----------
    effect: float
        The exponential coefficient between the treatment and the outcome
    """
    return _apply_effect(exponential_coefficient_evaluator, df, treatment_column, outcome_column)


@curry
def logistic_coefficient_effect(df: pd.DataFrame, treatment_column: str, outcome_column: str) -> float:
    """
    Computes the logistic coefficient between the treatment and the outcome. Finds a1 in the following equation
    outcome = logistic(a0 + a1 treatment)


    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame with target and prediction scores.

    treatment_column : str
        The name of the treatment column in `df`.

    outcome_column : str
        The name of the outcome column in `df`.


    Returns
    ----------
    effect: float
        The logistic coefficient between the treatment and the outcome
    """
    return _apply_effect(logistic_coefficient_evaluator, df, treatment_column, outcome_column)
