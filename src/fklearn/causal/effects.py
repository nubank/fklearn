import pandas as pd
from toolz import curry

from fklearn.validation.evaluators import (spearman_evaluator, correlation_evaluator, linear_coefficient_evaluator,
                                           exponential_coefficient_evaluator, logistic_coefficient_evaluator)


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
    return linear_coefficient_evaluator(df, treatment, outcome, eval_name="effect").get("effect")


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

    return spearman_evaluator(df, treatment, outcome, eval_name="effect").get("effect")


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
    return correlation_evaluator(df, treatment, outcome, eval_name="effect").get("effect")


@curry
def exponential_coefficient_effect(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """
    Computes the exponential coefficient between the treatment and the outcome. Finds a1 in the following equation
    outcome = exp(a0 + a1 treatment) + error


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
        The exponential coefficient between the treatment and the outcome
    """
    return exponential_coefficient_evaluator(df, treatment, outcome, eval_name="effect").get("effect")


@curry
def logistic_coefficient_effect(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """
    Computes the logistic coefficient between the treatment and the outcome. Finds a1 in the following equation
    outcome = logistic(a0 + a1 treatment)


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
        The logistic coefficient between the treatment and the outcome
    """
    return logistic_coefficient_evaluator(df, treatment, outcome, eval_name="effect").get("effect")
