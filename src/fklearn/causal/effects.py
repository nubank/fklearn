import numpy as np
import pandas as pd
from toolz import curry


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

