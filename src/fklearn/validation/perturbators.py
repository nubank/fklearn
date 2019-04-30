import random
from typing import List

import numpy as np
from toolz import curry
import pandas as pd

from fklearn.types import ColumnWisePerturbFnType


@curry
def shift_mu(col: pd.Series, perc: float) -> pd.Series:
    """
    Shift the mean of column by a given percentage

    Parameters
    ----------
    col : pd.Series
        A Pandas' Series

    perc : float
        How much to shift the mu percentually (can be negative)

    Returns
    ----------
    A transformed pd.Series
    """
    mu = np.mean(col)
    col = col + mu * perc
    return col


@curry
def random_noise(col: pd.Series, mag: float) -> pd.Series:
    """
    Fit a gaussian to column, then sample and add to each entry with a magnification parameter

    Parameters
    ----------
    col : pd.Series
        A Pandas' Series

    mag : float
        Multiplies the noise to control scaling

    Returns
    ----------
    A transformed pd.Series
    """
    mu = np.mean(col)
    std = np.std(col)
    noise = np.random.normal(mu, std, len(col)) * mag
    return col + noise


@curry
def nullify(col: pd.Series, perc: float = 1) -> pd.Series:
    """
    Replace a percenteage of values in  the input Series by np.nan

    Parameters
    ----------
    col : pd.Series
        A Pandas' Series

    perc : float
        Percentage to be replaced by no.nan

    Returns
    ----------
    A transformed pd.Series
    """
    # default behavior to nullify whole column
    n = len(col)
    ix_to_nan = random.sample(range(n), int(n * perc))
    ret = col.copy(deep=True)
    ret.iloc[ix_to_nan] = np.nan
    return ret


@curry
def sample_columns(data: pd.DataFrame, perc: float) -> List[str]:
    """
    Helper function that picks randomly a percentage of the columns

    Parameters
    ----------
    data : pd.DataFrame
        A Pandas' DataFrame

    perc : float
        Percentage of columns to be sampled

    Returns
    ----------
    A list of column names
    """
    return random.sample(list(data.columns), int(len(data.columns) * perc))


@curry
def perturbator(data: pd.DataFrame,
                cols: List[str],
                corruption_fn: ColumnWisePerturbFnType) -> pd.DataFrame:
    """
    transforms specific columns of a dataset according to an artificial
    corruption function.

    Parameters
    ----------
    data : pandas.DataFrame
        A Pandas' DataFrame

    cols : List[str]
        A list of columns to apply the corruption function

    corruption_fn : function pandas.Series -> pandas.Series
        An arbitrary corruption function

    Returns
    ----------
    A transformed dataset
    """
    return data.assign(**{col: corruption_fn(data[col]) for col in cols})
