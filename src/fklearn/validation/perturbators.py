import numpy as np
import random
from toolz import curry
import pandas as pd
from typing import Callable, List

@curry
def shift_mu (col: pd.Series, perc: float) -> pd.Series:
    mu = np.mean(col)
    col = col + mu * perc
    return col

@curry
def random_noise (col: pd.Series) -> pd.Series:
    mu = np.mean(col)
    std = np.std(col)
    noise = np.random.normal(mu, std, len(col))
    return col + noise

@curry
def nullify(col: pd.Series) -> pd.Series:
    return pd.Series([np.nan] * len(col))
  
@curry
def sample_columns (data: pd.DataFrame, perc: float) -> List[str]:
    return random.sample(list(data.columns), int(len(data.columns) * perc))

@curry
def perturbator(data: pd.DataFrame, cols: List[str], corr_fn: Callable) -> pd.DataFrame:
    copy = data.copy(deep=True)
    for col in cols:
        copy.loc[:, col] = corr_fn(data[col])
    return copy 
    """
    transforms specific columns of a dataset according to an artificial
    corruption function.

    Parameters
    ----------
    data : pandas.DataFrame
        A Pandas' DataFrame

    cols : List[str]
        A list of columns to apply the corruption function

    corr_fn : function pandas.Series -> pandas.Series
        An arbitrary corruption function

    Returns
    ----------
    A list of log-like dictionary evaluations.
    """
