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
def sample_columns (df: pd.DataFrame, perc: float) -> List[str]:
    return random.sample(list(df.columns), int(len(df.columns) * perc))

@curry
def perturbator(df: pd.DataFrame, cols: List[str], corr_fn: Callable) -> pd.DataFrame:
    copy = df.copy(deep=True)
    for col in cols:
        copy.loc[:, col] = corr_fn(df[col])
    return copy 
