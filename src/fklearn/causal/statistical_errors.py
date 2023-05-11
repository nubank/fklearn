import numpy as np
import pandas as pd
from fklearn.causal.effects import linear_effect

def linear_standard_error(df: pd.DataFrame, treatment: str, outcome: str) -> float:
    """
    Linear Standard Error

    Returns a Float: the linear standard error of a linear regression 
    of the outcome as a function of the treatment.

    Parameters
    ----------

    df : Pandas DataFrame
        A Pandas' DataFrame with with treatment, outcome and confounder columns

    treatment : str
        The name of the column in `df` with the treatment.

    outcome : str
        The name of the column in `df` with the outcome.

    Returns
    ----------
    se : Float
        A Float of the linear standard error extracted by using the formula for 
        the simple linear regression.
    """
    n = df.shape[0]
    t_bar = df[treatment].mean()
    beta1 = linear_effect(df, treatment, outcome)
    beta0 = df[outcome].mean() - beta1 * t_bar
    e = df[outcome] - (beta0 + beta1 * df[treatment])
    se = np.sqrt(((1 / (n - 2)) * np.sum(e**2)) / np.sum((df[treatment] - t_bar)**2))
    return se
