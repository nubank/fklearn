from typing import List

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from statsmodels.formula.api import ols
from toolz import curry, merge


@curry
def debias_with_regression_formula(df: pd.DataFrame,
                                   treatment: str,
                                   outcome: str,
                                   confounder_formula: str,
                                   suffix: str = "_debiased",
                                   denoise: bool = True) -> pd.DataFrame:
    """
    Frisch-Waugh-Lovell style debiasing with linear regression. Uses statsmodels to fit OLS models with formulas.
    To debias, we
     1) fit a linear model to predict the treatment from the confounders and take the residuals from this fit
     (debias step)
     2) fit a linear model to predict the outcome from the confounders and take the residuals from this fit
     (denoise step).
    We then add back the average outcome and treatment so that their levels remain unchanged.

    Returns a dataframe with the debiased columns with suffix appended to the name

    Parameters
    ----------

    df : Pandas DataFrame
        A Pandas' DataFrame with with treatment, an outcome and confounder columns

    treatment : String
        The name of the column in `df` with the treatment.

    outcome : String
        The name of the column in `df` with the outcome.

    confounder_formula : String
        An R formula modeling the confounders. Check https://www.statsmodels.org/dev/example_formulas.html for examples.

    suffix : String
        A suffix to append to the returning debiased column names.

    denoise : Bool (Default=True)
        If it should denoise the outcome using the confounders or not

    Returns
    ----------
    debiased_df : Pandas DataFrame
        The original `df` dataframe with debiased columns added.
    """

    treatment_residual = {
        treatment + suffix: ols(f"{treatment}~{confounder_formula}", data=df).fit().resid + df[treatment].mean()}

    outcome_residual = {
        outcome + suffix: ols(f"{outcome}~{confounder_formula}", data=df).fit().resid + df[outcome].mean()
    } if denoise else dict()

    return df.assign(**merge(treatment_residual, outcome_residual))


@curry
def debias_with_regression(df: pd.DataFrame,
                           treatment: str,
                           outcome: str,
                           confounders: List[str],
                           suffix: str = "_debiased",
                           denoise: bool = True) -> pd.DataFrame:
    """
    Frisch-Waugh-Lovell style debiasing with linear regression.
    To debias, we
     1) fit a linear model to predict the treatment from the confounders and take the residuals from this fit
     (debias step)
     2) fit a linear model to predict the outcome from the confounders and take the residuals from this fit
     (denoise step).
    We then add back the average outcome and treatment so that their levels remain unchanged.

    Returns a dataframe with the debiased columns with suffix appended to the name

    Parameters
    ----------

    df : Pandas DataFrame
        A Pandas' DataFrame with with treatment, an outcome and confounder columns

    treatment : String
        The name of the column in `df` with the treatment.

    outcome : String
        The name of the column in `df` with the outcome.

    confounders : List of String
        A list of confounder present in df

    suffix : String
        A suffix to append to the returning debiased column names.

    denoise : Bool (Default=True)
        If it should denoise the outcome using the confounders or not

    Returns
    ----------
    debiased_df : Pandas DataFrame
        The original `df` dataframe with debiased columns added.
    """

    model = LinearRegression()
    cols_to_debias = [treatment, outcome] if denoise else [treatment]

    model.fit(df[confounders], df[cols_to_debias])

    debiased = (df[cols_to_debias] - model.predict(df[confounders]) + df[cols_to_debias].mean())

    return df.assign(**{c + suffix: debiased[c] for c in cols_to_debias})


@curry
def debias_with_double_ml(df: pd.DataFrame,
                          treatment: str,
                          outcome: str,
                          confounders: List[str],
                          ml_regressor: RegressorMixin = GradientBoostingRegressor,
                          hyperparam: dict = dict(max_depth=5),
                          cv: int = 5,
                          suffix: str = "_debiased",
                          denoise: bool = True,
                          seed: int = 123) -> pd.DataFrame:
    """
    Frisch-Waugh-Lovell style debiasing with ML model.
    To debias, we
     1) fit a regression ml model to predict the treatment from the confounders and take out of fold residuals from
      this fit (debias step)
     2) fit a regression ml model to predict the outcome from the confounders and take the out of fold residuals from
      this fit (denoise step).
    We then add back the average outcome and treatment so that their levels remain unchanged.

    Returns a dataframe with the debiased columns with suffix appended to the name

    Parameters
    ----------

    df : Pandas DataFrame
        A Pandas' DataFrame with with treatment, an outcome and confounder columns

    treatment : String
        The name of the column in `df` with the treatment.

    outcome : String
        The name of the column in `df` with the outcome.

    confounders : List of String
        A list of confounder present in df

    ml_regressor : Sklearn's RegressorMixin
        A regressor model that implements a fit and a predict method

    hyperparam : dict
        The hyper-parameters for the model

    cv : int
        The number of folds to cross predict

    suffix : String
        A suffix to append to the returning debiased column names.

    denoise : Bool (Default=True)
        If it should denoise the outcome using the confounders or not

    seed : int
        A seed for consistency in random computation

    Returns
    ----------
    debiased_df : Pandas DataFrame
        The original `df` dataframe with debiased columns added.
    """

    cols_to_debias = [treatment, outcome] if denoise else [treatment]

    np.random.seed(seed)

    def get_cv_resid(col_to_debias: str) -> np.ndarray:
        model = ml_regressor(**hyperparam)
        cv_pred = cross_val_predict(estimator=model, X=df[confounders], y=df[col_to_debias], cv=cv)
        return df[col_to_debias] - cv_pred + df[col_to_debias].mean()

    return df.assign(**{c + suffix: get_cv_resid(c) for c in cols_to_debias})
