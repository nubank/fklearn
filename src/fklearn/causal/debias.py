from typing import List

import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import cross_val_predict
from statsmodels.formula.api import ols
from toolz import curry, merge
from typing import Dict, Any


@curry
def debias_with_regression_formula(df: pd.DataFrame,
                                   treatment_column: str,
                                   outcome_column: str,
                                   confounder_formula: str,
                                   suffix: str = "_debiased",
                                   denoise: bool = True) -> pd.DataFrame:
    """
    Frisch-Waugh-Lovell style debiasing with linear regression. With R formula to define confounders.
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
        A Pandas' DataFrame with with treatment, outcome and confounder columns

    treatment_column : String
        The name of the column in `df` with the treatment.

    outcome_column : String
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

    cols_to_debias = [treatment_column, outcome_column] if denoise else [treatment_column]

    def get_resid(col_to_debias: str) -> np.ndarray:
        model = ols(f"{col_to_debias}~{confounder_formula}", data=df).fit()
        return model.resid + df[col_to_debias].mean()

    return df.assign(**{c + suffix: get_resid(c) for c in cols_to_debias})


@curry
def debias_with_regression(df: pd.DataFrame,
                           treatment_column: str,
                           outcome_column: str,
                           confounder_columns: List[str],
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
        A Pandas' DataFrame with with treatment, outcome and confounder columns

    treatment_column : String
        The name of the column in `df` with the treatment.

    outcome_column : String
        The name of the column in `df` with the outcome.

    confounder_columns : List of String
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
    cols_to_debias = [treatment_column, outcome_column] if denoise else [treatment_column]

    model.fit(df[confounder_columns], df[cols_to_debias])

    debiased = (df[cols_to_debias] - model.predict(df[confounder_columns]) + df[cols_to_debias].mean())

    return df.assign(**{c + suffix: debiased[c] for c in cols_to_debias})


@curry
def debias_with_fixed_effects(df: pd.DataFrame,
                              treatment_column: str,
                              outcome_column: str,
                              confounder_columns: List[str],
                              suffix: str = "_debiased",
                              denoise: bool = True) -> pd.DataFrame:
    """
    Returns a dataframe with the debiased columns with suffix appended to the name

    This is equivalent of debiasing with regression where the forumla is "C(x1) + C(x2) + ...".
    However, it is much more eficient than runing such a dummy variable regression.

    Parameters
    ----------

    df : Pandas DataFrame
        A Pandas' DataFrame with with treatment, outcome and confounder columns

    treatment_column : String
        The name of the column in `df` with the treatment.

    outcome_column : String
        The name of the column in `df` with the outcome.

    confounder_columns : List of String
        Confounders are categorical groups we wish to explain away. Some examples are units (ex: customers),
        and time (day, months...). We perform a group by on these columns, so they should not be continuous
        variables.

    suffix : String
        A suffix to append to the returning debiased column names.

    denoise : Bool (Default=True)
        If it should denoise the outcome using the confounders or not

    Returns
    ----------
    debiased_df : Pandas DataFrame
        The original `df` dataframe with debiased columns added.
    """

    cols_to_debias = [treatment_column, outcome_column] if denoise else [treatment_column]

    def debias_column(c: str) -> dict:
        mu = sum([df.groupby(x)[c].transform("mean") for x in confounder_columns])
        return {c + suffix: df[c] - mu + df[c].mean()}

    return df.assign(**merge(*[debias_column(c) for c in cols_to_debias]))


@curry
def debias_with_double_ml(df: pd.DataFrame,
                          treatment_column: str,
                          outcome_column: str,
                          confounder_columns: List[str],
                          ml_regressor: RegressorMixin = GradientBoostingRegressor,
                          extra_params: Dict[str, Any] = None,
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
        A Pandas' DataFrame with with treatment, outcome and confounder columns

    treatment_column : String
        The name of the column in `df` with the treatment.

    outcome_column : String
        The name of the column in `df` with the outcome.

    confounder_columns : List of String
        A list of confounder present in df

    ml_regressor : Sklearn's RegressorMixin
        A regressor model that implements a fit and a predict method

    extra_params : dict
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

    params = extra_params if extra_params else {}

    cols_to_debias = [treatment_column, outcome_column] if denoise else [treatment_column]

    np.random.seed(seed)

    def get_cv_resid(col_to_debias: str) -> np.ndarray:
        model = ml_regressor(**params)
        cv_pred = cross_val_predict(estimator=model, X=df[confounder_columns], y=df[col_to_debias], cv=cv)
        return df[col_to_debias] - cv_pred + df[col_to_debias].mean()

    return df.assign(**{c + suffix: get_cv_resid(c) for c in cols_to_debias})
