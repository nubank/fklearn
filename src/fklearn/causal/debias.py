from statsmodels.formula.api import ols
import pandas as pd
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
        outcome+suffix: ols(f"{outcome}~{confounder_formula}", data=df).fit().resid + df[outcome].mean()
    } if denoise else dict()

    return df.assign(**merge(treatment_residual, outcome_residual))
