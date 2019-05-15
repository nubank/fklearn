from typing import Any, List

import pandas as pd
from sklearn.impute import SimpleImputer
from toolz import curry

from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
from fklearn.types import LearnerReturnType
from fklearn.training.utils import log_learner_time


@curry
@log_learner_time(learner_name='imputer')
def imputer(df: pd.DataFrame,
            columns_to_impute: List[str],
            impute_strategy: str = 'median') -> LearnerReturnType:
    """
    Fits a missing value imputer to the dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with columns to impute missing values.
        It must contain all columns listed in `columns_to_impute`

    columns_to_impute : List of strings
        A list of names of the columns for missing value imputation.

    impute_strategy : String, (default="median")
        The imputation strategy.
        - If "mean", then replace missing values using the mean along the axis.
        - If "median", then replace missing values using the median along the axis.
        - If "most_frequent", then replace missing using the most frequent value along the axis.
    """

    imp = SimpleImputer(strategy=impute_strategy)

    imp.fit(df[columns_to_impute].values)

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        new_data = imp.transform(new_data_set[columns_to_impute])
        new_cols = pd.DataFrame(data=new_data, columns=columns_to_impute).to_dict('list')
        return new_data_set.assign(**new_cols)

    p.__doc__ = learner_pred_fn_docstring("imputer")

    log = {'imputer': {'impute_strategy': impute_strategy,
                       'columns_to_impute': columns_to_impute,
                       'training_proportion_of_nulls': df[columns_to_impute].isnull().mean(axis=0).to_dict(),
                       'statistics': imp.statistics_}}

    return p, p(df), log


imputer.__doc__ += learner_return_docstring("SimpleImputer")


@curry
@log_learner_time(learner_name='placeholder_imputer')
def placeholder_imputer(df: pd.DataFrame,
                        columns_to_impute: List[str],
                        placeholder_value: Any = -999) -> LearnerReturnType:
    """
    Fills missing values with a fixed value.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with columns to fill missing values.
        It must contain all columns listed in `columns_to_impute`

    columns_to_impute : List of strings
        A list of names of the columns for filling missing value.

    placeholder_value : Any, (default=-999)
        The value used to fill in missing values.
    """

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        new_cols = new_data_set[columns_to_impute].fillna(placeholder_value).to_dict('list')
        return new_data_set.assign(**new_cols)

    p.__doc__ = learner_pred_fn_docstring("placeholder_imputer")

    log = {
        'placeholder_imputer': {
            'columns_to_impute': columns_to_impute,
            'training_proportion_of_nulls': df[columns_to_impute].isnull().mean(axis=0).to_dict(),
            'placeholder_value': placeholder_value
        }
    }

    return p, p(df), log


placeholder_imputer.__doc__ += learner_return_docstring("Placeholder SimpleImputer")
