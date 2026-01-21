"""
Input validation utilities for fklearn.

This module provides curried validation functions following the fklearn functional
style with @curry decorators and composable design.
"""
from typing import Any, List, Optional, Set, Union

import pandas as pd
from toolz import curry

from fklearn.exceptions import (
    EmptyDataFrameError,
    InvalidParameterRangeError,
    InvalidParameterValueError,
    MissingColumnsError,
)


@curry
def validate_columns_exist(
    df: pd.DataFrame,
    columns: List[str],
    context: Optional[str] = None
) -> None:
    """
    Validate that all specified columns exist in the DataFrame.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    columns : List[str]
        List of column names that must exist in the DataFrame.
    context : Optional[str]
        Optional context string for error messages (e.g., function name).

    Raises
    ------
    MissingColumnsError
        If any of the specified columns are not present in the DataFrame.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2], 'b': [3, 4]})
    >>> validate_columns_exist(df, ['a', 'b'])  # No error
    >>> validate_columns_exist(df, ['a', 'c'])  # Raises MissingColumnsError
    """
    available_columns = set(df.columns)
    required_columns = set(columns)
    missing = required_columns - available_columns

    if missing:
        context_prefix = f"[{context}] " if context else ""
        msg = (
            f"{context_prefix}Missing required columns: {sorted(missing)}. "
            f"Available: {sorted(available_columns)}"
        )
        raise MissingColumnsError(
            missing_columns=sorted(missing),
            available_columns=sorted(available_columns),
            msg=msg
        )


@curry
def validate_nonempty_df(
    df: pd.DataFrame,
    context: Optional[str] = None
) -> None:
    """
    Validate that the DataFrame is not empty.

    Parameters
    ----------
    df : pd.DataFrame
        The DataFrame to validate.
    context : Optional[str]
        Optional context string for error messages (e.g., function name).

    Raises
    ------
    EmptyDataFrameError
        If the DataFrame has no rows.

    Examples
    --------
    >>> import pandas as pd
    >>> df = pd.DataFrame({'a': [1, 2]})
    >>> validate_nonempty_df(df)  # No error
    >>> validate_nonempty_df(pd.DataFrame())  # Raises EmptyDataFrameError
    """
    if len(df) == 0:
        context_prefix = f"[{context}] " if context else ""
        msg = f"{context_prefix}DataFrame is empty but non-empty data is required."
        raise EmptyDataFrameError(msg=msg)


@curry
def validate_range(
    value: Union[int, float],
    param_name: str,
    min_value: Optional[Union[int, float]] = None,
    max_value: Optional[Union[int, float]] = None
) -> None:
    """
    Validate that a numeric value is within the specified range.

    Parameters
    ----------
    value : Union[int, float]
        The value to validate.
    param_name : str
        Name of the parameter (for error messages).
    min_value : Optional[Union[int, float]]
        Minimum allowed value (inclusive). None means no lower bound.
    max_value : Optional[Union[int, float]]
        Maximum allowed value (inclusive). None means no upper bound.

    Raises
    ------
    InvalidParameterRangeError
        If the value is outside the specified range.

    Examples
    --------
    >>> validate_range(5, 'n_estimators', min_value=1, max_value=100)  # No error
    >>> validate_range(0, 'n_estimators', min_value=1)  # Raises InvalidParameterRangeError
    """
    if min_value is not None and value < min_value:
        raise InvalidParameterRangeError(
            param_name=param_name,
            value=value,
            min_value=min_value,
            max_value=max_value
        )
    if max_value is not None and value > max_value:
        raise InvalidParameterRangeError(
            param_name=param_name,
            value=value,
            min_value=min_value,
            max_value=max_value
        )


@curry
def validate_value_in_set(
    value: Any,
    param_name: str,
    allowed_values: Union[List[Any], Set[Any]]
) -> None:
    """
    Validate that a value is in the set of allowed values.

    Parameters
    ----------
    value : Any
        The value to validate.
    param_name : str
        Name of the parameter (for error messages).
    allowed_values : Union[List[Any], Set[Any]]
        Collection of allowed values.

    Raises
    ------
    InvalidParameterValueError
        If the value is not in the allowed set.

    Examples
    --------
    >>> validate_value_in_set('mean', 'strategy', ['mean', 'median'])  # No error
    >>> validate_value_in_set('mode', 'strategy', ['mean', 'median'])  # Raises InvalidParameterValueError
    """
    if value not in allowed_values:
        raise InvalidParameterValueError(
            param_name=param_name,
            value=value,
            allowed_values=list(allowed_values)
        )


@curry
def validate_positive_int(
    value: int,
    param_name: str
) -> None:
    """
    Validate that a value is a positive integer (>= 1).

    Parameters
    ----------
    value : int
        The value to validate.
    param_name : str
        Name of the parameter (for error messages).

    Raises
    ------
    InvalidParameterRangeError
        If the value is not a positive integer.

    Examples
    --------
    >>> validate_positive_int(5, 'n_folds')  # No error
    >>> validate_positive_int(0, 'n_folds')  # Raises InvalidParameterRangeError
    """
    if not isinstance(value, int) or value < 1:
        raise InvalidParameterRangeError(
            param_name=param_name,
            value=value,
            min_value=1,
            msg=f"Parameter '{param_name}' must be a positive integer (>= 1), got {value}."
        )
