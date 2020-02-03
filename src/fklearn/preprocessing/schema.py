import functools
import inspect
import pandas as pd
import toolz

from typing import Any, Callable, Dict, List, Optional

from ..types import LearnerLogType, LearnerReturnType


@toolz.curry
def feature_duplicator(df: pd.DataFrame,
                       columns_to_duplicate: Optional[List[str]] = None,
                       columns_mapping: Optional[Dict[str, str]] = None,
                       preffix: Optional[str] = None,
                       suffix: Optional[str] = None) -> LearnerReturnType:
    """
    Duplicates some columns in the dataframe

    Parameters
    ----------
    df: pandas.DataFrame
        A Pandas' DataFrame with columns_to_duplicate columns

    columns_to_duplicate: list of str
        List of columns names

    columns_mapping: int (default None)
        Mapping of source columns to destination columns

    preffix: int (default None)
        Preffix to add to columns to duplicate

    suffix: int (default None)
        Suffix to add to columns to duplicate

    Returns
    ----------
    increased_dataset : pandas.DataFrame
        A dataset with repeated columns
    """

    stackname = inspect.stack()[0].function

    if columns_to_duplicate:
        columns_final_mapping = {
            col: (preffix or '') + str(col) + (suffix or '')
            for col in columns_to_duplicate
        }
    else:
        columns_final_mapping = {}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        for src_col, dest_col in columns_final_mapping.items():
            new_df.insert(len(new_df.columns), dest_col, df[src_col])
        return new_df

    p.__doc__ = eval(inspect.stack()[0].function).__doc__

    log: LearnerLogType = {
        stackname: {
            'columns_to_duplicate': columns_to_duplicate,
            'columns_mapping': columns_mapping,
            'preffix': preffix,
            'suffix': suffix,
            'columns_final_mapping': columns_final_mapping,
        }
    }
    eval(stackname).log = log[stackname]

    return p, p(df), log


def column_duplicatable(columns_to_bind: str) -> Callable:
    """
    Decorator to duplicate some columns in the dataframe

    Parameters
    ----------
    columns_to_bind: list of str
        Duplicates these columns before applying an inplace learner
    """

    def _decorator(child: LearnerReturnType) -> LearnerReturnType:
        mixin = feature_duplicator

        def _init(*args: List[Any], **kwargs: Dict[str, Any]) -> LearnerReturnType:
            mixin_spec = inspect.getfullargspec(mixin)
            mixin_named_args = set(mixin_spec.args) | set(mixin_spec.kwonlyargs)

            child_spec = inspect.getfullargspec(child)
            child_named_args = set(child_spec.args) | set(child_spec.kwonlyargs)

            def _learn(df: pd.DataFrame) -> LearnerReturnType:
                mixin_kwargs = {key: value for key, value in kwargs.items() if key in mixin_named_args}

                if 'preffix' in kwargs.keys() or 'suffix' in kwargs.keys():
                    mixin_kwargs['columns_to_duplicate'] = kwargs[columns_to_bind]

                mixin_fn, mixin_df, mixin_log = mixin(df, **mixin_kwargs)

                child_kwargs: Dict[str, Any] = {key: value for key, value in kwargs.items() if key in child_named_args}
                child_fn, child_df, child_log = child(mixin_df, *args[1:], **child_kwargs)

                child_kwargs[columns_to_bind] = \
                    list(mixin_log['feature_duplicator']['columns_final_mapping'].values())

                return toolz.compose(child_fn, mixin_fn), child_df, {**mixin_log, **child_log}

            if not len(args):
                _learn.__doc__ = child.__doc__
                return _learn
            else:
                return _learn(args[0])

        callable_fn = functools.wraps(child)(_init)
        callable_fn.__doc__ = child.__doc__

        return callable_fn

    return _decorator
