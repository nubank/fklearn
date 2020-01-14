import functools
import inspect
import pandas as pd
import toolz

from typing import Any, Callable, Dict, List, Union, Optional

@curry
def feature_duplicator(df: pd.DataFrame,
                       columns_to_duplicate: Optional[List[str]] = None,
                       columns_mapping: Optional[Dict[str, str]] = None,
                       preffix: Optional[str] = None,
                       suffix: Optional[str] = None):
    """
    #TODO
    """
    stackname = inspect.stack()[0].function

    if columns_mapping is None:
        columns_mapping = {
            col: (preffix or '') + str(col) + (suffix or '')
            for col in columns_to_duplicate
        }

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        for src_col, dest_col in columns_mapping.items():
            new_df.insert(len(new_df.columns), dest_col, df[src_col])
        return new_df

    p.__doc__ = learner_pred_fn_docstring(stackname)

    log = {
        stackname: {
            'columns_to_duplicate': columns_to_duplicate,
            'columns_mapping': columns_mapping,
            'preffix': preffix,
            'suffix': suffix
        }
    }
    eval(stackname).log = log[stackname]

    return p, p(df), log


def column_duplicatable(columns_to_bind):
    """
    #TODO
    """

    def _decorator(child):
        mixin = feature_duplicator

        def _init(**kwargs):
            mixin_spec  = inspect.getfullargspec(mixin)
            mixin_kargs = set(mixin_spec.args) | set(mixin_spec.kwonlyargs)

            child_spec  = inspect.getfullargspec(child)
            child_kargs = set(child_spec.args) | set(child_spec.kwonlyargs)

            def _learn(df):
                mixin_fn, mixin_df, mixin_log = mixin(
                    df,
                    **{key: value for key, value in kwargs.items() if key in mixin_kargs})
                child_fn, child_df, child_log = child(
                    mixin_df, 
                    **{
                        **{
                            key: value 
                            for key, value in kwargs.items() 
                            if key in child_kargs
                        },
                        columns_to_bind: list(mixin_log['feature_duplicator']['columns_mapping'].values())})

                return toolz.compose(child_fn, mixin_fn), child_df, {**mixin_log, **child_log}
            
            _learn.__doc__ = child.__doc__

            return _learn

        callable_fn = functools.wraps(child)(_init)
        callable_fn.__doc__ = child.__doc__

        return callable_fn

    return _decorator
