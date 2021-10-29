from typing import List

import numpy as np
import pandas as pd
from sklearn import __version__ as sk_version
from sklearn.base import RegressorMixin
from sklearn.model_selection import KFold
from sklearn.ensemble import GradientBoostingRegressor
from toolz import curry

from fklearn.common_docstrings import learner_pred_fn_docstring, learner_return_docstring
from fklearn.training.utils import log_learner_time, expand_features_encoded
from fklearn.types import LearnerReturnType


def _cv_estimate(model: RegressorMixin,
                 train_data: pd.DataFrame,
                 features: List[str],
                 y: str,
                 n_splits: int):

    cv = KFold(n_splits=n_splits)
    models = []
    cv_pred = pd.Series(np.nan, index=train_data.index)

    for train, test in cv.split(train_data):
        m = model.fit(train_data[features].iloc[train], train_data[y].iloc[train])
        cv_pred.iloc[test] = m.predict(train_data[features].iloc[test])
        models += [m]

    return cv_pred, models


@curry
@log_learner_time(learner_name='non_parametric_double_ml_learner')
def non_parametric_double_ml_learner(df: pd.DataFrame,
                                     feature_column: List[str],
                                     treatment_column: str,
                                     outcome_column: str,
                                     debias_model: RegressorMixin = GradientBoostingRegressor(),
                                     denoise_model: RegressorMixin = GradientBoostingRegressor(),
                                     final_model: RegressorMixin = GradientBoostingRegressor(),
                                     prediction_column: str = "prediction",
                                     cv_splits: int = 5,
                                     encode_extra_cols: bool = True) -> LearnerReturnType:

    features = feature_column if not encode_extra_cols else expand_features_encoded(df, feature_column)

    t_hat, mts = _cv_estimate(debias_model, df, features, treatment_column, cv_splits)
    y_hat, mys = _cv_estimate(denoise_model, df, features, outcome_column, cv_splits)

    y_res = df[treatment_column] - y_hat
    t_res = df[outcome_column] - t_hat

    final_target = y_res / t_res
    weights = t_res ** 2

    model_final_fitted = final_model.fit(X=df[features], y=final_target, sample_weight=weights)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{prediction_column: model_final_fitted.predict(new_df[features].values)})

    p.__doc__ = learner_pred_fn_docstring("non_parametric_double_ml_learner")

    log = {'non_parametric_double_ml_learner': {
        'features': feature_column,
        'outcome_column': outcome_column,
        'treatment_column': treatment_column,
        'prediction_column': prediction_column,
        'package': "sklearn",
        'package_version': sk_version,
        'feature_importance': None,
        'training_samples': len(df)},
        'debias_models': mts,
        'denoise_models': mys,
        'cv_splits': cv_splits,
        'object': model_final_fitted}

    return p, p(df), log


non_parametric_double_ml_learner.__doc__ += learner_return_docstring("Non Parametric Double/ML")
