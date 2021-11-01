from typing import List, Tuple

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
                 n_splits: int) -> Tuple[pd.Series, List[RegressorMixin]]:

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
                                     feature_columns: List[str],
                                     treatment_column: str,
                                     outcome_column: str,
                                     debias_model: RegressorMixin = GradientBoostingRegressor(),
                                     debias_feature_columns: List[str] = None,
                                     denoise_model: RegressorMixin = GradientBoostingRegressor(),
                                     denoise_feature_columns: List[str] = None,
                                     final_model: RegressorMixin = GradientBoostingRegressor(),
                                     final_model_feature_columns: List[str] = None,
                                     prediction_column: str = "prediction",
                                     cv_splits: int = 2,
                                     encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    Fits an Non-Parametric Double/ML Meta Learner for Conditional Average Treatment Effect Estimation. It implements the
    following steps:
    1) fits k instances of the debias model to predict the treatment from the features and get out-of-fold residuals
        t_res=t-t_hat;
    2) fits k instances of the denoise model to predict the outcome from the features and get out-of-fold residuals
        y_res=y-y_hat;
    3) fits a final ML model to predict y_res / t_res from the features using weighted regression with weights set to
        t_res^2. Trained like this, the final model will output treatment effect predictions.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features, treatment and target columns.
        The model will be trained to predict the target column
        from the features.

    feature_columns : list of str
        A list os column names that are used as features for the denoise, debias and final models in double-ml. All
         this names should be in `df`.

    treatment_column : str
        The name of the column in `df` that should be used as treatment for the double-ml model. It will learn the
         impact of this column with respect to the outcome column.

    outcome_column : str
        The name of the column in `df` that should be used as outcome for the double-ml model. It will learn the impact
        of the treatment column on this outcome column.

    debias_model : RegressorMixin (default GradientBoostingRegressor())
        The estimator for fitting the treatment from the features. Must implement fit and predict methods. It can be an
        scikit-learn regressor.

    debias_feature_columns : list of str (default None)
        A list os column names to be used only for the debias model. If not None, it will replace feature_columns when
        fitting the debias model.

    denoise_model : RegressorMixin (default GradientBoostingRegressor())
        The estimator for fitting the outcome from the features. Must implement fit and predict methods. It can be an
        scikit-learn regressor.

    denoise_feature_columns : list of str (default None)
        A list os column names to be used only for the denoise model. If not None, it will replace feature_columns when
        fitting the denoise model.

    final_model : RegressorMixin (default GradientBoostingRegressor())
        The estimator for fitting the outcome residuals from the treatment residuals. Must implement fit and predict
        methods. It can be an arbitrary scikit-learn regressor. The fit method must accept sample_weight as a keyword
        argument.

    final_model_feature_columns : list of str (default None)
        A list os column names to be used only for the final model. If not None, it will replace feature_columns when
        fitting the final model.

    prediction_column : str (default "prediction")
        The name of the column with the treatment effect predictions from the final model.

    cv_splits : int (default 2)
        Number of folds to split the training data when fitting the debias and denoise models

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """

    features = feature_columns if not encode_extra_cols else expand_features_encoded(df, feature_columns)

    t_hat, mts = _cv_estimate(debias_model, df,
                              features if debias_feature_columns is None else debias_feature_columns,
                              treatment_column, cv_splits)
    y_hat, mys = _cv_estimate(denoise_model, df,
                              features if denoise_feature_columns is None else denoise_feature_columns,
                              outcome_column, cv_splits)

    y_res = df[outcome_column] - y_hat
    t_res = df[treatment_column] - t_hat

    final_target = y_res / t_res
    weights = t_res ** 2
    final_model_x = features if final_model_feature_columns is None else final_model_feature_columns

    model_final_fitted = final_model.fit(X=df[final_model_x],
                                         y=final_target,
                                         sample_weight=weights)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{prediction_column: model_final_fitted.predict(new_df[final_model_x].values)})

    p.__doc__ = learner_pred_fn_docstring("non_parametric_double_ml_learner")

    log = {'non_parametric_double_ml_learner': {
        'features': feature_columns,
        'debias_feature_columns': debias_feature_columns,
        'denoise_feature_columns': denoise_feature_columns,
        'final_model_feature_columns': final_model_feature_columns,
        'outcome_column': outcome_column,
        'treatment_column': treatment_column,
        'prediction_column': prediction_column,
        'package': "sklearn",
        'package_version': sk_version,
        'feature_importance': dict(zip(features, model_final_fitted.feature_importances_)),
        'training_samples': len(df)},
        'debias_models': mts,
        'denoise_models': mys,
        'cv_splits': cv_splits,
        'object': model_final_fitted}

    return p, p(df), log


non_parametric_double_ml_learner.__doc__ += learner_return_docstring("Non Parametric Double/ML")
