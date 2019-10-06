from typing import Any, Dict, List, Union

import numpy as np
import pandas as pd
from toolz import merge, curry, assoc
from sklearn.linear_model import LinearRegression
from sklearn.gaussian_process import GaussianProcessRegressor, kernels
from sklearn import __version__ as sk_version

from fklearn.common_docstrings import learner_pred_fn_docstring, learner_return_docstring
from fklearn.types import LearnerReturnType
from fklearn.training.utils import log_learner_time, expand_features_encoded


@curry
@log_learner_time(learner_name='linear_regression_learner')
def linear_regression_learner(df: pd.DataFrame,
                              features: List[str],
                              target: str,
                              params: Dict[str, Any] = None,
                              prediction_column: str = "prediction",
                              weight_column: str = None,
                              encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    Fits an linear regression classifier to the dataset. Return the predict function
    for the model and the predictions for the input dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be continuous, since this is a regression model.

    params : dict
        The LinearRegression parameters in the format {"par_name": param}. See:
        http://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html

    prediction_column : str
        The name of the column with the predictions from the model.

    weight_column : str, optional
        The name of the column with scores to weight the data.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """

    def_params = {"fit_intercept": True}
    params = def_params if not params else merge(def_params, params)

    weights = df[weight_column].values if weight_column else None

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    regr = LinearRegression(**params)
    regr.fit(df[features].values, df[target].values, sample_weight=weights)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{prediction_column: regr.predict(new_df[features].values)})

    p.__doc__ = learner_pred_fn_docstring("linear_regression_learner")

    log = {'linear_regression_learner': {
        'features': features,
        'target': target,
        'parameters': params,
        'prediction_column': prediction_column,
        'package': "sklearn",
        'package_version': sk_version,
        'feature_importance': dict(zip(features, regr.coef_.flatten())),
        'training_samples': len(df)},
        'object': regr}

    return p, p(df), log


linear_regression_learner.__doc__ += learner_return_docstring("Linear Regression")


@curry
@log_learner_time(learner_name='xgb_regression_learner')
def xgb_regression_learner(df: pd.DataFrame,
                           features: List[str],
                           target: str,
                           learning_rate: float = 0.1,
                           num_estimators: int = 100,
                           extra_params: Dict[str, Any] = None,
                           prediction_column: str = "prediction",
                           weight_column: str = None,
                           encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    Fits an XGBoost regressor to the dataset. It first generates a DMatrix
    with the specified features and labels from `df`. Then it fits a XGBoost
    model to this DMatrix. Return the predict function for the model and the
    predictions for the input dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be numerical and continuous, since this is a regression model.

    learning_rate : float
        Float in range [0,1].
        Step size shrinkage used in update to prevents overfitting. After each boosting step,
        we can directly get the weights of new features. and eta actually shrinks the
        feature weights to make the boosting process more conservative.
        See the eta hyper-parameter in:
        http://xgboost.readthedocs.io/en/latest/parameter.html

    num_estimators : int
        Int in range [0, inf]
        Number of boosted trees to fit.
        See the n_estimators hyper-parameter in:
        http://xgboost.readthedocs.io/en/latest/python/python_api.html

    extra_params : dict, optional
        Dictionary in the format {"hyperparameter_name" : hyperparameter_value.
        Other parameters for the XGBoost model. See the list in:
        http://xgboost.readthedocs.io/en/latest/parameter.html
        If not passed, the default will be used.

    prediction_column : str
        The name of the column with the predictions from the model.

    weight_column : str, optional
        The name of the column with scores to weight the data.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """

    import xgboost as xgb

    weights = df[weight_column].values if weight_column else None
    params = extra_params if extra_params else {}
    params = assoc(params, "eta", learning_rate)
    params = params if "objective" in params else assoc(params, "objective", 'reg:linear')

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    dtrain = xgb.DMatrix(df[features].values, label=df[target].values, weight=weights, feature_names=map(str, features))

    bst = xgb.train(params, dtrain, num_estimators)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        dtest = xgb.DMatrix(new_df[features].values, feature_names=map(str, features))
        col_dict = {prediction_column: bst.predict(dtest)}

        if apply_shap:
            import shap
            explainer = shap.TreeExplainer(bst)
            shap_values = list(explainer.shap_values(new_df[features]))
            shap_expected_value = explainer.expected_value

            shap_output = {"shap_values": shap_values,
                           "shap_expected_value": np.repeat(shap_expected_value, len(shap_values))}

            col_dict = merge(col_dict, shap_output)

        return new_df.assign(**col_dict)

    p.__doc__ = learner_pred_fn_docstring("xgb_regression_learner", shap=True)

    log = {'xgb_regression_learner': {
        'features': features,
        'target': target,
        'prediction_column': prediction_column,
        'package': "xgboost",
        'package_version': xgb.__version__,
        'parameters': assoc(params, "num_estimators", num_estimators),
        'feature_importance': bst.get_score(),
        'training_samples': len(df)},
        'object': bst}

    return p, p(df), log


xgb_regression_learner.__doc__ += learner_return_docstring("XGboost Regressor")


@curry
@log_learner_time(learner_name='catboost_regressor_learner')
def catboost_regressor_learner(df: pd.DataFrame,
                               features: List[str],
                               target: str,
                               learning_rate: float = 0.1,
                               num_estimators: int = 100,
                               extra_params: Dict[str, Any] = None,
                               prediction_column: str = "prediction",
                               weight_column: str = None) -> LearnerReturnType:
    """
    Fits an CatBoost regressor to the dataset. It first generates a Pool
    with the specified features and labels from `df`. Then it fits a CatBoost
    model to this Pool. Return the predict function for the model and the
    predictions for the input dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be numerical and continuous, since this is a regression model.

    learning_rate : float
        Float in range [0,1].
        Step size shrinkage used in update to prevents overfitting. After each boosting step,
        we can directly get the weights of new features. and eta actually shrinks the
        feature weights to make the boosting process more conservative.
        See the eta hyper-parameter in:
        https://catboost.ai/docs/concepts/python-reference_parameters-list.html

    num_estimators : int
        Int in range [0, inf]
        Number of boosted trees to fit.
        See the n_estimators hyper-parameter in:
        https://catboost.ai/docs/concepts/python-reference_parameters-list.html

    extra_params : dict, optional
        Dictionary in the format {"hyperparameter_name" : hyperparameter_value.
        Other parameters for the CatBoost model. See the list in:
        https://catboost.ai/docs/concepts/python-reference_catboostregressor.html
        If not passed, the default will be used.

    prediction_column : str
        The name of the column with the predictions from the model.

    weight_column : str, optional
        The name of the column with scores to weight the data.
    """
    from catboost import Pool, CatBoostRegressor
    import catboost

    weights = df[weight_column].values if weight_column else None
    params = extra_params if extra_params else {}
    params = assoc(params, "eta", learning_rate)

    dtrain = Pool(df[features].values, df[target].values, weight=weights, feature_names=list(map(str, features)))
    cat_boost_regressor = CatBoostRegressor(iterations=num_estimators, **params)
    cbr = cat_boost_regressor.fit(dtrain, verbose=0)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        dtest = Pool(new_df[features].values, feature_names=list(map(str, features)))
        col_dict = {prediction_column: cbr.predict(dtest)}

        if apply_shap:
            import shap
            explainer = shap.TreeExplainer(cbr)
            shap_values = list(explainer.shap_values(new_df[features]))
            shap_expected_value = explainer.expected_value

            shap_output = {"shap_values": shap_values,
                           "shap_expected_value": np.repeat(shap_expected_value, len(shap_values))}

            col_dict = merge(col_dict, shap_output)

        return new_df.assign(**col_dict)

    p.__doc__ = learner_pred_fn_docstring("CatBoostRegressor", shap=False)

    log = {'catboost_regression_learner': {
        'features': features,
        'target': target,
        'prediction_column': prediction_column,
        'package': "catboost",
        'package_version': catboost.__version__,
        'parameters': assoc(params, "num_estimators", num_estimators),
        'feature_importance': cbr.feature_importances_,
        'training_samples': len(df)},
        'object': cbr}

    return p, p(df), log


catboost_regressor_learner.__doc__ += learner_return_docstring("CatBoostRegressor")


@curry
@log_learner_time(learner_name='gp_regression_learner')
def gp_regression_learner(df: pd.DataFrame,
                          features: List[str],
                          target: str,
                          kernel: kernels.Kernel = None,
                          alpha: float = 0.1,
                          extra_variance: Union[str, float] = "fit",
                          return_std: bool = False,
                          extra_params: Dict[str, Any] = None,
                          prediction_column: str = "prediction",
                          encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    Fits an gaussian process regressor to the dataset.

    Parameters
    ----------

    df: pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features: list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target: str
        The name of the column in `df` that should be used as target for the model.
        This column should be numerical and continuous, since this is a regression model.

    kernel: sklearn.gaussian_process.kernels
        The kernel specifying the covariance function of the GP. If None is passed,
        the kernel "1.0 * RBF(1.0)" is used as default. Note that the kernel's hyperparameters
        are optimized during fitting.

    alpha: float
        Value added to the diagonal of the kernel matrix during fitting. Larger values correspond to increased
        noise level in the observations. This can also prevent a potential numerical issue during fitting,
        by ensuring that the calculated values form a positive definite matrix.

    extra_variance: float
        The amount of extra variance to scale to the predictions in standard deviations. If left as the default "fit",
        Uses the standard deviation of the target.

    return_std: bool
        If True, the standard-deviation of the predictive distribution at the query points is returned
        along with the mean.

    extra_params: dict {"hyperparameter_name" : hyperparameter_value}, optional
        Other parameters for the GaussianProcessRegressor model. See the list in:
        http://scikit-learn.org/stable/modules/generated/sklearn.gaussian_process.GaussianProcessRegressor.html
        If not passed, the default will be used.

    prediction_column : str
        The name of the column with the predictions from the model.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """

    params = extra_params if extra_params else {}

    params['alpha'] = alpha
    params['kernel'] = kernel

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    gp = GaussianProcessRegressor(**params)
    gp.fit(df[features], df[target])

    extra_variance = df[target].std() if extra_variance == "fit" else extra_variance if extra_variance else 1

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        if return_std:
            pred_mean, pred_std = gp.predict(df[features], return_std=True)
            pred_std *= extra_variance
            return new_df.assign(**{prediction_column: pred_mean, prediction_column + "_std": pred_std})
        else:
            return new_df.assign(**{prediction_column: gp.predict(df[features])})

    p.__doc__ = learner_pred_fn_docstring("gp_regression_learner")

    log = {'gp_regression_learner': {
        'features': features,
        'target': target,
        'parameters': merge(params, {'extra_variance': extra_variance,
                                     'return_std': return_std}),
        'prediction_column': prediction_column,
        'package': "sklearn",
        'package_version': sk_version,
        'training_samples': len(df)},
        'object': gp}

    return p, p(df), log


gp_regression_learner.__doc__ += learner_return_docstring("Gaussian Process Regressor")


@curry
@log_learner_time(learner_name='lgbm_regression_learner')
def lgbm_regression_learner(df: pd.DataFrame,
                            features: List[str],
                            target: str,
                            learning_rate: float = 0.1,
                            num_estimators: int = 100,
                            extra_params: Dict[str, Any] = None,
                            prediction_column: str = "prediction",
                            weight_column: str = None,
                            encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    Fits an LGBM regressor to the dataset.

    It first generates a Dataset with the specified features and labels
    from `df`. Then, it fits a LGBM model to this Dataset. Return the predict
    function for the model and the predictions for the input dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.
        This column should be binary, since this is a classification model.

    learning_rate : float
        Float in the range (0, 1]
        Step size shrinkage used in update to prevents overfitting. After each boosting step,
        we can directly get the weights of new features. and eta actually shrinks the
        feature weights to make the boosting process more conservative.
        See the learning_rate hyper-parameter in:
        https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

    num_estimators : int
        Int in the range (0, inf)
        Number of boosted trees to fit.
        See the num_iterations hyper-parameter in:
        https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst

    extra_params : dict, optional
        Dictionary in the format {"hyperparameter_name" : hyperparameter_value}.
        Other parameters for the LGBM model. See the list in:
        https://github.com/Microsoft/LightGBM/blob/master/docs/Parameters.rst
        If not passed, the default will be used.

    prediction_column : str
        The name of the column with the predictions from the model.

    weight_column : str, optional
        The name of the column with scores to weight the data.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """

    import lightgbm as lgbm

    params = extra_params if extra_params else {}
    params = assoc(params, "eta", learning_rate)
    params = params if "objective" in params else assoc(params, "objective", 'regression')

    weights = df[weight_column].values if weight_column else None

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    dtrain = lgbm.Dataset(df[features].values, label=df[target], feature_name=list(map(str, features)), weight=weights,
                          silent=True)

    bst = lgbm.train(params, dtrain, num_estimators)

    def p(new_df: pd.DataFrame, apply_shap: bool = False) -> pd.DataFrame:
        col_dict = {prediction_column: bst.predict(new_df[features].values)}

        if apply_shap:
            import shap
            explainer = shap.TreeExplainer(bst)
            shap_values = list(explainer.shap_values(new_df[features]))
            shap_expected_value = explainer.expected_value

            shap_output = {"shap_values": shap_values,
                           "shap_expected_value": np.repeat(shap_expected_value, len(shap_values))}

            col_dict = merge(col_dict, shap_output)

        return new_df.assign(**col_dict)

    p.__doc__ = learner_pred_fn_docstring("lgbm_regression_learner", shap=True)

    log = {'lgbm_regression_learner': {
        'features': features,
        'target': target,
        'prediction_column': prediction_column,
        'package': "lightgbm",
        'package_version': lgbm.__version__,
        'parameters': assoc(params, "num_estimators", num_estimators),
        'feature_importance': dict(zip(features, bst.feature_importance().tolist())),
        'training_samples': len(df)},
        'object': bst}

    return p, p(df), log


lgbm_regression_learner.__doc__ += learner_return_docstring("LGBM Regressor")


@curry
@log_learner_time(learner_name='custom_supervised_model_learner')
def custom_supervised_model_learner(df: pd.DataFrame,
                                    features: List[str],
                                    target: str,
                                    model: Any,
                                    supervised_type: str,
                                    log: Dict[str, Dict],
                                    prediction_column: str = "prediction") -> LearnerReturnType:
    """
    Fits a custom model to the dataset.
    Return the predict function, the predictions for the input dataset and a log describing the model.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    target : str
        The name of the column in `df` that should be used as target for the model.

    model: Object
        Machine learning model to be used for regression or clasisfication.
        model object must have ".fit" attribute to train the data.
        For classification problems, it also needs ".predict_proba" attribute.
        For regression problemsm it needs ".predict" attribute.

    supervised_type: str
        Type of supervised learning to be used
        The options are: 'classification' or 'regression'

    log : Dict[str, Dict]
        Log with additional information of the custom model used.
        It must start with just one element with the model name.

    prediction_column : str
        The name of the column with the predictions from the model.
        For classification problems, all probabilities wiill be added: for i in range(0,n_classes).
        For regression just prediction_column will be added.
    """

    if len(log) != 1:
        raise ValueError("\'log\' dictionary must start with model name")
    if supervised_type not in ('classification', 'regression'):
        raise TypeError("supervised_type options are: \'classification\' or \'regression\'")
    if not hasattr(model, 'fit'):
        raise AttributeError("\'model\' object must have \'fit\' attribute")
    if supervised_type == 'classification' and not hasattr(model, 'predict_proba'):
        raise AttributeError("\'model\' object for classification must have \'predict_proba\' attribute")
    if supervised_type == 'regression' and not hasattr(model, 'predict'):
        raise AttributeError("\'model\' object for regression must have \'predict\' attribute")

    model.fit(df[features].values, df[target].values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        if supervised_type == 'classification':
            pred = model.predict_proba(new_df[features].values)
            col_dict = {}
            for (key, value) in enumerate(pred.T):
                col_dict.update({prediction_column + "_" + str(key): value})
        elif supervised_type == 'regression':
            col_dict = {prediction_column: model.predict(new_df[features].values)}

        return new_df.assign(**col_dict)

    p.__doc__ = learner_pred_fn_docstring("custom_supervised_model_learner")

    log["object"] = model

    return p, p(df), log


custom_supervised_model_learner.__doc__ += learner_return_docstring("Custom Supervised Model Learner")
