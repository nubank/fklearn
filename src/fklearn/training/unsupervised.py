from typing import Any, Dict, List

import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.cluster import KMeans
import sklearn
from toolz import curry, merge

from fklearn.common_docstrings import learner_pred_fn_docstring, learner_return_docstring
from fklearn.types import LearnerReturnType
from fklearn.training.utils import log_learner_time, expand_features_encoded


@curry
@log_learner_time(learner_name='isolation_forest_learner')
def isolation_forest_learner(df: pd.DataFrame,
                             features: List[str],
                             params: Dict[str, Any] = None,
                             prediction_column: str = "prediction",
                             encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    Fits an anomaly detection algorithm (Isolation Forest) to the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    params : dict
        The IsolationForest parameters in the format {"par_name": param}. See:
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

    prediction_column : str
        The name of the column with the predictions from the model.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """

    model = IsolationForest()

    default_params: Dict[str, Any] = {"n_jobs": -1, "random_state": 1729, "contamination": 0.1}
    # Remove this when we stop supporting scikit-learn<0.24 as this param is deprecated
    if "behaviour" in model.get_params():
        default_params["behaviour"] = "new"
    params = default_params if not params else merge(default_params, params)
    model.set_params(**params)

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    model.fit(df[features].values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        output_col = {prediction_column: model.decision_function(
            new_df[features])}

        return new_df.assign(**output_col)

    p.__doc__ = learner_pred_fn_docstring("isolation_forest_learner")

    log = {'isolation_forest_learner': {
        'features': features,
        'parameters': params,
        'prediction_column': prediction_column,
        'package': "sklearn",
        'package_version': sklearn.__version__,
        'training_samples': len(df)}}

    return p, p(df), log


isolation_forest_learner.__doc__ += learner_return_docstring("Isolation Forest")


@curry
@log_learner_time(learner_name='kmeans_learner')
def kmeans_learner(df: pd.DataFrame,
                   features: List[str],
                   n_clusters: int = 8,
                   extra_params: Dict[str, Any] = None,
                   prediction_column: str = "prediction",
                   encode_extra_cols: bool = True) -> LearnerReturnType:
    """
    The KMeans algorithm clusters data by trying to separate samples in n groups of equal variance, minimizing a
    criterion known as the inertia or within-cluster sum-of-squares (see below). This algorithm requires the number of
    clusters to be specified. For now, the implementation is limited to euclidean distance.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame with features.
        The model will be trained to split data into k groups
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    n_clusters : int
        The number of clusters to form as well as the number of centroids to generate.

    extra_params : dict
        The KMeans parameters in the format {"par_name": param}. See:
        https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html

    prediction_column : str
        The name of the column with the predictions from the model.

    encode_extra_cols : bool (default: True)
        If True, treats all columns in `df` with name pattern fklearn_feat__col==val` as feature columns.
    """

    default_params = {"init": "k-means++", "n_init": 10, "max_iter": 300, "tol": 1e-4}
    params = default_params if not extra_params else merge(default_params, extra_params)

    features = features if not encode_extra_cols else expand_features_encoded(df, features)

    model = KMeans(n_clusters=n_clusters)
    model.set_params(**params)
    model.fit(df[features].values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        output_col = {prediction_column: model.predict(
            new_df[features])}

        return new_df.assign(**output_col)

    p.__doc__ = learner_pred_fn_docstring("kmeans_learner")

    log = {'kmeans_learner': {
        'features': features,
        'n_clusters': n_clusters,
        'centers': {i: model.cluster_centers_[i].tolist() for i in range(model.n_clusters)},
        'parameters': params,
        'prediction_column': prediction_column,
        'package': "sklearn",
        'package_version': sklearn.__version__,
        'training_samples': len(df)}}

    return p, p(df), log


kmeans_learner.__doc__ += learner_return_docstring("K-Means clustering")
