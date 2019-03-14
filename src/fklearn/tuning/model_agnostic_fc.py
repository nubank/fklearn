from typing import List

import pandas as pd
from numpy import tril
from toolz import curry

from fklearn.types import LogType


@curry
def correlation_feature_selection(train_set: pd.DataFrame,
                                  features: List[str],
                                  threshold: float = 1.0) -> LogType:
    """
    Feature selection based on correlation

    Parameters
    ----------
    train_set : pd.DataFrame
        A Pandas' DataFrame with the training data

    features : list of str
        The list of features to consider when dropping with correlation

    threshold : float
        The correlation threshold. Will drop features with correlation equal or
        above this threshold

    Returns
    ----------
    log with feature correlation, features to drop and final features
    """

    correlogram = train_set[features].corr()
    correlogram_diag = pd.DataFrame(tril(correlogram.values),
                                    columns=correlogram.columns,
                                    index=correlogram.index)

    features_to_drop = pd.melt(correlogram_diag.reset_index(), id_vars='index') \
        .query('index!=variable') \
        .query('abs(value)>0.0') \
        .query('abs(value)>=%f' % threshold)["variable"].tolist()

    final_features = list(set(features) - set(features_to_drop))

    return {"feature_corr": correlogram.to_dict(),
            "features_to_drop": features_to_drop,
            "final_features": final_features}


@curry
def variance_feature_selection(train_set: pd.DataFrame, features: List[str], threshold: float = 0.0) -> LogType:
    """
    Feature selection based on variance

    Parameters
    ----------
    train_set : pd.DataFrame
        A Pandas' DataFrame with the training data

    features : list of str
        The list of features to consider when dropping with variance

    threshold : float
        The variance threshold. Will drop features with variance equal or
        bellow this threshold

    Returns
    ----------
    log with feature variance, features to drop and final features
    """

    feature_var = train_set[features].var()

    features_to_drop = feature_var[feature_var <= threshold].index.tolist()

    final_features = list(set(features) - set(features_to_drop))

    return {"feature_var": feature_var.to_dict(),
            "features_to_drop": features_to_drop,
            "final_features": final_features}
