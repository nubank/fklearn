from typing import List

import pandas as pd
from numpy import tril
from toolz import curry
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.cluster import AgglomerativeClustering
import itertools

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


@curry
def feature_clustering_selection(train_set: pd.DataFrame,
                                 features: List[str],
                                 dissimilarity_threshold: float = 0.5) -> LogType:
    """
    Feature selection based on feature clustering with absolute correlarion as distance metric.
    One feature is selected from cluster, using the selection criteria of lower feature
    "1 - R2 ratio". "1 - R2 ratio" = (1 - "own cluster R2") / (1 - "nearest cluster R2").
    The higher is "own cluster R2", or the more the feature explain from the other features
    from the cluster, the lower is "1 - R2 ratio". In parallel, the lower is "nearest cluster R2",
    or the less the feature explain from the features from the nearest cluster, the lower is
    "1 - R2 ratio".
    The intuition behind this is to keep the most heterogenious information from the dataset,
    keeping features that most represent the information present in the other features from its
    cluster, and that is not already explained by the features from the nearest cluster.

    Parameters
    ----------
    train_set : pd.DataFrame
        A Pandas' DataFrame with the training data

    features : list of str
        The list of features to consider when dropping with correlation

    dissimilarity_threshold : float
        The dissimilarity (1 - absolute correlation) threshold. It will only cluster features
        in which the dissimilarity were equal or under this threshold. Or, intuitively, will only
        cluster features in which absolute correlation were equal or above (1 - this threshold).

    Returns
    ----------
    log with feature scores ("own cluster R", "nearest cluster R2" and "1 - R2 ratio"), features to
    drop and final features
    """

    # correlation matrix
    corr_matrix = train_set[features].corr()

    # dissimilarity matrix
    dissimilarity_matrix = 1 - np.abs(corr_matrix)

    # feature clustering
    clustering = AgglomerativeClustering(
        affinity='precomputed',
        linkage='average',
        n_clusters=None,
        distance_threshold=dissimilarity_threshold)
    clustering.fit(dissimilarity_matrix)

    # unique labels
    unique_labels = np.sort(np.unique(clustering.labels_))

    unique_cluster_features = [(
        cluster,
        list(
            dissimilarity_matrix.columns[np.where(clustering.labels_ == cluster)])) for cluster in unique_labels]

    features_cluster_regression_data = [(
        own_cluster_feature,
        cluster,
        tuple(
            own_cluster_other_feature for own_cluster_other_feature
            in own_cluster_features if own_cluster_feature != own_cluster_other_feature
        )) for cluster, own_cluster_features in unique_cluster_features
        for own_cluster_feature in own_cluster_features]

    # feature scores
    features_r2_scores = [(
        f[0],
        f[1],
        0 if len(f[2]) == 0 else LinearRegression().fit(
            train_set[features][list(f[2])],
            train_set[features][f[0]]).score(
                train_set[features][list(f[2])],
                train_set[features][f[0]])) for f in features_cluster_regression_data]

    df_features_ratio_scores = pd.DataFrame([(
        f,
        cluster,
        own_cluster_score) for f, cluster, own_cluster_score in features_r2_scores],
        columns=[
            'feature',
            'cluster',
            'own_cluster_R2'])

    # cluster pairs mean distances
    cluster_pairs = [
        element for element in itertools.product(*[unique_labels, unique_labels]) if element[0] < element[1]]
    pairs = [pair for pair in cluster_pairs]
    distances = [(
        pair[0],
        pair[1],
        np.mean(dissimilarity_matrix.loc[
            dissimilarity_matrix.columns[
                np.where(clustering.labels_ == pair[0])[0]]][dissimilarity_matrix.columns[
                    np.where(clustering.labels_ == pair[1])[0]]].values)) for pair in pairs]

    # A->B and B->A cluster distances
    distances = distances + [(distance[1], distance[0], distance[2]) for distance in distances]
    df_clusters_distances = pd.DataFrame(
        distances, columns=['cluster', 'neighbor', 'dissimilarity'])

    # nearest cluster
    df_nearest_cluster_pairs = df_clusters_distances.sort_values('dissimilarity').groupby('cluster').head(1)

    # cluster pairs feature list
    nearest_cluster_pairs_features = [(
        row['cluster'],
        list(dissimilarity_matrix.columns[
            np.where(clustering.labels_ == row['cluster'])]),
        row['neighbor'],
        list(dissimilarity_matrix.columns[
            np.where(clustering.labels_ == row['neighbor'])])) for ix, row in df_nearest_cluster_pairs.iterrows()]

    features_nearest_cluster_regression_data = [(
        own_cluster_feature,
        own_cluster,
        nearest_cluster,
        nearest_cluster_features) for own_cluster, own_cluster_features, nearest_cluster, nearest_cluster_features
        in nearest_cluster_pairs_features for own_cluster_feature in own_cluster_features]

    # feature scores
    dict_nearest_cluster_r2_scores = {
        f[0]: {
            'nearest_cluster': f[2],
            'nearest_cluster_R2': LinearRegression().fit(
                train_set[features][list(f[3])],
                train_set[features][f[0]]).score(
                    train_set[features][list(f[3])],
                    train_set[features][f[0]])} for f in features_nearest_cluster_regression_data}

    # nearest cluster data
    df_features_ratio_scores['nearest_cluster'] = df_features_ratio_scores['feature'].apply(
        lambda x: None if x not in dict_nearest_cluster_r2_scores
        else dict_nearest_cluster_r2_scores[x]['nearest_cluster'])

    df_features_ratio_scores['nearest_cluster_R2'] = df_features_ratio_scores['feature'].apply(
        lambda x: None if x not in dict_nearest_cluster_r2_scores
        else dict_nearest_cluster_r2_scores[x]['nearest_cluster_R2'])

    # final scores (1-R2_ratio)
    df_features_ratio_scores['1-R2_ratio'] = (
        (1 - df_features_ratio_scores['own_cluster_R2']) / (1 - df_features_ratio_scores['nearest_cluster_R2']))

    # best features
    df_features_ratio_scores = df_features_ratio_scores.sort_values('cluster')
    df_best_features = df_features_ratio_scores.sort_values(
        ['1-R2_ratio', 'own_cluster_R2'],
        ascending=[True, False]).groupby('cluster').head(1)

    final_features = list(df_best_features['feature'].values)
    features_to_drop = list(set(features) - set(final_features))

    df_features_ratio_scores['is_selected'] = df_features_ratio_scores['feature'].apply(
        lambda x: x in final_features)

    return {
        "df_feature_clustering": df_features_ratio_scores.sort_values(
            ['cluster', '1-R2_ratio', 'own_cluster_R2'],
            ascending=[True, True, False]).reset_index(drop=True).to_dict(),
        "features_to_drop": features_to_drop,
        "final_features": final_features}
