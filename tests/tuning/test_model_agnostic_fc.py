import pandas as pd
from fklearn.tuning.model_agnostic_fc import variance_feature_selection, \
    correlation_feature_selection, feature_clustering_selection


def test_correlation_feature_selection():

    train_set = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id3"],
        'date': pd.to_datetime(["2016-01-01", "2016-02-01", "2016-03-01", "2016-04-01"]),
        'x': [.2, .9, .3, .3],
        'y': [.2, .91, .3, .3],
        'z': [.4, .4, .4, .4001],
        'a': [4, 2, 3, 1],
        'b': [-4, -2.1, -3.1, -1.1],
        'target': [0, 1, 0, 1]
    })

    features = ["x", "y", "z", "a", "b"]

    result = correlation_feature_selection(train_set, features, threshold=.99)

    assert set(result["features_to_drop"]) == {'x', 'a'}
    assert set(result["final_features"]) == {'b', 'z', 'y'}


def test_variance_feature_selection():

    train_set = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id3"],
        'date': pd.to_datetime(["2016-01-01", "2016-02-01", "2016-03-01", "2016-04-01"]),
        'x': [.2, .9, .3, .3],
        'y': [.2, .91, .3, .3],
        'z': [.4, .4, .4, .4001],
        'a': [4, 2, 3, 1],
        'b': [-4, -2.1, -3.1, -1.1],
        'c': [-4, -4, -4, -4],
        'target': [0, 1, 0, 1]
    })

    features = ["x", "y", "z", "a", "b", "c"]

    result = variance_feature_selection(train_set, features)

    assert set(result["features_to_drop"]) == {'c'}
    assert set(result["final_features"]) == {'x', 'y', 'z', 'a', 'b'}


def test_feature_clustering_selection():

    train_set = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4", "id5", "id6", "id7", "id8", "id9", "id10",
               "id11", "id12", "id13", "id14", "id15", "id16", "id17", "id18", "id19", "id20"],
        'x': [1.58, 2.82, 4.71, 2.2, 4.59, 7.03, 3.3, 4.64, 3.48, 5.31,
              -1.29, 4.29, 6.86, 1.42, 6.47, 5.24, 7.46, 5.45, 5.71, 6.23],
        'y': [4.5, -1.07, 0.61, 1.97, 2.96, -3.02, 3.37, -0.06, 1.73, -1.94,
              8.01, 2.66, -0.08, 2.38, -0.68, -1.01, -5.9, 0.7, -0.76, -1.24],
        'z': [11.93, 11.49, 10.09, 11.93, 10.15, 8.97, 10.55, 10.74, 11.12, 9.47,
              13.91, 10.19, 8.73, 12.23, 9.3, 9.9, 8.63, 9.47, 9.81, 9.64],
        'q': [40.57, 59.36, 36.24, 36.64, 126.97, 5.23, 38.58, 10.01, 105.32, 6.23,
              11.27, 13.89, 21.11, 7.5, 165.71, 16.66, 12.35, 43.63, 39.77, 6.99],
        'a': [48.96, 51.59, 49.2, 52.07, 51.83, 50.11, 51.28, 46.29, 52.45, 52.17,
              50.64, 45.41, 45.44, 47.31, 52.72, 51.09, 49.56, 52.12, 48.99, 47.97],
        'b': [6.89, 8.83, 6.68, 7.87, 8.11, 7.57, 9.0, 5.89, 7.75, 7.86,
              8.94, 6.17, 3.13, 7.99, 8.76, 3.71, 7.73, 8.36, 6.45, 6.23]
    })

    features = ["x", "y", "z", "q", "b", "a"]

    dissimilarity_threshold = 0.5
    result = feature_clustering_selection(
        train_set=train_set,
        features=features,
        dissimilarity_threshold=dissimilarity_threshold)

    assert set(result["features_to_drop"]) == {'y', 'z', 'a'}
    assert set(result["final_features"]) == {'x', 'b', 'q'}

    dissimilarity_threshold = 0.0
    result = feature_clustering_selection(
        train_set=train_set,
        features=features,
        dissimilarity_threshold=dissimilarity_threshold)

    assert set(result["features_to_drop"]) == set()
    assert set(result["final_features"]) == {'q', 'a', 'b', 'y', 'z', 'x'}

    dissimilarity_threshold = 1.0
    result = feature_clustering_selection(
        train_set=train_set,
        features=features,
        dissimilarity_threshold=dissimilarity_threshold)

    assert set(result["features_to_drop"]) == {'a', 'y', 'q', 'b', 'z'}
    assert set(result["final_features"]) == {'x'}