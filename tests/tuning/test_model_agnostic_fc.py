import pandas as pd
from fklearn.tuning.model_agnostic_fc import variance_feature_selection, correlation_feature_selection


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
