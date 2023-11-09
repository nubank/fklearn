# -*- coding: utf-8 -*-
from collections import Counter

import pandas as pd

from fklearn.training.unsupervised import isolation_forest_learner, kmeans_learner


def test_anomaly_learner():
    df_train_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 100.0, 13.0],
        'x2': [0, 1, 100, 0],
        'w': [2, 1, 2, 0.5],
        'y': [0, 1, 0, 1]
    })

    df_test_binary = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [1200.0, 19000.0, -400.0, 0.0],
        'x2': [1, 101111, 111110, 1],
        'w': [1, 2, 0, 0.5],
        'y': [1, 0, 0, 1]
    })

    # Standard Behavior
    predict_fn, pred_train, log = isolation_forest_learner(df_train_binary,
                                                           features=["x1", "x2"])

    pred_test = predict_fn(df_test_binary)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()


def test_kmeans_learner():
    df_train_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 100.0, 13.0],
        'x2': [0, 1, 100, 0]
    })

    df_test_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [1200.0, 19000.0, -400.0, 0.0],
        'x2': [1, 101111, 111110, 1]
    })

    # Standard Behavior
    predict_fn, pred_train, log = kmeans_learner(df_train_binary,
                                                 features=["x1", "x2"],
                                                 n_clusters=2,
                                                 extra_params={"random_state": 42})

    pred_test = predict_fn(df_test_binary)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()
