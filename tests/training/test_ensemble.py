# -*- coding: utf-8 -*-
from fklearn.training.ensemble import xgb_octopus_classification_learner
from collections import Counter
import pandas as pd


def test_xgb_octopus_classification_learner():
    df = pd.DataFrame({
        'split_col': [1, 2, 1, 1, 2, 2, 1],
        'x1': [10.0, 13.0, 10.0, 13.0, 14.0, 11.0, 12.0],
        'x2': [0, 1, 1, 1, 0, 0, 1],
        'y': [0, 1, 0, 1, 1, 1, 0]
    })

    df_test = pd.DataFrame({
        'split_col': [1, 1, 2, 1, 2, 2, 2],
        'x1': [10.0, 13.0, 10.0, 13.0, 14.0, 11.0, 12.0],
        'x2': [0, 1, 1, 1, 0, 0, 1],
        'y': [0, 1, 0, 1, 1, 1, 0]
    })

    train_split_bins = [1, 2]
    train_split_col = "split_col"
    target = "y"

    learning_rate_by_bin = {1: 0.08, 2: 0.1}
    num_estimators_by_bin = {1: 10, 2: 20}
    extra_params_by_bin = {split_bin: {'reg_alpha': 0.0, 'colsample_bytree': 0.3, 'subsample': 0.7, 'reg_lambda': 3}
                           for split_bin in train_split_bins}

    features_by_bin = {1: ["x1"], 2: ["x1", "x2"]}

    train_fn = xgb_octopus_classification_learner(learning_rate_by_bin=learning_rate_by_bin,
                                                  num_estimators_by_bin=num_estimators_by_bin,
                                                  extra_params_by_bin=extra_params_by_bin,
                                                  features_by_bin=features_by_bin,
                                                  train_split_col=train_split_col,
                                                  train_split_bins=train_split_bins,
                                                  nthread=4,
                                                  target_column=target,
                                                  prediction_column="predictions")

    pred_fn, pred_train, logs = train_fn(df)

    pred_test = pred_fn(df_test)

    expected_col_train = df.columns.tolist() + ["predictions", "predictions_bin_1", "predictions_bin_2"]
    expected_col_test = df_test.columns.tolist() + ["predictions", "predictions_bin_1", "predictions_bin_2"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert pred_test["predictions"].max() < 1
    assert pred_test["predictions"].min() > 0
    assert (pred_test.columns == pred_train.columns).all()
