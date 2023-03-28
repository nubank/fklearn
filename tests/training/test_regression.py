from collections import Counter

import numpy as np
import pandas as pd

from fklearn.training.regression import \
    linear_regression_learner, gp_regression_learner, \
    xgb_regression_learner, lgbm_regression_learner, catboost_regressor_learner, \
    custom_supervised_model_learner, elasticnet_regression_learner


def test_elasticnet_regression_learner():
    df_train = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [2.3, 4.0, 100.0, -3.9]
    })

    df_test = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1.3, -4.0, 0.0, 49]
    })

    learner = elasticnet_regression_learner(features=["x1", "x2"],
                                            target="y",
                                            params=None,
                                            prediction_column="prediction",
                                            weight_column="w")

    predict_fn, pred_train, log = learner(df_train)

    pred_test = predict_fn(df_test)

    expected_col_train = df_train.columns.tolist() + ["prediction"]
    expected_col_test = df_test.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()
    assert "prediction" in pred_test.columns


def test_linear_regression_learner():
    df_train = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [2.3, 4.0, 100.0, -3.9]
    })

    df_test = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1.3, -4.0, 0.0, 49]
    })

    learner = linear_regression_learner(features=["x1", "x2"],
                                        target="y",
                                        params=None,
                                        prediction_column="prediction",
                                        weight_column="w")

    predict_fn, pred_train, log = learner(df_train)

    pred_test = predict_fn(df_test)

    expected_col_train = df_train.columns.tolist() + ["prediction"]
    expected_col_test = df_test.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()
    assert "prediction" in pred_test.columns


def test_gp_regression_learner():
    df_train = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        'y': [2.3, 4.0, 100.0, -3.9]
    })

    df_test = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        'y': [1.3, -4.0, 0.0, 49]
    })

    from sklearn.gaussian_process.kernels import RBF, WhiteKernel, DotProduct

    kernel = RBF() + WhiteKernel() + DotProduct()

    learner = gp_regression_learner(features=["x1", "x2"],
                                    target="y",
                                    kernel=kernel,
                                    alpha=0.1,
                                    extra_variance="fit",
                                    return_std=True,
                                    extra_params=None,
                                    prediction_column="prediction")

    predict_fn, pred_train, log = learner(df_train)

    pred_test = predict_fn(df_test)

    expected_col_train = df_train.columns.tolist() + ["prediction", "prediction_std"]
    expected_col_test = df_test.columns.tolist() + ["prediction", "prediction_std"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()
    assert "prediction" in pred_test.columns


def test_xgb_regression_learner():
    df_train = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [2.3, 4.0, 100.0, -3.9]
    })

    df_test = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1.3, -4.0, 0.0, 49]
    })

    features = ["x1", "x2"]

    learner = xgb_regression_learner(features=features,
                                     target="y",
                                     learning_rate=0.1,
                                     num_estimators=20,
                                     extra_params={"max_depth": 2, "seed": 42},
                                     prediction_column="prediction",
                                     weight_column="w")

    predict_fn, pred_train, log = learner(df_train)

    pred_test = predict_fn(df_test)

    expected_col_train = df_train.columns.tolist() + ["prediction"]
    expected_col_test = df_test.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()
    assert "prediction" in pred_test.columns

    # SHAP test
    pred_shap = predict_fn(df_test, apply_shap=True)
    assert "shap_values" in pred_shap.columns
    assert "shap_expected_value" in pred_shap.columns
    assert np.vstack(pred_shap["shap_values"]).shape == (4, 2)


def test_lgbm_regression_learner():
    df_train = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [2.3, 4.0, 100.0, -3.9]
    })

    df_test = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1.3, -4.0, 0.0, 49]
    })

    features = ["x1", "x2"]

    learner = lgbm_regression_learner(features=features,
                                      target="y",
                                      learning_rate=0.1,
                                      num_estimators=20,
                                      extra_params={"max_depth": 2, "seed": 42},
                                      prediction_column="prediction",
                                      weight_column="w")

    predict_fn, pred_train, log = learner(df_train)

    pred_test = predict_fn(df_test)

    expected_col_train = df_train.columns.tolist() + ["prediction"]
    expected_col_test = df_test.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()
    assert "prediction" in pred_test.columns

    # SHAP test
    pred_shap = predict_fn(df_test, apply_shap=True)
    assert "shap_values" in pred_shap.columns
    assert "shap_expected_value" in pred_shap.columns
    assert np.vstack(pred_shap["shap_values"]).shape == (4, 2)


def test_catboost_regressor_learner():
    df_train = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [2.3, 4.0, 100.0, -3.9]
    })

    df_test = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1.3, -4.0, 0.0, 49]
    })

    features = ["x1", "x2"]

    learner = catboost_regressor_learner(features=features,
                                         target="y",
                                         learning_rate=0.1,
                                         num_estimators=20,
                                         extra_params={"max_depth": 2, "random_seed": 42},
                                         prediction_column="prediction",
                                         weight_column="w")

    predict_fn, pred_train, log = learner(df_train)

    pred_test = predict_fn(df_test)

    expected_col_train = df_train.columns.tolist() + ["prediction"]
    expected_col_test = df_test.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()
    assert "prediction" in pred_test.columns

    # SHAP test
    pred_shap = predict_fn(df_test, apply_shap=True)
    assert "shap_values" in pred_shap.columns
    assert "shap_expected_value" in pred_shap.columns
    assert np.vstack(pred_shap["shap_values"]).shape == (4, 2)


def test_custom_supervised_model_learner():
    from sklearn import tree
    from sklearn.gaussian_process import GaussianProcessRegressor

    df_train_classification = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        'y': [0, 1, 0, 1]
    })
    df_test_classification = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        'y': [1, 0, 0, 1]
    })
    df_train_regression = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        'y': [2.3, 4.0, 100.0, -3.9]
    })
    df_test_regression = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        'y': [1.3, -4.0, 0.0, 49]
    })

    features = ["x1", "x2"]
    custom_classifier_learner = custom_supervised_model_learner(
        features=features,
        target='y',
        model=tree.DecisionTreeClassifier(),
        supervised_type='classification',
        log={'DecisionTreeClassifier': None})
    predict_fn_classification, pred_train_classification, log = custom_classifier_learner(df_train_classification)
    pred_test_classification = predict_fn_classification(df_test_classification)

    expected_col_train = df_train_classification.columns.tolist() + ["prediction_0", "prediction_1"]
    expected_col_test = df_test_classification.columns.tolist() + ["prediction_0", "prediction_1"]
    assert (Counter(expected_col_train) == Counter(pred_train_classification.columns.tolist()))
    assert (Counter(expected_col_test) == Counter(pred_test_classification.columns.tolist()))
    assert (pred_test_classification.prediction_0.max() <= 1)
    assert (pred_test_classification.prediction_0.min() >= 0)
    assert (pred_test_classification.prediction_1.max() <= 1)
    assert (pred_test_classification.prediction_1.min() >= 0)

    custom_regression_learner = custom_supervised_model_learner(
        features=features,
        target='y',
        model=GaussianProcessRegressor(),
        supervised_type='regression',
        log={'GaussianProcessRegressor': None})
    predict_fn, pred_train, log = custom_regression_learner(df_train_regression)
    pred_test = predict_fn(df_test_regression)

    expected_col_train = df_train_regression.columns.tolist() + ["prediction"]
    expected_col_test = df_test_regression.columns.tolist() + ["prediction"]
    assert (Counter(expected_col_train) == Counter(pred_train.columns.tolist()))
    assert (Counter(expected_col_test) == Counter(pred_test.columns.tolist()))
    assert ((pred_test.columns == pred_train.columns).all())
    assert ("prediction" in pred_test.columns)
