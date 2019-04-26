# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import pandas as pd

from fklearn.training.classification import \
    logistic_classification_learner, xgb_classification_learner, \
    nlp_logistic_classification_learner, lgbm_classification_learner, \
    catboost_classification_learner


def test_logistic_classification_learner():
    df_train_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [0, 1, 0, 1]
    })

    df_test_binary = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1, 0, 0, 1]
    })

    df_train_multinomial = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4", "id4", "id6"],
        'x1': [10.0, 13.0, 10.0, 13.0, 20, 13],
        "x2": [0, 1, 1, 0, 1, 1],
        "w": [2, 1, 2, 0.5, 0.5, 3],
        'y': [0, 1, 2, 0, 1, 2]
    })

    df_test_multinomial = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6", "id6", "id7"],
        'x1': [12.0, 1000.0, -4.0, 0.0, 0.0, 1],
        "x2": [1, 1, 0, 1, 1, 0],
        "w": [1, 2, 0, 0.5, 0.1, 2],
        'y': [2, 0, 1, 1, 0, 2]
    })

    # test binomial case
    learner_binary = logistic_classification_learner(features=["x1", "x2"],
                                                     target="y",
                                                     params={"max_iter": 2000})

    predict_fn, pred_train, log = learner_binary(df_train_binary)

    pred_test = predict_fn(df_test_binary)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert pred_test.prediction.max() < 1
    assert pred_test.prediction.min() > 0
    assert (pred_test.columns == pred_train.columns).all()

    # test multinomial case
    learner_multinomial = logistic_classification_learner(features=["x1", "x2"],
                                                          target="y",
                                                          params={"multi_class": "multinomial",
                                                                  "solver": "sag",
                                                                  "max_iter": 2000},
                                                          weight_column="w")

    predict_fn, pred_train, log = learner_multinomial(df_train_multinomial)

    pred_test = predict_fn(df_test_multinomial)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2",
                                                             "prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2",
                                                           "prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()


def test_xgb_classification_learner():
    df_train_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [0, 1, 0, 1]
    })

    df_test_binary = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1, 0, 0, 1]
    })

    df_train_multinomial = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0, 1, 0],
        "w": [2, 1, 2, 0.5, 2, 0.5],
        'y': [0, 1, 2, 1, 2, 0]
    })

    df_test_multinomial = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1, 0, 1],
        "w": [1, 2, 0, 0.5, 0, 0.5],
        'y': [1, 2, 0, 1, 2, 0]
    })

    features = ["x1", "x2"]

    learner_binary = xgb_classification_learner(features=features,
                                                target="y",
                                                learning_rate=0.1,
                                                num_estimators=20,
                                                extra_params={"max_depth": 4, "seed": 42},
                                                prediction_column="prediction",
                                                weight_column="w")

    predict_fn_binary, pred_train_binary, log = learner_binary(df_train_binary)

    pred_test_binary = predict_fn_binary(df_test_binary)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train_binary.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test_binary.columns.tolist())
    assert pred_test_binary.prediction.max() < 1
    assert pred_test_binary.prediction.min() > 0
    assert (pred_test_binary.columns == pred_train_binary.columns).all()

    # SHAP test (binary only)
    pred_shap = predict_fn_binary(df_test_binary, apply_shap=True)
    assert "shap_values" in pred_shap.columns
    assert "shap_expected_value" in pred_shap.columns
    assert np.vstack(pred_shap["shap_values"]).shape == (4, 2)

    # test multinomial case
    learner_multinomial = xgb_classification_learner(features=features,
                                                     target="y",
                                                     learning_rate=0.1,
                                                     num_estimators=20,
                                                     extra_params={"max_depth": 2,
                                                                   "seed": 42,
                                                                   "objective": 'multi:softprob',
                                                                   "num_class": 3},
                                                     prediction_column="prediction")

    predict_fn_multinomial, pred_train_multinomial, log = learner_multinomial(df_train_multinomial)

    pred_test_multinomial = predict_fn_multinomial(df_test_multinomial)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2",
                                                             "prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2",
                                                           "prediction"]

    assert Counter(expected_col_train) == Counter(pred_train_multinomial.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test_multinomial.columns.tolist())
    assert (pred_test_multinomial.columns == pred_train_multinomial.columns).all()


def test_catboost_classification_learner():
    df_train_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [0, 1, 0, 1]
    })

    df_test_binary = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1, 0, 0, 1]
    })

    df_train_multinomial = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0, 1, 0],
        "w": [2, 1, 2, 0.5, 2, 0.5],
        'y': [0, 1, 2, 1, 2, 0]
    })

    df_test_multinomial = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1, 0, 1],
        "w": [1, 2, 0, 0.5, 0, 0.5],
        'y': [1, 2, 0, 1, 2, 0]
    })

    features = ["x1", "x2"]

    learner_binary = catboost_classification_learner(features=features,
                                                     target="y",
                                                     learning_rate=0.1,
                                                     num_estimators=20,
                                                     extra_params={"max_depth": 4, "random_seed": 42},
                                                     prediction_column="prediction",
                                                     weight_column="w")

    predict_fn_binary, pred_train_binary, log = learner_binary(df_train_binary)

    pred_test_binary = predict_fn_binary(df_test_binary)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train_binary.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test_binary.columns.tolist())
    assert pred_test_binary.prediction.max() < 1
    assert pred_test_binary.prediction.min() > 0
    assert (pred_test_binary.columns == pred_train_binary.columns).all()

    # SHAP test (binary only)
    pred_shap = predict_fn_binary(df_test_binary, apply_shap=True)
    assert "shap_values" in pred_shap.columns
    assert "shap_expected_value" in pred_shap.columns
    assert np.vstack(pred_shap["shap_values"]).shape == (4, 2)

    # test multinomial case
    learner_multinomial = lgbm_classification_learner(features=features,
                                                      target="y",
                                                      learning_rate=0.1,
                                                      num_estimators=20,
                                                      extra_params={"max_depth": 2,
                                                                    "seed": 42,
                                                                    "objective": 'multiclass',
                                                                    "num_class": 3},
                                                      prediction_column="prediction")

    predict_fn_multinomial, pred_train_multinomial, log = learner_multinomial(df_train_multinomial)

    pred_test_multinomial = predict_fn_multinomial(df_test_multinomial)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2"]

    assert Counter(expected_col_train) == Counter(pred_train_multinomial.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test_multinomial.columns.tolist())
    assert (pred_test_multinomial.columns == pred_train_multinomial.columns).all()


def test_nlp_logistic_classification_learner():
    df_train_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "text1": ["banana manga", "manga açaí", "banana banana", "Manga."],
        "text2": ["banana mamao", "manga açaí", "banana banana", "Manga."],
        'y': [0, 1, 0, 1]
    })

    df_test_binary = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6", "id5", "id6"],
        'x1': [0.0, 3.0, 2.0, -13.0, 2.0, -13.0],
        "text1": ["banana manga", "manga açaí", "banana banana", "Manga.", "banana manga", "manga açaí"],
        "text2": ["banana manga", "manga açaí", "jaca banana", "Manga.", "jaca banana", "Manga."],
        'y': [1, 0, 0, 1, 0, 1]
    })

    df_train_multinomial = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0, 10.0, 13.0],
        "text": ["banana manga", "manga açaí", "banana banana", "Manga.", "banana banana", "Manga."],
        'y': [0, 1, 2, 0, 1, 2]
    })

    df_test_multinomial = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6", "id5", "id6"],
        'x1': [0.0, 3.0, 2.0, -13.0, 2.0, -13.0],
        "text": ["abacaxi manga", "manga açaí", "banana banana", "Abacaxi.", "banana banana", "Manga."],
        'y': [0, 1, 2, 0, 2, 1]
    })

    # test binomial case
    learner_binary = nlp_logistic_classification_learner(text_feature_cols=["text1", "text2"],
                                                         target="y",
                                                         vectorizer_params={"min_df": 1},
                                                         logistic_params=None,
                                                         prediction_column="prediction")

    predict_fn, pred_train, log = learner_binary(df_train_binary)

    pred_test = predict_fn(df_test_binary)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert pred_test.prediction.max() < 1
    assert pred_test.prediction.min() > 0
    assert (pred_test.columns == pred_train.columns).all()

    # test multinomial case
    learner_multinomial = nlp_logistic_classification_learner(text_feature_cols=["text"],
                                                              target="y",
                                                              vectorizer_params={"min_df": 1},
                                                              logistic_params={"multi_class": "multinomial",
                                                                               "solver": "sag",
                                                                               "max_iter": 200},
                                                              prediction_column="prediction")

    predict_fn, pred_train, log = learner_multinomial(df_train_multinomial)

    pred_test = predict_fn(df_test_multinomial)

    expected_col_train = df_train_multinomial.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2"]
    expected_col_test = df_test_multinomial.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2"]
    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
    assert (pred_test.columns == pred_train.columns).all()


def test_lgbm_classification_learner():
    df_train_binary = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0],
        "w": [2, 1, 2, 0.5],
        'y': [0, 1, 0, 1]
    })

    df_test_binary = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1],
        "w": [1, 2, 0, 0.5],
        'y': [1, 0, 0, 1]
    })

    df_train_multinomial = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0, 10.0, 13.0],
        "x2": [0, 1, 1, 0, 1, 0],
        "w": [2, 1, 2, 0.5, 2, 0.5],
        'y': [0, 1, 2, 1, 2, 0]
    })

    df_test_multinomial = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0, -4.0, 0.0],
        "x2": [1, 1, 0, 1, 0, 1],
        "w": [1, 2, 0, 0.5, 0, 0.5],
        'y': [1, 2, 0, 1, 2, 0]
    })

    features = ["x1", "x2"]

    learner_binary = lgbm_classification_learner(features=features,
                                                 target="y",
                                                 learning_rate=0.1,
                                                 num_estimators=20,
                                                 extra_params={"max_depth": 4, "seed": 42},
                                                 prediction_column="prediction",
                                                 weight_column="w")

    predict_fn_binary, pred_train_binary, log = learner_binary(df_train_binary)

    pred_test_binary = predict_fn_binary(df_test_binary)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train_binary.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test_binary.columns.tolist())
    assert pred_test_binary.prediction.max() < 1
    assert pred_test_binary.prediction.min() > 0
    assert (pred_test_binary.columns == pred_train_binary.columns).all()

    # SHAP test (binary only)
    pred_shap = predict_fn_binary(df_test_binary, apply_shap=True)
    assert "shap_values" in pred_shap.columns
    assert "shap_expected_value" in pred_shap.columns
    assert np.vstack(pred_shap["shap_values"]).shape == (4, 2)

    # test multinomial case
    learner_multinomial = lgbm_classification_learner(features=features,
                                                      target="y",
                                                      learning_rate=0.1,
                                                      num_estimators=20,
                                                      extra_params={"max_depth": 2,
                                                                    "seed": 42,
                                                                    "objective": 'multiclass',
                                                                    "num_class": 3},
                                                      prediction_column="prediction")

    predict_fn_multinomial, pred_train_multinomial, log = learner_multinomial(df_train_multinomial)

    pred_test_multinomial = predict_fn_multinomial(df_test_multinomial)

    expected_col_train = df_train_binary.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2"]

    assert Counter(expected_col_train) == Counter(pred_train_multinomial.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test_multinomial.columns.tolist())
    assert (pred_test_multinomial.columns == pred_train_multinomial.columns).all()
