# -*- coding: utf-8 -*-
from collections import Counter

import numpy as np
import pandas as pd

from fklearn.training.classification import \
    logistic_classification_learner, xgb_classification_learner, \
    nlp_logistic_classification_learner, lgbm_classification_learner, \
    catboost_classification_learner
from unittest.mock import MagicMock, patch, Mock


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

    # SHAP test
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

    # SHAP test multinomial
    pred_shap_multinomial = predict_fn_multinomial(df_test_multinomial, apply_shap=True)

    expected_col_shap = expected_col_test + ["shap_values_0", "shap_values_1", "shap_values_2"] + \
        ["shap_expected_value_0", "shap_expected_value_1", "shap_expected_value_2"]
    assert Counter(expected_col_shap) == Counter(pred_shap_multinomial.columns.tolist())
    assert np.vstack(pred_shap_multinomial["shap_values_0"]).shape == (6, 2)


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

    df_train_categorical = pd.DataFrame({
        'id': ["id4", "id4", "id5"],
        "x1": ["a", "a", "c"],
        "x2": ["b", "b", "d"],
        "x3": [1, 4, 30],
        "x4": [1, 5, 40],
        "x5": [5, 6, 50],
        "x6": [6, 7, 60],
        "y": [1, 1, 0],
        "w": [1, 1, 1]
    })

    df_test_categorical = pd.DataFrame({
        'id': ["id4", "id4"],
        "x1": ["a", "a"],
        "x2": ["b", "b"],
        "x3": [2, 5],
        "x4": [2, 6],
        "x5": [5, 6],
        "x6": [6, 7],
        "y": [1, 1],
        "w": [1, 1]
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

    # SHAP test
    pred_shap = predict_fn_binary(df_test_binary, apply_shap=True)
    assert "shap_values" in pred_shap.columns
    assert "shap_expected_value" in pred_shap.columns
    assert np.vstack(pred_shap["shap_values"]).shape == (4, 2)

    # test multinomial case
    learner_multinomial = catboost_classification_learner(features=features,
                                                          target="y",
                                                          learning_rate=0.1,
                                                          num_estimators=20,
                                                          extra_params={"max_depth": 2,
                                                                        "random_seed": 42,
                                                                        "objective": 'MultiClass'},
                                                          prediction_column="prediction")

    predict_fn_multinomial, pred_train_multinomial, log = learner_multinomial(df_train_multinomial)

    pred_test_multinomial = predict_fn_multinomial(df_test_multinomial)

    expected_col_train = df_train_binary.columns.tolist(
    ) + ["prediction_0", "prediction_1", "prediction_2", "prediction"]
    expected_col_test = df_test_binary.columns.tolist() + ["prediction_0", "prediction_1", "prediction_2", "prediction"]

    assert Counter(expected_col_train) == Counter(pred_train_multinomial.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test_multinomial.columns.tolist())
    assert (pred_test_multinomial.columns == pred_train_multinomial.columns).all()

    # SHAP test multinomial
    pred_shap_multinomial = predict_fn_multinomial(df_test_multinomial, apply_shap=True)

    expected_col_shap = expected_col_test + ["shap_values_0", "shap_values_1", "shap_values_2"] + \
        ["shap_expected_value_0", "shap_expected_value_1", "shap_expected_value_2"]
    assert Counter(expected_col_shap) == Counter(pred_shap_multinomial.columns.tolist())
    assert np.vstack(pred_shap_multinomial["shap_values_0"]).shape == (6, 2)

    # test categorical case
    features = ["x1", "x2", "x3", "x4", "x5", "x6"]

    learner_binary = catboost_classification_learner(features=features,
                                                     target="y",
                                                     learning_rate=0.1,
                                                     num_estimators=20,
                                                     extra_params={"max_depth": 2,
                                                                   "random_seed": 42, 'cat_features': [0, 1]},
                                                     prediction_column="prediction",
                                                     weight_column="w")

    predict_fn_categorical, pred_train_categorical, log = learner_binary(df_train_categorical)

    pred_test_categorical = predict_fn_categorical(df_train_categorical)

    expected_col_train = df_train_categorical.columns.tolist() + ["prediction"]
    expected_col_test = df_test_categorical.columns.tolist() + ["prediction"]

    assert Counter(expected_col_train) == Counter(pred_train_categorical.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test_categorical.columns.tolist())
    assert pred_test_categorical.prediction.max() < 1
    assert pred_test_categorical.prediction.min() > 0
    assert (pred_test_binary.columns == pred_train_binary.columns).all()


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

    # SHAP test
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

    # SHAP test multinomial
    pred_shap_multinomial = predict_fn_multinomial(df_test_multinomial, apply_shap=True)

    expected_col_shap = expected_col_test + ["shap_values_0", "shap_values_1", "shap_values_2"] + \
        ["shap_expected_value_0", "shap_expected_value_1", "shap_expected_value_2"]
    assert Counter(expected_col_shap) == Counter(pred_shap_multinomial.columns.tolist())
    assert np.vstack(pred_shap_multinomial["shap_values_0"]).shape == (6, 2)


def test_lgbm_classification_learner_params():
    import lightgbm
    # Test input parameters

    df = pd.DataFrame(
        {"feat1": [1, 2, 1, 1, 1, 0],
         "feat2": [0.1, 0.5, 0.2, 0.5, 0.0, 0.1],
         "target": [1, 0, 1, 1, 0, 0]
         }
    )

    features = ["feat1", "feat2"]
    target = "target"

    df_result = pd.DataFrame(
        {"feat1": [1, 2, 1, 1, 1, 0],
         "feat2": [0.1, 0.5, 0.2, 0.5, 0.0, 0.1],
         "target": [1, 0, 1, 1, 0, 0],
         "prediction": [0.9, 0.0, 1.0, 1.0, 0.0, 0.0],
         }
    )

    lgbm_dataset = lightgbm.Dataset(df[features].values, label=df[target])

    mock_lgbm = MagicMock()
    mock_lgbm.predict.return_value = df_result["prediction"]
    mock_lgbm.Dataset.return_value = lgbm_dataset
    mock_lgbm.train.return_value = mock_lgbm

    mock_lgbm.__version__ = Mock(return_value='1.0')

    with patch.dict("sys.modules", lightgbm=mock_lgbm):
        # default settings
        lgbm_classification_learner(df=df,
                                    features=["feat1", "feat2"],
                                    target="target",
                                    learning_rate=0.1,
                                    num_estimators=100
                                    )

        mock_lgbm.train.assert_called()
        mock_lgbm.train.assert_called_with(
            params={'eta': 0.1, 'objective': 'binary'},
            train_set=lgbm_dataset,
            num_boost_round=100,
            valid_sets=None,
            valid_names=None,
            feval=None,
            init_model=None,
            feature_name='auto',
            categorical_feature='auto',
            keep_training_booster=False,
            callbacks=None
        )

        # Non default value for keep training booster
        lgbm_classification_learner(
            df=df,
            features=["feat1", "feat2"],
            target="target",
            learning_rate=0.1,
            num_estimators=100,
            valid_sets=None,
            valid_names=None,
            feval=None,
            init_model=None,
            feature_name='auto',
            categorical_feature='auto',
            keep_training_booster=True,
            callbacks=None,
            dataset_init_score=None
        )

        mock_lgbm.train.assert_called_with(
            params={'eta': 0.1, 'objective': 'binary'},
            train_set=lgbm_dataset,
            num_boost_round=100,
            valid_sets=None,
            valid_names=None,
            feval=None,
            init_model=None,
            feature_name='auto',
            categorical_feature='auto',
            keep_training_booster=True,
            callbacks=None
        )
