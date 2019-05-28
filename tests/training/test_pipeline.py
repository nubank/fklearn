import itertools

import numpy as np
import pandas as pd
import pytest
import toolz as fp

from fklearn.training.imputation import placeholder_imputer
from fklearn.training.pipeline import build_pipeline
from fklearn.training.regression import xgb_regression_learner
from fklearn.training.transformation import count_categorizer, onehot_categorizer


def test_build_pipeline():
    df_train = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0, None, 13.0],
        "x2": [0, 1, 1, 0, 1, 0],
        "cat": ["c1", "c1", "c2", None, "c2", "c4"],
        'y': [2.3, 4.0, 100.0, -3.9, 100.0, -3.9]
    })

    df_test = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0, -4.0, 0.0],
        "x2": [1, 1, 0, None, 0, 1],
        "cat": ["c1", "c2", "c5", None, "c2", "c3"],
        'y': [1.3, -4.0, 0.0, 49, 0.0, 49]
    })

    features = ["x1", "x2", "cat"]
    target = "y"

    train_fn = build_pipeline(
        placeholder_imputer(columns_to_impute=features, placeholder_value=-999),
        count_categorizer(columns_to_categorize=["cat"]),
        xgb_regression_learner(features=features,
                               target=target,
                               num_estimators=20,
                               extra_params={"seed": 42}))

    predict_fn, pred_train, log = train_fn(df_train)

    pred_test_with_shap = predict_fn(df_test, apply_shap=True)
    assert set(pred_test_with_shap.columns) - set(pred_train.columns) == {"shap_values", "shap_expected_value"}

    pred_test_without_shap = predict_fn(df_test)
    assert set(pred_test_without_shap.columns) == set(pred_train.columns)

    pd.util.testing.assert_frame_equal(pred_test_with_shap[pred_test_without_shap.columns], pred_test_without_shap)


def test_build_pipeline_no_side_effects():
    test_df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    orig_df = test_df.copy()

    def side_effect_learner(df):
        df['side_effect1'] = df['x'] * 2
        return lambda dataset: dataset, df, {}

    def kwargs_learner(df):
        df['side_effect2'] = df['y'] * 2

        def p(dataset, mult=2):
            return dataset.assign(x=dataset.x * mult)

        return p, p(df), {}

    side_effect_pipeline = build_pipeline(side_effect_learner, kwargs_learner)
    side_effect_pipeline(test_df)

    pd.util.testing.assert_frame_equal(test_df, orig_df)


def test_build_pipeline_idempotency():
    test_df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})
    orig_df = test_df.copy()

    mult_constant = 2
    expected_df = pd.DataFrame({"x": np.array([1, 2, 3, 4, 5]) * mult_constant, "y": [2, 4, 6, 8, 10]})

    def kwargs_learner(df):
        def p(dataset, mult):
            return dataset.assign(x=dataset.x * mult)

        return p, p(df, mult_constant), {"kwargs_learner": {"mult_constant": mult_constant}}

    def dummy_learner(df):
        return lambda dataset: dataset, df, {"dummy_learner": {"dummy": {}}}

    for variation in itertools.permutations([dummy_learner, kwargs_learner, dummy_learner]):
        side_effect_pipeline = build_pipeline(*variation)
        predict_fn, result_df, log = side_effect_pipeline(test_df)

        pd.util.testing.assert_frame_equal(test_df, orig_df)
        pd.util.testing.assert_frame_equal(result_df, expected_df)
        pd.util.testing.assert_frame_equal(predict_fn(test_df, mult=mult_constant), expected_df)


def test_build_pipeline_learner_assertion():
    @fp.curry
    def learner(df, a, b, c=3):
        return lambda dataset: dataset + a + b + c, df, {}

    learner_fn = learner(b=2)

    with pytest.raises(AssertionError):
        build_pipeline(learner_fn)

    learner_fn = learner(a=1, b=2)

    build_pipeline(learner_fn)


def test_build_pipeline_predict_arguments_assertion():
    test_df = pd.DataFrame({"x": [1, 2, 3, 4, 5], "y": [2, 4, 6, 8, 10]})

    @fp.curry
    def invalid_learner(df):
        def p(dataset, *a, **b):
            return dataset + len(a) + len(b)

        return p, df, {}

    with pytest.raises(AssertionError):
        build_pipeline(invalid_learner)(test_df)


def test_build_pipeline_serialisation():
    df_train = pd.DataFrame({
        'id': ["id1"],
        'x1': [10.0],
        'y': [2.3]
    })

    fn = lambda x: x

    @fp.curry
    def dummy_learner(df, fn, call):
        return fn, df, {f"dummy_learner_{call}": {}}

    @fp.curry
    def dummy_learner_2(df, fn, call):
        return dummy_learner(df, fn, call)

    @fp.curry
    def dummy_learner_3(df, fn, call):
        return fn, df, {f"dummy_learner_{call}": {}, "obj": "a"}

    train_fn = build_pipeline(
        dummy_learner(fn=fn, call=1),
        dummy_learner_2(fn=fn, call=2),
        dummy_learner_3(fn=fn, call=3))

    predict_fn, pred_train, log = train_fn(df_train)

    fkml = {"pipeline": ["dummy_learner", "dummy_learner_2", "dummy_learner_3"],
            "output_columns": ['id', 'x1', 'y'],
            "features": ['id', 'x1', 'y'],
            "learners": {"dummy_learner": {"fn": fn, "log": {"dummy_learner_1": {}}},
                         "dummy_learner_2": {"fn": fn, "log": {"dummy_learner_2": {}}},
                         "dummy_learner_3": {"fn": fn, "log": {"dummy_learner_3": {}}, "obj": "a"}}}

    assert log["__fkml__"] == fkml
    assert "obj" not in log.keys()


def test_build_pipeline_with_onehotencoder():
    df_train = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4", "id3", "id4"],
        'x1': [10.0, 13.0, 10.0, 13.0, None, 13.0],
        "x2": [0, 1, 1, 0, 1, 0],
        "cat": ["c1", "c1", "c2", None, "c2", "c4"],
        'y': [2.3, 4.0, 100.0, -3.9, 100.0, -3.9]
    })

    df_test = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6", "id5", "id6"],
        'x1': [12.0, 1000.0, -4.0, 0.0, -4.0, 0.0],
        "x2": [1, 1, 0, None, 0, 1],
        "cat": ["c1", "c2", "c5", None, "c2", "c3"],
        'y': [1.3, -4.0, 0.0, 49, 0.0, 49]
    })

    features = ["x1", "x2", "cat"]
    target = "y"

    train_fn = build_pipeline(
        placeholder_imputer(columns_to_impute=["x1", "x2"], placeholder_value=-999),
        onehot_categorizer(columns_to_categorize=["cat"], hardcode_nans=True),
        xgb_regression_learner(features=features,
                               target=target,
                               num_estimators=20,
                               extra_params={"seed": 42}))

    predict_fn, pred_train, log = train_fn(df_train)

    pred_test = predict_fn(df_test)

    expected_feature_columns_after_encoding = ["x1", "x2", "fklearn_feat__cat==c1", "fklearn_feat__cat==c2",
                                               "fklearn_feat__cat==c4", "fklearn_feat__cat==nan"]

    assert set(pred_test.columns) == set(expected_feature_columns_after_encoding + ["id", target, "prediction"])
