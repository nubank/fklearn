import pandas as pd
import numpy as np
import itertools
import toolz as fp
import pytest

from fklearn.training.pipeline import build_pipeline
from fklearn.log_schemata import pipeline_schema


@fp.curry
def dummy_imputer(df, columns_to_impute, placeholder_value):
    def p(new_data_set):
        new_cols = new_data_set[columns_to_impute].fillna(placeholder_value).to_dict('list')
        return new_data_set.assign(**new_cols)

    log = {}
    return p, p(df), log


@fp.curry
def dummy_categorizer(df, columns_to_categorize, replace_unseen=-1):
    categ_getter = lambda col: df[col].value_counts().to_dict()
    vec = {column: categ_getter(column) for column in columns_to_categorize}

    def p(new_df):
        column_categorizer = lambda col: new_df[col].apply(lambda x: (np.nan
                                                                      if isinstance(x, float) and np.isnan(x)
                                                                      else vec[col].get(x, replace_unseen)))
        categ_columns = {col: column_categorizer(col) for col in columns_to_categorize}
        return new_df.assign(**categ_columns)

    log = {}

    return p, p(df), log


@fp.curry
def dummy_learner(df, **kwargs):
    return fp.identity, df, {'dummy_learner': {'dummy': 0}}


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
        dummy_imputer(columns_to_impute=features, placeholder_value=-999),
        dummy_categorizer(columns_to_categorize=["cat"]),
        dummy_learner(features=features,
                      target=target,
                      num_estimators=20,
                      extra_params={"seed": 42}))

    predict_fn, pred_train, log = train_fn(df_train)

    pipeline_schema.validate(log)

    pred_test = predict_fn(df_test)
    assert set(pred_test.columns) == set(pred_train.columns)


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

    for variation in itertools.permutations([dummy_learner, kwargs_learner, dummy_learner]):
        side_effect_pipeline = build_pipeline(*variation)
        predict_fn, result_df, log = side_effect_pipeline(test_df)
        pipeline_schema.validate(log)

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
