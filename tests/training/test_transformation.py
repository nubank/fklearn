from collections import OrderedDict

import math
import pandas as pd
from numpy import nan, round, sqrt, floor, log as ln
from numpy.testing import assert_almost_equal

from fklearn.training.transformation import \
    selector, capper, floorer, prediction_ranger, count_categorizer, label_categorizer, quantile_biner, \
    truncate_categorical, rank_categorical, onehot_categorizer, target_categorizer, standard_scaler, ecdfer, \
    discrete_ecdfer, custom_transformer, value_mapper, null_injector, missing_warner


def test_selector():
    input_df = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [100, 200, 300],
        'target': [0, 1, 0]
    })

    expected = pd.DataFrame({
        'feat1': [1, 2, 3],
        'target': [0, 1, 0]
    })

    expected2 = pd.DataFrame({
        'feat1': [1, 2, 3],
    })

    pred_fn, data, log = selector(input_df, ["feat1", "target"], ["feat1"])

    # the transformed input df should contain both feat1 and target
    assert expected.equals(data)

    # but applying the result function should result in a df containing only feat1
    assert expected2.equals(pred_fn(input_df))


def test_capper():
    input_df = pd.DataFrame({
        'feat1': [10, 13, 50],
        'feat2': [50, 75, None],
    })

    input_df2 = pd.DataFrame({
        'feat1': [7, 15],
        'feat2': [200, None],
    })

    expected1 = pd.DataFrame({
        'feat1': [9, 9, 9],
        'feat2': [50, 75, None],
    })

    expected2 = pd.DataFrame({
        'feat1': [7, 9],
        'feat2': [75, None],
    })

    pred_fn, data, log = capper(input_df, ["feat1", "feat2"], {'feat1': 9.0})

    assert expected1.equals(data)

    assert expected2.equals(pred_fn(input_df2))


def test_floorer():
    input_df = pd.DataFrame({
        'feat1': [10, 13, 10],
        'feat2': [50, 75, None],
    })

    input_df2 = pd.DataFrame({
        'feat1': [7, 15],
        'feat2': [15, None],
    })

    expected1 = pd.DataFrame({
        'feat1': [11, 13, 11],
        'feat2': [50, 75, None],
    })

    expected2 = pd.DataFrame({
        'feat1': [11, 15],
        'feat2': [50, None],
    })

    pred_fn, data, log = floorer(input_df, ["feat1", "feat2"], {'feat1': 11})

    assert expected1.equals(data)

    assert expected2.equals(pred_fn(input_df2))


def test_prediction_ranger():
    input_df = pd.DataFrame({
        'feat1': [10, 13, 10, 15],
        'prediction': [100, 200, 300, None],
    })

    pred_fn, data, log = prediction_ranger(input_df, 150, 250)

    expected = pd.DataFrame({
        'feat1': [10, 13, 10, 15],
        'prediction': [150, 200, 250, None],
    })

    assert expected.equals(data)


def test_value_mapper():
    input_df = pd.DataFrame({
        'feat1': [10, 10, 13, 15],
        'feat2': [100, 200, 300, None],
        'feat3': ['a', 'b', 'c', 'b']
    })

    value_maps = {'feat1': {10: 1, 13: 2},
                  'feat2': {100: [1, 2, 3]},
                  'feat3': {'a': 'b', 'b': nan}}

    pred_fn, data_ignore, log = value_mapper(input_df, value_maps)
    pred_fn, data_not_ignore, log = value_mapper(input_df, value_maps, ignore_unseen=False)

    expected_ignore = pd.DataFrame({
        'feat1': [1, 1, 2, 15],
        'feat2': [[1, 2, 3], 200, 300, None],
        'feat3': ['b', nan, 'c', nan]
    })

    expected_not_ignore = pd.DataFrame({
        'feat1': [1, 1, 2, nan],
        'feat2': [[1, 2, 3], nan, nan, nan],
        'feat3': ['b', nan, nan, nan]
    })

    assert expected_ignore.equals(data_ignore)
    assert expected_not_ignore.equals(data_not_ignore)


def test_count_categorizer():
    input_df_train = pd.DataFrame({
        "feat1_num": [1, 0.5, nan, 100],
        "feat2_cat": ["a", "a", "a", "b"],
        "feat3_cat": ["c", "c", "c", nan]
    })

    expected_output_train = pd.DataFrame({
        "feat1_num": [1, 0.5, nan, 100],
        "feat2_cat": [3, 3, 3, 1],
        "feat3_cat": [3, 3, 3, nan]
    })

    input_df_test = pd.DataFrame({
        "feat1_num": [2, 20, 200, 2000],
        "feat2_cat": ["a", "b", "b", "d"],
        "feat3_cat": [nan, nan, "c", "c"]
    })

    expected_output_test = pd.DataFrame({
        "feat1_num": [2, 20, 200, 2000],
        "feat2_cat": [3, 1, 1, 1],  # replace unseen vars with constant (1)
        "feat3_cat": [nan, nan, 3, 3]
    })

    categorizer_learner = count_categorizer(columns_to_categorize=["feat2_cat", "feat3_cat"],
                                            replace_unseen=1)

    pred_fn, data, log = categorizer_learner(input_df_train)

    test_result = pred_fn(input_df_test)

    assert data.equals(expected_output_train)
    assert test_result.equals(expected_output_test)


def test_label_categorizer():
    input_df_train = pd.DataFrame({
        "feat1_num": [1, 0.5, nan, 100],
        "feat2_cat": ["a", "a", "a", "b"],
        "feat3_cat": ["c", "c", "c", nan]
    })

    expected_output_train = pd.DataFrame({
        "feat1_num": [1, 0.5, nan, 100],
        "feat2_cat": [0, 0, 0, 1],
        "feat3_cat": [0, 0, 0, nan]
    })

    input_df_test = pd.DataFrame({
        "feat1_num": [2, 20, 200, 2000],
        "feat2_cat": ["a", "b", "b", "d"],
        "feat3_cat": [nan, nan, "c", "c"]
    })

    expected_output_test = pd.DataFrame({
        "feat1_num": [2, 20, 200, 2000],
        "feat2_cat": [0, 1, 1, -99],  # replace unseen vars with constant (1)
        "feat3_cat": [nan, nan, 0, 0]
    })

    categorizer_learner = label_categorizer(columns_to_categorize=["feat2_cat", "feat3_cat"],
                                            replace_unseen=-99)

    pred_fn, data, log = categorizer_learner(input_df_train)
    test_result = pred_fn(input_df_test)

    assert data.equals(expected_output_train)
    assert test_result.equals(expected_output_test)


def test_quantile_biner():
    input_df_train = pd.DataFrame({
        "col": [1., 2, 3, 4, 5, 6, 7, 8, 9, 10, nan],
        "y": [1., 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    })

    input_df_test = pd.DataFrame({
        "col": [-1., 2, 3, 4, 20, 6, 7, 8, 9, -20, nan],
        "y": [1., 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    })

    expected_output_train = pd.DataFrame({
        "col": [0., 1., 1., 2., 2., 3., 3., 4., 4., 4., nan],
        "y": [1., 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    })

    expected_output_test = pd.DataFrame({
        "col": [0., 1, 1, 2, 5, 3, 3, 4, 4, 0, nan],
        "y": [1., 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    })

    biner_learner = quantile_biner(columns_to_bin=["col"], q=4, right=True)

    pred_fn, data, log = biner_learner(input_df_train)
    test_result = pred_fn(input_df_test)

    assert data.equals(expected_output_train)
    assert test_result.equals(expected_output_test)


def test_truncate_categorical():
    input_df_train = pd.DataFrame({
        "col": ["a", "a", "a", "b", "b", "b", "b", "c", "d", "f", nan],
        "y": [1., 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    })

    input_df_test = pd.DataFrame({
        "col": ["a", "a", "b", "c", "d", "f", "e", nan],
        "y": [1., 0, 1, 1, 1, 0, 1, 1]
    })

    expected_output_train = pd.DataFrame({
        "col": ["a", "a", "a", "b", "b", "b", "b", -9999, -9999, -9999, nan],
        "y": [1., 0, 1, 1, 1, 0, 1, 0, 1, 0, 1]
    })

    expected_output_test = pd.DataFrame({
        "col": ["a", "a", "b", -9999, -9999, -9999, -9999, nan],
        "y": [1., 0, 1, 1, 1, 0, 1, 1]
    })

    truncate_learner = truncate_categorical(columns_to_truncate=["col"], percentile=0.1)

    pred_fn, data, log = truncate_learner(input_df_train)
    test_result = pred_fn(input_df_test)

    assert data.equals(expected_output_train)
    assert test_result.equals(expected_output_test)


def test_rank_categorical():
    input_df_train = pd.DataFrame({
        "col": ["a", "b", "b", "c", "c", "d", "d", "d", nan, nan, nan]
    })

    input_df_test = pd.DataFrame({
        "col": ["a", "b", "c", "d", "d", nan, nan]
    })

    expected_output_train = pd.DataFrame({
        "col": [4, 2, 2, 3, 3, 1, 1, 1, nan, nan, nan]
    })

    expected_output_test = pd.DataFrame({
        "col": [4, 2, 3, 1, 1, nan, nan]
    })

    pred_fn, data, log = rank_categorical(input_df_train, ["col"])
    test_result = pred_fn(input_df_test)

    assert expected_output_train.equals(data), "rank_categorical is not working as expected in train."
    assert expected_output_test.equals(test_result), "rank_categorical is not working as expected in test."


def test_onehot_categorizer():
    input_df_train = pd.DataFrame({
        "feat1_num": [1, 0.5, nan, 100],
        "sex": ["female", "male", "male", "male"],
        "region": ["SP", "RG", "MG", nan]
    })

    expected_output_train_no_drop_first_no_hardcode = pd.DataFrame(OrderedDict((
        ("feat1_num", [1, 0.5, nan, 100]),
        ("fklearn_feat__sex==female", [1, 0, 0, 0]),
        ("fklearn_feat__sex==male", [0, 1, 1, 1]),
        ("fklearn_feat__region==MG", [0, 0, 1, 0]),
        ("fklearn_feat__region==RG", [0, 1, 0, 0]),
        ("fklearn_feat__region==SP", [1, 0, 0, 0])
    )))

    expected_output_train_no_drop_first_hardcode = pd.DataFrame(OrderedDict((
        ("feat1_num", [1, 0.5, nan, 100]),
        ("fklearn_feat__sex==female", [1, 0, 0, 0]),
        ("fklearn_feat__sex==male", [0, 1, 1, 1]),
        ("fklearn_feat__sex==nan", [0, 0, 0, 0]),
        ("fklearn_feat__region==MG", [0, 0, 1, 0]),
        ("fklearn_feat__region==RG", [0, 1, 0, 0]),
        ("fklearn_feat__region==SP", [1, 0, 0, 0]),
        ("fklearn_feat__region==nan", [0, 0, 0, 1])
    )))

    expected_output_train_drop_first_hardcode = pd.DataFrame(OrderedDict((
        ("feat1_num", [1, 0.5, nan, 100]),
        ("fklearn_feat__sex==male", [0, 1, 1, 1]),
        ("fklearn_feat__sex==nan", [0, 0, 0, 0]),
        ("fklearn_feat__region==RG", [0, 1, 0, 0]),
        ("fklearn_feat__region==SP", [1, 0, 0, 0]),
        ("fklearn_feat__region==nan", [0, 0, 0, 1])
    )))

    expected_output_train_drop_first_no_hardcode = pd.DataFrame(OrderedDict((
        ("feat1_num", [1, 0.5, nan, 100]),
        ("fklearn_feat__sex==male", [0, 1, 1, 1]),
        ("fklearn_feat__region==RG", [0, 1, 0, 0]),
        ("fklearn_feat__region==SP", [1, 0, 0, 0])
    )))

    input_df_test = pd.DataFrame({
        "feat1_num": [2, 20, 200, 2000],
        "sex": ["male", "female", "male", "nonbinary"],
        "region": [nan, nan, "SP", "RG"]
    })

    expected_output_test_no_drop_first_no_hardcode = pd.DataFrame(OrderedDict((
        ("feat1_num", [2, 20, 200, 2000]),
        ("fklearn_feat__sex==female", [0, 1, 0, 0]),
        ("fklearn_feat__sex==male", [1, 0, 1, 0]),
        ("fklearn_feat__region==MG", [0, 0, 0, 0]),
        ("fklearn_feat__region==RG", [0, 0, 0, 1]),
        ("fklearn_feat__region==SP", [0, 0, 1, 0])
    )))

    expected_output_test_no_drop_first_hardcode = pd.DataFrame(OrderedDict((
        ("feat1_num", [2, 20, 200, 2000]),
        ("fklearn_feat__sex==female", [0, 1, 0, 0]),
        ("fklearn_feat__sex==male", [1, 0, 1, 0]),
        ("fklearn_feat__sex==nan", [0, 0, 0, 1]),
        ("fklearn_feat__region==MG", [0, 0, 0, 0]),
        ("fklearn_feat__region==RG", [0, 0, 0, 1]),
        ("fklearn_feat__region==SP", [0, 0, 1, 0]),
        ("fklearn_feat__region==nan", [1, 1, 0, 0])
    )))

    expected_output_test_drop_first_no_hardcode = pd.DataFrame(OrderedDict((
        ("feat1_num", [2, 20, 200, 2000]),
        ("fklearn_feat__sex==male", [1, 0, 1, 0]),
        ("fklearn_feat__region==RG", [0, 0, 0, 1]),
        ("fklearn_feat__region==SP", [0, 0, 1, 0])
    )))

    expected_output_test_drop_first_hardcode = pd.DataFrame(OrderedDict((
        ("feat1_num", [2, 20, 200, 2000]),
        ("fklearn_feat__sex==male", [1, 0, 1, 0]),
        ("fklearn_feat__sex==nan", [0, 0, 0, 1]),
        ("fklearn_feat__region==RG", [0, 0, 0, 1]),
        ("fklearn_feat__region==SP", [0, 0, 1, 0]),
        ("fklearn_feat__region==nan", [1, 1, 0, 0])
    )))

    # Test without hardcoding NaNs and not dropping first column
    categorizer_learner = onehot_categorizer(
        columns_to_categorize=["sex", "region"], hardcode_nans=False, drop_first_column=False)

    pred_fn, data, log = categorizer_learner(input_df_train)

    test_result = pred_fn(input_df_test)

    assert (test_result.equals(expected_output_test_no_drop_first_no_hardcode))
    assert (data.equals(expected_output_train_no_drop_first_no_hardcode))

    # Test with hardcoding NaNs and not dropping first column
    categorizer_learner = onehot_categorizer(
        columns_to_categorize=["sex", "region"], hardcode_nans=True, drop_first_column=False)

    pred_fn, data, log = categorizer_learner(input_df_train)

    test_result = pred_fn(input_df_test)

    assert (test_result.equals(expected_output_test_no_drop_first_hardcode))
    assert (data.equals(expected_output_train_no_drop_first_hardcode))

    # Testing dropping the first column and without hardcoding NaNs
    categorizer_learner = onehot_categorizer(
        columns_to_categorize=["sex", "region"], hardcode_nans=False,
        drop_first_column=True)

    pred_fn, data, log = categorizer_learner(input_df_train)

    test_result = pred_fn(input_df_test)

    assert (test_result.equals(expected_output_test_drop_first_no_hardcode))
    assert (data.equals(expected_output_train_drop_first_no_hardcode))

    # Testing dropping the first column and hardcoding NaNs
    categorizer_learner = onehot_categorizer(
        columns_to_categorize=["sex", "region"], hardcode_nans=True,
        drop_first_column=True)

    pred_fn, data, log = categorizer_learner(input_df_train)

    test_result = pred_fn(input_df_test)

    assert (test_result.equals(expected_output_test_drop_first_hardcode))
    assert (data.equals(expected_output_train_drop_first_hardcode))


def test_target_categorizer():
    input_df_train_binary_target = pd.DataFrame({
        "feat1_num": [1, 0.5, nan, 100, 10, 0.7],
        "feat2_cat": ["a", "a", "a", "b", "c", "c"],
        "feat3_cat": ["c", "c", "c", "a", "a", "a"],
        "target": [1, 0, 0, 1, 0, 1]
    })

    expected_output_train_binary_target = pd.DataFrame(OrderedDict((
        ("feat1_num", [1, 0.5, nan, 100, 10, 0.7]),
        ("feat2_cat", [0.375, 0.375, 0.375, 0.75, 0.5, 0.5]),
        ("feat3_cat", [0.375, 0.375, 0.375, 0.625, 0.625, 0.625]),
        ("target", [1, 0, 0, 1, 0, 1])
    )))

    input_df_test_binary_target = pd.DataFrame({
        "feat1_num": [2.0, 4.0, 8.0],
        "feat2_cat": ["b", "a", "c"],
        "feat3_cat": ["c", "b", "a"]
    })

    expected_output_test_binary_target = pd.DataFrame(OrderedDict((
        ("feat1_num", [2.0, 4.0, 8.0]),
        ("feat2_cat", [0.75, 0.375, 0.5]),
        ("feat3_cat", [0.375, nan, 0.625])
    )))

    input_df_train_continuous_target = pd.DataFrame({
        "feat1_num": [1, 0.5, nan, 100, 10, 0.7],
        "feat2_cat": ["a", "a", "a", nan, "c", "c"],
        "target": [41., 10.5, 23., 4., 5.5, 60.]
    })

    expected_output_train_continuous_target = pd.DataFrame(OrderedDict((
        ("feat1_num", [1, 0.5, nan, 100, 10, 0.7]),
        ("feat2_cat", [24.625, 24.625, 24.625, nan, 29.83333, 29.83333]),
        ("target", [41., 10.5, 23., 4., 5.5, 60.])
    )))

    input_df_test_continuous_target = pd.DataFrame({
        "feat1_num": [2.0, 4.0, 8.0],
        "feat2_cat": ["b", "a", "c"],
    })

    expected_output_test_continuous_target = pd.DataFrame(OrderedDict((
        ("feat1_num", [2.0, 4.0, 8.0]),
        ("feat2_cat", [24., 24.625, 29.83333]),
    )))

    # Test with binary target
    categorizer_learner = target_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"], target_column="target")

    pred_fn, data, log = categorizer_learner(input_df_train_binary_target)

    test_result = pred_fn(input_df_test_binary_target)

    assert (test_result[expected_output_test_binary_target.columns].  # we don't care about output order
            equals(expected_output_test_binary_target))

    assert (data[expected_output_train_binary_target.columns].  # we don't care about output order
            equals(expected_output_train_binary_target))

    # Test with continuous target
    categorizer_learner = target_categorizer(
        columns_to_categorize=["feat2_cat"], target_column="target", ignore_unseen=False)

    pred_fn, data, log = categorizer_learner(input_df_train_continuous_target)

    test_result = pred_fn(input_df_test_continuous_target)

    assert_almost_equal(test_result[expected_output_test_continuous_target.columns].values,
                        expected_output_test_continuous_target.values, decimal=5)

    assert_almost_equal(data[expected_output_train_continuous_target.columns].values,
                        expected_output_train_continuous_target.values, decimal=5)


def test_standard_scaler():
    input_df_train = pd.DataFrame({
        "feat1_num": [1.0, 0.5, 100.0],
    })

    expected_output_train = pd.DataFrame({
        "feat1_num": [-0.70175673, -0.71244338, 1.4142001],
    })

    input_df_test = pd.DataFrame({
        "feat1_num": [2.0, 4.0, 8.0],
    })

    expected_output_test = pd.DataFrame({
        "feat1_num": [-0.68038342, -0.63763682, -0.55214362],
    })

    pred_fn, train_result, log = standard_scaler(input_df_train, ["feat1_num"])
    test_result = pred_fn(input_df_test)

    assert_almost_equal(expected_output_train.values, train_result.values, decimal=5)

    assert_almost_equal(test_result.values, expected_output_test.values, decimal=5)


def test_ecdfer():
    fit_df = pd.DataFrame({
        "prediction": [0.1, 0.1, 0.3, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9]
    })

    input_df = pd.DataFrame({
        "prediction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
    })

    expected_df = pd.DataFrame({
        "prediction_ecdf": [200.0, 200.0, 300.0, 300.0, 500.0, 700.0, 700.0, 800.0, 900.0, 1000.0, 1000.0]
    })

    ascending = True
    prediction_column = "prediction"
    ecdf_column = "prediction_ecdf"
    max_range = 1000

    pred_fn, data, log = ecdfer(fit_df, ascending, prediction_column, ecdf_column, max_range)
    actual_df = pred_fn(input_df)

    assert_almost_equal(expected_df[ecdf_column].values, actual_df[ecdf_column].values, decimal=5)

    ascending = False
    pred_fn, data, log = ecdfer(fit_df, ascending, prediction_column, ecdf_column, max_range)

    expected_df = pd.DataFrame({
        "prediction_ecdf": [800.0, 800.0, 700.0, 700.0, 500.0, 300.0, 300.0, 200.0, 100.0, 0.0, 0.0]
    })
    actual_df = pred_fn(input_df)
    assert_almost_equal(expected_df[ecdf_column].values, actual_df[ecdf_column].values, decimal=5)


def test_discrete_ecdfer():
    fit_df = pd.DataFrame({
        "prediction": [0.1, 0.1, 0.3, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9]
    })

    input_df = pd.DataFrame({
        "prediction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]
    })

    ascending = True
    prediction_column = "prediction"
    ecdf_column = "prediction_ecdf"
    max_range = 1000

    ecdfer_fn, _, _ = ecdfer(fit_df, ascending, prediction_column, ecdf_column, max_range)
    ecdfer_df = ecdfer_fn(input_df)

    discrete_ecdfer_fn, _, _ = discrete_ecdfer(
        fit_df, ascending, prediction_column, ecdf_column, max_range, round_method=round)
    discrete_ecdfer_df = discrete_ecdfer_fn(input_df)

    assert_almost_equal(ecdfer_df[ecdf_column].values, discrete_ecdfer_df[ecdf_column].values, decimal=5)

    ascending = False
    ecdfer_fn, data, log = ecdfer(fit_df, ascending, prediction_column, ecdf_column, max_range)
    ecdfer_df = ecdfer_fn(input_df)

    discrete_ecdfer_fn, _, _ = discrete_ecdfer(
        fit_df, ascending, prediction_column, ecdf_column, max_range, round_method=float)
    discrete_ecdfer_df = discrete_ecdfer_fn(input_df)

    assert_almost_equal(discrete_ecdfer_df[ecdf_column].values, ecdfer_df[ecdf_column].values, decimal=5)


def test_custom_transformer():
    input_df = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [math.e, math.e ** 2, math.e ** 3],
        'feat3': [1.5, 2.5, 3.5],
        'target': [1, 4, 9]
    })

    expected = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [math.e, math.e ** 2, math.e ** 3],
        'feat3': [1.5, 2.5, 3.5],
        'target': [1.0, 2.0, 3.0]
    })

    expected2 = pd.DataFrame({
        'feat1': [1, 4, 9],
        'feat2': [math.e, math.e ** 2, math.e ** 3],
        'feat3': [1.5, 2.5, 3.5],
        'target': [1, 4, 9]
    })

    transformer_fn, data, log = custom_transformer(input_df, ["target"], sqrt)

    # the transformed input df should contain the square root of the target column
    assert expected.equals(data)

    transformer_fn, data, log = custom_transformer(input_df, ["feat1"], lambda x: x ** 2)

    # the transformed input df should contain the squared value of the feat1 column
    assert expected2.equals(data)

    expected3 = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [1.0, 2.0, 3.0],
        'feat3': [1.5, 2.5, 3.5],
        'target': [1, 4, 9]
    })

    expected4 = pd.DataFrame({
        'feat1': [1, 2, 3],
        'feat2': [math.e, math.e ** 2, math.e ** 3],
        'feat3': [1.0, 2.0, 3.0],
        'target': [1, 4, 9]
    })

    transformer_fn, data, log = custom_transformer(input_df, ["feat2"], ln, is_vectorized=True)

    # the transformed input df should contain the square root of the target column
    assert expected3.equals(data)

    transformer_fn, data, log = custom_transformer(input_df, ["feat3"], floor, is_vectorized=True)

    # the transformed input df should contain the squared value of the feat1 column
    assert expected4.equals(data)


def test_null_injector():
    train = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        'b': [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        'c': [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        'd': [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0],
    })

    test = pd.DataFrame({
        'a': [5.0, 6.0, 7.0],
        'b': [1.0, 0.0, 0.0],
        'c': [8.0, 9.0, 9.0],
        'd': [1.0, 1.0, 1.0]
    })

    p, result, log = null_injector(train, 0.3, ["a", "b"], seed=42)

    expected = pd.DataFrame({
        'a': [1.0, nan, nan, 4.0, 5.0, 6.0, 7.0],
        'b': [1.0, 0.0, 0.0, 1.0, 1.0, nan, 1.0],
        'c': [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        'd': [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0],
    })

    assert expected.equals(result)

    assert p(test).equals(test), "test must be left unchanged"

    # test group nans
    p, result, log = null_injector(train, 0.3, groups=[["a"], ["b", "c"]], seed=42)

    expected = pd.DataFrame({
        'a': [1.0, nan, nan, 4.0, 5.0, 6.0, 7.0],
        'b': [1.0, 0.0, 0.0, 1.0, 1.0, nan, 1.0],
        'c': [9.0, 8.0, 7.0, 6.0, 5.0, nan, 3.0],
        'd': [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0],
    })

    assert expected.equals(result)

    assert p(test).equals(test), "test must be left unchanged"


def test_missing_warner():
    train = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        'b': [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        'c': [9.0, 8.0, nan, 6.0, 5.0, 4.0, 3.0],
        'd': [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0]
    })

    test = pd.DataFrame({
        'a': [5.0, nan, nan, 3.0, 3.0],
        'b': [1.0, nan, 0.0, 2.0, 2.0],
        'c': [nan, 9.0, 9.0, 7.0, 4.0],
        'd': [1.0, 1.0, 1.0, nan, 6.0]
    })

    p, result, log = missing_warner(train, ["a", "b", "c"], "missing_alert_col_name")

    expected_train_1 = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        'b': [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        'c': [9.0, 8.0, nan, 6.0, 5.0, 4.0, 3.0],
        'd': [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0]
    })

    expected_test_1 = pd.DataFrame({
        'a': [5.0, nan, nan, 3.0, 3.0],
        'b': [1.0, nan, 0.0, 2.0, 2.0],
        'c': [nan, 9.0, 9.0, 7.0, 4.0],
        'd': [1.0, 1.0, 1.0, nan, 6.0],
        'missing_alert_col_name': [False, True, True, False, False]
    })

    # train data should not change
    assert expected_train_1.equals(result)

    assert expected_test_1.equals(p(test))

    p, result, log = missing_warner(train, ["a", "b", "c"], "missing_alert_col_name", True, "missing_alert_explaining")

    expected_train_2 = pd.DataFrame({
        'a': [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
        'b': [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
        'c': [9.0, 8.0, nan, 6.0, 5.0, 4.0, 3.0],
        'd': [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0]
    })

    expected_test_2 = pd.DataFrame({
        'a': [5.0, nan, nan, 3.0, 3.0],
        'b': [1.0, nan, 0.0, 2.0, 2.0],
        'c': [nan, 9.0, 9.0, 7.0, 4.0],
        'd': [1.0, 1.0, 1.0, nan, 6.0],
        'missing_alert_col_name': [False, True, True, False, False],
        'missing_alert_explaining': [[], ["a", "b"], ["a"], [], []]
    })

    # train data should not change
    assert expected_train_2.equals(result)

    assert expected_test_2.equals(p(test))

    # checking when test has no nulls

    test_2 = pd.DataFrame({
        'a': [5.0, 3.0, 2.0, 3.0, 3.0],
        'b': [1.0, 1.0, 0.0, 2.0, 2.0],
        'c': [3.0, 9.0, 9.0, 7.0, 4.0],
        'd': [1.0, 1.0, 1.0, 4.0, 6.0]
    })

    expected_test_3 = pd.DataFrame({
        'a': [5.0, 3.0, 2.0, 3.0, 3.0],
        'b': [1.0, 1.0, 0.0, 2.0, 2.0],
        'c': [3.0, 9.0, 9.0, 7.0, 4.0],
        'd': [1.0, 1.0, 1.0, 4.0, 6.0],
        'missing_alert_col_name': [False, False, False, False, False],
        'missing_alert_explaining': [[], [], [], [], []]
    })

    assert expected_test_3.equals(p(test_2))
