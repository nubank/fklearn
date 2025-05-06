from collections import OrderedDict

import math
import pandas as pd
import pytest
from numpy import nan, round, sqrt, floor, log as ln
from numpy.testing import assert_almost_equal
from pandas.testing import assert_frame_equal

from fklearn.training.transformation import (
    selector,
    capper,
    floorer,
    prediction_ranger,
    count_categorizer,
    label_categorizer,
    quantile_biner,
    truncate_categorical,
    rank_categorical,
    onehot_categorizer,
    target_categorizer,
    standard_scaler,
    minmax_scaler,
    ecdfer,
    discrete_ecdfer,
    custom_transformer,
    value_mapper,
    null_injector,
    missing_warner,
)


def test_selector():
    input_df = pd.DataFrame(
        {"feat1": [1, 2, 3], "feat2": [100, 200, 300], "target": [0, 1, 0]}
    )

    expected = pd.DataFrame({"feat1": [1, 2, 3], "target": [0, 1, 0]})

    expected2 = pd.DataFrame({"feat1": [1, 2, 3]})

    pred_fn, data, log = selector(input_df, ["feat1", "target"], ["feat1"])

    # the transformed input df should contain both feat1 and target
    assert expected.equals(data)

    # but applying the result function should result in a df containing only feat1
    assert expected2.equals(pred_fn(input_df))


def test_capper():
    input_df = pd.DataFrame({"feat1": [10, 13, 50], "feat2": [50, 75, None]})

    input_df2 = pd.DataFrame({"feat1": [7, 15], "feat2": [200, None]})

    expected1 = pd.DataFrame({"feat1": [9, 9, 9], "feat2": [50, 75, None]})

    expected2 = pd.DataFrame({"feat1": [7, 9], "feat2": [75, None]})

    pred_fn1, data1, log = capper(input_df, ["feat1", "feat2"], {"feat1": 9})
    pred_fn2, data2, log = capper(
        input_df, ["feat1", "feat2"], {"feat1": 9}, suffix="_suffix"
    )
    pred_fn3, data3, log = capper(
        input_df, ["feat1", "feat2"], {"feat1": 9}, prefix="prefix_"
    )
    pred_fn4, data4, log = capper(
        input_df,
        ["feat1", "feat2"],
        {"feat1": 9},
        columns_mapping={"feat1": "feat1_raw", "feat2": "feat2_raw"},
    )

    assert expected1.equals(data1)
    assert expected2.equals(pred_fn1(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_suffix("_suffix")], axis=1
    ).equals(data2)
    assert pd.concat(
        [expected2, input_df2.copy().add_suffix("_suffix")], axis=1
    ).equals(pred_fn2(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_prefix("prefix_")], axis=1
    ).equals(data3)
    assert pd.concat(
        [expected2, input_df2.copy().add_prefix("prefix_")], axis=1
    ).equals(pred_fn3(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_suffix("_raw")], axis=1
    ).equals(data4)
    assert pd.concat(
        [expected2, input_df2.copy().add_suffix("_raw")], axis=1
    ).equals(pred_fn4(input_df2))


def test_floorer():
    input_df = pd.DataFrame({"feat1": [10, 13, 10], "feat2": [50, 75, None]})

    input_df2 = pd.DataFrame({"feat1": [7, 15], "feat2": [15, None]})

    expected1 = pd.DataFrame({"feat1": [11, 13, 11], "feat2": [50, 75, None]})

    expected2 = pd.DataFrame({"feat1": [11, 15], "feat2": [50, None]})

    pred_fn1, data1, log = floorer(input_df, ["feat1", "feat2"], {"feat1": 11})
    pred_fn2, data2, log = floorer(
        input_df, ["feat1", "feat2"], {"feat1": 11}, suffix="_suffix"
    )
    pred_fn3, data3, log = floorer(
        input_df, ["feat1", "feat2"], {"feat1": 11}, prefix="prefix_"
    )
    pred_fn4, data4, log = floorer(
        input_df,
        ["feat1", "feat2"],
        {"feat1": 11},
        columns_mapping={"feat1": "feat1_raw", "feat2": "feat2_raw"},
    )

    assert expected1.equals(data1)
    assert expected2.equals(pred_fn1(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_suffix("_suffix")], axis=1
    ).equals(data2)
    assert pd.concat(
        [expected2, input_df2.copy().add_suffix("_suffix")], axis=1
    ).equals(pred_fn2(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_prefix("prefix_")], axis=1
    ).equals(data3)
    assert pd.concat(
        [expected2, input_df2.copy().add_prefix("prefix_")], axis=1
    ).equals(pred_fn3(input_df2))

    assert pd.concat(
        [expected1, input_df.copy().add_suffix("_raw")], axis=1
    ).equals(data4)
    assert pd.concat(
        [expected2, input_df2.copy().add_suffix("_raw")], axis=1
    ).equals(pred_fn4(input_df2))


def test_prediction_ranger():
    input_df = pd.DataFrame(
        {"feat1": [10, 13, 10, 15], "prediction": [100, 200, 300, None]}
    )

    pred_fn, data, log = prediction_ranger(input_df, 150, 250)

    expected = pd.DataFrame(
        {"feat1": [10, 13, 10, 15], "prediction": [150, 200, 250, None]}
    )

    assert expected.equals(data)


def test_value_mapper():
    input_df = pd.DataFrame(
        {
            "feat1": [10, 10, 13, 15],
            "feat2": [100, 200, 300, None],
            "feat3": ["a", "b", "c", "b"],
        }
    )

    value_maps = {
        "feat1": {10: 1, 13: 2},
        "feat2": {100: [1, 2, 3]},
        "feat3": {"a": "b", "b": nan},
    }

    pred_fn, data_ignore, log = value_mapper(input_df, value_maps)
    pred_fn2, data_ignore2, log2 = value_mapper(input_df, value_maps, suffix="_suffix")
    pred_fn3, data_ignore3, log3 = value_mapper(input_df, value_maps, prefix="prefix_")
    pred_fn4, data_ignore4, log4 = value_mapper(input_df, value_maps,
                                                columns_mapping={"feat1": "feat1_raw",
                                                                 "feat2": "feat2_raw",
                                                                 "feat3": "feat3_raw"})
    pred_fn, data_not_ignore, log = value_mapper(
        input_df, value_maps, ignore_unseen=False
    )

    expected_ignore = pd.DataFrame(
        {
            "feat1": [1, 1, 2, 15],
            "feat2": [[1, 2, 3], 200, 300, None],
            "feat3": ["b", nan, "c", nan],
        }
    )

    expected_not_ignore = pd.DataFrame(
        {
            "feat1": [1, 1, 2, nan],
            "feat2": [[1, 2, 3], nan, nan, nan],
            "feat3": ["b", nan, nan, nan],
        }
    )

    assert expected_ignore.equals(data_ignore)
    assert expected_not_ignore.equals(data_not_ignore)

    assert pd.concat(
        [expected_ignore, input_df.copy().add_suffix("_suffix")], axis=1
    ).equals(data_ignore2)

    assert pd.concat(
        [expected_ignore, input_df.copy().add_prefix("prefix_")], axis=1
    ).equals(data_ignore3)

    assert pd.concat(
        [expected_ignore, input_df.copy().add_suffix("_raw")], axis=1
    ).equals(data_ignore4)


def test_truncate_categorical():
    input_df_train = pd.DataFrame(
        {
            "col": ["a", "a", "a", "b", "b", "b", "b", "c", "d", "f", nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    input_df_test = pd.DataFrame(
        {
            "col": ["a", "a", "b", "c", "d", "f", "e", nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 1],
        }
    )

    expected_output_train = pd.DataFrame(
        {
            "col": [
                "a",
                "a",
                "a",
                "b",
                "b",
                "b",
                "b",
                -9999,
                -9999,
                -9999,
                nan,
            ],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    expected_output_test = pd.DataFrame(
        {
            "col": ["a", "a", "b", -9999, -9999, -9999, -9999, nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 1],
        }
    )

    truncate_learner1 = truncate_categorical(
        columns_to_truncate=["col"], percentile=0.1
    )
    truncate_learner2 = truncate_categorical(
        columns_to_truncate=["col"], percentile=0.1, suffix="_suffix"
    )
    truncate_learner3 = truncate_categorical(
        columns_to_truncate=["col"], percentile=0.1, prefix="prefix_"
    )
    truncate_learner4 = truncate_categorical(
        columns_to_truncate=["col"],
        percentile=0.1,
        columns_mapping={"col": "col_raw"},
    )

    pred_fn1, data1, log = truncate_learner1(input_df_train)
    pred_fn2, data2, log = truncate_learner2(input_df_train)
    pred_fn3, data3, log = truncate_learner3(input_df_train)
    pred_fn4, data4, log = truncate_learner4(input_df_train)

    assert expected_output_train.equals(data1)
    assert expected_output_test.equals(pred_fn1(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(data2)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(pred_fn2(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(data3)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(pred_fn3(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(data4)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(pred_fn4(input_df_test))


def test_rank_categorical():
    input_df_train = pd.DataFrame(
        {"col": ["a", "b", "b", "c", "c", "d", "d", "d", nan, nan, nan]}
    )

    input_df_test = pd.DataFrame({"col": ["a", "b", "c", "d", "d", nan, nan]})

    expected_output_train = pd.DataFrame(
        {"col": [4, 2, 2, 3, 3, 1, 1, 1, nan, nan, nan]}
    )

    expected_output_test = pd.DataFrame({"col": [4, 2, 3, 1, 1, nan, nan]})

    pred_fn1, data1, log = rank_categorical(input_df_train, ["col"])
    pred_fn2, data2, log = rank_categorical(
        input_df_train, ["col"], suffix="_suffix"
    )
    pred_fn3, data3, log = rank_categorical(
        input_df_train, ["col"], prefix="prefix_"
    )
    pred_fn4, data4, log = rank_categorical(
        input_df_train, ["col"], columns_mapping={"col": "col_raw"}
    )

    assert expected_output_train.equals(data1)
    assert expected_output_test.equals(pred_fn1(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(data2)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(pred_fn2(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(data3)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(pred_fn3(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(data4)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(pred_fn4(input_df_test))


def test_count_categorizer():
    input_df_train = pd.DataFrame(
        {
            "feat1_num": [1, 0.5, nan, 100],
            "feat2_cat": ["a", "a", "a", "b"],
            "feat3_cat": ["c", "c", "c", nan],
        }
    )

    expected_output_train = pd.DataFrame(
        {
            "feat1_num": [1, 0.5, nan, 100],
            "feat2_cat": [3, 3, 3, 1],
            "feat3_cat": [3, 3, 3, nan],
        }
    )

    input_df_test = pd.DataFrame(
        {
            "feat1_num": [2, 20, 200, 2000],
            "feat2_cat": ["a", "b", "b", "d"],
            "feat3_cat": [nan, nan, "c", "c"],
        }
    )

    expected_output_test = pd.DataFrame(
        {
            "feat1_num": [2, 20, 200, 2000],
            "feat2_cat": [3, 1, 1, 1],  # replace unseen vars with constant (1)
            "feat3_cat": [nan, nan, 3, 3],
        }
    )

    categorizer_learner1 = count_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"], replace_unseen=1
    )
    categorizer_learner2 = count_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        replace_unseen=1,
        suffix="_suffix",
    )
    categorizer_learner3 = count_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        replace_unseen=1,
        prefix="prefix_",
    )
    categorizer_learner4 = count_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        replace_unseen=1,
        columns_mapping={
            "feat2_cat": "feat2_cat_raw",
            "feat3_cat": "feat3_cat_raw",
        },
    )

    pred_fn1, data1, log = categorizer_learner1(input_df_train)
    pred_fn2, data2, log = categorizer_learner2(input_df_train)
    pred_fn3, data3, log = categorizer_learner3(input_df_train)
    pred_fn4, data4, log = categorizer_learner4(input_df_train)

    assert expected_output_train.equals(data1)
    assert expected_output_test.equals(pred_fn1(input_df_test))

    categorized = ["feat2_cat", "feat3_cat"]
    assert pd.concat(
        [
            expected_output_train,
            input_df_train[categorized].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(data2)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[categorized].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(pred_fn2(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[categorized].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(data3)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[categorized].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(pred_fn3(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[categorized].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(data4)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[categorized].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(pred_fn4(input_df_test))


def test_label_categorizer():
    input_df_train = pd.DataFrame(
        {
            "feat1_num": [1, 0.5, nan, 100],
            "feat2_cat": ["a", "a", "a", "b"],
            "feat3_cat": ["c", "c", "c", nan],
        }
    )

    expected_output_train = pd.DataFrame(
        {
            "feat1_num": [1, 0.5, nan, 100],
            "feat2_cat": [0, 0, 0, 1],
            "feat3_cat": [0, 0, 0, nan],
        }
    )

    input_df_test = pd.DataFrame(
        {
            "feat1_num": [2, 20, 200, 2000],
            "feat2_cat": ["a", "b", "b", "d"],
            "feat3_cat": [nan, nan, "c", "c"],
        }
    )

    expected_output_test = pd.DataFrame(
        {
            "feat1_num": [2, 20, 200, 2000],
            "feat2_cat": [
                0,
                1,
                1,
                -99,
            ],  # replace unseen vars with constant (1)
            "feat3_cat": [nan, nan, 0, 0],
        }
    )

    categorizer_learner1 = label_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"], replace_unseen=-99
    )
    categorizer_learner2 = label_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        replace_unseen=-99,
        suffix="_suffix",
    )
    categorizer_learner3 = label_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        replace_unseen=-99,
        prefix="prefix_",
    )
    categorizer_learner4 = label_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        replace_unseen=-99,
        columns_mapping={
            "feat2_cat": "feat2_cat_raw",
            "feat3_cat": "feat3_cat_raw",
        },
    )

    pred_fn1, data1, log = categorizer_learner1(input_df_train)
    pred_fn2, data2, log = categorizer_learner2(input_df_train)
    pred_fn3, data3, log = categorizer_learner3(input_df_train)
    pred_fn4, data4, log = categorizer_learner4(input_df_train)

    assert expected_output_train.equals(data1)
    assert expected_output_test.equals(pred_fn1(input_df_test))

    categorized = ["feat2_cat", "feat3_cat"]
    assert pd.concat(
        [
            expected_output_train,
            input_df_train[categorized].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(data2)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[categorized].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(pred_fn2(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[categorized].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(data3)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[categorized].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(pred_fn3(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[categorized].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(data4)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[categorized].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(pred_fn4(input_df_test))


def test_quantile_biner():
    input_df_train = pd.DataFrame(
        {
            "col": [1.0, 2, 3, 4, 5, 6, 7, 8, 9, 10, nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    input_df_test = pd.DataFrame(
        {
            "col": [-1.0, 2, 3, 4, 20, 6, 7, 8, 9, -20, nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    expected_output_train = pd.DataFrame(
        {
            "col": [0.0, 1.0, 1.0, 2.0, 2.0, 3.0, 3.0, 4.0, 4.0, 4.0, nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    expected_output_test = pd.DataFrame(
        {
            "col": [0.0, 1, 1, 2, 5, 3, 3, 4, 4, 0, nan],
            "y": [1.0, 0, 1, 1, 1, 0, 1, 0, 1, 0, 1],
        }
    )

    biner_learner1 = quantile_biner(columns_to_bin=["col"], q=4, right=True)
    biner_learner2 = quantile_biner(
        columns_to_bin=["col"], q=4, right=True, suffix="_suffix"
    )
    biner_learner3 = quantile_biner(
        columns_to_bin=["col"], q=4, right=True, prefix="prefix_"
    )
    biner_learner4 = quantile_biner(
        columns_to_bin=["col"],
        q=4,
        right=True,
        columns_mapping={"col": "col_raw"},
    )

    pred_fn1, data1, log = biner_learner1(input_df_train)
    pred_fn2, data2, log = biner_learner2(input_df_train)
    pred_fn3, data3, log = biner_learner3(input_df_train)
    pred_fn4, data4, log = biner_learner4(input_df_train)

    assert expected_output_train.equals(data1)
    assert expected_output_test.equals(pred_fn1(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(data2)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(pred_fn2(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(data3)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(pred_fn3(input_df_test))

    assert pd.concat(
        [
            expected_output_train,
            input_df_train[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(data4)
    assert pd.concat(
        [
            expected_output_test,
            input_df_test[["col"]].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(pred_fn4(input_df_test))


@pytest.mark.parametrize(
    "df_train, df_test, columns_to_categorize, drop_first, hardcode, expected_output_train, expected_output_test",
    [(  # no drop_first - no hardcode
        pd.DataFrame({
            "feat1_num": [1, 0.5, nan, 100],
            "sex": ["female", "male", "male", "male"],
            "region": ["SP", "RG", "MG", nan]
        }),
        pd.DataFrame({
            "feat1_num": [2, 20, 200, 2000],
            "sex": ["male", "female", "male", "nonbinary"],
            "region": [nan, nan, "SP", "RG"]
        }),
        ["sex", "region"],
        False, False,
        pd.DataFrame({
            "feat1_num": [1, 0.5, nan, 100],
            "fklearn_feat__sex==female": [1, 0, 0, 0],
            "fklearn_feat__sex==male": [0, 1, 1, 1],
            "fklearn_feat__region==MG": [0, 0, 1, 0],
            "fklearn_feat__region==RG": [0, 1, 0, 0],
            "fklearn_feat__region==SP": [1, 0, 0, 0]
        }),
        pd.DataFrame({
            "feat1_num": [2, 20, 200, 2000],
            "fklearn_feat__sex==female": [0, 1, 0, 0],
            "fklearn_feat__sex==male": [1, 0, 1, 0],
            "fklearn_feat__region==MG": [0, 0, 0, 0],
            "fklearn_feat__region==RG": [0, 0, 0, 1],
            "fklearn_feat__region==SP": [0, 0, 1, 0]
        })
    ), (  # no drop_first - hardcode
        pd.DataFrame({
            "feat1_num": [1, 0.5, nan, 100],
            "sex": ["female", "male", "male", "male"],
            "region": ["SP", "RG", "MG", nan]
        }),
        pd.DataFrame({
            "feat1_num": [2, 20, 200, 2000],
            "sex": ["male", "female", "male", "nonbinary"],
            "region": [nan, nan, "SP", "RG"]
        }),
        ["sex", "region"],
        False, True,
        pd.DataFrame({
            "feat1_num": [1, 0.5, nan, 100],
            "fklearn_feat__sex==female": [1, 0, 0, 0],
            "fklearn_feat__sex==male": [0, 1, 1, 1],
            "fklearn_feat__sex==nan": [0, 0, 0, 0],
            "fklearn_feat__region==MG": [0, 0, 1, 0],
            "fklearn_feat__region==RG": [0, 1, 0, 0],
            "fklearn_feat__region==SP": [1, 0, 0, 0],
            "fklearn_feat__region==nan": [0, 0, 0, 1]
        }),
        pd.DataFrame({
            "feat1_num": [2, 20, 200, 2000],
            "fklearn_feat__sex==female": [0, 1, 0, 0],
            "fklearn_feat__sex==male": [1, 0, 1, 0],
            "fklearn_feat__sex==nan": [0, 0, 0, 1],
            "fklearn_feat__region==MG": [0, 0, 0, 0],
            "fklearn_feat__region==RG": [0, 0, 0, 1],
            "fklearn_feat__region==SP": [0, 0, 1, 0],
            "fklearn_feat__region==nan": [1, 1, 0, 0]
        }),
    ), (  # drop_first - hardcode
        pd.DataFrame({
            "feat1_num": [1, 0.5, nan, 100],
            "sex": ["female", "male", "male", "male"],
            "region": ["SP", "RG", "MG", nan]
        }),
        pd.DataFrame({
            "feat1_num": [2, 20, 200, 2000],
            "sex": ["male", "female", "male", "nonbinary"],
            "region": [nan, nan, "SP", "RG"]
        }),
        ["sex", "region"],
        True, True,
        pd.DataFrame({
            "feat1_num": [1, 0.5, nan, 100],
            "fklearn_feat__sex==male": [0, 1, 1, 1],
            "fklearn_feat__sex==nan": [0, 0, 0, 0],
            "fklearn_feat__region==RG": [0, 1, 0, 0],
            "fklearn_feat__region==SP": [1, 0, 0, 0],
            "fklearn_feat__region==nan": [0, 0, 0, 1]
        }),
        pd.DataFrame({
            "feat1_num": [2, 20, 200, 2000],
            "fklearn_feat__sex==male": [1, 0, 1, 0],
            "fklearn_feat__sex==nan": [0, 0, 0, 1],
            "fklearn_feat__region==RG": [0, 0, 0, 1],
            "fklearn_feat__region==SP": [0, 0, 1, 0],
            "fklearn_feat__region==nan": [1, 1, 0, 0],
        }),
    ), (  # drop_first - not hardcode
        pd.DataFrame({
            "feat1_num": [1, 0.5, nan, 100],
            "sex": ["female", "male", "male", "male"],
            "region": ["SP", "RG", "MG", nan]
        }),
        pd.DataFrame({
            "feat1_num": [2, 20, 200, 2000],
            "sex": ["male", "female", "male", "nonbinary"],
            "region": [nan, nan, "SP", "RG"]
        }),
        ["sex", "region"],
        True, False,
        pd.DataFrame({
            "feat1_num": [1, 0.5, nan, 100],
            "fklearn_feat__sex==male": [0, 1, 1, 1],
            "fklearn_feat__region==RG": [0, 1, 0, 0],
            "fklearn_feat__region==SP": [1, 0, 0, 0],
        }),
        pd.DataFrame({
            "feat1_num": [2, 20, 200, 2000],
            "fklearn_feat__sex==male": [1, 0, 1, 0],
            "fklearn_feat__region==RG": [0, 0, 0, 1],
            "fklearn_feat__region==SP": [0, 0, 1, 0]
        }),
    ),
    ]
)
def test_onehot_categorizer(
        df_train, df_test, columns_to_categorize, drop_first, hardcode, expected_output_train, expected_output_test
):

    categorizer_learner = onehot_categorizer(
        columns_to_categorize=columns_to_categorize, hardcode_nans=hardcode, drop_first_column=drop_first)

    pred_fn, data, log = categorizer_learner(df_train)
    test_result = pred_fn(df_test)

    assert_frame_equal(test_result, expected_output_test)
    assert_frame_equal(data, expected_output_train)


def test_target_categorizer():
    input_df_train_binary_target = pd.DataFrame(
        {
            "feat1_num": [1, 0.5, nan, 100, 10, 0.7],
            "feat2_cat": ["a", "a", "a", "b", "c", "c"],
            "feat3_cat": ["c", "c", "c", "a", "a", "a"],
            "target": [1, 0, 0, 1, 0, 1],
        }
    )

    expected_output_train_binary_target = pd.DataFrame(
        OrderedDict(
            (
                ("feat1_num", [1, 0.5, nan, 100, 10, 0.7]),
                ("feat2_cat", [0.375, 0.375, 0.375, 0.75, 0.5, 0.5]),
                ("feat3_cat", [0.375, 0.375, 0.375, 0.625, 0.625, 0.625]),
                ("target", [1, 0, 0, 1, 0, 1]),
            )
        )
    )

    input_df_test_binary_target = pd.DataFrame(
        {
            "feat1_num": [2.0, 4.0, 8.0],
            "feat2_cat": ["b", "a", "c"],
            "feat3_cat": ["c", "b", "a"],
        }
    )

    expected_output_test_binary_target = pd.DataFrame(
        OrderedDict(
            (
                ("feat1_num", [2.0, 4.0, 8.0]),
                ("feat2_cat", [0.75, 0.375, 0.5]),
                ("feat3_cat", [0.375, nan, 0.625]),
            )
        )
    )

    input_df_train_continuous_target = pd.DataFrame(
        {
            "feat1_num": [1, 0.5, nan, 100, 10, 0.7],
            "feat2_cat": ["a", "a", "a", nan, "c", "c"],
            "target": [41.0, 10.5, 23.0, 4.0, 5.5, 60.0],
        }
    )

    expected_output_train_continuous_target = pd.DataFrame(
        OrderedDict(
            (
                ("feat1_num", [1, 0.5, nan, 100, 10, 0.7]),
                (
                    "feat2_cat",
                    [24.625, 24.625, 24.625, nan, 29.83333, 29.83333],
                ),
                ("target", [41.0, 10.5, 23.0, 4.0, 5.5, 60.0]),
            )
        )
    )

    input_df_test_continuous_target = pd.DataFrame(
        {"feat1_num": [2.0, 4.0, 8.0], "feat2_cat": ["b", "a", "c"]}
    )

    expected_output_test_continuous_target = pd.DataFrame(
        OrderedDict(
            (
                ("feat1_num", [2.0, 4.0, 8.0]),
                ("feat2_cat", [24.0, 24.625, 29.83333]),
            )
        )
    )

    # Test with binary target
    categorizer_learner1 = target_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"], target_column="target"
    )
    categorizer_learner2 = target_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        target_column="target",
        suffix="_suffix",
    )
    categorizer_learner3 = target_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        target_column="target",
        prefix="prefix_",
    )
    categorizer_learner4 = target_categorizer(
        columns_to_categorize=["feat2_cat", "feat3_cat"],
        target_column="target",
        columns_mapping={
            "feat2_cat": "feat2_cat_raw",
            "feat3_cat": "feat3_cat_raw",
        },
    )

    pred_fn1, data1, log = categorizer_learner1(input_df_train_binary_target)
    pred_fn2, data2, log = categorizer_learner2(input_df_train_binary_target)
    pred_fn3, data3, log = categorizer_learner3(input_df_train_binary_target)
    pred_fn4, data4, log = categorizer_learner4(input_df_train_binary_target)

    assert expected_output_train_binary_target.equals(data1)
    assert expected_output_test_binary_target.equals(
        pred_fn1(input_df_test_binary_target)
    )

    categorized = ["feat2_cat", "feat3_cat"]
    assert pd.concat(
        [
            expected_output_train_binary_target,
            input_df_train_binary_target[categorized]
            .copy()
            .add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(data2)
    assert pd.concat(
        [
            expected_output_test_binary_target,
            input_df_test_binary_target[categorized]
            .copy()
            .add_suffix("_suffix"),
        ],
        axis=1,
    ).equals(pred_fn2(input_df_test_binary_target))

    assert pd.concat(
        [
            expected_output_train_binary_target,
            input_df_train_binary_target[categorized]
            .copy()
            .add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(data3)
    assert pd.concat(
        [
            expected_output_test_binary_target,
            input_df_test_binary_target[categorized]
            .copy()
            .add_prefix("prefix_"),
        ],
        axis=1,
    ).equals(pred_fn3(input_df_test_binary_target))

    assert pd.concat(
        [
            expected_output_train_binary_target,
            input_df_train_binary_target[categorized].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(data4)
    assert pd.concat(
        [
            expected_output_test_binary_target,
            input_df_test_binary_target[categorized].copy().add_suffix("_raw"),
        ],
        axis=1,
    ).equals(pred_fn4(input_df_test_binary_target))

    # Test with continuous target
    categorizer_learner1 = target_categorizer(
        columns_to_categorize=["feat2_cat"],
        ignore_unseen=False,
        target_column="target",
    )
    categorizer_learner2 = target_categorizer(
        columns_to_categorize=["feat2_cat"],
        ignore_unseen=False,
        target_column="target",
        suffix="_suffix",
    )
    categorizer_learner3 = target_categorizer(
        columns_to_categorize=["feat2_cat"],
        ignore_unseen=False,
        target_column="target",
        prefix="prefix_",
    )
    categorizer_learner4 = target_categorizer(
        columns_to_categorize=["feat2_cat"],
        ignore_unseen=False,
        target_column="target",
        columns_mapping={"feat2_cat": "feat2_cat_raw"},
    )

    pred_fn1, data1, log = categorizer_learner1(
        input_df_train_continuous_target
    )
    pred_fn2, data2, log = categorizer_learner2(
        input_df_train_continuous_target
    )
    pred_fn3, data3, log = categorizer_learner3(
        input_df_train_continuous_target
    )
    pred_fn4, data4, log = categorizer_learner4(
        input_df_train_continuous_target
    )

    assert_almost_equal(
        expected_output_train_continuous_target.values, data1.values, decimal=5
    )
    assert_almost_equal(
        expected_output_test_continuous_target.values,
        pred_fn1(input_df_test_continuous_target).values,
        decimal=5,
    )

    assert_almost_equal(
        expected_output_train_continuous_target.values,
        data2.drop(columns=["feat2_cat_suffix"]).values,
        decimal=5,
    )
    assert (
        input_df_train_continuous_target[["feat2_cat"]]
        .copy()
        .add_suffix("_suffix")
        .equals(data2[["feat2_cat_suffix"]])
    )
    assert_almost_equal(
        expected_output_test_continuous_target.values,
        pred_fn2(input_df_test_continuous_target)
        .drop(columns=["feat2_cat_suffix"])
        .values,
        decimal=5,
    )
    assert (
        input_df_test_continuous_target[["feat2_cat"]]
        .copy()
        .add_suffix("_suffix")
        .equals(pred_fn2(input_df_test_continuous_target)[["feat2_cat_suffix"]])
    )

    assert_almost_equal(
        expected_output_train_continuous_target.values,
        data3.drop(columns=["prefix_feat2_cat"]).values,
        decimal=5,
    )
    assert (
        input_df_train_continuous_target[["feat2_cat"]]
        .copy()
        .add_prefix("prefix_")
        .equals(data3[["prefix_feat2_cat"]])
    )
    assert_almost_equal(
        expected_output_test_continuous_target.values,
        pred_fn3(input_df_test_continuous_target)
        .drop(columns=["prefix_feat2_cat"])
        .values,
        decimal=5,
    )
    assert (
        input_df_test_continuous_target[["feat2_cat"]]
        .copy()
        .add_prefix("prefix_")
        .equals(pred_fn3(input_df_test_continuous_target)[["prefix_feat2_cat"]])
    )

    assert_almost_equal(
        expected_output_train_continuous_target.values,
        data4.drop(columns=["feat2_cat_raw"]).values,
        decimal=5,
    )
    assert (
        input_df_train_continuous_target[["feat2_cat"]]
        .copy()
        .rename(columns={"feat2_cat": "feat2_cat_raw"})
        .equals(data4[["feat2_cat_raw"]])
    )
    assert_almost_equal(
        expected_output_test_continuous_target.values,
        pred_fn4(input_df_test_continuous_target)
        .drop(columns=["feat2_cat_raw"])
        .values,
        decimal=5,
    )
    assert (
        input_df_test_continuous_target[["feat2_cat"]]
        .copy()
        .rename(columns={"feat2_cat": "feat2_cat_raw"})
        .equals(pred_fn4(input_df_test_continuous_target)[["feat2_cat_raw"]])
    )


def test_standard_scaler():
    input_df_train = pd.DataFrame({"feat1_num": [1.0, 0.5, 100.0]})

    expected_output_train = pd.DataFrame(
        {"feat1_num": [-0.70175673, -0.71244338, 1.4142001]}
    )

    input_df_test = pd.DataFrame({"feat1_num": [2.0, 4.0, 8.0]})

    expected_output_test = pd.DataFrame(
        {"feat1_num": [-0.68038342, -0.63763682, -0.55214362]}
    )

    pred_fn1, data1, log = standard_scaler(input_df_train, ["feat1_num"])
    pred_fn2, data2, log = standard_scaler(
        input_df_train, ["feat1_num"], suffix="_suffix"
    )
    pred_fn3, data3, log = standard_scaler(
        input_df_train, ["feat1_num"], prefix="prefix_"
    )
    pred_fn4, data4, log = standard_scaler(
        input_df_train,
        ["feat1_num"],
        columns_mapping={"feat1_num": "feat1_num_raw"},
    )

    assert_almost_equal(expected_output_train.values, data1.values, decimal=5)
    assert_almost_equal(
        expected_output_test.values, pred_fn1(input_df_test).values, decimal=5
    )

    assert_almost_equal(
        pd.concat(
            [
                expected_output_train,
                input_df_train[["feat1_num"]].copy().add_suffix("_suffix"),
            ],
            axis=1,
        ).values,
        data2.values,
        decimal=5,
    )
    assert_almost_equal(
        pd.concat(
            [
                expected_output_test,
                input_df_test[["feat1_num"]].copy().add_suffix("_suffix"),
            ],
            axis=1,
        ).values,
        pred_fn2(input_df_test).values,
        decimal=5,
    )

    assert_almost_equal(
        pd.concat(
            [
                expected_output_train,
                input_df_train[["feat1_num"]].copy().add_prefix("prefix_"),
            ],
            axis=1,
        ).values,
        data3.values,
        decimal=5,
    )
    assert_almost_equal(
        pd.concat(
            [
                expected_output_test,
                input_df_test[["feat1_num"]].copy().add_prefix("prefix_"),
            ],
            axis=1,
        ).values,
        pred_fn3(input_df_test).values,
        decimal=5,
    )

    assert_almost_equal(
        pd.concat(
            [
                expected_output_train,
                input_df_train[["feat1_num"]].copy().add_suffix("_raw"),
            ],
            axis=1,
        ).values,
        data4.values,
        decimal=5,
    )
    assert_almost_equal(
        pd.concat(
            [
                expected_output_test,
                input_df_test[["feat1_num"]].copy().add_suffix("_raw"),
            ],
            axis=1,
        ).values,
        pred_fn4(input_df_test).values,
        decimal=5,
    )


def test_minmax_scaler():
    input_df_train = pd.DataFrame({"feat1_num": [1.0, 0.5, 100.0],
                                   "feat2_num": [-4.5, 10.2, 7.4]})

    expected_output_train = pd.DataFrame(
        {"feat1_num": [0.005025, 0.0, 1.0],
         "feat2_num": [0.0, 1.0, 0.809524]}
    )

    input_df_test = pd.DataFrame({"feat1_num": [2.0, 4.0, 8.0],
                                  "feat2_num": [-8.0, -4.0, 8.0]})

    expected_output_test = pd.DataFrame(
        {"feat1_num": [0.015075, 0.035176, 0.075377],
         "feat2_num": [-0.238095, 0.034014, 0.850340]}
    )

    pred_fn1, data1, log = minmax_scaler(input_df_train, ["feat1_num", "feat2_num"])
    pred_fn2, data2, log = minmax_scaler(
        input_df_train, ["feat1_num", "feat2_num"], suffix="_suffix"
    )
    pred_fn3, data3, log = minmax_scaler(
        input_df_train, ["feat1_num", "feat2_num"], prefix="prefix_"
    )
    pred_fn4, data4, log = minmax_scaler(
        input_df_train,
        ["feat1_num"],
        columns_mapping={"feat1_num": "feat1_num_raw"},
    )

    assert_almost_equal(expected_output_train.values, data1.values, decimal=5)
    assert_almost_equal(
        expected_output_test.values, pred_fn1(input_df_test).values, decimal=5
    )

    assert_almost_equal(
        pd.concat(
            [
                expected_output_train,
                input_df_train[["feat1_num", "feat2_num"]].copy().add_suffix("_suffix"),
            ],
            axis=1,
        ).values,
        data2.values,
        decimal=5,
    )
    assert_almost_equal(
        pd.concat(
            [
                expected_output_test,
                input_df_test[["feat1_num", "feat2_num"]].copy().add_suffix("_suffix"),
            ],
            axis=1,
        ).values,
        pred_fn2(input_df_test).values,
        decimal=5,
    )

    assert_almost_equal(
        pd.concat(
            [
                expected_output_train,
                input_df_train[["feat1_num", "feat2_num"]].copy().add_prefix("prefix_"),
            ],
            axis=1,
        ).values,
        data3.values,
        decimal=5,
    )
    assert_almost_equal(
        pd.concat(
            [
                expected_output_test,
                input_df_test[["feat1_num", "feat2_num"]].copy().add_prefix("prefix_"),
            ],
            axis=1,
        ).values,
        pred_fn3(input_df_test).values,
        decimal=5,
    )

    assert_almost_equal(
        pd.concat(
            [
                expected_output_train[["feat1_num"]],
                input_df_train[["feat1_num"]].copy().add_suffix("_raw"),
            ],
            axis=1,
        ).values,
        data4[["feat1_num", "feat1_num_raw"]].values,
        decimal=5,
    )
    assert_almost_equal(
        pd.concat(
            [
                expected_output_test[["feat1_num"]],
                input_df_test[["feat1_num"]].copy().add_suffix("_raw"),
            ],
            axis=1,
        ).values,
        pred_fn4(input_df_test)[["feat1_num", "feat1_num_raw"]].values,
        decimal=5,
    )


def test_custom_transformer():
    input_df = pd.DataFrame(
        {
            "feat1": [1, 2, 3],
            "feat2": [math.e, math.e ** 2, math.e ** 3],
            "feat3": [1.5, 2.5, 3.5],
            "target": [1, 4, 9],
        }
    )

    expected = pd.DataFrame(
        {
            "feat1": [1, 2, 3],
            "feat2": [math.e, math.e ** 2, math.e ** 3],
            "feat3": [1.5, 2.5, 3.5],
            "target": [1.0, 2.0, 3.0],
        }
    )

    expected2 = pd.DataFrame(
        {
            "feat1": [1, 4, 9],
            "feat2": [math.e, math.e ** 2, math.e ** 3],
            "feat3": [1.5, 2.5, 3.5],
            "target": [1, 4, 9],
        }
    )

    # the transformed input df should contain the square root of the target column
    pred_fn1, data1, log = custom_transformer(input_df, ["target"], sqrt)
    pred_fn2, data2, log = custom_transformer(
        input_df, ["target"], sqrt, suffix="_suffix"
    )
    pred_fn3, data3, log = custom_transformer(
        input_df, ["target"], sqrt, prefix="prefix_"
    )
    pred_fn4, data4, log = custom_transformer(
        input_df, ["target"], sqrt, columns_mapping={"target": "target_raw"}
    )

    assert expected.equals(data1)
    assert pd.concat(
        [expected, input_df[["target"]].copy().add_suffix("_suffix")], axis=1
    ).equals(data2)
    assert pd.concat(
        [expected, input_df[["target"]].copy().add_prefix("prefix_")], axis=1
    ).equals(data3)
    assert pd.concat(
        [expected, input_df[["target"]].copy().add_suffix("_raw")], axis=1
    ).equals(data4)

    # the transformed input df should contain the squared value of the feat1 column
    pred_fn1, data1, log = custom_transformer(
        input_df, ["feat1"], lambda x: x ** 2
    )
    pred_fn2, data2, log = custom_transformer(
        input_df, ["feat1"], lambda x: x ** 2, suffix="_suffix"
    )
    pred_fn3, data3, log = custom_transformer(
        input_df, ["feat1"], lambda x: x ** 2, prefix="prefix_"
    )
    pred_fn4, data4, log = custom_transformer(
        input_df,
        ["feat1"],
        lambda x: x ** 2,
        columns_mapping={"feat1": "feat1_raw"},
    )

    assert expected2.equals(data1)
    assert pd.concat(
        [expected2, input_df[["feat1"]].copy().add_suffix("_suffix")], axis=1
    ).equals(data2)
    assert pd.concat(
        [expected2, input_df[["feat1"]].copy().add_prefix("prefix_")], axis=1
    ).equals(data3)
    assert pd.concat(
        [expected2, input_df[["feat1"]].copy().add_suffix("_raw")], axis=1
    ).equals(data4)

    expected3 = pd.DataFrame(
        {
            "feat1": [1, 2, 3],
            "feat2": [1.0, 2.0, 3.0],
            "feat3": [1.5, 2.5, 3.5],
            "target": [1, 4, 9],
        }
    )

    expected4 = pd.DataFrame(
        {
            "feat1": [1, 2, 3],
            "feat2": [math.e, math.e ** 2, math.e ** 3],
            "feat3": [1.0, 2.0, 3.0],
            "target": [1, 4, 9],
        }
    )

    # the transformed input df should contain the log of the target column
    pred_fn1, data1, log = custom_transformer(
        input_df, ["feat2"], ln, is_vectorized=True
    )
    pred_fn2, data2, log = custom_transformer(
        input_df, ["feat2"], ln, is_vectorized=True, suffix="_suffix"
    )
    pred_fn3, data3, log = custom_transformer(
        input_df, ["feat2"], ln, is_vectorized=True, prefix="prefix_"
    )
    pred_fn4, data4, log = custom_transformer(
        input_df,
        ["feat2"],
        ln,
        is_vectorized=True,
        columns_mapping={"feat2": "feat2_raw"},
    )

    assert_frame_equal(expected3, data1)
    assert_frame_equal(pd.concat(
        [expected3, input_df[["feat2"]].copy().add_suffix("_suffix")], axis=1
    ), data2)
    assert_frame_equal(pd.concat(
        [expected3, input_df[["feat2"]].copy().add_prefix("prefix_")], axis=1
    ), data3)
    assert_frame_equal(pd.concat(
        [expected3, input_df[["feat2"]].copy().add_suffix("_raw")], axis=1
    ), data4)

    # the transformed input df should contain the floor value of the feat1 column
    pred_fn1, data1, log = custom_transformer(
        input_df, ["feat3"], floor, is_vectorized=True
    )
    pred_fn2, data2, log = custom_transformer(
        input_df, ["feat3"], floor, is_vectorized=True, suffix="_suffix"
    )
    pred_fn3, data3, log = custom_transformer(
        input_df, ["feat3"], floor, is_vectorized=True, prefix="prefix_"
    )
    pred_fn4, data4, log = custom_transformer(
        input_df,
        ["feat3"],
        floor,
        is_vectorized=True,
        columns_mapping={"feat3": "feat3_raw"},
    )

    assert expected4.equals(data1)
    assert pd.concat(
        [expected4, input_df[["feat3"]].copy().add_suffix("_suffix")], axis=1
    ).equals(data2)
    assert pd.concat(
        [expected4, input_df[["feat3"]].copy().add_prefix("prefix_")], axis=1
    ).equals(data3)
    assert pd.concat(
        [expected4, input_df[["feat3"]].copy().add_suffix("_raw")], axis=1
    ).equals(data4)


def test_null_injector():
    train = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "c": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
            "d": [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0],
        }
    )

    test = pd.DataFrame(
        {
            "a": [5.0, 6.0, 7.0],
            "b": [1.0, 0.0, 0.0],
            "c": [8.0, 9.0, 9.0],
            "d": [1.0, 1.0, 1.0],
        }
    )

    p, result, log = null_injector(train, 0.3, ["a", "b"], seed=42)

    expected = pd.DataFrame(
        {
            "a": [1.0, nan, nan, 4.0, 5.0, 6.0, 7.0],
            "b": [1.0, 0.0, 0.0, 1.0, 1.0, nan, 1.0],
            "c": [9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
            "d": [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0],
        }
    )

    assert expected.equals(result)

    assert p(test).equals(test), "test must be left unchanged"

    # test group nans
    p, result, log = null_injector(
        train, 0.3, groups=[["a"], ["b", "c"]], seed=42
    )

    expected = pd.DataFrame(
        {
            "a": [1.0, nan, nan, 4.0, 5.0, 6.0, 7.0],
            "b": [1.0, 0.0, 0.0, 1.0, 1.0, nan, 1.0],
            "c": [9.0, 8.0, 7.0, 6.0, 5.0, nan, 3.0],
            "d": [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0],
        }
    )

    assert expected.equals(result)

    assert p(test).equals(test), "test must be left unchanged"


def test_ecdfer():
    fit_df = pd.DataFrame(
        {"prediction": [0.1, 0.1, 0.3, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9]}
    )

    input_df = pd.DataFrame(
        {"prediction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]}
    )

    expected_df = pd.DataFrame(
        {
            "prediction_ecdf": [
                200.0,
                200.0,
                300.0,
                300.0,
                500.0,
                700.0,
                700.0,
                800.0,
                900.0,
                1000.0,
                1000.0,
            ]
        }
    )

    ascending = True
    prediction_column = "prediction"
    ecdf_column = "prediction_ecdf"
    max_range = 1000

    pred_fn, data, log = ecdfer(
        fit_df, ascending, prediction_column, ecdf_column, max_range
    )
    actual_df = pred_fn(input_df)

    assert_almost_equal(
        expected_df[ecdf_column].values,
        actual_df[ecdf_column].values,
        decimal=5,
    )

    ascending = False
    pred_fn, data, log = ecdfer(
        fit_df, ascending, prediction_column, ecdf_column, max_range
    )

    expected_df = pd.DataFrame(
        {
            "prediction_ecdf": [
                800.0,
                800.0,
                700.0,
                700.0,
                500.0,
                300.0,
                300.0,
                200.0,
                100.0,
                0.0,
                0.0,
            ]
        }
    )
    actual_df = pred_fn(input_df)
    assert_almost_equal(
        expected_df[ecdf_column].values,
        actual_df[ecdf_column].values,
        decimal=5,
    )


def test_discrete_ecdfer():
    fit_df = pd.DataFrame(
        {"prediction": [0.1, 0.1, 0.3, 0.5, 0.5, 0.6, 0.6, 0.7, 0.8, 0.9]}
    )

    input_df = pd.DataFrame(
        {"prediction": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.65, 0.7, 0.8, 0.9, 1.0]}
    )

    ascending = True
    prediction_column = "prediction"
    ecdf_column = "prediction_ecdf"
    max_range = 1000

    ecdfer_fn, _, _ = ecdfer(
        fit_df, ascending, prediction_column, ecdf_column, max_range
    )
    ecdfer_df = ecdfer_fn(input_df)

    discrete_ecdfer_fn, _, _ = discrete_ecdfer(
        fit_df,
        ascending,
        prediction_column,
        ecdf_column,
        max_range,
        round_method=round,
    )
    discrete_ecdfer_df = discrete_ecdfer_fn(input_df)

    assert_almost_equal(
        ecdfer_df[ecdf_column].values,
        discrete_ecdfer_df[ecdf_column].values,
        decimal=5,
    )

    ascending = False
    ecdfer_fn, data, log = ecdfer(
        fit_df, ascending, prediction_column, ecdf_column, max_range
    )
    ecdfer_df = ecdfer_fn(input_df)

    discrete_ecdfer_fn, _, _ = discrete_ecdfer(
        fit_df,
        ascending,
        prediction_column,
        ecdf_column,
        max_range,
        round_method=float,
    )
    discrete_ecdfer_df = discrete_ecdfer_fn(input_df)

    assert_almost_equal(
        discrete_ecdfer_df[ecdf_column].values,
        ecdfer_df[ecdf_column].values,
        decimal=5,
    )


def test_missing_warner():
    train = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "c": [9.0, 8.0, nan, 6.0, 5.0, 4.0, 3.0],
            "d": [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0],
        }
    )

    test = pd.DataFrame(
        {
            "a": [5.0, nan, nan, 3.0, 3.0],
            "b": [1.0, nan, 0.0, 2.0, 2.0],
            "c": [nan, 9.0, 9.0, 7.0, 4.0],
            "d": [1.0, 1.0, 1.0, nan, 6.0],
        }
    )

    p, result, log = missing_warner(
        train, ["a", "b", "c"], "missing_alert_col_name"
    )

    expected_train_1 = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "c": [9.0, 8.0, nan, 6.0, 5.0, 4.0, 3.0],
            "d": [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0],
        }
    )

    expected_test_1 = pd.DataFrame(
        {
            "a": [5.0, nan, nan, 3.0, 3.0],
            "b": [1.0, nan, 0.0, 2.0, 2.0],
            "c": [nan, 9.0, 9.0, 7.0, 4.0],
            "d": [1.0, 1.0, 1.0, nan, 6.0],
            "missing_alert_col_name": [False, True, True, False, False],
        }
    )

    # train data should not change
    assert expected_train_1.equals(result)

    assert expected_test_1.equals(p(test))

    p, result, log = missing_warner(
        train,
        ["a", "b", "c"],
        "missing_alert_col_name",
        True,
        "missing_alert_explaining",
    )

    expected_train_2 = pd.DataFrame(
        {
            "a": [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0],
            "b": [1.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0],
            "c": [9.0, 8.0, nan, 6.0, 5.0, 4.0, 3.0],
            "d": [1.0, 8.0, 1.0, 4.0, 3.0, 4.0, 3.0],
        }
    )

    expected_test_2 = pd.DataFrame(
        {
            "a": [5.0, nan, nan, 3.0, 3.0],
            "b": [1.0, nan, 0.0, 2.0, 2.0],
            "c": [nan, 9.0, 9.0, 7.0, 4.0],
            "d": [1.0, 1.0, 1.0, nan, 6.0],
            "missing_alert_col_name": [False, True, True, False, False],
            "missing_alert_explaining": [[], ["a", "b"], ["a"], [], []],
        }
    )

    # train data should not change
    assert expected_train_2.equals(result)

    assert expected_test_2.equals(p(test))

    # checking when test has no nulls

    test_2 = pd.DataFrame(
        {
            "a": [5.0, 3.0, 2.0, 3.0, 3.0],
            "b": [1.0, 1.0, 0.0, 2.0, 2.0],
            "c": [3.0, 9.0, 9.0, 7.0, 4.0],
            "d": [1.0, 1.0, 1.0, 4.0, 6.0],
        }
    )

    expected_test_3 = pd.DataFrame(
        {
            "a": [5.0, 3.0, 2.0, 3.0, 3.0],
            "b": [1.0, 1.0, 0.0, 2.0, 2.0],
            "c": [3.0, 9.0, 9.0, 7.0, 4.0],
            "d": [1.0, 1.0, 1.0, 4.0, 6.0],
            "missing_alert_col_name": [False, False, False, False, False],
            "missing_alert_explaining": [[], [], [], [], []],
        }
    )

    assert expected_test_3.equals(p(test_2))
