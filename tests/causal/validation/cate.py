import pandas as pd
import pytest

from fklearn.causal.validation.cate import (
    cate_mean_by_bin,
    cate_mean_by_bin_meta_evaluator,
)
from fklearn.validation.evaluators import r2_evaluator


DF_INPUT = pd.DataFrame(
    {
        "group_column": ["1", "1", "1", "1", "2", "2", "2", "2"],
        "bin_column": [1, 1, 2, 2, 1, 1, 2, 2],
        "prediction_column": [5, 3, 4, 7, 8, 6, 1, 2],
        "target_column": [10, 20, 50, 40, 70, 30, 50, 60],
    }
)


def test_delta_mean_by_group_and_bin():
    df_expected = pd.DataFrame(
        {"prediction_column": [3.0, -4.0], "target_column": [35.0, 10.0]}
    )

    df_result = cate_mean_by_bin(
        DF_INPUT,
        "group_column",
        "1",
        "bin_column",
        2,
        False,
        "prediction_column",
        "target_column",
    )
    pd.testing.assert_frame_equal(df_expected, df_result)


def test_cate_mean_by_bin_meta_evaluator():
    evaluation = cate_mean_by_bin_meta_evaluator(
        test_data=DF_INPUT,
        group_column="group_column",
        control_group_name="1",
        bin_column="bin_column",
        n_bins=2,
        allow_dropped_bins=False,
        inner_evaluator=r2_evaluator,
        eval_name="cate_r2_evaluator",
        prediction_column="prediction_column",
        target_column="target_column",
    )

    assert evaluation.get("cate_r2_evaluator") == -2.904


PROBLEMATIC_PARAMS_PAIRS = [
    (
        dict(test_data=DF_INPUT,
             group_column="group_column",
             control_group_name="3",  # invalid control group name
             bin_column="bin_column",
             n_bins=2,
             allow_dropped_bins=False,
             inner_evaluator=r2_evaluator,
             eval_name="cate_r2_evaluator",
             prediction_column="prediction_column",
             target_column="target_column"),
        ValueError
    ),
    (
        dict(test_data=DF_INPUT.assign(group_column="2"),  # everyone belongs to the same group
             group_column="group_column",
             control_group_name="2",
             bin_column="bin_column",
             n_bins=2,
             allow_dropped_bins=False,
             inner_evaluator=r2_evaluator,
             eval_name="cate_r2_evaluator",
             prediction_column="prediction_column",
             target_column="target_column"),
        RuntimeError
    ),
    (
        dict(test_data=DF_INPUT.assign(group_column=["1", "1", "1", "1", "1", "1", "1", "2"]),
             group_column="group_column",
             control_group_name="2",
             bin_column="bin_column",
             n_bins=3,
             allow_dropped_bins=False,  # won't be able to create three different bins
             inner_evaluator=r2_evaluator,
             eval_name="cate_r2_evaluator",
             prediction_column="prediction_column",
             target_column="target_column"),
        ValueError
    )
]


@pytest.mark.parametrize("problematic_params_pair", PROBLEMATIC_PARAMS_PAIRS)
def test_cate_mean_by_bin_meta_evaluator_errors(problematic_params_pair):
    params, err = problematic_params_pair
    with pytest.raises(err):
        cate_mean_by_bin_meta_evaluator(**params)
