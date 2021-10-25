import pandas as pd
from toolz import curry

from fklearn.types import EvalReturnType, UncurriedEvalFnType


def _validate_test_and_control_groups(
    test_data: pd.DataFrame, group_column: str, control_group_name: str
) -> bool:
    unique_values = test_data[group_column].unique()

    if control_group_name not in unique_values:
        raise ValueError("control group '{}' not found".format(control_group_name))

    n_groups = len(unique_values)
    if n_groups != 2:
        raise RuntimeError(
            "Exactly 2 groups are required for delta evaluations. found {}".format(
                n_groups
            )
        )
    test_group_name = (
        unique_values[0] if control_group_name == unique_values[1] else unique_values[1]
    )
    return test_group_name > control_group_name


def cate_mean_by_bin(
    test_data: pd.DataFrame,
    group_column: str,
    test_after_control: bool,
    bin_column: str,
    n_bins: int,
    allow_dropped_bins: bool,
    prediction_column: str,
    target_column: str,
) -> pd.DataFrame:
    quantile_column = bin_column + "_q" + str(n_bins)
    duplicates = "drop" if allow_dropped_bins else "raise"
    test_data_binned = test_data.assign(
        **{
            quantile_column: pd.qcut(
                test_data[bin_column], n_bins, duplicates=duplicates
            )
        }
    )

    gb_columns = [group_column, quantile_column]

    gb = (
        test_data_binned[gb_columns + [prediction_column, target_column]]
        .groupby(gb_columns)
        .mean()
        .sort_index(level=group_column, ascending=test_after_control)
    )
    return gb.groupby(quantile_column).diff().dropna().reset_index(drop=True)


@curry
def cate_mean_by_bin_meta_evaluator(
    test_data: pd.DataFrame,
    group_column: str,
    control_group_name: str,
    bin_column: str,
    n_bins: int,
    allow_dropped_bins: bool,
    inner_evaluator: UncurriedEvalFnType,
    eval_name: str = None,
    prediction_column: str = "prediction",
    target_column: str = "target",
) -> EvalReturnType:
    test_after_control = _validate_test_and_control_groups(test_data, group_column, control_group_name)

    try:
        gb = cate_mean_by_bin(
            test_data,
            group_column,
            test_after_control,
            bin_column,
            n_bins,
            allow_dropped_bins,
            prediction_column,
            target_column,
        )
    except ValueError:
        raise ValueError(
            "can't create {} bins for column '{}'. use 'allow_dropped_bins=True' to drop duplicated bins".format(
                n_bins, bin_column
            )
        )

    if eval_name is None:
        eval_name = (
            "cate_mean_by_bin_"
            + bin_column
            + "[{}q]".format(n_bins)
            + "__"
            + inner_evaluator.__name__
        )

    return inner_evaluator(
        test_data=gb,
        prediction_column=prediction_column,
        target_column=target_column,
        eval_name=eval_name,
    )
