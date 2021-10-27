import pandas as pd
from toolz import curry

from fklearn.types import EvalReturnType, UncurriedEvalFnType
from fklearn.validation.evaluators import r2_evaluator


def _validate_test_and_control_groups(
    test_data: pd.DataFrame, group_column: str, control_group_name: str
) -> str:
    """
    Checks whether `test_data` has data on exactly two different experiment groups: test and control. Also returns the
    name of the test group.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with `group_column` as a column.

    group_column : String
        The name of the column that tells whether rows belong to the test or control group.

    control_group_name : String
        The name of the control group.

    Returns
    ----------
    test_group_name: String
        The name of the test group.
    """
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
    return (
        unique_values[0] if control_group_name == unique_values[1] else unique_values[1]
    )


def _cate_mean_by_bin(
    test_data: pd.DataFrame,
    group_column: str,
    control_group_name: str,
    bin_column: str,
    n_bins: int,
    allow_dropped_bins: bool,
    prediction_column: str,
    target_column: str,
) -> pd.DataFrame:
    """
    Computes a dataframe with predicted and actual CATEs by bins of a given column.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with `group_column` as a column.

    group_column : String
        The name of the column that tells whether rows belong to the test or control group.

    control_group_name : String
        The name of the control group.

    bin_column : String
        The name of the column from which the quantiles will be created.

    n_bins : String
        The number of bins to be created.

    allow_dropped_bins : bool
        Whether to allow the function to drop duplicated quantiles.

    prediction_column : String
        The name of the column containing the predictions from the model being evaluated.

    target_column : String
        The name of the column containing the actual outcomes of the treatment.

    Returns
    ----------
    gb: pd.DataFrame
        The grouped dataframe with actual and predicted CATEs by bin.
    """
    test_group_name = _validate_test_and_control_groups(
        test_data, group_column, control_group_name
    )

    test_after_control = test_group_name > control_group_name

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
    allow_dropped_bins: bool = False,
    inner_evaluator: UncurriedEvalFnType = r2_evaluator,
    eval_name: str = None,
    prediction_column: str = "prediction",
    target_column: str = "target",
) -> EvalReturnType:
    """
    Evaluates the predictions of a causal model that outputs treatment outcomes w.r.t. its capabilities to predict the
    CATE.

    Due to the fundamental lack of counterfactual data, the CATEs are computed for bins of a given column. This function
    then applies a fklearn-like evaluator on top of the aggregated dataframe.

    Parameters
    ----------
    test_data : Pandas' DataFrame
        A Pandas' DataFrame with `group_column` as a column.

    group_column : String
        The name of the column that tells whether rows belong to the test or control group.

    control_group_name : String
        The name of the control group.

    bin_column : String
        The name of the column from which the quantiles will be created.

    n_bins : String
        The number of bins to be created.

    allow_dropped_bins : bool, optional (default=False)
        Whether to allow the function to drop duplicated quantiles.

    inner_evaluator : UncurriedEvalFnType, optional (default=r2_evaluator)
        An instance of a fklearn-like evaluator, which will be applied to the .

    eval_name : String, optional (default=None)
        The name of the evaluator as it will appear in the logs.

    prediction_column : String, optional (default=None)
        The name of the column containing the predictions from the model being evaluated.

    target_column : String, optional (default=None)
        The name of the column containing the actual outcomes of the treatment.

    Returns
    ----------
    log: dict
        A log-like dictionary with the evaluation by `inner_evaluator`
    """
    try:
        gb = _cate_mean_by_bin(
            test_data,
            group_column,
            control_group_name,
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
