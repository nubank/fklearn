import pandas as pd
import pytest

from fklearn.training.imputation import imputer, placeholder_imputer


def test_imputer():
    input_df = pd.DataFrame({"col1": [10, 13, 10], "col2": [50, 100, None]})

    input_df2 = pd.DataFrame({"col1": [10, None], "col2": [None, 100]})

    expected1 = pd.DataFrame({"col1": [10.0, 13.0, 10.0], "col2": [50.0, 100.0, 75.0]})

    expected2 = pd.DataFrame({"col1": [10, 11.0], "col2": [75.0, 100]})

    pred_fn, data, log = imputer(input_df, ["col1", "col2"], "mean")

    assert expected1.equals(data)
    assert expected2.equals(pred_fn(input_df2))


def test_imputer_with_fill_value():
    input_df = pd.DataFrame({"col1": [10, 13, 10], "col2": [50, 100, None], "col3": [None, None, None]})

    df = pd.DataFrame({"col1": [10.0, 13.0, 10.0], "col2": [50.0, 100.0, 75.0], "col3": [10.0, None, None]})

    expected = pd.DataFrame({"col1": [10.0, 13.0, 10.0], "col2": [50.0, 100.0, 75.0], "col3": [10.0, -999, -999]})

    pred_fn, data, log = imputer(input_df, ["col1", "col2", "col3"], "mean", placeholder_value=-999)

    assert expected.equals(pred_fn(df))


def _cells_equal(actual, expected):
    # Compare element-wise treating NaN/None as equal to NaN/None, so the
    # per-strategy expectations below can mix real numbers with missing values.
    return all((pd.isna(a) and pd.isna(e)) or a == e for a, e in zip(actual, expected))


@pytest.mark.parametrize(
    "impute_strategy, expected_train_col2, expected_pred_col2",
    [
        # Numeric strategies impute the all-missing (empty) feature with 0.0 when
        # `keep_empty_features=True`.
        ("mean", [0.0, 0.0, 0.0], [0.0, 100.0]),
        ("median", [0.0, 0.0, 0.0], [0.0, 100.0]),
        # `most_frequent` has no modal value for an all-missing column, so the
        # empty feature is kept but left as a missing sentinel rather than being
        # coerced to 0.0. `keep_empty_features` thus affects each strategy's
        # handling of fully-missing columns differently.
        ("most_frequent", [None, None, None], [float("nan"), 100.0]),
    ],
)
def test_imputer_all_na_column_without_placeholder(impute_strategy, expected_train_col2, expected_pred_col2):
    # Regression test for #124: an entirely-NaN column in `columns_to_impute`
    # used to crash `imputer` with the default `placeholder_value=None`, because
    # SimpleImputer silently dropped the empty feature and the output no longer
    # matched `columns_imputable`. With `keep_empty_features=True` the column is
    # preserved for every strategy; the value it is filled with is strategy
    # dependent, so we assert the concrete behaviour of each one here.
    input_df = pd.DataFrame({"col1": [10, 13, 10], "col2": [None, None, None]})

    input_df2 = pd.DataFrame({"col1": [10, None], "col2": [None, 100]})

    pred_fn, data, log = imputer(input_df, ["col1", "col2"], impute_strategy)

    # the entirely-NaN column is preserved (3 rows) instead of being dropped
    assert len(data["col2"]) == 3
    assert _cells_equal(list(data["col2"]), expected_train_col2)
    # on new data, present values are preserved and missing entries take the
    # strategy's default fill for the empty feature
    assert _cells_equal(list(pred_fn(input_df2)["col2"]), expected_pred_col2)


def test_placeholder_imputer():
    input_df = pd.DataFrame({"col1": [10, 13, 10], "col2": [50, 100, None]})

    input_df2 = pd.DataFrame({"col1": [10, None], "col2": [None, 100]})

    expected1 = pd.DataFrame({"col1": [10, 13, 10], "col2": [50.0, 100.0, -999.0]})

    expected2 = pd.DataFrame({"col1": [10, -999.0], "col2": [-999.0, 100]})

    pred_fn, data, log = placeholder_imputer(input_df, ["col1", "col2"], -999)

    assert expected1.equals(data)
    assert expected2.equals(pred_fn(input_df2))
