import pandas as pd

from fklearn.training.imputation import imputer, placeholder_imputer


def test_imputer():
    input_df = pd.DataFrame({
        'col1': [10, 13, 10],
        'col2': [50, 100, None]
    })

    input_df2 = pd.DataFrame({
        'col1': [10, None],
        'col2': [None, 100]
    })

    expected1 = pd.DataFrame({
        'col1': [10.0, 13.0, 10.0],
        'col2': [50.0, 100.0, 75.0]
    })

    expected2 = pd.DataFrame({
        'col1': [10, 11.0],
        'col2': [75.0, 100]
    })

    pred_fn, data, log = imputer(input_df, ["col1", "col2"], "mean")

    assert expected1.equals(data)
    assert expected2.equals(pred_fn(input_df2))


def test_imputer_with_fill_value():
    input_df = pd.DataFrame({
        'col1': [10, 13, 10],
        'col2': [50, 100, None],
        'col3': [None, None, None]
    })

    expected = pd.DataFrame({
        'col1': [10.0, 13.0, 10.0],
        'col2': [50.0, 100.0, 75.0],
        'col3': [0, 0, 0]
    })

    pred_fn, data, log = imputer(input_df, ["col1", "col2", "col3"], "mean", fill_value=0)

    assert expected.equals(data)


def test_placeholder_imputer():
    input_df = pd.DataFrame({
        'col1': [10, 13, 10],
        'col2': [50, 100, None]
    })

    input_df2 = pd.DataFrame({
        'col1': [10, None],
        'col2': [None, 100]
    })

    expected1 = pd.DataFrame({
        'col1': [10, 13, 10],
        'col2': [50.0, 100.0, -999.0]
    })

    expected2 = pd.DataFrame({
        'col1': [10, -999.0],
        'col2': [-999.0, 100]
    })

    pred_fn, data, log = placeholder_imputer(input_df, ["col1", "col2"], -999)

    assert expected1.equals(data)
    assert expected2.equals(pred_fn(input_df2))
