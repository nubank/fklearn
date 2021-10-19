import pandas as pd
from fklearn.causal.debias import debias_with_regression_formula, debias_with_regression, debias_with_double_ml


def test_debias_with_regression_formula():
    df = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],  # confounder
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 2, 2, 2, 3, 2, 2, 3],
    ))

    result = debias_with_regression_formula(df, "t", "y", "C(x)", suffix="_d").round(3)

    expected = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 2, 2, 2, 3, 2, 2, 3],
        t_d=[2., 2., 3., 2., 2., 3., 2., 2., 3.],
        y_d=[1.667, 1.667, 2.667, 1.667, 1.667, 2.667, 1.667, 1.667, 2.667],
    ))

    pd.testing.assert_frame_equal(expected, result)

    result2 = debias_with_regression_formula(df, "t", "y", "C(x)", suffix="_d", denoise=False).round(3)

    pd.testing.assert_frame_equal(expected.drop(columns=["y_d"]), result2)


def test_debias_with_regression():
    df = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],  # confounder
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 2, 2, 2, 3, 2, 2, 3],
    ))

    result = debias_with_regression(df, "t", "y", ["x"], suffix="_d").round(3)

    expected = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 2, 2, 2, 3, 2, 2, 3],
        t_d=[2., 2., 3., 2., 2., 3., 2., 2., 3.],
        y_d=[1.5, 1.5, 2.5, 2.0, 2.0, 3.0, 1.5, 1.5, 2.5],
    ))

    pd.testing.assert_frame_equal(expected, result)

    result2 = debias_with_regression(df, "t", "y", ["x"], suffix="_d", denoise=False).round(3)

    pd.testing.assert_frame_equal(expected.drop(columns=["y_d"]), result2)


def test_debias_with_double_ml():
    df = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],  # confounder
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 2, 2, 2, 3, 2, 2, 3],
    ))

    result = debias_with_double_ml(df, "t", "y", ["x"], suffix="_d").round(3)

    expected = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 2, 2, 2, 3, 2, 2, 3],
        t_d=[1.333, 1.333, 3.333, 1.833, 2.333, 3.333, 1.333, 1.333, 3.333],
        y_d=[1.0, 1.0, 3.0, 1.5, 2.0, 3.0, 1.0, 1.0, 3.0],
    ))

    pd.testing.assert_frame_equal(expected, result)

    result2 = debias_with_double_ml(df, "t", "y", ["x"], suffix="_d", denoise=False).round(3)

    pd.testing.assert_frame_equal(expected.drop(columns=["y_d"]), result2)
