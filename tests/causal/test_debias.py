import pandas as pd
from fklearn.causal.debias import (debias_with_regression_formula, debias_with_regression, debias_with_double_ml,
                                   debias_with_fixed_effects)


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


def test_debias_with_fixed_effects():

    # te = 2
    df = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],  # confounder
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 3, 2, 2, 4, 3, 3, 5],
    ))

    result = debias_with_fixed_effects(df, "t", "y", ["x"], suffix="_d").round(3)

    expected = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 3, 2, 2, 4, 3, 3, 5],
        t_d=[2., 2., 3., 2., 2., 3., 2., 2., 3.],
        y_d=[2., 2., 4., 2., 2., 4., 2., 2., 4.],
    ))

    pd.testing.assert_frame_equal(expected, result)

    # te = 2
    df2 = pd.DataFrame(dict(
        x1=[0, 0, 0, 0, 1, 1, 1, 1],  # confounder
        x2=[0, 0, 1, 1, 1, 1, 0, 0],  # confounder
        t=[1, 2, 1 + 1, 2 + 1, 1 + 2, 2 + 2, 1 + 1, 2 + 1],
        y=[1, 3, 1 - 1, 3 - 1, 1 - 2, 3 - 2, 1 - 1, 3 - 1],
    ))

    result2 = debias_with_fixed_effects(df2, "t", "y", ["x1", "x2"], suffix="_d").round(3)

    expected = pd.DataFrame(dict(
        x1=[0, 0, 0, 0, 1, 1, 1, 1],
        x2=[0, 0, 1, 1, 1, 1, 0, 0],
        t=[1, 2, 1 + 1, 2 + 1, 1 + 2, 2 + 2, 1 + 1, 2 + 1],
        y=[1, 3, 1 - 1, 3 - 1, 1 - 2, 3 - 2, 1 - 1, 3 - 1],
        t_d=[-.5, .5, -.5, .5, -.5, .5, -.5, .5],
        y_d=[-1., 1., -1., 1., -1., 1., -1., 1.],
    ))

    pd.testing.assert_frame_equal(expected, result2)


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
