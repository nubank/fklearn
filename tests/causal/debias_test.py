import pandas as pd
from fklearn.causal.debias import debias_with_regression_formula


def test_debias_with_regression_formula():
    df = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],  # confounder
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 2, 2, 2, 3, 2, 2, 3],
    ))

    result = debias_with_regression_formula(df, "t", "y", "C(x)", suffix="_d")

    expected = pd.DataFrame(dict(
        x=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        t=[3, 3, 4, 2, 2, 3, 1, 1, 2],
        y=[1, 1, 2, 2, 2, 3, 2, 2, 3],
        t_d=[2., 2., 3., 2., 2., 3., 2., 2., 3.],
        y_d=[1.666, 1.666, 2.666, 1.666, 1.666, 2.666, 1.666, 1.666, 2.666],
    ))

    pd.testing.assert_frame_equal(expected, result, rtol=1.0e-2, atol=1.0e-2)