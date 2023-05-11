import numpy as np
import pandas as pd

from fklearn.causal.validation.statistical_errors import linear_standard_error


def test_linear_standard_error():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = linear_standard_error(df, treatment="t", outcome="y")
    expected = 0.48795003647426655

    np.testing.assert_array_almost_equal(result, expected, decimal=4)
