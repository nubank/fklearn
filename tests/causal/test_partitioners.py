import numpy as np
import pandas as pd
from fklearn.causal.partitioners import quantile_partitioner


def test_quantile_partitioner():

    series = pd.Series([1, 1, 2, 2, 3, 3])

    result = quantile_partitioner(series, segments=2)
    expected = np.array([1., 2., 3.])

    np.testing.assert_array_equal(result, expected)