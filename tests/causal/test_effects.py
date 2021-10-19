import numpy as np
import pandas as pd

from fklearn.causal.effects import (linear_effect, linear_effect_ci, spearman_effect, pearson_effect)


def test_linear_effect():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = linear_effect(df, treatment="t", outcome="y")
    expected = 2.0

    assert expected == result


def test_linear_effect_ci():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = linear_effect_ci(df, treatment="t", outcome="y", z=1.96)
    expected = np.array([-1.19736616,  1.90324851])

    np.testing.assert_allclose(result, expected)

    result = linear_effect_ci(df, treatment="t", outcome="y", z=1.64)
    expected = np.array([-0.94425475,  1.65013711])

    np.testing.assert_allclose(result, expected)


def test_spearman_effect():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = spearman_effect(df, treatment="t", outcome="y")
    assert round(result, 3) == 0.888


def test_pearson_effect():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = pearson_effect(df, treatment="t", outcome="y")
    assert round(result, 3) == 0.840

