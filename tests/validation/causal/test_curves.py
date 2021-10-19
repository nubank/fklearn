import numpy as np
import pandas as pd

from fklearn.causal.effects import linear_effect
from fklearn.validation.causal.curves import (effect_by_segment, cumulative_effect_curve, cumulative_gain_curve,
                                              relative_cumulative_gain_curve)


def test_effect_by_segment():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = effect_by_segment(df, prediction="x", outcome="y", treatment="t", segments=3, effect_fn=linear_effect)
    expected = pd.Series([1., 2., 3.], index=result.index)

    pd.testing.assert_series_equal(result, expected)


def test_cumulative_effect_curve():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    expected = np.array([3., 3., 2.92857143, 2.5, 2.5, 2.46153846, 2.])

    result = cumulative_effect_curve(df, prediction="x", outcome="y", treatment="t", min_rows=3, steps=df.shape[0],
                                     effect_fn=linear_effect)

    np.testing.assert_allclose(expected, result, rtol=1e-07)


def test_cumulative_gain_curve():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    expected = np.array([1., 1.33333333, 1.62698413, 1.66666667, 1.94444444, 2.18803419, 2.])

    result = cumulative_gain_curve(df, prediction="x", outcome="y", treatment="t", min_rows=3, steps=df.shape[0],
                                   effect_fn=linear_effect)

    np.testing.assert_allclose(expected, result, rtol=1e-07)


def test_relative_cumulative_gain_curve():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    expected = np.array([0.33333333, 0.44444444, 0.51587302, 0.33333333, 0.38888889, 0.41025641, 0.])

    result = relative_cumulative_gain_curve(df, prediction="x", outcome="y", treatment="t", min_rows=3,
                                            steps=df.shape[0], effect_fn=linear_effect)

    np.testing.assert_allclose(expected, result, rtol=1e-07)
