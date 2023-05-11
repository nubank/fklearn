import numpy as np
import pandas as pd

from fklearn.causal.effects import linear_effect
from fklearn.causal.validation.curves import (effect_by_segment, cumulative_effect_curve, cumulative_gain_curve,
                                              relative_cumulative_gain_curve, effect_curves)
from fklearn.causal.validation.statistical_errors import linear_standard_error


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


def test_effect_curves():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    expected = pd.DataFrame({
        "samples_count": [3, 4, 5, 6, 7, 8, 9],
        "cumulative_effect_curve": [3., 3., 2.92857143, 2.5, 2.5, 2.46153846, 2.],
        "samples_fraction": [0.3333333, 0.4444444, 0.5555555, 0.6666666, 0.7777777, 0.8888888, 1.],
        "cumulative_gain_curve": [1., 1.33333333, 1.62698413, 1.66666667, 1.94444444, 2.18803419, 2.],
        "random_model_cumulative_gain_curve": [0.6666666, 0.8888888, 1.1111111, 1.3333333, 1.5555555, 1.7777777, 2.],
        "relative_cumulative_gain_curve": [0.33333333, 0.44444444, 0.51587302, 0.33333333, 0.38888889, 0.41025641, 0.],
        "cumulative_effect_curve_error": [0.0 , 0.0 , 0.30583887, 0.39528471, 0.32084447, 0.39055247, 0.48795004],
        "cumulative_gain_curve_error": [0.0, 0.0, 0.16991048, 0.26352313, 0.24954570, 0.34715774, 0.48795003],
    })

    result = effect_curves(df, prediction="x", outcome="y", treatment="t", min_rows=3, steps=df.shape[0],
                           effect_fn=linear_effect, error_fn=linear_standard_error)

    pd.testing.assert_frame_equal(result, expected, atol=1e-07)
