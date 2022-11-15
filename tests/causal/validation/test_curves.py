import numpy as np
import pandas as pd

from fklearn.causal.effects import linear_effect
from fklearn.causal.partitioners import quantile_partitioner
from fklearn.causal.validation.curves import (effect_by_segment, cumulative_effect_curve, cumulative_gain_curve,
                                              relative_cumulative_gain_curve, effect_curves)


def test_effect_by_segment():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = effect_by_segment(df, prediction="x", outcome="y", treatment="t", segments=3, effect_fn=linear_effect, partition_fn=quantile_partitioner)
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
    })

    result = effect_curves(df, prediction="x", outcome="y", treatment="t", min_rows=3, steps=df.shape[0],
                           effect_fn=linear_effect)

    pd.testing.assert_frame_equal(result, expected, atol=1e-07)
