import pandas as pd

from fklearn.causal.effects import linear_effect
from fklearn.validation.causal.auc import (area_under_the_cumulative_gain_curve, area_under_the_cumulative_effect_curve,
                                           area_under_the_relative_cumulative_gain_curve)


def test_area_under_the_cumulative_effect_curve():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = area_under_the_cumulative_effect_curve(df, prediction="x", outcome="y", treatment="t", min_rows=3,
                                                    steps=df.shape[0], effect_fn=linear_effect)
    assert round(result, 3) == 6.390


def test_area_under_the_cumulative_gain_curve():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = area_under_the_cumulative_gain_curve(df, prediction="x", outcome="y", treatment="t", min_rows=3,
                                                  steps=df.shape[0], effect_fn=linear_effect)
    assert round(result, 3) == 13.759


def test_area_under_the_relative_cumulative_gain_curve():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        x=[1, 2, 3, 1, 2, 3, 1, 2, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = area_under_the_relative_cumulative_gain_curve(df, prediction="x", outcome="y", treatment="t", min_rows=3,
                                                           steps=df.shape[0], effect_fn=linear_effect)
    assert round(result, 3) == 3.093
