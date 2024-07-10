import pandas as pd
import numpy as np

from fklearn.causal.effects import (linear_effect, spearman_effect, pearson_effect, exponential_coefficient_effect,
                                    logistic_coefficient_effect)


def test_linear_effect():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = linear_effect(df, treatment_column="t", outcome_column="y")
    expected = 2.0

    assert expected == result


def test_spearman_effect():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = spearman_effect(df, treatment_column="t", outcome_column="y")
    assert round(result, 3) == 0.888


def test_pearson_effect():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        y=[1, 1, 1, 2, 3, 4, 3, 5, 7],
    ))

    result = pearson_effect(df, treatment_column="t", outcome_column="y")
    assert round(result, 3) == 0.840


def test_exponential_coefficient_effect():
    a1 = -10
    a0 = -2
    t = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1])

    df = pd.DataFrame(dict(
        t=t,
        y=np.exp(a0 + a1 * t)
    ))

    result = exponential_coefficient_effect(df, treatment_column="t", outcome_column="y")
    assert round(result, 3) == a1


def test_logistic_coefficient_effect():

    df = pd.DataFrame(dict(
        t=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        y=[0, 0, 0, 0, 0, 0, 1, 1, 1]
    ))

    result = logistic_coefficient_effect(df, treatment_column="t", outcome_column="y")

    SKLEARN_GTE_1_4_RESULT = 17.922
    SKLEARN_LT_1_4_RESULT = 20.645
    expected_result_range = {SKLEARN_GTE_1_4_RESULT, SKLEARN_LT_1_4_RESULT}

    assert round(result, 3) in expected_result_range
