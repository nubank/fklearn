from fklearn.validation.perturbators import \
    shift_mu, random_noise, nullify, sample_columns, perturbator

import pandas as pd
import numpy as np


def test_shift_mu():
    series = pd.Series([1, 3, 5, 7, 166])
    shift_by = 0.7
    expected = pd.Series([26.48,  28.48,  30.48,  32.48, 191.48])
    new_series = shift_mu(col=series, perc=shift_by)
    map(np.testing.assert_approx_equal, zip(expected, new_series))


def test_random_noise():
    series = pd.Series([1, 3, 5, 7, 166])
    random_noise(col=series, mag=0.14)


def test_nullify():
    series = pd.Series([1, 3, 5, 7, 166])
    expected_nan_count = 3
    new_series = nullify(col=series, perc=0.6)
    new_nan_count = sum(new_series.isna())
    assert expected_nan_count == new_nan_count


def test_sample_columns():
    df = pd.DataFrame(columns=['feature1', 'feature2', 'feature3', 'feature4'])
    expected_len = 2
    found = sample_columns(data=df, perc=0.5)
    assert expected_len == len(found)
    assert all([el in df.columns for el in found])


def test_perturbator():
    test_df = pd.DataFrame(
        {
            'a': [1, 1, 0],
            'bb': [2, 0, 0],
            'target': [0, 1, 2]
        }
    )

    expected_df = pd.DataFrame(
        {
            'a': [np.nan, np.nan, np.nan],
            'bb': [2, 0, 0],
            'target': [0, 1, 2]
        }
    )

    out_df = perturbator(data=test_df, cols=['a'], corruption_fn=nullify())

    assert expected_df.equals(out_df)
