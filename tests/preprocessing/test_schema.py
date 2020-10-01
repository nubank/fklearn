import pandas as pd

from fklearn.preprocessing.schema import feature_duplicator


def test_feature_duplicator():
    input_df = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
    })

    expected1 = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'prefix__a': [1, 2, 3],
    })

    expected2 = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'a__suffix': [1, 2, 3],
        'b__suffix': [4, 5, 6],
    })

    expected3 = pd.DataFrame({
        'a': [1, 2, 3],
        'b': [4, 5, 6],
        'c': [4, 5, 6],
    })

    assert expected1.equals(
        feature_duplicator(
            input_df.copy(),
            columns_to_duplicate=['a'],
            prefix='prefix__',
        )[1])
    assert expected2.equals(
        feature_duplicator(
            input_df.copy(),
            columns_to_duplicate=['a', 'b'],
            suffix='__suffix',
        )[1])
    assert expected3.equals(
        feature_duplicator(
            input_df.copy(),
            columns_mapping={'b': 'c'},
        )[1])
