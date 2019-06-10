# -*- coding: utf-8 -*-
from collections import Counter

import pandas as pd

from fklearn.training.utils import expand_features_encoded


def test_expand_features_encoded():
    df_A = pd.DataFrame({
        'a': ["id1", "id2", "id3", "id4"],
        'b': [10.0, 13.0, 100.0, 13.0],
        'c': [0, 1, 100, 0],
        'd': [2, 1, 2, 0.5],
        'e': [0, 1, 0, 1]
    })

    df_B = pd.DataFrame({
        'fklearn_feat__a==1': ["id1", "id2", "id3", "id4"],
        'fklearn_feat__a==2': [10.0, 13.0, 100.0, 13.0],
        'fklearn_feat__a==nan': [0, 1, 100, 0],
        'b': [2, 1, 2, 0.5],
        'c': [0, 1, 0, 1]
    })

    df_C = pd.DataFrame({
        'fklearn_feat__a==1': ["id1", "id2", "id3", "id4"],
        'fklearn_feat__a==2': [10.0, 13.0, 100.0, 13.0],
        'fklearn_feat__a==nan': [0, 1, 100, 0],
        'b': [2, 1, 2, 0.5],
        'fklearn_feat__c==10': [0, 1, 0, 1]
    })

    features_all = ["a", "b", "c", "d", "e"]
    features_partial = ["a", "b", "c"]
    features_partialler = ["a", "b"]

    transformed_1 = expand_features_encoded(df_A, features_all)
    expected_1 = ["a", "b", "c", "d", "e"]

    transformed_2 = expand_features_encoded(df_A, features_partial)
    expected_2 = ["a", "b", "c"]

    transformed_3 = expand_features_encoded(df_B, features_partial)
    expected_3 = ["fklearn_feat__a==1", "fklearn_feat__a==2", "fklearn_feat__a==nan", "b", "c"]

    transformed_4 = expand_features_encoded(df_C, features_partial)
    expected_4 = ["fklearn_feat__a==1", "fklearn_feat__a==2", "fklearn_feat__a==nan", "b", "fklearn_feat__c==10"]

    transformed_5 = expand_features_encoded(df_C, features_partialler)
    expected_5 = ["fklearn_feat__a==1", "fklearn_feat__a==2", "fklearn_feat__a==nan", "b"]

    assert Counter(transformed_1) == Counter(expected_1)
    assert Counter(transformed_2) == Counter(expected_2)
    assert Counter(transformed_3) == Counter(expected_3)
    assert Counter(transformed_4) == Counter(expected_4)
    assert Counter(transformed_5) == Counter(expected_5)
