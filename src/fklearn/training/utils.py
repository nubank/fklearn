from functools import reduce, wraps
from time import time
import re
from typing import Any, List

import pandas as pd
from toolz import curry
import toolz as fp

from fklearn.types import LearnerReturnType, UncurriedLearnerFnType


@curry
def log_learner_time(learner: UncurriedLearnerFnType, learner_name: str) -> UncurriedLearnerFnType:
    @wraps(learner)
    def timed_learner(*args: Any, **kwargs: Any) -> LearnerReturnType:
        t0 = time()
        (p, d, l) = learner(*args, **kwargs)
        return p, d, fp.assoc_in(l, [learner_name, 'running_time'], "%2.3f s" % (time() - t0))

    return timed_learner


@curry
def print_learner_run(learner: UncurriedLearnerFnType, learner_name: str) -> UncurriedLearnerFnType:
    @wraps(learner)
    def printed_learner(*args: Any, **kwargs: Any) -> LearnerReturnType:
        print('%s running' % learner_name)
        return learner(*args, **kwargs)

    return printed_learner


def expand_features_encoded(df: pd.DataFrame,
                            features: List[str]) -> List[str]:

    """
    Expand the list of features to include features created automatically
    by fklearn in encoders such as Onehot-encoder.
    All features created by fklearn have the naming pattern `fklearn_feat__col==val`.
    This function looks for these names in the DataFrame columns, checks if they can
    be derivative of any of the features listed in `features`, adds them to the new
    list of features and removes the original names from the list.

    E.g. df has columns `col1` with values 0 and 1 and `col2`. After Onehot-encoding
    `col1` df will have columns `fklearn_feat_col1==0`, `fklearn_feat_col1==1`, `col2`.
    This function will then add `fklearn_feat_col1==0` and `fklearn_feat_col1==1` to
    the list of features and remove `col1`. If for some reason df also has another
    column `fklearn_feat_col3==x` but `col3` is not on the list of features, this
    column will not be added.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas' DataFrame with all features.

    features : list of str
        The original list of features.
    """

    def fklearn_features(df: pd.DataFrame) -> List[str]:
        return list(filter(lambda col: col.startswith("fklearn_feat__"), df.columns))

    def feature_prefix(feature: str) -> str:
        return feature.split("==")[0]

    def filter_non_listed_features(fklearn_features: List[str], features: List[str]) -> List[str]:
        possible_prefixes_with_listed_features = ["fklearn_feat__" + f for f in features]
        return list(filter(lambda col: feature_prefix(col) in possible_prefixes_with_listed_features, fklearn_features))

    def remove_original_pre_encoded_features(features: List[str], encoded_features: List[str]) -> List[str]:
        expr = r"fklearn_feat__(.*)=="
        original_preencoded_features: List[str] = reduce(lambda x, y: x + y,
                                                         (map(lambda x: re.findall(expr, x),
                                                              encoded_features)),
                                                         [])
        return list(filter(lambda col: col not in set(original_preencoded_features), features))

    all_fklearn_features = fklearn_features(df)
    encoded_features = filter_non_listed_features(all_fklearn_features, features)
    not_encoded_features = remove_original_pre_encoded_features(features, encoded_features)
    return not_encoded_features + encoded_features
