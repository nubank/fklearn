from functools import wraps
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
    def extract_original_name(encoded_name: str) -> str:
        return re.search("fklearn_feat__(.*)==", encoded_name).group(1)

    encode_name_pat = "fklearn_feat_"
    features_from_encoding = df.columns[df.columns.str.contains(encode_name_pat)].tolist()
    if len(features_from_encoding):
        original_encoded_feature_names = set([extract_original_name(f) for f in features_from_encoding])
        not_encoded_features = [f for f in features if f not in original_encoded_feature_names]
        return not_encoded_features + features_from_encoding
    else:
        return features
