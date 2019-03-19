from functools import wraps
from time import time
from typing import Any

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
