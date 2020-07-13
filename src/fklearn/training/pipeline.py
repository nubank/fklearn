from collections import defaultdict
from inspect import Parameter, signature
from typing import Dict

import pandas as pd
import toolz as fp

from fklearn.types import LearnerFnType, LearnerReturnType, PredictFnType


def build_pipeline(*learners: LearnerFnType, has_repeated_learners: bool = False) -> LearnerFnType:
    """
    Builds a pipeline of different chained learners functions with the possibility of using keyword arguments
    in the predict functions of the pipeline.

    Say you have two learners, you create a pipeline with `pipeline = build_pipeline(learner1, learner2)`.
    Those learners must be functions with just one unfilled argument (the dataset itself).

    Then, you train the pipeline with `predict_fn, transformed_df, logs = pipeline(df)`,
    which will be like applying the learners in the following order: `learner2(learner1(df))`.

    Finally, you predict on different datasets with `pred_df = predict_fn(new_df)`, with optional kwargs.
    For example, if you have XGBoost or LightGBM, you can get SHAP values with `predict_fn(new_df, apply_shap=True)`.

    Parameters
    ----------
    learners : partially-applied learner functions.

    has_repeated_learners : bool
        Boolean value indicating wheter the pipeline contains learners with the same name or not.

    Returns
    ----------
    p : function pandas.DataFrame, **kwargs -> pandas.DataFrame
        A function that when applied to a DataFrame will apply all learner
        functions in sequence, with optional kwargs.

    new_df : pandas.DataFrame
        A DataFrame that is the result of applying all learner function
        in sequence.

    log : dict
        A log-like Dict that stores information of all learner functions.
    """

    def _has_one_unfilled_arg(learner: LearnerFnType) -> None:
        no_default_list = [p for p, a in signature(learner).parameters.items() if a.default == '__no__default__']
        if len(no_default_list) > 1:
            raise ValueError("Learner {0} has more than one unfilled argument: {1}\n"
                             "Make sure all learners are curried properly and only require one argument,"
                             " which is the dataset (usually `df`)."
                             .format(learner.__name__, ', '.join(no_default_list)))

    def _no_variable_args(learner: LearnerFnType, predict_fn: PredictFnType) -> None:
        invalid_parameter_kinds = (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        var_args = [p for p, a in signature(predict_fn).parameters.items() if a.kind in invalid_parameter_kinds]
        if len(var_args) != 0:
            raise ValueError("Predict function of learner {0} contains variable length arguments: {1}\n"
                             "Make sure no predict function uses arguments like *args or **kwargs."
                             .format(learner.__name__, ', '.join(var_args)))

    # Check for unfilled arguments of learners
    for learner in learners:
        _has_one_unfilled_arg(learner)

    def pipeline(data: pd.DataFrame) -> LearnerReturnType:
        current_data = data.copy()
        features = list(data.columns)
        fns = []
        logs = []
        pipeline = []
        serialisation = defaultdict(list)  # type: dict

        for learner in learners:
            learner_fn, new_data, learner_log = learner(current_data)
            # Check for invalid predict fn arguments
            _no_variable_args(learner, learner_fn)

            learner_name = learner.__name__
            fns.append(learner_fn)
            pipeline.append(learner_name)
            current_data = new_data

            model_objects = {}
            if learner_log.get("obj"):
                model_objects["obj"] = learner_log.pop("obj")

            serialisation[learner_name].append({"fn": learner_fn, "log": learner_log, **model_objects})
            logs.append(learner_log)

        merged_logs = fp.merge(logs)

        def predict_fn(df: pd.DataFrame, **kwargs: Dict) -> pd.DataFrame:
            # Get the proper arguments for each predict function (based on their signature)
            fns_args = [{k: v for k, v in kwargs.items() if k in signature(f).parameters} for f in fns]
            # Partially apply the arguments to the predict functions when applicable
            fns_with_args = [fp.curry(fn)(**args) if len(args) > 0 else fn for fn, args in zip(fns, fns_args)]
            return fp.pipe(df, *fns_with_args)

        serialisation_logs = {k: v if has_repeated_learners else v[-1] for k, v in serialisation.items()}

        merged_logs["__fkml__"] = {"pipeline": pipeline,
                                   "output_columns": list(current_data.columns),
                                   "features": features,
                                   "learners": {**serialisation_logs}}

        return predict_fn, current_data, merged_logs

    return pipeline
