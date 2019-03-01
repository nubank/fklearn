from inspect import signature, Parameter
import toolz as fp


def build_pipeline(*learners):
    """
    Builds a pipeline of chained learners functions with the possibility of using keyword arguments
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

    def _has_one_unfilled_arg(learner):
        no_default_list = [p for p, a in signature(learner).parameters.items() if a.default == '__no__default__']
        assert len(no_default_list) <= 1, "Learner {0} has more than one unfilled argument: {1}\n" \
                                          "Make sure all learners are curried properly and only require one argument," \
                                          " which is the dataset (usually `df`).".format(
            learner.__name__,
            ', '.join(no_default_list)
        )

    def _no_variable_args(learner, predict_fn):
        invalid_parameter_kinds = (Parameter.VAR_POSITIONAL, Parameter.VAR_KEYWORD)
        var_args = [p for p, a in signature(predict_fn).parameters.items() if a.kind in invalid_parameter_kinds]
        assert len(var_args) == 0, "Predict function of learner {0} contains variable length arguments: {1}\n" \
                                   "Make sure no predict function uses arguments like *args or **kwargs.".format(
            learner.__name__,
            ', '.join(var_args)
        )

    # Check for unfilled arguments of learners
    for l in learners:
        _has_one_unfilled_arg(l)

    def pipeline(data):
        current_data = data.copy()
        fns = []
        logs = []

        for learner in learners:
            learner_fn, new_data, learner_log = learner(current_data)
            _no_variable_args(learner, learner_fn)  # Check for invalid predict fn arguments
            fns.append(learner_fn)
            logs.append(learner_log)
            current_data = new_data

        logs = fp.merge(logs)

        def predict_fn(df, **kwargs):
            # Get the proper arguments for each predict function (based on their signature)
            fns_args = [{k: v for k, v in kwargs.items() if k in signature(f).parameters} for f in fns]
            # Partially apply the arguments to the predict functions when applicable
            fns_with_args = [fp.curry(fn)(**args) if len(args) > 0 else fn for fn, args in zip(fns, fns_args)]
            return fp.pipe(df, *fns_with_args)

        return predict_fn, current_data, logs

    return pipeline
