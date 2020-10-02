import collections
from datetime import datetime
from itertools import chain, repeat

import pandas as pd
from toolz import curry
from numpy import nan


@curry
def evaluator_extractor(result, evaluator_name):
    metric_value = result[evaluator_name] if result else nan
    return pd.DataFrame({evaluator_name: [metric_value]})


@curry
def combined_evaluator_extractor(result, base_extractors):
    return pd.concat([x(result) for x in base_extractors], axis=1)


@curry
def split_evaluator_extractor_iteration(split_value, result, split_col, base_extractor, eval_name=None):
    if eval_name is None:
        eval_name = 'split_evaluator__' + split_col

    key = eval_name + '_' + str(split_value)

    return (base_extractor(result.get(key, {}))
            .assign(**{eval_name: split_value}))


@curry
def split_evaluator_extractor(result, split_col, split_values, base_extractor, eval_name=None):
    return pd.concat(
        list(map(split_evaluator_extractor_iteration(result=result, split_col=split_col, base_extractor=base_extractor,
                                                     eval_name=eval_name),
                 split_values)))


@curry
def temporal_split_evaluator_extractor(result, time_col, base_extractor, time_format="%Y-%m", eval_name=None):
    if eval_name is None:
        eval_name = 'split_evaluator__' + time_col

    split_keys = [key for key in result.keys() if eval_name in key]
    split_values = []
    for key in split_keys:
        date = key.split(eval_name)[1][1:]
        try:
            # just check time format
            datetime.strptime(date, time_format)
            split_values.append(date)
        except ValueError:
            # this might happen if result has temporal splitters using different data formats
            pass

    return split_evaluator_extractor(result, time_col, split_values, base_extractor)


@curry
def learning_curve_evaluator_extractor(result, base_extractor):
    return base_extractor(result).assign(lc_period_end=result['lc_period_end'])


@curry
def reverse_learning_curve_evaluator_extractor(result, base_extractor):
    return base_extractor(result).assign(reverse_lc_period_start=result['reverse_lc_period_start'])


@curry
def stability_curve_evaluator_extractor(result, base_extractor):
    return base_extractor(result).assign(sc_period=result['sc_period'])


@curry
def repeat_split_log(split_log, results_len):
    if isinstance(split_log, collections.Iterable):
        n_repeat = results_len // len(split_log)
        # The logic below makes [1, 2, 3] into [1, 1, 1, 2, 2, 2, 3, 3, 3] for n_repeat=3
        return list(chain.from_iterable(zip(*repeat(split_log, n_repeat))))
    else:
        return split_log


@curry
def extract_base_iteration(result, extractor):
    extracted_results = pd.concat(list(map(extractor, result['eval_results'])))
    repeat_fn = repeat_split_log(results_len=len(extracted_results))

    keys = result['split_log'].keys()
    assignments = {k: repeat_fn(result['split_log'][k]) for k in keys}

    return (extracted_results
            .assign(fold_num=result['fold_num'])
            .assign(**assignments))


@curry
def extract(validator_results, extractor):
    return pd.concat(list(map(extract_base_iteration(extractor=extractor), validator_results)))


@curry
def extract_lc(validator_results, extractor):
    return extract(validator_results, learning_curve_evaluator_extractor(base_extractor=extractor))


@curry
def extract_reverse_lc(validator_results, extractor):
    return extract(validator_results, reverse_learning_curve_evaluator_extractor(base_extractor=extractor))


@curry
def extract_sc(validator_results, extractor):
    return extract(validator_results, stability_curve_evaluator_extractor(base_extractor=extractor))


@curry
def extract_param_tuning_iteration(iteration, tuning_log, base_extractor, model_learner_name):
    iter_df = base_extractor(tuning_log[iteration]["validator_log"])
    return iter_df.assign(**tuning_log[iteration]["train_log"][model_learner_name]["parameters"])


@curry
def extract_tuning(tuning_log, base_extractor, model_learner_name):
    iter_fn = extract_param_tuning_iteration(tuning_log=tuning_log, base_extractor=base_extractor,
                                             model_learner_name=model_learner_name)
    return pd.concat(list(map(iter_fn, range(len(tuning_log)))))


@curry
def permutation_extractor(results, base_extractor):
    df = pd.concat(base_extractor(r) for r in results['permutation_importance'].values())
    df.index = results['permutation_importance'].keys()
    if 'permutation_importance_baseline' in results:  # With baseline comparison
        baseline = base_extractor(results['permutation_importance_baseline'])
        baseline.index = ["baseline"]
        df = pd.concat((df, baseline))
        for c in baseline.columns:
            df[c + '_delta_from_baseline'] = baseline[c].iloc[0] - df[c]
    return df
