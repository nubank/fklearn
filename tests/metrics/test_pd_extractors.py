import numpy as np
import pandas as pd
from datetime import timedelta
from sklearn.datasets import load_boston

from fklearn.training.regression import linear_regression_learner
from fklearn.validation.splitters import time_learning_curve_splitter, stability_curve_time_splitter, \
    forward_stability_curve_time_splitter, out_of_time_and_space_splitter
from fklearn.validation.evaluators import r2_evaluator, spearman_evaluator, combined_evaluators, split_evaluator
from fklearn.validation.validator import validator
from fklearn.metrics.pd_extractors import combined_evaluator_extractor, split_evaluator_extractor, \
    evaluator_extractor, extract


def test_extract():
    boston = load_boston()
    df = pd.DataFrame(boston['data'], columns=boston['feature_names'])
    df['target'] = boston['target']
    df['time'] = pd.date_range(start='2015-01-01', periods=len(df))
    np.random.seed(42)
    df['space'] = np.random.randint(0, 100, size=len(df))

    # Define train function
    train_fn = linear_regression_learner(features=boston['feature_names'].tolist(), target="target")

    # Define evaluator function
    base_evaluator = combined_evaluators(evaluators=[
        r2_evaluator(target_column='target', prediction_column='prediction'),
        spearman_evaluator(target_column='target', prediction_column='prediction')
    ])

    splitter = split_evaluator(eval_fn=base_evaluator, split_col='RAD', split_values=[4.0, 5.0, 24.0])

    eval_fn = combined_evaluators(evaluators=[base_evaluator, splitter])

    # Define splitters
    cv_split_fn = out_of_time_and_space_splitter(
        n_splits=5, in_time_limit='2016-01-01', time_column='time', space_column='space'
    )

    tlc_split_fn = time_learning_curve_splitter(training_time_limit='2016-01-01', time_column='time', min_samples=0)

    sc_split_fn = stability_curve_time_splitter(training_time_limit='2016-01-01', time_column='time', min_samples=0)

    fw_sc_split_fn = forward_stability_curve_time_splitter(
        training_time_start="2015-01-01",
        training_time_end="2016-01-01",
        holdout_gap=timedelta(days=30),
        holdout_size=timedelta(days=30),
        step=timedelta(days=30),
        time_column='time'
    )

    # Validate results
    cv_results = validator(df, cv_split_fn, train_fn, eval_fn)['validator_log']
    tlc_results = validator(df, tlc_split_fn, train_fn, eval_fn)['validator_log']
    sc_results = validator(df, sc_split_fn, train_fn, eval_fn)['validator_log']
    fw_sc_results = validator(df, fw_sc_split_fn, train_fn, eval_fn)['validator_log']

    # Define extractors
    base_extractors = combined_evaluator_extractor(base_extractors=[
        evaluator_extractor(evaluator_name="r2_evaluator__target"),
        evaluator_extractor(evaluator_name="spearman_evaluator__target")
    ])

    splitter_extractor = split_evaluator_extractor(split_col='RAD', split_values=[4.0, 5.0, 24.0],
                                                   base_extractor=base_extractors)

    assert extract(cv_results, base_extractors).shape == (5, 9)
    assert extract(cv_results, splitter_extractor).shape == (15, 10)

    assert extract(tlc_results, base_extractors).shape == (12, 9)
    assert extract(tlc_results, splitter_extractor).shape == (36, 10)

    assert extract(sc_results, base_extractors).shape == (5, 9)
    assert extract(sc_results, splitter_extractor).shape == (15, 10)

    assert extract(fw_sc_results, base_extractors).shape == (3, 9)
    assert extract(fw_sc_results, splitter_extractor).shape == (9, 10)
