from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.datasets import load_boston

from fklearn.metrics.pd_extractors import (combined_evaluator_extractor,
                                           evaluator_extractor, extract,
                                           split_evaluator_extractor,
                                           temporal_split_evaluator_extractor)
from fklearn.training.regression import linear_regression_learner
from fklearn.validation.evaluators import (combined_evaluators, r2_evaluator,
                                           spearman_evaluator, split_evaluator,
                                           temporal_split_evaluator)
from fklearn.validation.splitters import (
    forward_stability_curve_time_splitter, out_of_time_and_space_splitter,
    stability_curve_time_splitter, time_learning_curve_splitter)
from fklearn.validation.validator import validator


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
    temporal_week_splitter = temporal_split_evaluator(eval_fn=base_evaluator, time_col='time', time_format='%Y-%W')
    temporal_year_splitter = temporal_split_evaluator(eval_fn=base_evaluator, time_col='time', time_format='%Y')

    eval_fn = combined_evaluators(evaluators=[base_evaluator, splitter])
    temporal_week_eval_fn = combined_evaluators(evaluators=[base_evaluator, temporal_week_splitter])
    temporal_year_eval_fn = combined_evaluators(evaluators=[base_evaluator, temporal_year_splitter])

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

    # temporal evaluation results
    predict_fn, _, _ = train_fn(df)
    temporal_week_results = temporal_week_eval_fn(predict_fn(df))
    temporal_year_results = temporal_year_eval_fn(predict_fn(df))

    # Define extractors
    base_extractors = combined_evaluator_extractor(base_extractors=[
        evaluator_extractor(evaluator_name="r2_evaluator__target"),
        evaluator_extractor(evaluator_name="spearman_evaluator__target")
    ])

    splitter_extractor = split_evaluator_extractor(split_col='RAD', split_values=[4.0, 5.0, 24.0],
                                                   base_extractor=base_extractors)

    temporal_week_splitter_extractor = temporal_split_evaluator_extractor(
        time_col='time', time_format='%Y-%W', base_extractor=base_extractors)

    temporal_year_splitter_extractor = temporal_split_evaluator_extractor(
        time_col='time', time_format='%Y', base_extractor=base_extractors)

    assert extract(cv_results, base_extractors).shape == (5, 9)
    assert extract(cv_results, splitter_extractor).shape == (15, 10)

    assert extract(tlc_results, base_extractors).shape == (12, 9)
    assert extract(tlc_results, splitter_extractor).shape == (36, 10)

    assert extract(sc_results, base_extractors).shape == (5, 9)
    assert extract(sc_results, splitter_extractor).shape == (15, 10)

    assert extract(fw_sc_results, base_extractors).shape == (3, 9)
    assert extract(fw_sc_results, splitter_extractor).shape == (9, 10)

    n_time_week_folds = len(df['time'].dt.strftime('%Y-%W').unique())
    n_time_year_folds = len(df['time'].dt.strftime('%Y').unique())
    assert temporal_week_splitter_extractor(temporal_week_results).shape == (n_time_week_folds, 3)
    assert temporal_year_splitter_extractor(temporal_year_results).shape == (n_time_year_folds, 3)
