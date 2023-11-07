from datetime import timedelta

import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import fetch_california_housing

from fklearn.data.datasets import make_tutorial_data
from fklearn.metrics.pd_extractors import (combined_evaluator_extractor,
                                           evaluator_extractor, extract,
                                           split_evaluator_extractor,
                                           split_evaluator_extractor_iteration,
                                           temporal_split_evaluator_extractor)
from fklearn.training.regression import linear_regression_learner
from fklearn.validation.evaluators import (combined_evaluators, r2_evaluator,
                                           spearman_evaluator, split_evaluator,
                                           temporal_split_evaluator, mse_evaluator)
from fklearn.validation.splitters import (
    forward_stability_curve_time_splitter, out_of_time_and_space_splitter,
    stability_curve_time_splitter, time_learning_curve_splitter)
from fklearn.validation.validator import validator


@pytest.fixture
def create_split_logs_and_evaluator():
    def _create_split_logs_and_evaluator(eval_name):
        logs = {
            eval_name + '_0': {'roc_auc': 0.48},
            eval_name + '_1': {'roc_auc': 0.52},
        }
        base_evaluator = evaluator_extractor(evaluator_name='roc_auc')
        return logs, base_evaluator

    return _create_split_logs_and_evaluator


@pytest.mark.parametrize('eval_name, split_kwargs', [
    ('split_evaluator__split', {'split_col': 'split'}),
    ('named_eval', {'split_col': 'irrelevant', 'eval_name': 'named_eval'})
])
def test__split_evaluator_extractor_iteration(eval_name, split_kwargs, create_split_logs_and_evaluator):
    logs, base_evaluator = create_split_logs_and_evaluator(eval_name)

    expected_df = pd.DataFrame({'roc_auc': [0.52], eval_name: [1]})
    actual_df = split_evaluator_extractor_iteration(split_value=1,
                                                    result=logs,
                                                    base_extractor=base_evaluator,
                                                    **split_kwargs
                                                    ).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_df, expected_df)


@pytest.mark.parametrize('eval_name, split_kwargs', [
    ('split_evaluator__split', {'split_col': 'split'}),
    ('named_eval', {'split_col': 'irrelevant', 'eval_name': 'named_eval'})
])
def test__split_evaluator_extractor(eval_name, split_kwargs, create_split_logs_and_evaluator):
    logs, base_evaluator = create_split_logs_and_evaluator(eval_name)

    expected_df = pd.DataFrame({'roc_auc': [0.48, 0.52], eval_name: [0, 1]})
    actual_df = split_evaluator_extractor(logs,
                                          base_extractor=base_evaluator,
                                          split_values=[0, 1],
                                          **split_kwargs
                                          ).reset_index(drop=True)

    pd.testing.assert_frame_equal(actual_df, expected_df)


def test__split_evaluator_extractor__when_split_value_is_missing():
    expected = [{'mse_evaluator__target': np.nan,
                 'split_evaluator__date': pd.Timestamp('2015-01-06 00:00:00'),
                 'split_evaluator__feature3': 'a'},
                {'mse_evaluator__target': 0.0,
                 'split_evaluator__date': pd.Timestamp('2015-01-06 00:00:00'),
                 'split_evaluator__feature3': 'b'},
                {'mse_evaluator__target': np.nan,
                 'split_evaluator__date': pd.Timestamp('2015-01-14 00:00:00'),
                 'split_evaluator__feature3': 'a'},
                {'mse_evaluator__target': 0.0,
                 'split_evaluator__date': pd.Timestamp('2015-01-14 00:00:00'),
                 'split_evaluator__feature3': 'b'},
                {'mse_evaluator__target': np.nan,
                 'split_evaluator__date': pd.Timestamp('2015-01-22 00:00:00'),
                 'split_evaluator__feature3': 'a'},
                {'mse_evaluator__target': 0.0,
                 'split_evaluator__date': pd.Timestamp('2015-01-22 00:00:00'),
                 'split_evaluator__feature3': 'b'},
                {'mse_evaluator__target': 0.0,
                 'split_evaluator__date': pd.Timestamp('2015-01-30 00:00:00'),
                 'split_evaluator__feature3': 'a'},
                {'mse_evaluator__target': np.nan,
                 'split_evaluator__date': pd.Timestamp('2015-01-30 00:00:00'),
                 'split_evaluator__feature3': 'b'},
                {'mse_evaluator__target': 0.0,
                 'split_evaluator__date': pd.Timestamp('2015-03-08 00:00:00'),
                 'split_evaluator__feature3': 'a'},
                {'mse_evaluator__target': np.nan,
                 'split_evaluator__date': pd.Timestamp('2015-03-08 00:00:00'),
                 'split_evaluator__feature3': 'b'},
                {'mse_evaluator__target': 0.0,
                 'split_evaluator__date': pd.Timestamp('2015-03-09 00:00:00'),
                 'split_evaluator__feature3': 'a'},
                {'mse_evaluator__target': np.nan,
                 'split_evaluator__date': pd.Timestamp('2015-03-09 00:00:00'),
                 'split_evaluator__feature3': 'b'},
                {'mse_evaluator__target': np.nan,
                 'split_evaluator__date': pd.Timestamp('2015-04-04 00:00:00'),
                 'split_evaluator__feature3': 'a'},
                {'mse_evaluator__target': 0.0,
                 'split_evaluator__date': pd.Timestamp('2015-04-04 00:00:00'),
                 'split_evaluator__feature3': 'b'}]
    expected_df = pd.DataFrame.from_dict(expected)
    data = make_tutorial_data(50).dropna(subset=["feature3"]).assign(prediction=lambda d: d.target)

    feature3_evaluator = split_evaluator(eval_fn=mse_evaluator, split_col="feature3")
    feature3_date_evaluator = split_evaluator(eval_fn=feature3_evaluator, split_col="date")

    results = feature3_date_evaluator(data)

    date_values = pd.to_datetime([
        np.datetime64("2015-01-06T00:00:00.000000000"),
        np.datetime64("2015-01-14T00:00:00.000000000"),
        np.datetime64("2015-01-22T00:00:00.000000000"),
        np.datetime64("2015-01-30T00:00:00.000000000"),
        np.datetime64("2015-03-08T00:00:00.000000000"),
        np.datetime64("2015-03-09T00:00:00.000000000"),
        np.datetime64("2015-04-04T00:00:00.000000000"),
    ])

    base_evaluator = evaluator_extractor(evaluator_name="mse_evaluator__target")
    feature3_extractor = split_evaluator_extractor(
        base_extractor=base_evaluator, split_col="feature3", split_values=["a", "b"]
    )
    feature3_date_extractor = split_evaluator_extractor(
        base_extractor=feature3_extractor, split_col="date", split_values=date_values
    )

    actual_df = feature3_date_extractor(results).reset_index(drop=True)
    pd.testing.assert_frame_equal(actual_df, expected_df, check_like=True)


def test_extract():
    california = fetch_california_housing()
    df = pd.DataFrame(california['data'], columns=california['feature_names'])
    df['target'] = california['target']
    df['time'] = pd.date_range(start='2015-01-01', periods=len(df))
    np.random.seed(42)
    df['space'] = np.random.randint(0, 100, size=len(df))

    # Define train function
    train_fn = linear_regression_learner(features=california['feature_names'], target="target")

    # Define evaluator function
    base_evaluator = combined_evaluators(evaluators=[
        r2_evaluator(target_column='target', prediction_column='prediction'),
        spearman_evaluator(target_column='target', prediction_column='prediction')
    ])

    splitter = split_evaluator(eval_fn=base_evaluator, split_col='MedInc', split_values=[0.5, 10.0, 20.0])
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

    assert extract(sc_results, base_extractors).shape == (667, 9)
    assert extract(sc_results, splitter_extractor).shape == (2001, 10)

    assert extract(fw_sc_results, base_extractors).shape == (674, 9)
    assert extract(fw_sc_results, splitter_extractor).shape == (2022, 10)

    n_time_week_folds = len(df['time'].dt.strftime('%Y-%W').unique())
    n_time_year_folds = len(df['time'].dt.strftime('%Y').unique())
    assert temporal_week_splitter_extractor(temporal_week_results).shape == (n_time_week_folds, 3)
    assert temporal_year_splitter_extractor(temporal_year_results).shape == (n_time_year_folds, 3)
