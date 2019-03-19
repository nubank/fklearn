import numpy as np
import pandas as pd

from fklearn.validation.evaluators import \
    r2_evaluator, mse_evaluator, combined_evaluators, mean_prediction_evaluator, auc_evaluator, \
    precision_evaluator, recall_evaluator, fbeta_score_evaluator, logloss_evaluator, brier_score_evaluator, \
    expected_calibration_error_evaluator, correlation_evaluator, spearman_evaluator, split_evaluator, \
    temporal_split_evaluator, permutation_evaluator


def test_combined_evaluators():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 2],
            'prediction': [0.5, 0.9, 1.5]
        }
    )

    eval_fn1 = r2_evaluator
    eval_fn2 = mse_evaluator

    result = combined_evaluators(predictions, [eval_fn1, eval_fn2])

    assert result['mse_evaluator__target'] == 0.17
    assert result['r2_evaluator__target'] == 0.745


def test_mean_prediction_evaluator():
    predictions = pd.DataFrame(
        {
            'prediction': [1, 0.9, 40]
        }
    )

    eval_fn = mean_prediction_evaluator(prediction_column="prediction",
                                        eval_name="eval_name")

    result = eval_fn(predictions)

    assert result["eval_name"] == (1 + 0.9 + 40) / 3


def test_auc_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 0, 1],
            'prediction': [.2, .9, .3, .3]
        }
    )

    eval_fn = auc_evaluator(prediction_column="prediction",
                            target_column="target",
                            eval_name="eval_name")

    result = eval_fn(predictions)

    assert result["eval_name"] == 0.875


def test_precision_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 0, 1],
            'prediction': [.2, .9, .3, .3]
        }
    )

    eval_fn = precision_evaluator(prediction_column="prediction",
                                  threshold=0.5,
                                  target_column="target",
                                  eval_name="eval_name")

    result = eval_fn(predictions)

    assert result["eval_name"] == 1.0


def test_recall_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 0, 1],
            'prediction': [.2, .9, .3, .3]
        }
    )

    eval_fn = recall_evaluator(prediction_column="prediction",
                               threshold=0.5,
                               target_column="target",
                               eval_name="eval_name")

    result = eval_fn(predictions)

    assert result["eval_name"] == 0.5


def test_fbeta_score_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 0, 1],
            'prediction': [.2, .9, .3, .8]
        }
    )

    eval_fn = fbeta_score_evaluator(prediction_column="prediction",
                                    threshold=0.5,
                                    beta=1,
                                    target_column="target",
                                    eval_name="eval_name")

    result = eval_fn(predictions)

    assert result["eval_name"] == 1.0


def test_logloss_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 0, 1],
            'prediction': [.2, .9, .3, .3]
        }
    )

    eval_fn = logloss_evaluator(prediction_column="prediction",
                                target_column="target",
                                eval_name="eval_name")

    result = eval_fn(predictions)

    assert abs(result["eval_name"] - 0.4722879) < 0.0001


def test_brier_score_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 0, 1],
            'prediction': [.2, .9, .3, .3]
        }
    )

    eval_fn = brier_score_evaluator(prediction_column="prediction",
                                    target_column="target",
                                    eval_name="eval_name")

    result = eval_fn(predictions)

    assert abs(result["eval_name"] - 0.1575) < 0.0001


def test_binary_calibration_evaluator():
    np.random.seed(42)
    probs = np.linspace(1e-10, 1.0 - 1e-10, 100000)
    target = np.random.binomial(n=1, p=probs, size=100000)

    predictions = pd.DataFrame(
        {
            'target': target,
            'prediction': probs
        }
    )

    eval_fn = expected_calibration_error_evaluator(
        prediction_column="prediction",
        target_column="target",
        eval_name="eval_name",
        n_bins=100,
        bin_choice="count"
    )

    result_count = eval_fn(predictions)

    assert result_count["eval_name"] < 0.1

    eval_fn = expected_calibration_error_evaluator(
        prediction_column="prediction",
        target_column="target",
        eval_name="eval_name",
        n_bins=100,
        bin_choice="prob"
    )

    result_prob = eval_fn(predictions)

    assert result_prob["eval_name"] < 0.1

    assert abs(result_count["eval_name"] - result_prob["eval_name"]) < 1e-3


def test_r2_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 2],
            'prediction': [0.5, 0.9, 1.5]
        }
    )

    result = r2_evaluator(predictions)

    assert result['r2_evaluator__target'] == 0.745


def test_mse_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 2],
            'prediction': [0.5, 0.9, 1.5]
        }
    )

    result = mse_evaluator(predictions)

    assert result['mse_evaluator__target'] == 0.17


def test_correlation_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 2],
            'prediction': [0.5, 1.0, 1.5]
        }
    )

    result = correlation_evaluator(predictions)

    assert result['correlation_evaluator__target'] == 1.0


def test_spearman_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 2],
            'prediction': [0.5, 0.9, 1.5]
        }
    )

    result = spearman_evaluator(predictions)

    assert result['spearman_evaluator__target'] == 1.0


def test_split_evaluator():
    predictions = pd.DataFrame(
        {
            'split_col_a': [1, 1, 0],
            'split_col_b': [2, 0, 0],
            'target': [0, 1, 2],
            'prediction': [0.5, 0.9, 1.5]
        }
    )

    base_eval = mean_prediction_evaluator
    split_eval = split_evaluator(eval_fn=base_eval, split_col='split_col_a', split_values=[1])

    result = split_evaluator(predictions, split_eval, 'split_col_b', [2])

    assert \
        result['split_evaluator__split_col_b_2']['split_evaluator__split_col_a_1']['mean_evaluator__prediction'] == 0.5


def test_temporal_split_evaluator():
    predictions = pd.DataFrame(
        {
            'time': pd.date_range("2017-01-01", periods=10, freq="W"),
            'target': [0, 1, 0, 1, 0, 1, 2, 0, 0, 0],
            'prediction': [1.0, 0.0, 1.0, 0.0, 1.0, 2.0, 2.0, 2.0, 2.0, 0.0]
        }
    )

    base_eval = mean_prediction_evaluator
    result = temporal_split_evaluator(predictions, eval_fn=base_eval, time_col='time')

    expected = {
        'split_evaluator__time_2017-01': {'mean_evaluator__prediction': 0.6},
        'split_evaluator__time_2017-02': {'mean_evaluator__prediction': 2.0},
        'split_evaluator__time_2017-03': {'mean_evaluator__prediction': 0.0}
    }

    assert result == expected


def test_permutation_evaluator():
    test_df = pd.DataFrame(
        {
            'a': [1, 1, 0],
            'bb': [2, 0, 0],
            'target': [0, 1, 2]
        }
    )

    base_eval = r2_evaluator

    def fake_predict(df):
        return df.assign(prediction=[0.5, 0.9, 1.5])

    expected_results = {'r2_evaluator__target': 0.745}

    pimp1 = permutation_evaluator(test_df, fake_predict, base_eval, features=["a"], baseline=True,
                                  shuffle_all_at_once=False)

    assert pimp1['permutation_importance']['a'] == expected_results
    assert pimp1['permutation_importance_baseline'] == expected_results

    pimp2 = permutation_evaluator(test_df, fake_predict, base_eval, features=["a", "bb"], baseline=False,
                                  shuffle_all_at_once=False)

    assert pimp2['permutation_importance']['a'] == expected_results
    assert pimp2['permutation_importance']['bb'] == expected_results

    pimp3 = permutation_evaluator(test_df, fake_predict, base_eval, features=["a", "bb"], baseline=True,
                                  shuffle_all_at_once=True)

    assert pimp3['permutation_importance']['a-bb'] == expected_results
    assert pimp3['permutation_importance_baseline'] == expected_results

    test_df2 = pd.DataFrame(
        {
            'abc': np.linspace(0, 1, 100),
            'abcd': 1.0 - np.linspace(0, 1, 100),
            'target': np.ones(100)
        }
    )

    def fake_predict2(df):
        return df.assign(prediction=df['abc'] + df['abcd'])

    expected_results2 = {'r2_evaluator__target': 1.0}

    pimp4 = permutation_evaluator(test_df2, fake_predict2, base_eval, features=["abc"], baseline=True,
                                  shuffle_all_at_once=False, random_state=0)

    assert pimp4['permutation_importance']['abc'] != expected_results2
    assert pimp4['permutation_importance_baseline'] == expected_results2
