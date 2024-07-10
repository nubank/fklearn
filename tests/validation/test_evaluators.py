import sys
import string

import numpy as np
import pandas as pd
import pytest

from fklearn.validation.evaluators import (
    auc_evaluator, brier_score_evaluator, combined_evaluators,
    correlation_evaluator, expected_calibration_error_evaluator,
    fbeta_score_evaluator, hash_evaluator, logloss_evaluator,
    mean_prediction_evaluator, mse_evaluator, permutation_evaluator,
    pr_auc_evaluator, precision_evaluator, r2_evaluator, recall_evaluator,
    roc_auc_evaluator, spearman_evaluator, linear_coefficient_evaluator, ndcg_evaluator, split_evaluator,
    temporal_split_evaluator, exponential_coefficient_evaluator, logistic_coefficient_evaluator)


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


def test_auc_evaluator_with_weights():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 0, 1],
            'prediction': [.2, .9, .3, .3],
            'weights': [1, 1, 1, 3],
        }
    )

    eval_fn = auc_evaluator(prediction_column="prediction",
                            target_column="target",
                            weight_column="weights",
                            eval_name="eval_name")

    result = eval_fn(predictions)

    assert result["eval_name"] == 0.8125


def test_roc_auc_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 0, 1],
            'prediction': [.2, .9, .3, .3]
        }
    )

    eval_fn = roc_auc_evaluator(prediction_column="prediction",
                                target_column="target",
                                eval_name="eval_name")

    result = eval_fn(predictions)

    assert result["eval_name"] == 0.875


def test_pr_auc_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [0, 1, 0, 1],
            'prediction': [.2, .9, .3, .3]
        }
    )

    eval_fn = pr_auc_evaluator(prediction_column="prediction",
                               target_column="target",
                               eval_name="eval_name")

    result = eval_fn(predictions)

    assert result["eval_name"] == pytest.approx(0.833333)


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


def test_linear_coefficient_evaluator():
    predictions = pd.DataFrame(
        {
            'target': [1, 2, 3],
            'prediction': [2, 4, 6]
        }
    )

    result = linear_coefficient_evaluator(predictions)

    assert result['linear_coefficient_evaluator__target'] == 0.5


@pytest.mark.parametrize("exponential_gain", [False, True])
def test_ndcg_evaluator(exponential_gain):
    predictions = pd.DataFrame(
        {
            'target': [1.0, 0.5, 1.5],
            'prediction': [0.9, 0.3, 1.2]
        }
    )

    k_raises = [-1, 0, 4]
    for k in k_raises:
        with pytest.raises(ValueError):
            ndcg_evaluator(
                predictions,
                k=k,
                exponential_gain=exponential_gain
            )

    k_not_raises = [None, 1, 2, 3]
    for k in k_not_raises:
        result = ndcg_evaluator(
            predictions,
            k=k,
            exponential_gain=exponential_gain
        )
        assert result['ndcg_evaluator__target'] == 1.0


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


def test_hash_evaluator():
    rows = 50
    np.random.seed(42)

    # Generate 120 different categories
    categories = [''.join(np.random.choice(list(string.ascii_uppercase + string.digits), size=10)) for _ in range(120)]
    # Create a dataframe with different datatypes
    df1 = pd.DataFrame({
        "feature1": np.random.normal(size=rows),
        "featureCategorical": np.random.choice(categories, size=rows),
        "featureDuplicate": np.repeat([100, 200], int(rows / 2))
    })

    # Create a dataframe with shuffled rows
    df2 = df1.copy().sample(frac=1.).reset_index(drop=True)
    df2["featureDuplicate"] = np.repeat([900, 1000], int(rows / 2))
    # create a dataframe changing one value of a feature in the row
    df3 = df1.copy()
    df3.iloc[0, 0] = 999.9

    # evaluate only in feature1 and featureCategorical
    eval_fn = hash_evaluator(hash_columns=["feature1", "featureCategorical"],
                             eval_name="eval_name")
    # evaluate hash in all columns
    eval_fn_all = hash_evaluator(eval_name="eval_name")
    eval_fn_order = hash_evaluator(hash_columns=["feature1", "featureCategorical"],
                                   eval_name="eval_name",
                                   consider_index=True)

    # shuffle preserves the hash with the default parameters
    assert eval_fn(df1)["eval_name"] == eval_fn(df2)["eval_name"]
    # if considering the index, the order matters
    assert eval_fn_order(df1)["eval_name"] != eval_fn_order(df2)["eval_name"]
    # changing one value of the feature should change the hash
    assert eval_fn(df1)["eval_name"] != eval_fn(df3)["eval_name"]
    # if we consider all the features in the dataframe, it should return different hashes for different dataframes
    assert eval_fn_all(df1)["eval_name"] != eval_fn_all(df2)["eval_name"]

    # Assert that the hashes stay the same everytime this is run.
    # The hash function is update in python 3.9 requiring different checks for each version.
    python_version = sys.version_info
    if python_version.minor == 8:
        assert eval_fn_all(df1)["eval_name"] == -6356943988420224450
        assert eval_fn_all(df2)["eval_name"] == -4865376220991082723
        assert eval_fn_all(df3)["eval_name"] == 141388279445698461
    else:
        assert eval_fn_all(df1)["eval_name"] == 12089800085289327166
        assert eval_fn_all(df2)["eval_name"] == 13581367852718468893
        assert eval_fn_all(df3)["eval_name"] == 141388279445698461


def test_exponential_coefficient_evaluator():

    a1 = -10
    a0 = -2

    prediction = np.array([0.0, 0.02, 0.04, 0.06, 0.08, 0.1])

    predictions = pd.DataFrame(dict(
        prediction=prediction,
        target=np.exp(a0 + a1 * prediction)
    ))

    result = exponential_coefficient_evaluator(predictions)

    assert result['exponential_coefficient_evaluator__target'] == pytest.approx(a1)


def test_logistic_coefficient_evaluator():

    predictions = pd.DataFrame(dict(
        prediction=[1, 1, 1, 2, 2, 2, 3, 3, 3],
        target=[0, 0, 0, 0, 0, 0, 1, 1, 1]
    ))

    result = logistic_coefficient_evaluator(predictions)

    SKLEARN_GTE_1_4_RESULT = 17.922
    SKLEARN_LT_1_4_RESULT = 20.645
    expected_result_range = {SKLEARN_GTE_1_4_RESULT, SKLEARN_LT_1_4_RESULT}

    assert round(result['logistic_coefficient_evaluator__target'], 3) in expected_result_range
