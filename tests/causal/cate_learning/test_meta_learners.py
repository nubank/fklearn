import numpy as np
import pandas as pd
from pandas import DataFrame

from fklearn.training.classification import logistic_classification_learner
from fklearn.causal.cate_learning.meta_learners import (
    TREATMENT_FEATURE,
    _append_treatment_feature,
    _get_learner_features,
    _get_unique_treatments,
    _filter_by_treatment,
    _create_treatment_flag,
    _fit_by_treatment,
    _predict_by_treatment_flag,
    _simulate_treatment_effect,
)


def test__append_treatment_feature():
    features = ["feat1", "feat2", "feat3"]
    treatment_feature = "treatment"

    assert _append_treatment_feature(features, treatment_feature) == features + [
        treatment_feature
    ]
    assert len(features) > 0
    assert treatment_feature


# def test__get_learner_features():
#   assert


def test__get_unique_treatments():

    df = pd.DataFrame(
        {
            "feature": [1.0, 4.0, 1.0, 5.0, 3.0],
            "treatment": [
                "treatment_A",
                "treatment_C",
                "treatment_B",
                "treatment_A",
                "control",
            ],
            "target": [1, 1, 0, 0, 1],
        }
    )

    assert _get_unique_treatments(
        df, treatment_col="treatment", control_name="control"
    ) == ["treatment_A", "treatment_B", "treatment_C"]


def test__filter_by_treatment():
    values = [
        [1.0, "treatment_A", 1],
        [4.0, "treatment_C", 1],
        [1.0, "treatment_B", 0],
        [5.0, "treatment_A", 0],
        [3.0, "control", 1],
    ]

    df = pd.DataFrame(data=values, columns=["feat1", "treatment", "target"])

    selected_treatment = "treatment_A"

    expected_values = [
        [1.0, "treatment_A", 1],
        [5.0, "treatment_A", 0],
        [3.0, "control", 1],
    ]

    expected_df: DataFrame = pd.DataFrame(
        data=expected_values, columns=["feat1", "treatment", "target"]
    )

    assert (
        _filter_by_treatment(
            df,
            treatment_col="treatment",
            treatment_name=selected_treatment,
            control_name="control",
        )
        == expected_df
    )


def test__create_treatment_flag():
    df = pd.DataFrame(
        {
            "feature": [1.3, 1.0, 1.8, -0.1],
            "group": ["treatment", "treatment", "control", "control"],
            "target": [1, 1, 1, 0],
        }
    )

    expected_df = pd.DataFrame(
        {
            "feature": [1.3, 1.0, 1.8, -0.1],
            "group": ["treatment", "treatment", "control", "control"],
            "target": [1, 1, 1, 0],
            TREATMENT_FEATURE: [1.0, 1.0, 0.0, 0.0],
        }
    )

    assert (
        _create_treatment_flag(df, treatment_col="group", treatment_name="treatment")
        == expected_df
    )


def test__fit_by_treatment():
    df = pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1, 0.0, 1.0, 2.2, 0.4, -5.0],
            "x2": [10, 4, 15, 6, 5, 12, 14, 5, 12],
            "treatment": [
                "A",
                "B",
                "A",
                "A",
                "B",
                "control",
                "control",
                "B",
                "control",
            ],
            "target": [1, 1, 1, 0, 0, 1, 0, 0, 1],
        }
    )

    learner_binary = logistic_classification_learner(
        features=["x1", "x2", TREATMENT_FEATURE],
        target="target",
        params={"max_iter": 10},
    )

    treatments = ["A", "B"]

    learners, logs = _fit_by_treatment(
        df,
        learner=learner_binary,
        treatment_col="treatment",
        control_name="control",
        treatments=treatments,
    )

    assert len(learners) == len(treatments)
    assert len(logs) == len(treatments)


def test__predict_by_treatment_flag_positive():
    df = pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1],
            "x2": [10, 4, 15, 6],
            TREATMENT_FEATURE: [1.0, 1.0, 0.0, 0.0],
            "target": [1, 1, 1, 0],
        }
    )

    learner_binary = logistic_classification_learner(
        features=["x1", "x2", TREATMENT_FEATURE],
        target="target",
        params={"max_iter": 10},
    )

    predict_fn, pred_df, log = learner_binary(df)

    prediction_array = _predict_by_treatment_flag(
        df, learner_fcn=predict_fn, is_treatment=True, prediction_column="prediction"
    )

    expected_array = np.array([0.79878432, 0.65191703, 0.88361953, 0.68358276])

    assert TREATMENT_FEATURE in df.columns
    assert prediction_array == expected_array


def test__predict_by_treatment_flag_negative():
    df = pd.DataFrame(
        {
            "x1": [1.3, 1.0, 1.8, -0.1],
            "x2": [10, 4, 15, 6],
            TREATMENT_FEATURE: [1.0, 1.0, 0.0, 0.0],
            "target": [1, 1, 1, 0],
        }
    )

    learner_binary = logistic_classification_learner(
        features=["x1", "x2", TREATMENT_FEATURE],
        target="target",
        params={"max_iter": 10},
    )

    predict_fn, pred_df, log = learner_binary(df)

    prediction_array = _predict_by_treatment_flag(
        df, learner_fcn=predict_fn, is_treatment=False, prediction_column="prediction"
    )

    expected_array = np.array([0.78981053, 0.63935056, 0.87785064, 0.67158357])

    assert TREATMENT_FEATURE in df.columns
    assert prediction_array == expected_array


# def test__simulate_treatment_effect():
#     df = pd.DataFrame({
#         'x1': [1.3, 1.0, 1.8, -0.1, 0.0, 1.0, 2.2, 0.4, -5.0],
#         "x2": [10, 4, 15, 6, 5, 12, 14, 5, 12],
#         "treatment": ["A", "B", "A", "A", "B", "control", "control", "B", "control"],
#         'target': [1, 1, 1, 0, 0, 1, 0, 0, 1]
#     })

#     treatments = ["A", "B"]

#     learner_binary = logistic_classification_learner(features=["x1", "x2", TREATMENT_FEATURE],
#                                                      target="target",
#                                                      params={"max_iter": 10})

#     learners, _ = _fit_by_treatment(df,
#                                     learner=learner_binary,
#                                     treatment_col="treatment",
#                                     control_name="control",
#                                     treatments=treatments)
