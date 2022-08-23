import copy
import inspect
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd
from toolz import curry

from fklearn.common_docstrings import (
    learner_pred_fn_docstring,
    learner_return_docstring,
)
from fklearn.exceptions.exceptions import (
    MissingControlError,
    MissingTreatmentError,
    MultipleTreatmentsError,
)
from fklearn.training.pipeline import build_pipeline
from fklearn.types import LearnerFnType, LearnerReturnType, PredictFnType

TREATMENT_FEATURE = "is_treatment"


def _append_treatment_feature(features: list, treatment_feature: str) -> list:
    return features + [treatment_feature]


def _get_learner_features(learner: Callable) -> list:
    return inspect.signature(learner).parameters["features"].default


def _get_unique_treatments(
    df: pd.DataFrame, treatment_col: str, control_name: str
) -> list:
    if control_name not in df[treatment_col].unique():
        raise MissingControlError()

    return [col for col in df[treatment_col].unique() if col != control_name]


def _filter_by_treatment(
    df: pd.DataFrame, treatment_col: str, treatment_name: str, control_name: str
) -> pd.DataFrame:
    treatment_control_values = df[treatment_col].unique()

    if control_name not in treatment_control_values:
        raise MissingControlError()

    if treatment_name not in treatment_control_values:
        raise MissingTreatmentError()

    treatment_control_df = df.loc[
        (df[treatment_col] == treatment_name) | (df[treatment_col] == control_name), :
    ]
    return treatment_control_df


def _create_treatment_flag(
    df: pd.DataFrame, treatment_col: str, treatment_name: str, control_name: str
) -> pd.DataFrame:
    df = df.copy()

    treatment_control_values = df[treatment_col].unique()

    if len(_get_unique_treatments(df, treatment_col, control_name)) > 1:
        raise MultipleTreatmentsError()

    if control_name not in treatment_control_values:
        raise MissingControlError()

    if treatment_name not in treatment_control_values:
        raise MissingTreatmentError()

    treatment_flag = np.where(df[treatment_col] == treatment_name, 1.0, 0.0)
    df[TREATMENT_FEATURE] = treatment_flag

    return df


def _fit_by_treatment(
    df: pd.DataFrame,
    learner: LearnerFnType,
    treatment_col: str,
    control_name: str,
    treatments: list,
) -> Tuple[dict, dict]:
    fitted_learners = {}
    learners_logs = {}
    for treatment in treatments:
        treatment_control_df = _filter_by_treatment(
            df=df,
            treatment_col=treatment_col,
            treatment_name=treatment,
            control_name=control_name,
        )
        treatment_control_df = _create_treatment_flag(
            df=treatment_control_df,
            treatment_col=treatment_col,
            treatment_name=treatment,
            control_name=control_name,
        )
        learner_fcn, _, learner_log = learner(treatment_control_df)
        fitted_learners[treatment] = learner_fcn
        learners_logs[treatment] = learner_log

    return fitted_learners, learners_logs


def _predict_by_treatment_flag(
    df: pd.DataFrame,
    learner_fcn: PredictFnType,
    is_treatment: bool,
    prediction_column: str,
) -> np.ndarray:

    treatment_flag = np.ones(df.shape[0]) if is_treatment else np.zeros(df.shape[0])
    df[TREATMENT_FEATURE] = treatment_flag
    prediction_df = learner_fcn(df)
    df.drop(columns=[TREATMENT_FEATURE], inplace=True)

    return prediction_df[prediction_column].values


def _simulate_treatment_effect(
    df: pd.DataFrame,
    treatments: list,
    control_name: str,
    learners: dict,
    prediction_column: str,
) -> pd.DataFrame:
    uplift_cols = []
    scored_df = df.copy()
    for treatment in treatments:
        learner_fcn = learners[treatment]

        scored_df[
            f"treatment_{treatment}__{prediction_column}_on_treatment"
        ] = _predict_by_treatment_flag(
            df=scored_df,
            learner_fcn=learner_fcn,
            is_treatment=True,
            prediction_column=prediction_column,
        )
        scored_df[
            f"treatment_{treatment}__{prediction_column}_on_control"
        ] = _predict_by_treatment_flag(
            df=scored_df,
            learner_fcn=learner_fcn,
            is_treatment=False,
            prediction_column=prediction_column,
        )

        uplift_cols.append(f"treatment_{treatment}__uplift")
        scored_df[uplift_cols[-1]] = (
            scored_df[f"treatment_{treatment}__{prediction_column}_on_treatment"]
            - scored_df[f"treatment_{treatment}__{prediction_column}_on_control"]
        )

    scored_df["uplift"] = scored_df[uplift_cols].max(axis=1).values
    scored_df["suggested_treatment"] = np.where(
        scored_df["uplift"].values <= 0,
        control_name,
        scored_df[uplift_cols].idxmax(axis=1).values,
    )
    scored_df["suggested_treatment"] = (
        scored_df["suggested_treatment"]
        .apply(lambda x: x.replace("__uplift", ""))
        .values
    )

    return scored_df


@curry
def causal_s_classification_learner(
    df: pd.DataFrame,
    treatment_col: str,
    control_name: str,
    prediction_column: str,
    learner: Callable,
    learner_transformers: List[LearnerFnType] = None,
) -> LearnerReturnType:
    """
    Fits a Causal S-Learner classifier. The S-learner is a meta-learner which
    learns the Conditional Average Treatment Effect (CATE) through the creation
    of an auxiliary binary feature T that indicates if the samples is in the
    treatment (T = 1) or in the control (T = 0) group. Then, this feature can
    then be used to perform inference by artificially simulating the conversion
    of a new sample for both scenarios, i.e., with T = 0 and T = 1. The CATE τ
    is defined as τ(xi) = M(X=xi, T=1) - M(X=xi, T=0), being M a Machine Learning
    Model.
    References:
    [1] https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html
    [2] https://causalml.readthedocs.io/en/latest/methodology.html
    Parameters
    ----------
    df : pd.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.
    treatment_col: str
        The name of the column in `df` which contains the names of
        the treatments or control to which each data sample was subjected.
    control_name: str
        The name of the control group.
    prediction_column : str
        The name of the column with the predictions from the provided learner.
    learner: Callable
        A fklearn classification learner function.
    learner_transformers: list
        A list of fklearn transformer functions to be applied after the learner and before estimating the CATE.
        This parameter may be useful, for example, to estimate the CATE with calibrated classifiers.
    """

    learner = copy.deepcopy(learner)
    features = _get_learner_features(learner)
    features_with_treatment = _append_treatment_feature(features, TREATMENT_FEATURE)
    unique_treatments = _get_unique_treatments(df, treatment_col, control_name)

    if learner_transformers is not None:
        learner_transformers = copy.deepcopy(learner_transformers)
        learner_pipe = build_pipeline(
            *[
                learner(features=features_with_treatment),
            ]
            + learner_transformers
        )
    else:
        learner_pipe = learner(features=features_with_treatment)

    fitted_learners, learners_logs = _fit_by_treatment(
        df=df,
        learner=learner_pipe,
        treatment_col=treatment_col,
        control_name=control_name,
        treatments=unique_treatments,
    )

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        scored_df = _simulate_treatment_effect(
            df=new_df,
            treatments=unique_treatments,
            learners=fitted_learners,
            control_name=control_name,
            prediction_column=prediction_column,
        )
        return scored_df

    p.__doc__ = learner_pred_fn_docstring("causal_s_classification_learner")
    log = {
        "causal_s_classification_learner": {
            **learners_logs,
            "causal_features": features_with_treatment,
        }
    }

    return p, p(df), log


causal_s_classification_learner.__doc__ += learner_return_docstring(
    "Causal S-Learner Classifier"
)


def _simulate_t_learner_treatment_effect(
    df: pd.DataFrame,
    learners: dict,
    treatments: list,
    control_name: str,
    prediction_column: str,
) -> pd.DataFrame:
    """
    Simulate the treatment effect for the T-Leaner.
    """
    control_fcn = learners[control_name]
    control_conversion_probability = control_fcn(df)[prediction_column].values

    scored_df = df.copy()

    uplift_cols = []
    for treatment_name in treatments:
        treatment_fcn = learners[treatment_name]
        treatment_conversion_probability = treatment_fcn(df)[prediction_column].values

        uplift_cols.append(f"treatment_{treatment_name}__uplift")
        scored_df[uplift_cols[-1]] = (
            treatment_conversion_probability - control_conversion_probability
        )

    scored_df["uplift"] = scored_df[uplift_cols].max(axis=1).values
    scored_df["suggested_treatment"] = np.where(
        scored_df["uplift"].values <= 0,
        control_name,
        scored_df[uplift_cols].idxmax(axis=1).values,
    )
    scored_df["suggested_treatment"] = (
        scored_df["suggested_treatment"]
        .apply(lambda x: x.replace("__uplift", ""))
        .values
    )

    return scored_df


def _get_model_fcn(
    df: pd.DataFrame,
    treatment_col: str,
    treatment_name: str,
    learner: Callable,
) -> Tuple[Callable, dict, dict]:
    """
    Returns a function that predicts the target column from the features.
    """
    df = df.loc[df[treatment_col] == treatment_name].reset_index(drop=True).copy()

    return learner(df)


def _get_learners(
    df: pd.DataFrame,
    unique_treatments: list,
    control_learner: Callable,
    control_name: str,
    treatment_learner: Callable,
    treatment_col: str,
) -> dict:
    learners: Dict[str, Callable] = {}

    learner_fcn, _, _ = _get_model_fcn(
        df, treatment_col, control_name, control_learner
    )
    learners[control_name] = learner_fcn

    for treatment_name in unique_treatments:
        learner_fcn, _, _ = _get_model_fcn(
            df, treatment_col, treatment_name, treatment_learner
        )
        learners[treatment_name] = learner_fcn

    return learners


@curry
def causal_t_classification_learner(
    df: pd.DataFrame,
    treatment_col: str,
    control_name: str,
    prediction_column: str,
    learner: Callable,
    treatment_learner: Callable = None,
    learner_transformers: List[LearnerFnType] = None,
) -> LearnerReturnType:
    """
    Fits a Causal T-Learner classifier. The T-Learner is a meta-learner which learns the
    Conditional Average Treatment Effect (CATE) through the use of one Machine Learning
    model for each treatment and for the control group. Each model is fitted in a subset of 
    the data, according to the treatment: the CATE $\tau$ is defined as
    $\tau(x_{i}) = M_{1}(X=x_{i}, T=1) - M_{0}(X=x_{i}, T=0)$, being $M_{1}$ a model fitted 
    with treatment data and $M_{0}$ a model fitted with control data. Notice that $M_{0}$
    and $M_{1}$ are traditional Machine Learning Model such as a LightGBM Classifier and 
    that $x_{i}$ is the feature set of sample $i$.

    References:
    [1] https://matheusfacure.github.io/python-causality-handbook/21-Meta-Learners.html
    [2] https://causalml.readthedocs.io/en/latest/methodology.html

    Parameters
    ----------

    df : pd.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    treatment_col: str
        The name of the column in `df` which contains the names of
        the treatments or control to which each data sample was subjected.

    control_name: str
        The name of the control group.

    prediction_column : str
        The name of the column with the predictions from the provided learner.

    learner: LearnerMutableParametersFnType
        A fklearn classification learner function.

    treatment_learner: LearnerFnType
        An optional fklearn classification learner function.

    learner_transformers: list
        A list of fklearn transformer functions to be applied after the learner and before estimating the CATE.
        This parameter may be useful, for example, to estimate the CATE with calibrated classifiers.
    """

    control_learner = copy.deepcopy(learner)

    if treatment_learner is None:
        treatment_learner = copy.deepcopy(learner)

    control_features = _get_learner_features(control_learner)
    treatment_features = _get_learner_features(treatment_learner)

    # pipeline
    if learner_transformers is not None:
        learner_transformers = copy.deepcopy(learner_transformers)
        control_learner_pipe = build_pipeline(
            *[control_learner(features=control_features)] + learner_transformers
        )

        treatment_learner_pipe = build_pipeline(
            *[treatment_learner(features=treatment_features)] + learner_transformers
        )
    else:
        control_learner_pipe = control_learner(features=control_features)
        treatment_learner_pipe = treatment_learner(features=treatment_features)

    # learners
    unique_treatments = _get_unique_treatments(df, treatment_col, control_name)

    learners = _get_learners(
        df=df,
        unique_treatments=unique_treatments,
        control_learner=control_learner_pipe,
        control_name=control_name,
        treatment_learner=treatment_learner_pipe,
        treatment_col=treatment_col,
    )

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return _simulate_t_learner_treatment_effect(
            new_df,
            learners,
            unique_treatments,
            control_name,
            prediction_column,
        )

    p.__doc__ = learner_pred_fn_docstring("causal_t_classification_learner", shap=True)
    partial_log = {
        "causal_features": {
            "control": control_features,
            "treatment": treatment_features,
        },
    }

    log = {"causal_t_classification_learner": partial_log}

    return p, p(df), log


causal_t_classification_learner.__doc__ = learner_return_docstring(
    "Causal T-Learner Classifier"
)
