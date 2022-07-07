import copy
import inspect
from typing import List, Tuple

import numpy as np
import pandas as pd
from fklearn.common_docstrings import (learner_pred_fn_docstring,
                                       learner_return_docstring)
from fklearn.training.classification import lgbm_classification_learner
from fklearn.training.pipeline import build_pipeline
from fklearn.types import LearnerFnType, LearnerReturnType, PredictFnType
from toolz import curry

TREATMENT_FEATURE = "is_treatment"


def _append_treatment_feature(features: list, treatment_feature: str) -> list:
    return features + [treatment_feature]


def _get_learner_features(learner: LearnerFnType) -> list:
    return inspect.signature(learner).parameters["features"].default


def _get_unique_treatments(
    df: pd.DataFrame, treatment_col: str, control_name: str
) -> list:
    return [col for col in df[treatment_col].unique() if col != control_name]


def _filter_by_treatment(
    df: pd.DataFrame, treatment_col: str, treatment_name: str, control_name: str
) -> pd.DataFrame:
    treatment_control_df = df.loc[
        (df[treatment_col] == treatment_name) | (df[treatment_col] == control_name), :
    ]
    return treatment_control_df


def _create_treatment_flag(
    df: pd.DataFrame, treatment_col: str, treatment_name: str
) -> pd.DataFrame:
    df = df.copy()
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
        )
        learner_fcn, _, learner_log = learner(treatment_control_df)
        fitted_learners[treatment] = learner_fcn
        learners_logs[treatment] = learner_log

    return fitted_learners, learners_logs


def _predict_for_control(
    df: pd.DataFrame, learner_fcn: PredictFnType, prediction_column: str
) -> np.array:
    control_flag = np.zeros(df.shape[0])
    df[TREATMENT_FEATURE] = control_flag
    control_pred_df = learner_fcn(df)
    return control_pred_df[prediction_column].values


def _predict_for_treatment(
    df: pd.DataFrame, learner_fcn: PredictFnType, prediction_column: str
) -> np.array:
    treatment_flag = np.ones(df.shape[0])
    df[TREATMENT_FEATURE] = treatment_flag
    treatment_pred_df = learner_fcn(df)
    return treatment_pred_df[prediction_column].values


def _prediction_by_treatment_flag(
    df: pd.DataFrame, treatments: list, learners: dict, prediction_column: str
) -> pd.DataFrame:
    uplift_cols = []
    scored_df = df.copy()
    for treatment in treatments:

        learner_fcn = learners[treatment]

        scored_df[
            f"treatment_{treatment}__{prediction_column}_on_treatment"
        ] = _predict_for_treatment(
            df=scored_df, learner_fcn=learner_fcn, prediction_column=prediction_column
        )
        scored_df[
            f"treatment_{treatment}__{prediction_column}_on_control"
        ] = _predict_for_control(
            df=scored_df, learner_fcn=learner_fcn, prediction_column=prediction_column
        )

        uplift_cols.append(f"treatment_{treatment}__uplift")
        scored_df[uplift_cols[-1]] = (
            scored_df[f"treatment_{treatment}__{prediction_column}_on_treatment"]
            - scored_df[f"treatment_{treatment}__{prediction_column}_on_control"]
        )

    scored_df["uplift"] = scored_df[uplift_cols].max(axis=1).values
    scored_df["suggested_treatment"] = scored_df[uplift_cols].idxmax(axis=1).values
    scored_df["suggested_treatment"] = (
        scored_df["suggested_treatment"]
        .apply(lambda x: x.replace("_uplift", ""))
        .values
    )
    scored_df.drop(columns=[TREATMENT_FEATURE], inplace=True)

    return scored_df


@curry
def causal_s_classification_learner(
    df: pd.DataFrame,
    treatment_col: str,
    control_name: str,
    prediction_column: str,
    learner: LearnerFnType = None,
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

    Parameters
    ----------

    df : pd.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    treatment_col: str
        The name of the column in `df` which contains the names of
        the treatments or control to which each data sample was subjected.

    prediction_column : str
        The name of the column with the predictions from the model.
        If a multiclass problem, additional prediction_column_i columns will be added for i in range(0,n_classes).

    learner: LearnerFnType
        A fklearn classification learner function.

    learner_transformers: list
        A list of fklearn transformer functions to be applied after the learner and before estimating the CATE.
        This parameter may be useful, for example, to estimate the CATE with calibrated classifiers.
    """

    if learner is None:
        learner = copy.deepcopy(lgbm_classification_learner)
    else:
        learner = copy.deepcopy(learner)

    features = _get_learner_features(learner)
    features_with_treatment = _append_treatment_feature(features, TREATMENT_FEATURE)
    unique_treatments = _get_unique_treatments(df, treatment_col, control_name)

    if not isinstance(learner_transformers, list):
        learner_transformers = [learner_transformers]

    if learner_transformers is not None:
        learner_transformers = copy.deepcopy(learner_transformers)
        learner_pipe = build_pipeline(
            *[
                learner(features=features_with_treatment),
            ]
            + learner_transformers
        )
    else:
        learner_pipe = learner

    fitted_learners, learners_logs = _fit_by_treatment(
        df=df,
        learner=learner_pipe,
        treatment_col=treatment_col,
        control_name=control_name,
        treatments=unique_treatments,
    )

    def p(new_df: pd.DataFrame):
        scored_df = _prediction_by_treatment_flag(
            df=new_df,
            treatments=unique_treatments,
            learners=fitted_learners,
            prediction_column=prediction_column,
        )
        return scored_df

    p.__doc__ = learner_pred_fn_docstring("causal_s_classification_learner", shap=True)

    partial_log = {"causal_features": features_with_treatment}
    partial_log.update(learners_logs)

    log = {"causal_s_classification_learner": partial_log}

    return p, p(df), log


causal_s_classification_learner.__doc__ += learner_return_docstring(
    "Causal S-Learner Classifier"
)
