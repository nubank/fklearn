import pandas as pd
import sklearn

from sklearn.isotonic import IsotonicRegression
from toolz import curry

from fklearn.common_docstrings import learner_pred_fn_docstring, learner_return_docstring
from fklearn.types import LearnerReturnType
from fklearn.training.utils import log_learner_time


@curry
@log_learner_time(learner_name='isotonic_calibration_learner')
def isotonic_calibration_learner(df: pd.DataFrame,
                                 target_column: str = "target",
                                 prediction_column: str = "prediction",
                                 output_column: str = "calibrated_prediction",
                                 y_min: float = 0.0,
                                 y_max: float = 1.0) -> LearnerReturnType:
    """
    Fits a single feature isotonic regression to the dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    target_column : str
        The name of the column in `df` that should be used as target for the model.
        This column should be binary, since this is a classification model.

    prediction_column : str
        The name of the column with the uncalibrated predictions from the model.

    output_column : str
        The name of the column with the calibrated predictions from the model.

    y_min: float
        Lower bound of Isotonic Regression

    y_max: float
        Upper bound of Isotonic Regression

    """

    clf = IsotonicRegression(y_min=y_min, y_max=y_max, out_of_bounds='clip')

    clf.fit(df[prediction_column], df[target_column])

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{output_column: clf.predict(new_df[prediction_column])})

    p.__doc__ = learner_pred_fn_docstring("isotonic_calibration_learner")

    log = {'isotonic_calibration_learner': {
        'output_column': output_column,
        'target_column': target_column,
        'prediction_column': prediction_column,
        'package': "sklearn",
        'package_version': sklearn.__version__,
        'training_samples': len(df)},
        'object': clf}

    return p, p(df), log


isotonic_calibration_learner.__doc__ += learner_return_docstring("Isotonic Calibration")


@curry
@log_learner_time(learner_name='find_thresholds_with_same_risk')
def find_thresholds_with_same_risk(df: pd.DataFrame,
                                   sensitive_factor: str,
                                   unfair_band_column: str,
                                   model_prediction_output: str,
                                   target_column: str = "target",
                                   output_column_name: str = "fair_band") -> LearnerReturnType:
    """
    Calculate fair calibration, where for each band any sensitive factor group have the same target mean.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    sensitive_factor: str
        Column where we have the different group classifications that we want to have the same target mean

    unfair_band_column: str
        Column with the original bands

    model_prediction_output : str
        Risk model's output

    target_column : str
        The name of the column in `df` that should be used as target for the model.
        This column should be binary, since this is a classification model.

    output_column_name : str
        The name of the column with the fair bins.
    """
    sorted_df = df.sort_values(by=model_prediction_output).reset_index(drop=True)

    def _find_thresholds_with_same_risk(df: pd.DataFrame,
                                        metric_by_band: pd.DataFrame) -> list:
        current_threshold = -1
        fair_thresholds = [current_threshold]

        for band, metric in metric_by_band.iterrows():
            df = df[df[model_prediction_output] > current_threshold]
            if df.empty:
                break
            df["cumulative_risk"] = df[target_column].expanding(min_periods=1).mean()
            df["distance"] = abs(df["cumulative_risk"] - metric[target_column])
            threshold = df.sort_values(by="distance").iloc[0][model_prediction_output]

            fair_thresholds.append(threshold)

            current_threshold = threshold

        fair_thresholds[-1] = df[model_prediction_output].max()

        return fair_thresholds

    unfair_bands = sorted(sorted_df[unfair_band_column].unique())
    metric_by_band = sorted_df.groupby(unfair_band_column).agg({target_column: "mean"})
    sensitive_groups = list(filter(lambda x: x, sorted_df[sensitive_factor].unique()))
    fair_thresholds = {}

    for group in sensitive_groups:
        raw_ecdf_with_target = sorted_df[sorted_df[sensitive_factor] == group][[model_prediction_output, target_column]]
        fair_thresholds[group] = _find_thresholds_with_same_risk(raw_ecdf_with_target,
                                                                 metric_by_band)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        new_df_copy = new_df.copy()
        new_df_copy[output_column_name] = pd.Series(dtype='int')
        for group in sensitive_groups:
            group_filter = new_df_copy[sensitive_factor] == group
            n_of_bands = len(fair_thresholds[group]) - 1
            new_df_copy.loc[group_filter, output_column_name] = pd.cut(
                new_df_copy.loc[group_filter, model_prediction_output],
                bins=fair_thresholds[group],
                labels=unfair_bands[:n_of_bands]).astype(float)
        return new_df_copy[output_column_name]

    p.__doc__ = learner_pred_fn_docstring("find_thresholds_with_same_risk")

    log = {'find_thresholds_with_same_risk': {
        'output_column': output_column_name,
        'prediction_ecdf': model_prediction_output,
        'target_column': target_column,
        'unfair_band_column': unfair_band_column,
        'sensitive_factor': sensitive_factor,
        'fair_thresholds': fair_thresholds}}

    return p, p(df), log


find_thresholds_with_same_risk.__doc__ += learner_return_docstring("find_thresholds_with_same_risk")
