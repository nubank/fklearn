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
                                 output_column: str = "calibrated_prediction") -> LearnerReturnType:
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

    """

    clf = IsotonicRegression(y_min=0, y_max=1, out_of_bounds='clip')
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
        'training_samples': len(df)
    }}

    return p, p(df), log


isotonic_calibration_learner.__doc__ += learner_return_docstring("Isotonic Calibration")
