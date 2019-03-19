from typing import Any, Dict, List

import pandas as pd
from sklearn.ensemble import IsolationForest
import sklearn
from toolz import curry, merge

from fklearn.common_docstrings import learner_pred_fn_docstring, learner_return_docstring
from fklearn.types import LearnerReturnType
from fklearn.training.utils import log_learner_time


@curry
@log_learner_time(learner_name='isolation_forest_learner')
def isolation_forest_learner(df: pd.DataFrame,
                             features: List[str],
                             params: Dict[str, Any] = None,
                             prediction_column: str = "prediction") -> LearnerReturnType:
    """Fits an anomaly detection algorithm (Isolation Forest) to the dataset

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame with features and target columns.
        The model will be trained to predict the target column
        from the features.

    features : list of str
        A list os column names that are used as features for the model. All this names
        should be in `df`.

    params : dict
        The IsolationForest parameters in the format {"par_name": param}. See:
        http://scikit-learn.org/stable/modules/generated/sklearn.ensemble.IsolationForest.html

    prediction_column : str
        The name of the column with the predictions from the model.
    """

    default_params = {"n_jobs": -1, "random_state": 1729}
    params = default_params if not params else merge(default_params, params)

    model = IsolationForest()
    model.set_params(**params)
    model.fit(df[features].values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        output_col = {prediction_column: model.decision_function(
            new_df[features])}

        return new_df.assign(**output_col)

    p.__doc__ = learner_pred_fn_docstring("isolation_forest_learner")

    log = {'isolation_forest_learner': {
        'features': features,
        'parameters': params,
        'prediction_column': prediction_column,
        'package': "sklearn",
        'package_version': sklearn.__version__,
        'training_samples': len(df)}}

    return p, p(df), log


isolation_forest_learner.__doc__ += learner_return_docstring("Isolation Forest")
