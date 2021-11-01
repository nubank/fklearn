from collections import Counter

import pandas as pd
import numpy as np
from fklearn.causal.cate_learning.double_machine_learning import non_parametric_double_ml_learner


def test_non_parametric_double_ml_learner():

    df_train = pd.DataFrame(dict(
        # TE = 1       TE = 2      TE = 3
        x=[1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3, 3] * 5,
        t=[1 + 1, 1 + 1, 2 + 1, 3 + 1, 1 + 2, 1 + 2, 2 + 2, 3 + 2, 1 + 3, 1 + 3, 2 + 3, 3 + 3] * 5,
        y=[1 - 1, 1 - 1, 2 - 1, 3 - 1, 1 - 2, 1 - 2, 3 - 2, 5 - 2, 1 - 3, 1 - 3, 4 - 3, 7 - 3] * 5,
    ))

    df_test = pd.DataFrame(dict(x=[1, 2, 3]))

    np.random.seed(123)
    predict_fn, pred_train, log = non_parametric_double_ml_learner(df=df_train,
                                                                   feature_columns=["x"],
                                                                   treatment_column="t",
                                                                   outcome_column="y",
                                                                   cv_splits=5,
                                                                   prediction_column="test_prediction")

    pred_test = predict_fn(df_test).round(3)
    expected_test_pred = pd.DataFrame(dict(
        x=[1, 2, 3],
        test_prediction=[1.0, 2.0, 3.0]
    ))

    pd.testing.assert_frame_equal(expected_test_pred, pred_test)

    expected_col_train = df_train.columns.tolist() + ["test_prediction"]
    expected_col_test = df_test.columns.tolist() + ["test_prediction"]

    assert Counter(expected_col_train) == Counter(pred_train.columns.tolist())
    assert Counter(expected_col_test) == Counter(pred_test.columns.tolist())
