import pandas as pd
import numpy as np

from fklearn.training.calibration import (isotonic_calibration_learner,
                                          find_thresholds_with_same_risk)


def test_isotonic_calibration_learner():
    df_train = pd.DataFrame({
        'id': ["id1", "id2", "id3", "id4"],
        "pred": [0.25, 0.64, 0.12, 0.9],
        'y': [0, 1, 0, 1]
    })

    df_test = pd.DataFrame({
        'id': ["id4", "id4", "id5", "id6", "id7"],
        "pred": [0.55, 0.12, 0.13, 0.9, 0.95],
        'y': [1, 0, 0, 1, 1]
    })

    learner = isotonic_calibration_learner(prediction_column="pred", target_column="y")

    predict_fn, pred_train, log = learner(df_train)

    pred_test = predict_fn(df_test)

    assert "calibrated_prediction" in pred_train.columns.values
    assert "calibrated_prediction" in pred_test.columns.values
    assert pred_test.calibrated_prediction.max() <= 1
    assert pred_test.calibrated_prediction.min() >= 0
    assert pred_test.calibrated_prediction.isnull().max() == 0


def test_find_thresholds_with_same_risk():
    df_with_ecdf = pd.DataFrame([["group_1", 0.0, 416],
                                 ["group_2", 0.0, 328],
                                 ["group_2", 0.0, 670],
                                 ["group_2", 0.0, 105],
                                 ["group_2", 0.0, 427],
                                 ["group_1", 1.0, 672],
                                 ["group_1", 0.0, 635],
                                 [None, 0.0, 158],
                                 ["group_2", 0.0, 152],
                                 ["group_1", 0.0, 305]],
                                columns=["sensitive_factor", "target",
                                         "prediction_ecdf"])

    band_size = 10
    df_with_ecdf["unfair_band"] = np.floor(df_with_ecdf["prediction_ecdf"] / band_size) * band_size

    df_expected = df_with_ecdf.copy()
    df_expected["fair"] = pd.Series([10, 20, 40, 0, 30, 30, 20, None, 10, 0])
    fair_thresholds = {'group_2': [-1, 105.0, 152.0, 328.0, 427.0, 670.0],
                       'group_1': [-1, 305.0, 416.0, 635.0, 672.0]}

    learner = find_thresholds_with_same_risk(sensitive_factor="sensitive_factor", unfair_band_column="unfair_band")

    predict_fn, pred_df, log = learner(df_with_ecdf)

    assert "fair" in pred_df.columns.values
    assert fair_thresholds == log["find_thresholds_with_same_risk"]["fair_thresholds"]
    pd.util.testing.assert_frame_equal(df_expected, pred_df)
