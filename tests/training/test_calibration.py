import pandas as pd

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
    df_with_ecdf = pd.DataFrame([["group_1", 0.0, 416, 3],
                                 ["group_2", 0.0, 328, 2],
                                 ["group_2", 0.0, 670, 4],
                                 ["group_2", 0.0, 105, 1],
                                 ["group_2", 0.0, 427, 3],
                                 ["group_1", 1.0, 672, 4],
                                 ["group_1", 0.0, 635, 4],
                                 [None, 0.0, 158, 1],
                                 ["group_2", 0.0, 152, 4],
                                 ["group_1", 0.0, 305, 2]],
                                columns=["sensitive_factor", "target",
                                         "prediction_ecdf", "unfair_band"])

    df_expected = df_with_ecdf.copy()
    df_expected["fair"] = pd.Series([2, 3, 4, 1, 4, 4, 3, None, 2, 1])
    fair_thresholds = {'group_2': [-1, 105.0, 152.0, 328.0, 670.0],
                       'group_1': [-1, 305.0, 416.0, 635.0, 672.0]}

    learner = find_thresholds_with_same_risk(sensitive_factor="sensitive_factor", unfair_band_column="unfair_band",
                                             model_prediction_output="prediction_ecdf")

    predict_fn, pred_df, log = learner(df_with_ecdf)
    df_with_ecdf["fair"] = pred_df

    assert fair_thresholds == log["find_thresholds_with_same_risk"]["fair_thresholds"]
    pd.testing.assert_frame_equal(df_expected, df_with_ecdf)
