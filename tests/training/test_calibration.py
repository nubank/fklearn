import pandas as pd

from fklearn.training.calibration import isotonic_calibration_learner


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
