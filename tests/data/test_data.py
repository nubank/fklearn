import numpy as np

from fklearn.data.datasets import make_tutorial_data, make_confounded_data


def test_make_tutorial_data():
    df = make_tutorial_data(1000)
    assert df.shape[0] == 1000


def test_make_confounded_data():
    f_rnd, df_obs, df_ctf = make_confounded_data(1000)
    traeat_corr = f_rnd.corr().loc[["age", "severity", "sex"], "medication"]

    assert f_rnd.shape[1] == df_obs.shape[1] == df_ctf.shape[1]
    assert np.all(np.abs(traeat_corr) < .05), "assignment is not random!"
