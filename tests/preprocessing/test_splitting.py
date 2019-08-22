from fklearn.preprocessing.splitting import space_time_split_dataset, time_split_dataset
import pandas as pd

df = pd.DataFrame(
    {
        'space': ['space1', 'space2', 'space1', 'space2'],
        'time': [pd.to_datetime("2016-10-01"), pd.to_datetime("2016-10-01"), pd.to_datetime("2016-11-01"),
                 pd.to_datetime("2016-11-01")]

    }
)


def test_time_split_dataset(test_df=df):
    in_time_train_set, out_time_test_set = time_split_dataset(dataset=test_df,
                                                              train_start_date="2016-10-01",
                                                              train_end_date="2016-11-01",
                                                              holdout_end_date="2016-12-01",
                                                              time_column="time")

    expected_train = pd.DataFrame(
        {
            'space': ['space1', 'space2'],
            'time': [pd.to_datetime("2016-10-01"), pd.to_datetime("2016-10-01")]
        }
    )

    expected_test = pd.DataFrame({
        'space': ['space1', 'space2'],
        'time': [pd.to_datetime("2016-11-01"), pd.to_datetime("2016-11-01")]
    })

    assert in_time_train_set.reset_index(drop=True).equals(expected_train)
    assert out_time_test_set.reset_index(drop=True).equals(expected_test)


def test_space_time_split_dataset(test_df=df):
    train_set, intime_outspace_hdout, outime_inspace_hdout, outime_outspace_hdout = \
        space_time_split_dataset(dataset=test_df,
                                 train_start_date="2016-10-01",
                                 train_end_date="2016-11-01",
                                 holdout_end_date="2016-12-01",
                                 split_seed=1,
                                 space_holdout_percentage=0.5,
                                 space_column="space",
                                 time_column="time")

    expected_train = pd.DataFrame({
        'space': ['space2'],
        'time': [pd.to_datetime("2016-10-01")]
    })

    expected_intime_outspace_holdout = pd.DataFrame({
        'space': ['space1'],
        'time': [pd.to_datetime("2016-10-01")]
    })

    expected_outime_outspace_holdout = pd.DataFrame({
        'space': ['space1'],
        'time': [pd.to_datetime("2016-11-01")]

    })

    expected_outime_inspace_holdout = pd.DataFrame({
        'space': ['space2'],
        'time': [pd.to_datetime("2016-11-01")]

    })

    assert train_set.reset_index(drop=True).equals(expected_train)
    assert intime_outspace_hdout.reset_index(drop=True).equals(expected_intime_outspace_holdout)
    assert outime_inspace_hdout.reset_index(drop=True).equals(expected_outime_inspace_holdout)
    assert outime_outspace_hdout.reset_index(drop=True).equals(expected_outime_outspace_holdout)
