import pandas as pd

from fklearn.preprocessing.splitting import space_time_split_dataset, time_split_dataset

df = pd.DataFrame(
    {
        'space': ['space1', 'space2', 'space1', 'space2', 'space1', 'space2'],
        'time': [pd.to_datetime("2016-10-01"), pd.to_datetime("2016-10-01"), pd.to_datetime("2016-11-01"),
                 pd.to_datetime("2016-11-01"), pd.to_datetime("2016-12-01"), pd.to_datetime("2016-12-01")]
    }
)

df_with_new_id = pd.DataFrame(
    {
        'space': ['space1', 'space2', 'space1', 'space2', 'space1', 'space2', 'space3'],
        'time': [pd.to_datetime("2016-10-01"), pd.to_datetime("2016-10-01"), pd.to_datetime("2016-11-01"),
                 pd.to_datetime("2016-11-01"), pd.to_datetime("2016-12-01"), pd.to_datetime("2016-12-01"),
                 pd.to_datetime("2016-11-01")]
    }
)

df_only_one_point_per_id = pd.DataFrame(
    {
        'space': ['space1', 'space2', 'space3', 'space4'],
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

    # Testing optional argument `holdout_start_date`
    in_time_train_set, out_time_test_set = time_split_dataset(dataset=test_df,
                                                              train_start_date="2016-10-01",
                                                              train_end_date="2016-11-01",
                                                              holdout_start_date="2016-11-03",
                                                              holdout_end_date="2017-01-01",
                                                              time_column="time")

    expected_train = pd.DataFrame({
        'space': ['space1', 'space2'],
        'time': [pd.to_datetime("2016-10-01"), pd.to_datetime("2016-10-01")]
    })

    expected_test = pd.DataFrame({
        'space': ['space1', 'space2'],
        'time': [pd.to_datetime("2016-12-01"), pd.to_datetime("2016-12-01")]
    })

    assert in_time_train_set.reset_index(drop=True).equals(expected_train)
    assert out_time_test_set.reset_index(drop=True).equals(expected_test)


def test_space_time_split_dataset(test_df=df,
                                  test_df_with_new_id=df_with_new_id,
                                  test_df_only_one_point_per_id=df_only_one_point_per_id):

    train_set, intime_outspace_hdout, outtime_inspace_hdout, outtime_outspace_hdout = \
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

    expected_outtime_outspace_holdout = pd.DataFrame({
        'space': ['space1'],
        'time': [pd.to_datetime("2016-11-01")]

    })

    expected_outtime_inspace_holdout = pd.DataFrame({
        'space': ['space2'],
        'time': [pd.to_datetime("2016-11-01")]

    })

    assert train_set.reset_index(drop=True).equals(expected_train)
    assert intime_outspace_hdout.reset_index(drop=True).equals(expected_intime_outspace_holdout)
    assert outtime_inspace_hdout.reset_index(drop=True).equals(expected_outtime_inspace_holdout)
    assert outtime_outspace_hdout.reset_index(drop=True).equals(expected_outtime_outspace_holdout)

    # Testing optional argument `holdout_start_date`
    train_set, intime_outspace_hdout, outtime_inspace_hdout, outtime_outspace_hdout = \
        space_time_split_dataset(dataset=test_df,
                                 train_start_date="2016-10-01",
                                 train_end_date="2016-11-01",
                                 holdout_start_date="2016-12-01",
                                 holdout_end_date="2017-01-01",
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

    expected_outtime_outspace_holdout = pd.DataFrame({
        'space': ['space1'],
        'time': [pd.to_datetime("2016-12-01")]

    })

    expected_outtime_inspace_holdout = pd.DataFrame({
        'space': ['space2'],
        'time': [pd.to_datetime("2016-12-01")]

    })

    assert train_set.reset_index(drop=True).equals(expected_train)
    assert intime_outspace_hdout.reset_index(drop=True).equals(expected_intime_outspace_holdout)
    assert outtime_inspace_hdout.reset_index(drop=True).equals(expected_outtime_inspace_holdout)
    assert outtime_outspace_hdout.reset_index(drop=True).equals(expected_outtime_outspace_holdout)

    # Testing new space id appearing in the holdout period
    train_set, intime_outspace_hdout, outtime_inspace_hdout, outtime_outspace_hdout = \
        space_time_split_dataset(dataset=test_df_with_new_id,
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

    expected_outtime_outspace_holdout = pd.DataFrame({
        'space': ['space1', 'space3'],
        'time': [pd.to_datetime("2016-11-01"), pd.to_datetime("2016-11-01")]

    })

    expected_outtime_inspace_holdout = pd.DataFrame({
        'space': ['space2'],
        'time': [pd.to_datetime("2016-11-01")]

    })

    assert train_set.reset_index(drop=True).equals(expected_train)
    assert intime_outspace_hdout.reset_index(drop=True).equals(expected_intime_outspace_holdout)
    assert outtime_inspace_hdout.reset_index(drop=True).equals(expected_outtime_inspace_holdout)
    assert outtime_outspace_hdout.reset_index(drop=True).equals(expected_outtime_outspace_holdout)

    # Testing only one point per space id
    train_set, intime_outspace_hdout, outtime_inspace_hdout, outtime_outspace_hdout = \
        space_time_split_dataset(dataset=test_df_only_one_point_per_id,
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

    expected_outtime_outspace_holdout = pd.DataFrame({
        'space': ['space3', 'space4'],
        'time': [pd.to_datetime("2016-11-01"), pd.to_datetime("2016-11-01")]

    })

    assert train_set.reset_index(drop=True).equals(expected_train)
    assert intime_outspace_hdout.reset_index(drop=True).equals(expected_intime_outspace_holdout)
    assert outtime_inspace_hdout.empty
    assert outtime_outspace_hdout.reset_index(drop=True).equals(expected_outtime_outspace_holdout)
