from collections import Counter

import hypothesis.strategies as st
import pandas as pd
from hypothesis import given, settings, HealthCheck
from hypothesis.extra.pandas import columns, data_frames, range_indexes
from pandas.testing import assert_frame_equal

from fklearn.preprocessing.splitting import space_time_split_dataset, time_split_dataset, stratified_split_dataset

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

MAX_STRATIFIED_SPLIT_SIZE_DIFFERENCE = 1


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


@st.composite
def gen_stratified_test_data(draw):
    column_name_strategy = st.text(st.characters(whitelist_categories=["Lu", "Ll"]), min_size=3)
    all_column_names = draw(st.lists(column_name_strategy, min_size=3, max_size=6, unique=True))
    target_column_name = all_column_names[-1]

    column_strategies = columns(all_column_names, dtype=int)
    data_set = draw(data_frames(column_strategies, index=range_indexes(min_size=50, max_size=100)))

    num_classes = draw(st.integers(min_value=2, max_value=5))
    data_set[target_column_name] = [i % num_classes for i in range(len(data_set))]

    return data_set, target_column_name, num_classes


def assert_sample_size_per_class(data, target_column_name, expected_samples_per_class):
    count_per_class = Counter(data[target_column_name]).values()

    for count in count_per_class:
        assert abs(count - expected_samples_per_class) <= MAX_STRATIFIED_SPLIT_SIZE_DIFFERENCE


@given(sample=gen_stratified_test_data(),
       random_state=st.integers(min_value=0, max_value=100),
       test_size=st.floats(min_value=0.2, max_value=0.8))
@settings(suppress_health_check={HealthCheck.too_slow})
def test_stratified_split_dataset(sample, random_state, test_size):
    expected_data, target_column_name, num_classes = sample

    train_data, test_data = stratified_split_dataset(expected_data, target_column_name, test_size=test_size,
                                                     random_state=random_state)

    total_samples = len(expected_data)
    expected_test_size = int(total_samples * test_size)
    expected_train_size = total_samples - expected_test_size

    expected_test_samples_per_class = expected_test_size / num_classes
    expected_train_samples_per_class = expected_train_size / num_classes

    data = pd.concat([train_data, test_data])

    assert abs(len(train_data) - expected_train_size) <= MAX_STRATIFIED_SPLIT_SIZE_DIFFERENCE
    assert abs(len(test_data) - expected_test_size) <= MAX_STRATIFIED_SPLIT_SIZE_DIFFERENCE

    assert_frame_equal(data, expected_data, check_like=True)
    assert_sample_size_per_class(train_data, target_column_name, expected_train_samples_per_class)
    assert_sample_size_per_class(test_data, target_column_name, expected_test_samples_per_class)
