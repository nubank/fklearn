from datetime import timedelta

import pandas as pd
import pytest
from fklearn.validation.splitters import \
    k_fold_splitter, out_of_time_and_space_splitter, spatial_learning_curve_splitter, time_learning_curve_splitter, \
    reverse_time_learning_curve_splitter, stability_curve_time_splitter, stability_curve_time_in_space_splitter, \
    stability_curve_time_space_splitter, forward_stability_curve_time_splitter, time_and_space_learning_curve_splitter

sample_data = pd.DataFrame({'space': ['a', 'a', 'b', 'b', 'a', 'c', 'c'],
                            'time': pd.to_datetime(
                                ['2015-01-01', '2015-02-02', '2016-04-25',
                                 '2015-03-03', '2015-07-07', '2015-08-08', '2015-09-09'])})

# fuck up index because we should't rely on them
sample_data.index = [1] * sample_data.shape[0]


def test_k_fold_splitter():
    # testing without stratification
    result, logs = k_fold_splitter(sample_data, 2, random_state=42)

    assert len(result) == 2

    train_1_idx = result[0][0]
    test_1_idx = result[0][1][0]
    train_2_idx = result[1][0]
    test_2_idx = result[1][1][0]

    assert set(train_1_idx) == set(test_2_idx)
    assert set(test_1_idx) == set(train_2_idx)

    # testing with stratification
    result, logs = k_fold_splitter(sample_data, 2, random_state=42, stratify_column='space')

    assert len(result) == 2

    train_1_idx = result[0][0]
    test_1_idx = result[0][1][0]
    train_2_idx = result[1][0]
    test_2_idx = result[1][1][0]

    assert set(train_1_idx) == set(test_2_idx)
    assert set(test_1_idx) == set(train_2_idx)

    train_1_strat = sample_data.iloc[train_1_idx]['space']
    test_1_strat = sample_data.iloc[test_1_idx]['space']
    train_2_strat = sample_data.iloc[train_2_idx]['space']
    test_2_strat = sample_data.iloc[test_2_idx]['space']

    assert train_1_strat.nunique() == test_2_strat.nunique()
    assert train_2_strat.nunique() == test_1_strat.nunique()


def test_out_of_time_and_space_splitter():
    result, logs = out_of_time_and_space_splitter(sample_data, 2, '2015-05-05', time_column='time',
                                                  space_column='space', holdout_gap=timedelta(days=31))

    assert len(result) == 2
    train_1 = sample_data.iloc[result[0][0]]
    test_1 = sample_data.iloc[result[0][1][0]]
    train_2 = sample_data.iloc[result[1][0]]
    test_2 = sample_data.iloc[result[1][1][0]]

    # there must be no overlap in space between folds
    assert len(train_1[train_1.space.isin(train_2.space)]) == 0
    assert len(train_2[train_2.space.isin(train_1.space)]) == 0

    # the training sets must have no dates after 2015-05-05
    assert len(train_1[train_1.time > '2015-05-05']) == 0
    assert len(train_2[train_2.time > '2015-05-05']) == 0

    # the test sets must have no dates before 2015-05-05 + 31 days
    assert len(test_1[test_1.time <= '2015-06-05']) == 0
    assert len(test_2[test_2.time <= '2015-06-05']) == 0

    # all rows with time before '2015-05-05' must be in a training set
    assert len(train_1) + len(train_2) == len(sample_data[sample_data.time <= '2015-05-05'])


def test_time_and_space_learning_curve_splitter():
    random_state = 21
    result, logs = time_and_space_learning_curve_splitter(sample_data, '2015-05-05', space_column='space',
                                                          time_column='time', holdout_gap=timedelta(days=31),
                                                          random_state=random_state, min_samples=0)

    assert len(result) == 4
    test_1 = sample_data.iloc[result[0][1][0]]
    train_4 = sample_data.iloc[result[3][0]]
    test_4 = sample_data.iloc[result[3][1][0]]

    # the test sets must all be the same, and have no dates before 2015-05-05 + 31 days
    assert set(result[0][1][0]) == set(result[1][1][0])
    assert set(result[1][1][0]) == set(result[2][1][0])
    assert set(result[2][1][0]) == set(result[3][1][0])
    assert len(test_1[test_1.time <= '2015-06-05']) == 0

    # every training set must contain the one before it, and have no dates after 2015-05-05
    assert set(result[0][0]).issubset(set(result[1][0]))
    assert set(result[1][0]).issubset(set(result[2][0]))
    assert set(result[2][0]).issubset(set(result[3][0]))
    assert len(train_4[train_4.time > '2015-05-05']) == 0

    # there must be no space overlap between training and test sets
    assert len(test_4[test_4.space.isin(train_4.space)]) == 0
    assert len(train_4[train_4.space.isin(test_4.space)]) == 0


def test_spatial_learning_curve_splitter():
    result, logs = spatial_learning_curve_splitter(
        sample_data,
        train_percentages=[0.5, 1.0],
        space_column="space",
        time_column="time",
        training_limit="2015-09-09",
        holdout_gap=timedelta(days=180),
        random_state=0
    )

    assert len(result) == 2

    train_1 = sample_data.iloc[result[0][0]]
    train_2 = sample_data.iloc[result[1][0]]
    test_1 = sample_data.iloc[result[0][1][0]]
    test_2 = sample_data.iloc[result[1][1][0]]

    assert test_1.equals(test_2)
    assert len(set(train_1["space"]).intersection(train_2["space"])) > 0
    assert train_1["time"].max() < test_1["time"].min()
    assert train_2["time"].max() < test_2["time"].min()
    assert test_1["time"].min() - train_1["time"].max() >= timedelta(days=180)
    assert test_2["time"].min() - train_2["time"].max() >= timedelta(days=180)
    assert len(train_2) > len(train_1)

    # should raise an exception when percentage is off bounds
    with pytest.raises(ValueError):
        result, logs = spatial_learning_curve_splitter(
            sample_data,
            train_percentages=[0.5, 1.1],
            space_column="space",
            time_column="time",
            training_limit="2015-09-09",
            holdout_gap=timedelta(days=180),
            random_state=0
        )
    with pytest.raises(ValueError):
        result, logs = spatial_learning_curve_splitter(
            sample_data,
            train_percentages=[-0.1, 1.0],
            space_column="space",
            time_column="time",
            training_limit="2015-09-09",
            holdout_gap=timedelta(days=180),
            random_state=0
        )


def test_time_learning_curve_splitter():
    result, logs = time_learning_curve_splitter(sample_data, '2015-05-05', time_column='time',
                                                holdout_gap=timedelta(days=31), min_samples=0)

    assert len(result) == 4
    train_1 = sample_data.iloc[result[0][0]]
    test_1 = sample_data.iloc[result[0][1][0]]
    train_2 = sample_data.iloc[result[1][0]]
    train_3 = sample_data.iloc[result[2][0]]
    train_4 = sample_data.iloc[result[3][0]]
    test_4 = sample_data.iloc[result[3][1][0]]

    # the test sets must all be the same, and have no dates before 2015-05-05 + 31 days
    assert set(result[0][1][0]) == set(result[1][1][0])
    assert set(result[1][1][0]) == set(result[2][1][0])
    assert set(result[2][1][0]) == set(result[3][1][0])
    assert len(test_1[test_1.time <= '2015-06-05']) == 0

    # every training set must contain the one before it, and have no dates after 2015-05-05
    assert set(result[0][0]).issubset(set(result[1][0]))
    assert set(result[1][0]).issubset(set(result[2][0]))
    assert set(result[2][0]).issubset(set(result[3][0]))
    assert len(train_4[train_4.time > '2015-05-05']) == 0

    # the test set will contain all space
    assert set(test_4.space) == set(sample_data.space)

    # every training set must have dates after the previous one
    assert train_2.time.max() > train_1.time.max()
    assert train_3.time.max() > train_2.time.max()


def test_reverse_time_learning_curve_splitter():
    result, logs = reverse_time_learning_curve_splitter(sample_data, time_column='time',
                                                        training_time_limit='2015-05-05',
                                                        holdout_gap=timedelta(days=31), min_samples=0)

    assert len(result) == 3
    train_1 = sample_data.iloc[result[0][0]]
    test_1 = sample_data.iloc[result[0][1][0]]
    train_2 = sample_data.iloc[result[1][0]]
    train_3 = sample_data.iloc[result[2][0]]

    # the test sets must all be the same, and have no dates before 2015-05-05 + 31 days
    assert set(result[0][1][0]) == set(result[1][1][0])
    assert set(result[1][1][0]) == set(result[2][1][0])
    assert len(test_1[test_1.time <= '2015-06-05']) == 0

    # every training set must contain the one before it, and have no dates after 2015-05-05
    assert set(result[0][0]).issubset(set(result[1][0]))
    assert set(result[1][0]).issubset(set(result[2][0]))
    assert len(train_3[train_3.time > '2015-05-05']) == 0

    # every training set must have dates before the previous one
    assert train_2.time.min() < train_1.time.min()
    assert train_3.time.min() < train_2.time.min()

    # the first training set must contain only 2015-03-03
    assert len(train_1) == 1
    # the last training set must contain 3 months
    assert len(train_3) == 3


def test_stability_curve_time_splitter():
    result, logs = stability_curve_time_splitter(sample_data, '2015-05-05', time_column='time', min_samples=0)

    assert len(result) == 1
    assert len(result[0]) == 2

    train_1 = sample_data.iloc[result[0][0]]
    test_1 = sample_data.iloc[result[0][1][0]]
    test_2 = sample_data.iloc[result[0][1][1]]
    test_3 = sample_data.iloc[result[0][1][2]]

    # the training set must contain no dates after 2015-05-05
    assert len(train_1[train_1.time > '2015-05-05']) == 0

    # the test sets must contain no dates before 2015-05-05
    assert len(test_1[test_1.time <= '2015-05-05']) == 0
    assert len(test_2[test_2.time <= '2015-05-05']) == 0
    assert len(test_3[test_3.time <= '2015-05-05']) == 0

    # the test sets must not overlap, and have increasing time
    assert test_1.time.max() < test_2.time.min()
    assert test_2.time.max() < test_3.time.min()


def test_stability_curve_time_in_space_splitter():
    result, logs = stability_curve_time_in_space_splitter(sample_data, '2015-05-05', random_state=25,
                                                          time_column='time',
                                                          min_samples=0, space_column='space', space_hold_percentage=.5)

    train_1 = sample_data.iloc[result[0][0]]
    test_1 = sample_data.iloc[result[0][1][0]]

    # the training set must contain no dates after 2015-05-05
    assert len(train_1[train_1.time > '2015-05-05']) == 0

    # the test sets must contain no dates before 2015-05-05
    assert len(test_1[test_1.time <= '2015-05-05']) == 0

    # all space in test set must be in the training set
    assert train_1.space.isin(test_1.space.values).any()


def test_stability_curve_time_space_splitter():
    result, logs = stability_curve_time_space_splitter(sample_data, '2015-05-05', random_state=25, time_column='time',
                                                       min_samples=0, space_column='space', space_hold_percentage=0.5)

    assert len(result) == 1
    assert len(result[0]) == 2
    train_1 = sample_data.iloc[result[0][0]]
    test_1 = sample_data.iloc[result[0][1][0]]
    test_2 = sample_data.iloc[result[0][1][1]]
    test_3 = sample_data.iloc[result[0][1][2]]

    # the training set must contain no dates after 2015-05-05
    assert len(train_1[train_1.time > '2015-05-05']) == 0

    # the test sets must contain no dates before 2015-05-05
    assert len(test_1[test_1.time <= '2015-05-05']) == 0
    assert len(test_2[test_2.time <= '2015-05-05']) == 0
    assert len(test_3[test_3.time <= '2015-05-05']) == 0

    # the test sets must not overlap, and have increasing time
    assert test_1.time.max() < test_2.time.min()
    assert test_2.time.max() < test_3.time.min()

    # there must be no space overlap between training and test sets
    assert len(test_1[test_1.space.isin(train_1.space)]) == 0
    assert len(train_1[train_1.space.isin(test_1.space)]) == 0

    assert len(test_2[test_2.space.isin(train_1.space)]) == 0
    assert len(train_1[train_1.space.isin(test_2.space)]) == 0

    assert len(test_3[test_3.space.isin(train_1.space)]) == 0
    assert len(train_1[train_1.space.isin(test_3.space)]) == 0


def test_forward_stability_curve_time_splitter():
    result, logs = forward_stability_curve_time_splitter(
        sample_data,
        training_time_start="2015-02-01",
        training_time_end="2015-05-05",
        holdout_gap=timedelta(days=0),
        holdout_size=timedelta(days=60),
        step=timedelta(days=60),
        time_column='time'
    )

    assert len(result) == 5
    assert len(result[0]) == 2

    # Training lengths
    assert len(result[0][0]) == 2
    assert len(result[1][0]) == 0
    assert len(result[2][0]) == 2
    assert len(result[3][0]) == 2

    # Test lengths
    assert len(result[0][1][0]) == 0
    assert len(result[1][1][0]) == 2
    assert len(result[2][1][0]) == 1
    assert len(result[3][1][0]) == 0

    train_1 = sample_data.iloc[result[0][0]]
    train_2 = sample_data.iloc[result[2][0]]
    train_3 = sample_data.iloc[result[3][0]]

    test_1 = sample_data.iloc[result[1][1][0]]
    test_2 = sample_data.iloc[result[2][1][0]]

    assert train_2.time.max() < test_2.time.min()

    assert train_1.time.min() < train_2.time.min()
    assert train_2.time.min() < train_3.time.min()

    assert test_1.time.min() < test_2.time.min()
