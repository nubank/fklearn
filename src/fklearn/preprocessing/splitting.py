from typing import Tuple

import numpy as np
from numpy.random import RandomState
import pandas as pd
from toolz import curry

from fklearn.types import DateType


@curry
def time_split_dataset(dataset: pd.DataFrame,
                       train_start_date: DateType,
                       train_end_date: DateType,
                       holdout_end_date: DateType,
                       time_column: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Splits temporal data into a training and testing datasets such that
    all training data comes before the testings one.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A Pandas' DataFrame with an Identifier Column and a Date Column.
        The model will be trained to predict the target column
        from the features.

    train_start_date : str
        A date string representing a the starting time of the training data.
        It should be in the same format as the Date Column in `dataset`.

    train_end_date : str
        A date string representing a the ending time of the training data.
        This will also be used as the start date of the holdout period.
        It should be in the same format as the Date Column in `dataset`.

    holdout_end_date : str
        A date string representing a the ending time of the holdout data.
        It should be in the same format as the Date Column in `dataset`.

    time_column : str
        The name of the Date column of `dataset`.


    Returns
    ----------
    train_set : pandas.DataFrame
        The in ID sample and in time training set.

    test_set : pandas.DataFrame
        The out of ID sample and in time hold out set.
    """

    train_set = dataset[
        (dataset[time_column] >= train_start_date) & (dataset[time_column] < train_end_date)]

    test_set = dataset[
        (dataset[time_column] >= train_end_date) & (dataset[time_column] < holdout_end_date)]

    return train_set, test_set


@curry
def space_time_split_dataset(dataset: pd.DataFrame,
                             train_start_date: DateType,
                             train_end_date: DateType,
                             holdout_end_date: DateType,
                             split_seed: int,
                             space_holdout_percentage: float,
                             space_column: str,
                             time_column: str,
                             holdout_space: np.ndarray = None) -> Tuple[pd.DataFrame, ...]:
    """
    Splits panel data using both ID and Time columns, resulting in four datasets

    1. A training set;
    2. An in training time, but out sample id hold out dataset;
    3. An out of training time, but in sample id hold out dataset;
    4. An out of training time and out of sample id hold out dataset.

    Parameters
    ----------
    dataset : pandas.DataFrame
        A Pandas' DataFrame with an Identifier Column and a Date Column.
        The model will be trained to predict the target column
        from the features.

    train_start_date : str
        A date string representing a the starting time of the training data.
        It should be in the same format as the Date Column in `dataset`.

    train_end_date : str
        A date string representing a the ending time of the training data.
        This will also be used as the start date of the holdout period.
        It should be in the same format as the Date Column in `dataset`.

    holdout_end_date : str
        A date string representing a the ending time of the holdout data.
        It should be in the same format as the Date Column in `dataset`.

    split_seed : int
        A seed used by the random number generator.

    space_holdout_percentage : float
        The out of id holdout size as a proportion of the in id training
        size.

    space_column : str
        The name of the Identifier column of `dataset`.

    time_column : str
        The name of the Date column of `dataset`.

    holdout_space : np.array
        An array containing the hold out IDs. If not specified,
        A random subset of IDs will be selected for holdout.

    Returns
    ----------
    train_set : pandas.DataFrame
        The in ID sample and in time training set.

    intime_outspace_hdout : pandas.DataFrame
        The out of ID sample and in time hold out set.

    outime_inspace_hdout : pandas.DataFrame
        The out of ID sample and in time hold out set.

    holdout_space : pandas.DataFrame
        The out of ID sample and in time hold out set.
    """
    train_period = dataset[
        (dataset[time_column] >= train_start_date) & (dataset[time_column] < train_end_date)]
    outime_inspace_hdout = dataset[
        (dataset[time_column] >= train_end_date) & (dataset[time_column] < holdout_end_date)]

    if holdout_space is None:
        train_period_space = train_period[space_column].unique()

        # for repeatability
        state = RandomState(split_seed)

        train_period_space = np.sort(train_period_space)

        # randomly sample accounts from the train period to hold out
        holdout_space = state.choice(train_period_space,
                                     int(space_holdout_percentage * len(train_period_space)),
                                     replace=False)

    train_set = train_period[~train_period[space_column].isin(holdout_space)]
    intime_outspace_hdout = train_period[train_period[space_column].isin(holdout_space)]
    outime_outspace_hdout = outime_inspace_hdout[outime_inspace_hdout[space_column].isin(holdout_space)]

    return train_set, intime_outspace_hdout, outime_inspace_hdout, outime_outspace_hdout
