from typing import Optional, Tuple

import numpy as np
from numpy.random import RandomState
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit
from toolz import curry

from fklearn.types import DateType


@curry
def time_split_dataset(dataset: pd.DataFrame,
                       train_start_date: DateType,
                       train_end_date: DateType,
                       holdout_end_date: DateType,
                       time_column: str,
                       holdout_start_date: DateType = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
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
        This will also be used as the start date of the holdout period if no `holdout_start_date` is given.
        It should be in the same format as the Date Column in `dataset`.

    holdout_end_date : str
        A date string representing a the ending time of the holdout data.
        It should be in the same format as the Date Column in `dataset`.

    time_column : str
        The name of the Date column of `dataset`.

    holdout_start_date: str
        A date string representing the starting time of the holdout data.
        If `None` is given it will be equal to `train_end_date`.
        It should be in the same format as the Date Column in `dataset`.

    Returns
    ----------
    train_set : pandas.DataFrame
        The in ID sample and in time training set.

    test_set : pandas.DataFrame
        The out of ID sample and in time hold out set.
    """

    holdout_start_date = holdout_start_date if holdout_start_date else train_end_date

    train_set = dataset[
        (dataset[time_column] >= train_start_date) & (dataset[time_column] < train_end_date)]

    test_set = dataset[
        (dataset[time_column] >= holdout_start_date) & (dataset[time_column] < holdout_end_date)]

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
                             holdout_space: np.ndarray = None,
                             holdout_start_date: DateType = None) -> Tuple[pd.DataFrame, ...]:
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
        This will also be used as the start date of the holdout period if no `holdout_start_date` is given.
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

    holdout_start_date: str
        A date string representing the starting time of the holdout data.
        If `None` is given it will be equal to `train_end_date`.
        It should be in the same format as the Date Column in `dataset`.

    Returns
    ----------
    train_set : pandas.DataFrame
        The in ID sample and in time training set.

    intime_outspace_hdout : pandas.DataFrame
        The out of ID sample and in time hold out set.

    outime_inspace_hdout : pandas.DataFrame
        The in ID sample and out of time hold out set.

    outime_outspace_hdout : pandas.DataFrame
        The out of ID sample and out of time hold out set.
    """
    holdout_start_date = holdout_start_date if holdout_start_date else train_end_date

    in_time_mask = (dataset[time_column] >= train_start_date) & (dataset[time_column] < train_end_date)
    out_time_mask = (dataset[time_column] >= holdout_start_date) & (dataset[time_column] < holdout_end_date)

    all_space_in_time = dataset[in_time_mask][space_column].unique()

    if holdout_space is None:
        # for repeatability
        state = RandomState(split_seed)
        train_period_space = np.sort(all_space_in_time)

        # randomly sample accounts from the train period to hold out
        partial_holdout_space = state.choice(train_period_space,
                                             int(space_holdout_percentage * len(train_period_space)),
                                             replace=False)

        in_space = pd.Index(all_space_in_time).difference(pd.Index(partial_holdout_space)).values

    else:
        in_space = pd.Index(all_space_in_time).difference(pd.Index(holdout_space)).values

    in_space_mask = dataset[space_column].isin(in_space)

    train_set = dataset[in_space_mask & in_time_mask]
    intime_outspace_hdout = dataset[~in_space_mask & in_time_mask]
    outtime_outspace_hdout = dataset[~in_space_mask & out_time_mask]
    outtime_inspace_hdout = dataset[in_space_mask & out_time_mask]

    return train_set, intime_outspace_hdout, outtime_inspace_hdout, outtime_outspace_hdout


@curry
def stratified_split_dataset(dataset: pd.DataFrame, target_column: str, test_size: float,
                             random_state: Optional[int] = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
        Splits data into a training and testing datasets such that
        they maintain the same class ratio of the original dataset.

        Parameters
        ----------
        dataset : pandas.DataFrame
            A Pandas' DataFrame with the target column.
            The model will be trained to predict the target column
            from the features.

        target_column : str
            The name of the target column of `dataset`.

        test_size : float
            Represent the proportion of the dataset to include in the test split.
            should be between 0.0 and 1.0.

        random_state : int or None, optional (default=None)
            If int, random_state is the seed used by the random number generator;
            If None, the random number generator is the RandomState instance used
            by `np.random`.

        Returns
        ----------
        train_set : pandas.DataFrame
            The train dataset sampled from the full dataset.

        test_set : pandas.DataFrame
            The test dataset sampled from the full dataset.
        """
    train_placeholder = np.zeros(len(dataset))
    target = dataset[target_column]

    splitter = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)

    train_indices, test_indices = next(splitter.split(train_placeholder, target))
    train_set = dataset.iloc[train_indices]
    test_set = dataset.iloc[test_indices]

    return train_set, test_set
