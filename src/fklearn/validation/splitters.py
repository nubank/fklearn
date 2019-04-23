import operator
from datetime import datetime, timedelta
from itertools import chain, repeat, starmap
from typing import Callable, Iterable, List, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold, KFold, StratifiedKFold
from sklearn.utils import check_random_state
from toolz.curried import curry, partial, pipe, assoc, accumulate, map, filter

from fklearn.common_docstrings import splitter_return_docstring
from fklearn.types import DateType, LogType, SplitterReturnType


def _log_time_fold(time_fold: Tuple[pd.Series, pd.Series]) -> LogType:
    train_time, test_time = time_fold
    return {"train_start": train_time.min(), "train_end": train_time.max(), "train_size": train_time.shape[0],
            "test_start": test_time.min(), "test_end": test_time.max(), "test_size": test_time.shape[0]}


def _get_lc_folds(date_range: Union[pd.DatetimeIndex, pd.PeriodIndex],
                  date_fold_filter_fn: Callable[[DateType], pd.DataFrame],
                  test_time: pd.Series,
                  time_column: str,
                  min_samples: int) -> List[Tuple[pd.Series, pd.Series]]:
    return pipe(date_range,
                map(date_fold_filter_fn),  # iteratively filter the dates
                map(lambda df: df[time_column]),  # keep only time column
                filter(lambda s: len(s.index) > min_samples),
                lambda train: zip(train, repeat(test_time)),
                list)


def _get_sc_folds(date_range: Union[pd.DatetimeIndex, pd.PeriodIndex],
                  date_fold_filter_fn: Callable[[DateType], pd.DataFrame],
                  time_column: str,
                  min_samples: int) -> List[Tuple[pd.Series, pd.Series]]:
    return pipe(date_range,
                map(date_fold_filter_fn),  # iteratively filter the dates
                map(lambda df: df[time_column]),  # keep only time column
                filter(lambda s: len(s.index) > min_samples),
                list)


def _get_sc_test_fold_idx_and_logs(test_data: pd.DataFrame,
                                   train_time: pd.Series,
                                   time_column: str,
                                   first_test_moment: DateType,
                                   last_test_moment: DateType,
                                   min_samples: int,
                                   freq: str) -> Tuple[List[LogType], List[List[pd.Index]]]:
    periods_range = pd.period_range(start=first_test_moment, end=last_test_moment, freq=freq)

    def date_filter_fn(period: DateType) -> pd.DataFrame:
        return test_data[test_data[time_column].dt.to_period(freq) == period]

    folds = _get_sc_folds(periods_range, date_filter_fn, time_column, min_samples)

    logs = list(map(_log_time_fold, zip(repeat(train_time), folds)))  # get fold logs
    test_indexes = list(map(lambda test: [test.index], folds))  # final formatting with idx
    return logs, test_indexes


def _lc_fold_to_indexes(folds: List[Tuple[pd.Series, pd.Series]]) -> List[Tuple[pd.Index, List[pd.Index]]]:
    return list(starmap(lambda train, test: (train.index, [test.index]), folds))


@curry
def k_fold_splitter(train_data: pd.DataFrame,
                    n_splits: int,
                    random_state: int = None,
                    stratify_column: str = None) -> SplitterReturnType:
    """
    Makes K random train/test split folds for cross validation.
    The folds are made so that every sample is used at least once for
    evaluating and K-1 times for training.

    If stratified is set to True, the split preserves the distribution of stratify_column

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split into K-Folds for cross validation.

    n_splits : int
        The number of folds K for the K-Fold cross validation strategy.

    random_state : int
        Seed to be used by the random number generator.

    stratify_column : string
        Column name in train_data to be used for stratified split.
    """

    if stratify_column is not None:
        folds = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state) \
            .split(train_data, train_data[stratify_column])
    else:
        folds = KFold(n_splits, shuffle=True, random_state=random_state).split(train_data)
    result = list(map(lambda f: (f[0], [f[1]]), folds))

    logs = [{"train_size": len(fold[0]), "test_size": train_data.shape[0] - len(fold[0])} for fold in result]

    return result, logs


k_fold_splitter.__doc__ += splitter_return_docstring


# generate splits by space_column
# train_indexes will have time_column <= in_time_limit
# test_indexes will have time_column > in_time_limit
@curry
def out_of_time_and_space_splitter(train_data: pd.DataFrame,
                                   n_splits: int,
                                   in_time_limit: DateType,
                                   time_column: str,
                                   space_column: str,
                                   holdout_gap: timedelta = timedelta(days=0)) -> SplitterReturnType:
    """
    Makes K grouped train/test split folds for cross validation.
    The folds are made so that every ID is used at least once for
    evaluating and K-1 times for training. Also, for each fold, evaluation
    will always be out-of-ID and out-of-time.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split into K out-of-time and ID
        folds for cross validation.

    n_splits : int
        The number of folds K for the K-Fold cross validation strategy.

    in_time_limit : str or datetime.datetime
        A String representing the end time of the training data.
        It should be in the same format as the Date column in `train_data`.

    time_column : str
        The name of the Date column of `train_data`.

    space_column : str
        The name of the ID column of `train_data`.

    holdout_gap: datetime.timedelta
        Timedelta of the gap between the end of the training period and the start of the validation period.
    """

    # first generate folds by space, using LabelKFold
    # GroupKFold is not supposed to be randomized, that's why there's no random_state here
    train_data = train_data.reset_index()
    space_folds = GroupKFold(n_splits).split(train_data, groups=train_data[space_column])

    if isinstance(in_time_limit, str):
        in_time_limit = datetime.strptime(in_time_limit, "%Y-%m-%d")

    # train_indexes have time_column <= in_time_limit
    # test_indexes have time_column > in_time_limit
    folds = pipe(space_folds,
                 partial(starmap, lambda f_train, f_test: [train_data.iloc[f_train][time_column],
                                                           train_data.iloc[f_test][time_column]]),
                 partial(starmap, lambda train, test: (train[train <= in_time_limit],  # filter train time
                                                       test[test > (in_time_limit + holdout_gap)])),  # filter test time
                 list)

    logs = list(map(_log_time_fold, folds))  # get fold logs
    folds_indexes = _lc_fold_to_indexes(folds)  # final formatting with idx
    return folds_indexes, logs


out_of_time_and_space_splitter.__doc__ += splitter_return_docstring


@curry
def time_and_space_learning_curve_splitter(train_data: pd.DataFrame,
                                           training_time_limit: str,
                                           space_column: str,
                                           time_column: str,
                                           freq: str = 'M',
                                           space_hold_percentage: float = 0.5,
                                           holdout_gap: timedelta = timedelta(days=0),
                                           random_state: int = None,
                                           min_samples: int = 1000) -> SplitterReturnType:
    """
    Splits the data into temporal buckets given by the specified frequency.
    Uses a fixed out-of-ID and time hold out set for every fold.
    Training size increases per fold, with more recent data being added in each fold.
    Useful for learning curve validation, that is, for seeing how hold out performance
    increases as the training size increases with more recent data.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split for learning curve estimation.

    training_time_limit : str
        The Date String for the end of the testing period. Should be of the same
        format as `time_column`.

    space_column : str
        The name of the ID column of `train_data`.

    time_column : str
        The name of the Date column of `train_data`.

    freq : str
        The temporal frequency.
        See: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    space_hold_percentage : float
        The proportion of hold out IDs.

    holdout_gap: datetime.timedelta
        Timedelta of the gap between the end of the training period and the start of the validation period.

    random_state : int
        A seed for the random number generator for ID sampling across train and
        hold out sets.

    min_samples : int
        The minimum number of samples required in the split to keep the split.
    """

    train_data = train_data.reset_index()
    first_moment = train_data[time_column].min()
    date_range = pd.date_range(start=first_moment, end=training_time_limit, freq=freq)

    # sklearn function that handles int or object random states, turns them into a rng
    rng = check_random_state(random_state)
    out_of_space_mask = pipe(train_data,
                             lambda df: df[df[time_column] > date_range[-1]],  # filter out of time
                             lambda df: df[space_column].unique(),  # get unique space
                             lambda array: rng.choice(array, int(len(array) * space_hold_percentage), replace=False),
                             lambda held_space: train_data[space_column].isin(held_space))  # filter out of space

    training_time_limit_dt = datetime.strptime(training_time_limit, "%Y-%m-%d") + holdout_gap
    test_time = train_data[(train_data[time_column] > training_time_limit_dt) & out_of_space_mask][time_column]

    def date_filter_fn(date: DateType) -> pd.DataFrame:
        return train_data[(train_data[time_column] <= date) & ~out_of_space_mask]

    folds = _get_lc_folds(date_range, date_filter_fn, test_time, time_column, min_samples)

    logs = list(map(_log_time_fold, folds))  # get fold logs
    folds_indexes = _lc_fold_to_indexes(folds)  # final formatting with idx

    return folds_indexes, logs


time_and_space_learning_curve_splitter.__doc__ += splitter_return_docstring


@curry
def time_learning_curve_splitter(train_data: pd.DataFrame,
                                 training_time_limit: DateType,
                                 time_column: str,
                                 freq: str = 'M',
                                 holdout_gap: timedelta = timedelta(days=0),
                                 min_samples: int = 1000) -> SplitterReturnType:
    """
    Splits the data into temporal buckets given by the specified frequency.

    Uses a fixed out-of-ID and time hold out set for every fold.
    Training size increases per fold, with more recent data being added in each fold.
    Useful for learning curve validation, that is, for seeing how hold out performance
    increases as the training size increases with more recent data.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split for learning curve estimation.

    training_time_limit : str
        The Date String for the end of the testing period. Should be of the same
        format as `time_column`.

    time_column : str
        The name of the Date column of `train_data`.

    freq : str
        The temporal frequency.
        See: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    holdout_gap: datetime.timedelta
        Timedelta of the gap between the end of the training period and the start of the validation period.

    min_samples : int
        The minimum number of samples required in the split to keep the split.
    """

    train_data = train_data.reset_index()
    first_moment = train_data[time_column].min()
    date_range = pd.date_range(start=first_moment, end=training_time_limit, freq=freq)

    # training will end at last timestamp in range
    effective_training_time_end = date_range[-1]

    test_time = train_data[train_data[time_column] > (effective_training_time_end + holdout_gap)][time_column]

    def date_filter_fn(date: DateType) -> pd.DataFrame:
        return train_data[train_data[time_column] <= date]

    folds = _get_lc_folds(date_range, date_filter_fn, test_time, time_column, min_samples)

    logs = list(map(_log_time_fold, folds))  # get fold logs
    folds_indexes = _lc_fold_to_indexes(folds)  # final formatting with idx
    return folds_indexes, logs


time_learning_curve_splitter.__doc__ += splitter_return_docstring


@curry
def reverse_time_learning_curve_splitter(train_data: pd.DataFrame,
                                         time_column: str,
                                         training_time_limit: DateType,
                                         lower_time_limit: DateType = None,
                                         freq: str = 'MS',
                                         holdout_gap: timedelta = timedelta(days=0),
                                         min_samples: int = 1000) -> SplitterReturnType:
    """
    Splits the data into temporal buckets given by the specified frequency.
    Uses a fixed out-of-ID and time hold out set for every fold.
    Training size increases per fold, with less recent data being added in each fold.
    Useful for inverse learning curve validation, that is, for seeing how hold out
    performance increases as the training size increases with less recent data.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split inverse learning curve estimation.

    time_column : str
        The name of the Date column of `train_data`.

    training_time_limit : str
        The Date String for the end of the testing period. Should be of the same
        format as `time_column`.

    lower_time_limit : str
        A Date String for the begining of the training period. This allows limiting
        the learning curve from bellow, avoiding heavy computation with very old data.

    freq : str
        The temporal frequency.
        See: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    holdout_gap: datetime.timedelta
        Timedelta of the gap between the end of the training period and the start of the validation period.

    min_samples : int
        The minimum number of samples required in the split to keep the split.
    """

    train_data = train_data.reset_index()
    first_moment = lower_time_limit if lower_time_limit else train_data[time_column].min()
    date_range = pd.date_range(start=first_moment, end=training_time_limit, freq=freq)

    # training will end at last timestamp in range
    effective_training_time_end = date_range[-1]

    train_range = train_data[train_data[time_column] <= effective_training_time_end]
    test_time = train_data[train_data[time_column] > (effective_training_time_end + holdout_gap)][time_column]

    def date_filter_fn(date: DateType) -> pd.DataFrame:
        return train_range.loc[train_data[time_column] >= date]

    folds = _get_lc_folds(date_range[::-1], date_filter_fn, test_time, time_column, min_samples)

    logs = list(map(_log_time_fold, folds))  # get fold logs
    folds_indexes = _lc_fold_to_indexes(folds)  # final formatting with idx

    return folds_indexes, logs


reverse_time_learning_curve_splitter.__doc__ += splitter_return_docstring


@curry
def spatial_learning_curve_splitter(train_data: pd.DataFrame,
                                    space_column: str,
                                    time_column: str,
                                    training_limit: DateType,
                                    holdout_gap: timedelta = timedelta(days=0),
                                    train_percentages: Iterable[float] = (0.25, 0.5, 0.75, 1.0),
                                    random_state: int = None) -> SplitterReturnType:
    """
    Splits the data for a spatial learning curve. Progressively adds more and
    more examples to the training in order to verify the impact of having more
    data available on a validation set.

    The validation set starts after the training set, with an optional time gap.

    Similar to the temporal learning curves, but with spatial increases in the training set.

    Parameters
    ----------

    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split for learning curve estimation.

    space_column : str
        The name of the ID column of `train_data`.

    time_column : str
        The name of the temporal column of `train_data`.

    training_limit: datetime or str
        The date limiting the training (after which the holdout begins).

    holdout_gap: timedelta
        The gap between the end of training and the start of the holdout.
        If you have censored data, use a gap similar to the censor time.

    train_percentages: list or tuple of floats
        A list containing the percentages of IDs to use in the training.
        Defaults to (0.25, 0.5, 0.75, 1.0). For example: For the default value,
        there would be four model trainings, containing respectively 25%, 50%,
        75%, and 100% of the IDs that are not part of the held out set.

    random_state : int
        A seed for the random number generator that shuffles the IDs.
    """
    assert 0.0 < np.min(train_percentages) <= 1.0, "Train percentages must be between 0 and 1"

    if isinstance(training_limit, str):
        training_limit = datetime.strptime(training_limit, "%Y-%m-%d")

    assert train_data[time_column].min() < training_limit < train_data[time_column].max(), (
        "Temporal training limit should be within datasets temporal bounds (min and max times)")
    assert timedelta(days=0) <= holdout_gap, "Holdout gap cannot be negative"
    assert holdout_gap < train_data[time_column].max() - training_limit, (
        "After taking the gap into account, there should be enough time for the holdout set")

    train_data = train_data.reset_index()

    # We need to sample the space column before getting its unique values so their order in the DF won't matter here
    spatial_ids = train_data[space_column].sample(frac=1, random_state=random_state).unique()

    cumulative_ids = pipe(
        spatial_ids,
        lambda ids: (np.array(train_percentages) * len(ids)).astype(int),  # Get the corresponding indices for each %
        lambda idx: np.split(spatial_ids, idx)[:-1],  # Split spatial ids by the indices
        lambda l: map(lambda x: x.tolist(), l),  # Transform sub-arrays into sub-lists
        lambda l: filter(None, l),  # Drop empty sub-lists
        accumulate(operator.add)  # Cumulative sum of lists
    )

    validation_set = train_data[train_data[time_column] > (training_limit + holdout_gap)]
    train_data = train_data[train_data[time_column] <= training_limit]

    folds = [(train_data[train_data[space_column].isin(ids)][time_column], validation_set[time_column])
             for ids in cumulative_ids]

    folds_indices = _lc_fold_to_indexes(folds)  # final formatting with idx

    logs = [assoc(l, "percentage", p) for l, p in zip(map(_log_time_fold, folds), train_percentages)]

    return folds_indices, logs


spatial_learning_curve_splitter.__doc__ += splitter_return_docstring


@curry
def stability_curve_time_splitter(train_data: pd.DataFrame,
                                  training_time_limit: DateType,
                                  time_column: str,
                                  freq: str = 'M',
                                  min_samples: int = 1000) -> SplitterReturnType:
    """
    Splits the data into temporal buckets given by the specified frequency.
    Training set is fixed before hold out and uses a rolling window hold out set.
    Each fold moves the hold out further into the future.
    Useful to see how model performance degrades as the training data gets more
    outdated. Training and holdout sets can have same IDs

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split for stability curve estimation.

    training_time_limit : str
        The Date String for the end of the testing period. Should be of the same
        format as `time_column`.

    time_column : str
        The name of the Date column of `train_data`.

    freq : str
        The temporal frequency.
        See: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    min_samples : int
        The minimum number of samples required in a split to keep it.
    """

    train_data = train_data.reset_index()

    train_time = train_data[train_data[time_column] <= training_time_limit][time_column]
    test_data = train_data[train_data[time_column] > training_time_limit]

    first_test_moment = test_data[time_column].min()
    last_test_moment = test_data[time_column].max()

    logs, test_indexes = _get_sc_test_fold_idx_and_logs(test_data, train_time, time_column, first_test_moment,
                                                        last_test_moment, min_samples, freq)

    # From "list of dicts" to "dict of lists" hack:
    logs = [{k: [dic[k] for dic in logs] for k in logs[0]}]

    # Flatten test_indexes:
    flattened_test_indices = list(chain.from_iterable(test_indexes))

    return [(train_time.index, flattened_test_indices)], logs


stability_curve_time_splitter.__doc__ += splitter_return_docstring


@curry
def stability_curve_time_in_space_splitter(train_data: pd.DataFrame,
                                           training_time_limit: DateType,
                                           space_column: str,
                                           time_column: str,
                                           freq: str = 'M',
                                           space_hold_percentage: float = 0.5,
                                           random_state: int = None,
                                           min_samples: int = 1000) -> SplitterReturnType:
    """
    Splits the data into temporal buckets given by the specified frequency.
    Training set is fixed before hold out and uses a rolling window hold out set.
    Each fold moves the hold out further into the future.
    Useful to see how model performance degrades as the training data gets more
    outdated. Folds are made so that ALL IDs in the holdout also appear in
    the training set.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split for stability curve estimation.

    training_time_limit : str
        The Date String for the end of the testing period. Should be of the same
        format as `time_column`.

    space_column : str
        The name of the ID column of `train_data`.

    time_column : str
        The name of the Date column of `train_data`.

    freq : str
        The temporal frequency.
        See: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    space_hold_percentage : float (default=0.5)
        The proportion of hold out IDs.

    random_state : int
        A seed for the random number generator for ID sampling across train and
        hold out sets.

    min_samples : int
        The minimum number of samples required in the split to keep the split.
    """

    train_data = train_data.reset_index()

    rng = check_random_state(random_state)

    train_time = train_data[train_data[time_column] <= training_time_limit][time_column]

    test_data = pipe(train_data,
                     lambda trand_df: trand_df.iloc[train_time.index][space_column].unique(),
                     lambda space: rng.choice(space, int(len(space) * space_hold_percentage), replace=False),
                     lambda held_space: train_data[(train_data[time_column] > training_time_limit)
                                                   & (train_data[space_column].isin(held_space))])

    first_test_moment = test_data[time_column].min()
    last_test_moment = test_data[time_column].max()

    logs, test_indexes = _get_sc_test_fold_idx_and_logs(test_data, train_time, time_column, first_test_moment,
                                                        last_test_moment, min_samples, freq)

    # From "list of dicts" to "dict of lists" hack:
    logs = [{k: [dic[k] for dic in logs] for k in logs[0]}]

    # Flatten test_indexes:
    flattened_test_indices = list(chain.from_iterable(test_indexes))

    return [(train_time.index, flattened_test_indices)], logs


stability_curve_time_in_space_splitter.__doc__ += splitter_return_docstring


@curry
def stability_curve_time_space_splitter(train_data: pd.DataFrame,
                                        training_time_limit: DateType,
                                        space_column: str,
                                        time_column: str,
                                        freq: str = 'M',
                                        space_hold_percentage: float = 0.5,
                                        random_state: int = None,
                                        min_samples: int = 1000) -> SplitterReturnType:
    """
    Splits the data into temporal buckets given by the specified frequency.
    Training set is fixed before hold out and uses a rolling window hold out set.
    Each fold moves the hold out further into the future.
    Useful to see how model performance degrades as the training data gets more
    outdated. Folds are made so that NONE of the IDs in the holdout appears in
    the training set.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split for stability curve estimation.

    training_time_limit : str
        The Date String for the end of the testing period. Should be of the same
        format as `time_column`

    space_column : str
        The name of the ID column of `train_data`

    time_column : str
        The name of the Date column of `train_data`

    freq : str
        The temporal frequency.
        See: http://pandas.pydata.org/pandas-docs/stable/timeseries.html#offset-aliases

    space_hold_percentage : float
        The proportion of hold out IDs

    random_state : int
        A seed for the random number generator for ID sampling across train and
        hold out sets.

    min_samples : int
        The minimum number of samples required in the split to keep the split.
    """
    train_data = train_data.reset_index()

    rng = check_random_state(random_state)

    train_time = train_data[train_data[time_column] <= training_time_limit][time_column]
    train_index = train_time.index.values
    train_space = train_data.iloc[train_index][space_column].unique()

    held_space = rng.choice(train_space, int(len(train_space) * space_hold_percentage), replace=False)

    test_data = train_data[
        (train_data[time_column] > training_time_limit) & (~train_data[space_column].isin(held_space))]
    train_index = train_data[
        (train_data[time_column] <= training_time_limit) & (train_data[space_column].isin(held_space))].index.values

    first_test_moment = test_data[time_column].min()
    last_test_moment = test_data[time_column].max()

    logs, test_indexes = _get_sc_test_fold_idx_and_logs(test_data, train_time, time_column, first_test_moment,
                                                        last_test_moment, min_samples, freq)

    # From "list of dicts" to "dict of lists" hack:
    logs = [{k: [dic[k] for dic in logs] for k in logs[0]}]

    # Flatten test_indexes:
    flattened_test_indices = list(chain.from_iterable(test_indexes))

    return [(train_index, flattened_test_indices)], logs


stability_curve_time_space_splitter.__doc__ += splitter_return_docstring


@curry
def forward_stability_curve_time_splitter(train_data: pd.DataFrame,
                                          training_time_start: DateType,
                                          training_time_end: DateType,
                                          time_column: str,
                                          holdout_gap: timedelta = timedelta(days=0),
                                          holdout_size: timedelta = timedelta(days=90),
                                          step: timedelta = timedelta(days=90),
                                          move_training_start_with_steps: bool = True) -> SplitterReturnType:
    """
    Splits the data into temporal buckets with both the training and testing folds both moving forward.
    The folds move forward by a fixed timedelta step.
    Optionally, there can be a gap between the end of the training period and the start of the holdout period.

    Similar to the stability curve time splitter, with the difference that the training period also
    moves forward with each fold.

    The clearest use case is to evaluate a periodic re-training framework.

    Parameters
    ----------
    train_data : pandas.DataFrame
        A Pandas' DataFrame that will be split for stability curve estimation.

    training_time_start: datetime.datetime or str
        Date for the start of the training period.
        If `move_training_start_with_steps` is `True`, each step will increase this date by `step`.

    training_time_end: datetime.datetime or str
        Date for the end of the training period.
        Each step increases this date by `step`.

    time_column : str
        The name of the Date column of `train_data`.

    holdout_gap: datetime.timedelta
        Timedelta of the gap between the end of the training period and the start of the validation period.

    holdout_size: datetime.timedelta
        Timedelta of the range between the start and the end of the holdout period.

    step: datetime.timedelta
        Timedelta that shifts both the training period and the holdout period by this value.

    move_training_start_with_steps: bool
        If True, the training start date will increase by `step` for each fold.
        If False, the training start date remains fixed at the `training_time_start` value.
    """

    if isinstance(training_time_start, str):
        training_time_start = datetime.strptime(training_time_start, "%Y-%m-%d")

    if isinstance(training_time_end, str):
        training_time_end = datetime.strptime(training_time_end, "%Y-%m-%d")

    train_data = train_data.reset_index()

    max_date = train_data[time_column].max()

    assert train_data[time_column].min() <= training_time_start < training_time_end <= max_date, (
        "Temporal training limits should be within datasets temporal bounds (min and max times)")
    assert timedelta(days=0) <= holdout_gap, "Holdout gap cannot be negative"
    assert timedelta(days=0) <= holdout_size, "Holdout size cannot be negative"

    n_folds = int(np.ceil((max_date - holdout_size - holdout_gap - training_time_end) / step))

    assert n_folds > 0, "After taking the gap and holdout into account, there should be enough time for the holdout set"

    train_ranges = [(training_time_start + i * step * move_training_start_with_steps, training_time_end + i * step)
                    for i in range(n_folds)]

    test_ranges = [
        (training_time_end + holdout_gap + i * step, training_time_end + holdout_gap + holdout_size + i * step)
        for i in range(n_folds)
    ]

    train_idx = [train_data[(train_data[time_column] >= start) & (train_data[time_column] < end)].index
                 for start, end in train_ranges]
    test_idx = [[train_data[(train_data[time_column] >= start) & (train_data[time_column] < end)].index]
                for start, end in test_ranges]

    logs = [_log_time_fold((train_data.iloc[i][time_column], train_data.iloc[j[0]][time_column])) for i, j in
            zip(train_idx, test_idx)]

    return list(zip(train_idx, test_idx)), logs


forward_stability_curve_time_splitter.__doc__ += splitter_return_docstring
