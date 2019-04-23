from typing import Any, Callable, Dict, List, Union, Optional

import numpy as np
import pandas as pd
from numpy import nan
import swifter  # NOQA
from sklearn.preprocessing import StandardScaler
from statsmodels.distributions import empirical_distribution as ed
from toolz import curry, merge, compose, mapcat
from fklearn.common_docstrings import learner_return_docstring, learner_pred_fn_docstring
from fklearn.training.utils import log_learner_time
from fklearn.types import LearnerReturnType, LearnerLogType


@curry
@log_learner_time(learner_name='selector')
def selector(df: pd.DataFrame,
             training_columns: List[str],
             predict_columns: List[str] = None) -> LearnerReturnType:
    """
    Filters a DataFrames by selecting only the desired columns.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns`

    training_columns : list of str
        A list of column names that will remain in the dataframe during training time (fit)

    predict_columns: list of str
        A list of column names that will remain in the dataframe during prediction time (transform)
        If None, it defaults to `training_columns`.
    """

    if predict_columns is None:
        predict_columns = training_columns

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        return new_data_set[predict_columns]

    p.__doc__ = learner_pred_fn_docstring("selector")

    log = {'selector': {
        'training_columns': training_columns,
        'predict_columns': predict_columns,
        'transformed_column': list(set(training_columns).union(predict_columns))}}

    return p, df[training_columns], log


selector.__doc__ += learner_return_docstring("Selector")


@curry
@log_learner_time(learner_name='capper')
def capper(df: pd.DataFrame,
           columns_to_cap: List[str],
           precomputed_caps: Dict[str, float] = None) -> LearnerReturnType:
    """
    Learns the maximum value for each of the `columns_to_cap`
    and used that as the cap for those columns. If precomputed caps
    are passed, the function uses that as the cap value instead of
    computing the maximum.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns_to_cap` columns.

    columns_to_cap : list of str
        A list os column names that should be caped.

    precomputed_caps : dict
        A dictionary on the format {"column_name" : cap_value}.
        That maps column names to pre computed cap values
    """

    if not precomputed_caps:
        precomputed_caps = {}

    caps = {col: precomputed_caps.get(col, df[col].max()) for col in columns_to_cap}

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        capped_cols = {col: new_data_set[col].clip(upper=caps[col]) for col in caps.keys()}
        return new_data_set.assign(**capped_cols)

    p.__doc__ = learner_pred_fn_docstring("capper")

    log = {'capper': {
        'caps': caps,
        'transformed_column': columns_to_cap,
        'precomputed_caps': precomputed_caps}}

    return p, p(df), log


capper.__doc__ += learner_return_docstring("Capper")


@curry
@log_learner_time(learner_name='floorer')
def floorer(df: pd.DataFrame,
            columns_to_floor: List[str],
            precomputed_floors: Dict[str, float] = None) -> LearnerReturnType:
    """
    Learns the minimum value for each of the `columns_to_floor`
    and used that as the floot for those columns. If precomputed floors
    are passed, the function uses that as the cap value instead of
    computing the minimun.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns_to_floor` columns.

    columns_to_floor : list of str
        A list os column names that should be floored.

    precomputed_floors : dict
        A dictionary on the format {"column_name" : floor_value}
        that maps column names to pre computed floor values
    """

    if not precomputed_floors:
        precomputed_floors = {}

    floors = {col: precomputed_floors.get(col, df[col].min()) for col in columns_to_floor}

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        capped_cols = {col: new_data_set[col].clip(lower=floors[col]) for col in floors.keys()}
        return new_data_set.assign(**capped_cols)

    p.__doc__ = learner_pred_fn_docstring("floorer")

    log = {'floorer': {
        'floors': floors,
        'transformed_column': columns_to_floor,
        'precomputed_floors': precomputed_floors}}

    return p, p(df), log


floorer.__doc__ += learner_return_docstring("Floorer")


@curry
@log_learner_time(learner_name='ecdfer')
def ecdfer(df: pd.DataFrame,
           ascending: bool = True,
           prediction_column: str = "prediction",
           ecdf_column: str = "prediction_ecdf",
           max_range: int = 1000) -> LearnerReturnType:
    """
    Learns an Empirical Cumulative Distribution Function from the specified column
    in the input DataFrame. It is usually used in the prediction column to convert
    a predicted probability into a score from 0 to 1000.

    Parameters
    ----------
    df : Pandas' pandas.DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    ascending : bool
        Whether to compute an ascending ECDF or a descending one.

    prediction_column : str
        The name of the column in `df` to learn the ECDF from.

    ecdf_column : str
        The name of the new ECDF column added by this function

    max_range : int
        The maximum value for the ECDF. It will go will go
         from 0 to max_range.
    """

    if ascending:
        base = 0
        sign = 1
    else:
        base = max_range
        sign = -1

    values = df[prediction_column]

    ecdf = ed.ECDF(values)

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(**{ecdf_column: (base + sign * max_range * ecdf(new_df[prediction_column]))})

    p.__doc__ = learner_pred_fn_docstring("ecdefer")

    log = {'ecdfer': {
        'nobs': len(values),
        'prediction_column': prediction_column,
        'ascending': ascending,
        'transformed_column': [ecdf_column]}}

    return p, p(df), log


ecdfer.__doc__ += learner_return_docstring("ECDFer")


@curry
@log_learner_time(learner_name='discrete_ecdfer')
def discrete_ecdfer(df: pd.DataFrame,
                    ascending: bool = True,
                    prediction_column: str = "prediction",
                    ecdf_column: str = "prediction_ecdf",
                    max_range: int = 1000,
                    round_method: Callable = int) -> LearnerReturnType:
    """
    Learns an Empirical Cumulative Distribution Function from the specified column
    in the input DataFrame. It is usually used in the prediction column to convert
    a predicted probability into a score from 0 to 1000.

    Parameters
    ----------
    df : Pandas' pandas.DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    ascending : bool
        Whether to compute an ascending ECDF or a descending one.

    prediction_column : str
        The name of the column in `df` to learn the ECDF from.

    ecdf_column : str
        The name of the new ECDF column added by this function.

    max_range : int
        The maximum value for the ECDF. It will go will go
         from 0 to max_range.

    round_method: Callable
        A function perform the round of transformed values for ex: (int, ceil, floor, round)
    """

    if ascending:
        base = 0
        sign = 1
    else:
        base = max_range
        sign = -1

    values = df[prediction_column]

    ecdf = ed.ECDF(values)

    df_ecdf = pd.DataFrame()
    df_ecdf['x'] = ecdf.x
    df_ecdf['y'] = pd.Series(base + sign * max_range * ecdf.y).apply(round_method)

    boundaries = df_ecdf.groupby("y").agg((min, max))["x"]["min"].reset_index()

    y = boundaries["y"]
    x = boundaries["min"]
    side = ecdf.side

    log = {'discrete_ecdfer': {
        'map': dict(zip(x, y)),
        'round_method': round_method,
        'nobs': len(values),
        'prediction_column': prediction_column,
        'ascending': ascending,
        'transformed_column': [ecdf_column]}}

    del ecdf
    del values
    del df_ecdf

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        if not ascending:
            tind = np.searchsorted(-x, -new_df[prediction_column])
        else:
            tind = np.searchsorted(x, new_df[prediction_column], side) - 1

        return new_df.assign(**{ecdf_column: y[tind].values})

    return p, p(df), log


discrete_ecdfer.__doc__ += learner_return_docstring("Discrete ECDFer")


@curry
def prediction_ranger(df: pd.DataFrame,
                      prediction_min: float,
                      prediction_max: float,
                      prediction_column: str = "prediction") -> LearnerReturnType:
    """
    Caps and floors the specified prediction column to a set range.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    prediction_min : float
        The floor for the prediction.

    prediction_max : float
        The cap for the prediction.

    prediction_column : str
        The name of the column in `df` to cap and floor
    """

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return new_df.assign(
            **{prediction_column: new_df[prediction_column].clip(lower=prediction_min, upper=prediction_max)}
        )

    p.__doc__ = learner_pred_fn_docstring("prediction_ranger")

    log = {'prediction_ranger': {
        'prediction_min': prediction_min,
        'prediction_max': prediction_max,
        'transformed_column': [prediction_column]}}

    return p, p(df), log


prediction_ranger.__doc__ += learner_return_docstring("Prediction Ranger")


def apply_replacements(df: pd.DataFrame,
                       columns: List[str],
                       vec: Dict[str, Dict],
                       replace_unseen: Any) -> pd.DataFrame:
    """
    Base function to apply the replacements values found on the
    "vec" vectors into the df DataFrame.

    Parameters
    -----------

    df: pandas.DataFrame
        A Pandas DataFrame containing the data to be replaced.

    columns : list of str
        The df columns names to perform the replacements.

    vec: dict
        A dict mapping a col to dict mapping a value to its replacement. For example:
        vec = {"feature1": {1: 2, 3: 5, 6: 8}}

    replace_unseen: Any
        Default value to replace when original value is not present in the `vec` dict for the feature

    """
    column_categorizer = lambda col: df[col].apply(lambda x: (np.nan
                                                              if isinstance(x, float) and np.isnan(x)
                                                              else vec[col].get(x, replace_unseen)))
    categ_columns = {col: column_categorizer(col) for col in columns}
    return df.assign(**categ_columns)


@curry
@log_learner_time(learner_name="value_mapper")
def value_mapper(df: pd.DataFrame,
                 value_maps: Dict[str, Dict],
                 ignore_unseen: bool = True,
                 replace_unseen_to: Any = np.nan) -> pd.DataFrame:
    """
    Map values in selected columns in the DataFrame according to dictionaries of replacements.
    Learner wrapper for apply_replacements

    Parameters
    -----------

    df: pandas.DataFrame
        A Pandas DataFrame containing the data to be replaced.

    value_maps: dict of dicts
        A dict mapping a col to dict mapping a value to its replacement. For example:
        value_maps = {"feature1": {1: 2, 3: 5, 6: 8}}

    ignore_unseen: bool
        If True, values not explicitly declared in value_maps will be left as is.
        If False, these will be replaced by replace_unseen_to.

    replace_unseen_to: Any
        Default value to replace when original value is not present in the `vec` dict for the feature.
    """

    def new_col_value_map(old_col_value_map: Dict[Any, Any],
                          new_keys: List[Any]) -> Dict[Any, Dict]:
        old_keys = old_col_value_map.keys()
        return {key: old_col_value_map[key] if key in old_keys else key for key in new_keys}

    columns = list(value_maps.keys())
    if ignore_unseen:
        value_maps = {col: new_col_value_map(value_maps[col], list(df[col].unique())) for col in columns}

    def p(df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(df, columns, value_maps, replace_unseen=replace_unseen_to)

    return p, p(df), {"value_maps": value_maps}


@curry
@log_learner_time(learner_name="truncate_categorical")
def truncate_categorical(df: pd.DataFrame,
                         columns_to_truncate: List[str],
                         percentile: float,
                         replacement: Union[str, float] = -9999,
                         replace_unseen: Union[str, float] = -9999,
                         store_mapping: bool = False) -> LearnerReturnType:
    """
    Truncate infrequent categories and replace them by a single one.
    You can think of it like "others" category.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    columns_to_truncate : list of str
        The df columns names to perform the truncation.

    percentile : float
        Categories less frequent than the percentile will be replaced by the
        same one.

    replacement: int, str, float or nan
        The value to use when a category is less frequent that the percentile
        variable.

    replace_unseen : int, str, float, or nan
        The value to impute unseen categories.

    store_mapping : bool (default: False)
        Whether to store the feature value -> integer dictionary in the log.
    """
    get_categs = lambda col: (df[col].value_counts() / len(df)).to_dict()
    update = lambda d: map(lambda kv: (kv[0], replacement) if kv[1] <= percentile else (kv[0], kv[0]), d.items())
    categs_to_dict = lambda categ_dict: dict(categ_dict)

    vec = {column: compose(categs_to_dict, update, get_categs)(column) for column in columns_to_truncate}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_truncate, vec, replace_unseen)

    p.__doc__ = learner_pred_fn_docstring("truncate_categorical")

    log: LearnerLogType = {'truncate_categorical': {
        'transformed_column': columns_to_truncate,
        'replace_unseen': replace_unseen}
    }

    if store_mapping:
        log["truncate_categorical"]["mapping"] = vec

    return p, p(df), log


truncate_categorical.__doc__ += learner_return_docstring("Truncate Categorical")


@curry
@log_learner_time(learner_name="rank_categorical")
def rank_categorical(df: pd.DataFrame,
                     columns_to_rank: List[str],
                     replace_unseen: Union[str, float] = nan,
                     store_mapping: bool = False) -> LearnerReturnType:
    """
    Rank categorical features by their frequency in the train set.

    Parameters
    ----------
    df : Pandas' DataFrame
        A Pandas' DataFrame that must contain a `prediction_column` columns.

    columns_to_rank : list of str
        The df columns names to perform the rank.

    replace_unseen : int, str, float, or nan
        The value to impute unseen categories.

    store_mapping : bool (default: False)
        Whether to store the feature value -> integer dictionary in the log
    """

    col_categ_getter = lambda col: (df[col]
                                    .value_counts()
                                    .reset_index()
                                    .sort_values([col, "index"], ascending=[False, True])
                                    .set_index("index")[col]
                                    .rank(method="first", ascending=False).to_dict())

    vec = {column: col_categ_getter(column) for column in columns_to_rank}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_rank, vec, replace_unseen)

    p.__doc__ = learner_pred_fn_docstring("rank_categorical")

    log: LearnerLogType = {'rank_categorical': {
        'transformed_column': columns_to_rank,
        'replace_unseen': replace_unseen}
    }

    if store_mapping:
        log['rank_categorical']['mapping'] = vec

    return p, p(df), log


rank_categorical.__doc__ += learner_return_docstring("Rank Categorical")


@curry
@log_learner_time(learner_name='count_categorizer')
def count_categorizer(df: pd.DataFrame,
                      columns_to_categorize: List[str],
                      replace_unseen: int = -1,
                      store_mapping: bool = False) -> LearnerReturnType:
    """
    Replaces categorical variables by count.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns_to_categorize` columns.

    columns_to_categorize : list of str
        A list of categorical column names.

    replace_unseen : int
        The value to impute unseen categories.

    store_mapping : bool (default: False)
        Whether to store the feature value -> integer dictionary in the log
    """

    categ_getter = lambda col: df[col].value_counts().to_dict()
    vec = {column: categ_getter(column) for column in columns_to_categorize}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_categorize, vec, replace_unseen)

    p.__doc__ = learner_pred_fn_docstring("count_categorizer")

    log: LearnerLogType = {'count_categorizer': {
        'transformed_column': columns_to_categorize,
        'replace_unseen': replace_unseen}
    }

    if store_mapping:
        log['count_categorizer']['mapping'] = vec

    return p, p(df), log


count_categorizer.__doc__ += learner_return_docstring("Count Categorizer")


@curry
@log_learner_time(learner_name='label_categorizer')
def label_categorizer(df: pd.DataFrame,
                      columns_to_categorize: List[str],
                      replace_unseen: Union[str, float] = nan,
                      store_mapping: bool = False) -> LearnerReturnType:
    """
    Replaces categorical variables with a numeric identifier.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns_to_categorize` columns.

    columns_to_categorize : list of str
        A list of categorical column names.

    replace_unseen : int, str, float, or nan
        The value to impute unseen categories.

    store_mapping : bool (default: False)
        Whether to store the feature value -> integer dictionary in the log
    """

    def categ_dict(series: pd.Series) -> Dict:
        categs = series.dropna().unique()
        return dict(map(reversed, enumerate(categs)))  # type: ignore

    vec = {column: categ_dict(df[column]) for column in columns_to_categorize}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        return apply_replacements(new_df, columns_to_categorize, vec, replace_unseen)

    p.__doc__ = learner_pred_fn_docstring("label_categorizer")

    log: LearnerLogType = {'label_categorizer': {
        'transformed_column': columns_to_categorize,
        'replace_unseen': replace_unseen}
    }

    if store_mapping:
        log['label_categorizer']['mapping'] = vec

    return p, p(df), log


label_categorizer.__doc__ += learner_return_docstring("Label Categorizer")


@curry
@log_learner_time(learner_name='quantile_biner')
def quantile_biner(df: pd.DataFrame,
                   columns_to_bin: List[str],
                   q: int = 4,
                   right: bool = False) -> LearnerReturnType:
    """
    Discretize continuous numerical columns into its quantiles. Uses pandas.qcut
    to find the bins and then numpy.digitize to fit the columns into bins.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns_to_categorize` columns.

    columns_to_bin : list of str
        A list of numerical column names.

    q : int
        Number of quantiles. 10 for deciles, 4 for quartiles, etc.
        Alternately array of quantiles, e.g. [0, .25, .5, .75, 1.] for quartiles.
        See https://pandas.pydata.org/pandas-docs/stable/generated/pandas.qcut.html

    right : bool
        Indicating whether the intervals include the right or the left bin edge.
        Default behavior is (right==False) indicating that the interval does not
        include the right edge. The left bin end is open in this case, i.e., bins[i-1]
        <= x < bins[i] is the default behavior for monotonically increasing bins.
        See https://docs.scipy.org/doc/numpy-1.13.0/reference/generated/numpy.digitize.html
    """

    bin_getter = lambda col: pd.qcut(df[col], q, retbins=True)[1]
    bins = {column: bin_getter(column) for column in columns_to_bin}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        col_biner = lambda col: np.where(new_df[col].isnull(), nan, np.digitize(new_df[col], bins[col], right=right))
        bined_columns = {col: col_biner(col) for col in columns_to_bin}
        return new_df.assign(**bined_columns)

    p.__doc__ = learner_pred_fn_docstring("quantile_biner")

    log = {'quantile_biner': {
        'transformed_column': columns_to_bin,
        'q': q}}

    return p, p(df), log


quantile_biner.__doc__ += learner_return_docstring("Quantile Biner")


@curry
@log_learner_time(learner_name='onehot_categorizer')
def onehot_categorizer(df: pd.DataFrame,
                       columns_to_categorize: List[str],
                       hardcode_nans: bool = False,
                       drop_first_column: bool = False,
                       store_mapping: bool = False) -> LearnerReturnType:
    """
    Onehot encoding on categorical columns.

    Parameters
    ----------
    df : pd.DataFrame
        A Pandas' DataFrame that must contain `columns_to_categorize` columns.

    columns_to_categorize : list of str
        A list of categorical column names. Must be non-empty.

    hardcode_nans : bool
        Hardcodes an extra column with: 1 if nan or unseen else 0.

    drop_first_column : bool
        Drops the first column to create (k-1)-sized one-hot arrays for k
        features per categorical column. Can be used to avoid colinearity.

    store_mapping : bool (default: False)
        Whether to store the feature value -> integer dictionary in the log
    """

    categ_getter = lambda col: list(np.sort(df[col].dropna(axis=0, how='any').unique())[int(drop_first_column):])
    vec = {column: categ_getter(column) for column in sorted(columns_to_categorize)}

    def p(new_df: pd.DataFrame) -> pd.DataFrame:
        make_dummies = lambda col: dict(map(lambda categ: (col + "==" + str(categ), (new_df[col] == categ).astype(int)),
                                            vec[col]))

        oh_cols = dict(mapcat(lambda col: merge(make_dummies(col),
                                                {col + "==nan": (~new_df[col].isin(vec[col])).astype(
                                                    int)} if hardcode_nans
                                                else {}).items(),
                              columns_to_categorize))

        return new_df.assign(**oh_cols).drop(columns_to_categorize, axis=1)

    p.__doc__ = learner_pred_fn_docstring("onehot_categorizer")

    log = {'onehot_categorizer': {
        'transformed_column': columns_to_categorize,
        'hardcode_nans': hardcode_nans,
        'drop_first_column': drop_first_column}}

    if store_mapping:
        log['onehot_categorizer']['mapping'] = vec

    return p, p(df), log


quantile_biner.__doc__ += learner_return_docstring("Onehot Categorizer")


@curry
@log_learner_time(learner_name='standard_scaler')
def standard_scaler(df: pd.DataFrame,
                    columns_to_scale: List[str]) -> LearnerReturnType:
    """
    Fits a standard scaler to the dataset.

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame with columns to scale.
        It must contain all columns listed in `columns_to_scale`.

    columns_to_scale : list of str
        A list of names of the columns for standard scaling.
    """

    scaler = StandardScaler()

    scaler.fit(df[columns_to_scale].values)

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        new_data = scaler.transform(new_data_set[columns_to_scale].values)
        new_cols = pd.DataFrame(data=new_data, columns=columns_to_scale).to_dict('list')
        return new_data_set.assign(**new_cols)

    p.__doc__ = learner_pred_fn_docstring("standard_scaler")

    log = {'standard_scaler': {
        'standard_scaler': scaler.get_params(),
        'transformed_column': columns_to_scale}}

    return p, p(df), log


standard_scaler.__doc__ += learner_return_docstring("Standard Scaler")


@curry
@log_learner_time(learner_name='custom_transformer')
def custom_transformer(df: pd.DataFrame,
                       columns_to_transform: List[str],
                       transformation_function: Callable[[pd.DataFrame], pd.DataFrame]) -> LearnerReturnType:
    """
    Applies a custom function to the desired columns.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns`

    columns_to_transform : list of str
        A list of column names that will remain in the dataframe during training time (fit)

    transformation_function : function(pandas.DataFrame) -> pandas.DataFrame
        A function that receives a DataFrame as input, performs a transformation on its columns
        and returns another DataFrame.

    """

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        new_data_set[columns_to_transform] = new_data_set[columns_to_transform].swifter.apply(transformation_function)

        return new_data_set

    p.__doc__ = learner_pred_fn_docstring("custom_transformer")

    log = {'custom_transformer': {
        'transformed_column': columns_to_transform,
        'transformation_function': transformation_function.__name__}
    }

    return p, p(df), log


custom_transformer.__doc__ += learner_return_docstring("Custom Transformer")


@curry
@log_learner_time(learner_name='null_injector')
def null_injector(df: pd.DataFrame,
                  proportion: float,
                  columns_to_inject: Optional[List[str]] = None,
                  groups: Optional[List[List[str]]] = None,
                  seed: int = 1) -> LearnerReturnType:
    """
    Applies a custom function to the desired columns.

    Parameters
    ----------
    df : pandas.DataFrame
        A Pandas' DataFrame that must contain `columns_to_inject` as columns

    columns_to_inject : list of str
        A list of features to inject nulls. If groups is not None it will be ignored.

    proportion : float
        Proportion of nulls to inject in the columns.

    groups : list of list of str (default = None)
        A list of group of features. If not None, feature in the same group will be set to NaN together.

    seed : int
        Random seed for consistency.
    """
    assert (proportion > 0.0) & (proportion < 1.0), "proportions must be between 0 and 1"
    assert (columns_to_inject is None) ^ (groups is None), "Either columns_to_inject or groups must be None."

    n_rows = df.shape[0]

    groups = [[f] for f in columns_to_inject] if columns_to_inject is not None else groups

    null_cols = {}  # type: ignore
    for seed_i, group in enumerate(groups):  # type: ignore
        np.random.seed(seed + seed_i)
        replace_mask = np.random.binomial(1, 1 - proportion, n_rows).astype(bool)
        null_cols = merge(null_cols, {feature: df[feature].where(replace_mask) for feature in group})

    null_data = df.assign(**null_cols)

    def p(new_data_set: pd.DataFrame) -> pd.DataFrame:
        return new_data_set

    p.__doc__ = learner_pred_fn_docstring("null_injector")

    log = {'null_injector': {
        "columns_to_inject": columns_to_inject,
        "proportion": proportion,
        "groups": groups
    }}

    return p, null_data, log


null_injector.__doc__ += learner_return_docstring("Null Injector")


@curry
@log_learner_time(learner_name='missing_warner')
def missing_warner(df: pd.DataFrame, cols_list: List[str],
                   new_column_name: str = "has_unexpected_missing",
                   detailed_warning: bool = False,
                   detailed_column_name: Optional[str] = None) -> LearnerReturnType:
    """
    Creates a new column to warn about rows that columns that don't have missing in the training set
    but have missing on the scoring

    Parameters
    ----------

    df : pandas.DataFrame
        A Pandas' DataFrame.

    cols_list : list of str
        List of columns to consider when evaluating missingness

    new_column_name : str
        Name of the column created to alert the existence of missing values
    """

    assert ((detailed_warning and detailed_column_name) or ((not detailed_warning) and (
        not detailed_column_name))), "Either detailed_warning and detailed_column_name " \
                                     "should be defined or both should be False."

    df_selected = df[cols_list]
    cols_without_missing = df_selected.loc[:, df_selected.isna().sum(axis=0) == 0].columns.tolist()

    def p(dataset: pd.DataFrame) -> pd.DataFrame:
        def detailed_assignment(df: pd.DataFrame, cols_to_check: List[str]) -> np.array:
            cols_with_missing = np.array([np.where(df[col].isna(), col, "") for col in cols_to_check]).T
            missing_by_row_list = np.array([list(filter(None, x)) for x in cols_with_missing]).reshape(-1, 1)
            if missing_by_row_list.size == 0:
                return np.empty((df.shape[0], 0)).tolist()
            else:
                return missing_by_row_list

        new_dataset = dataset.assign(**{new_column_name: lambda df: df[cols_without_missing].isna().sum(axis=1) > 0})
        if detailed_warning and detailed_column_name:
            missing_by_row_list = detailed_assignment(new_dataset, cols_without_missing)
            return new_dataset.assign(**{detailed_column_name: missing_by_row_list})
        else:
            return new_dataset

    p.__doc__ = learner_pred_fn_docstring("missing_warner")

    log = {"missing_warner": {
        "cols_list": cols_list,
        "cols_without_missing": cols_without_missing}
    }

    return p, df, log


missing_warner.__doc__ += learner_return_docstring("Missing Alerter")
