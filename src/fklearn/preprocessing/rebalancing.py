import pandas as pd
from toolz import curry, partial


@curry
def rebalance_by_categorical(dataset: pd.DataFrame, categ_column: str, max_lines_by_categ: int = None,
                             seed: int = 1) -> pd.DataFrame:
    """
    Resample dataset so that the result contains the same number of lines per category in  categ_column.

    Parameters
    ----------
    dataset: pandas.DataFrame
        A Pandas' DataFrame with an categ_column column

    categ_column: str
        The name of the categorical column

    max_lines_by_categ: int (default None)
        The maximum number of lines by category. If None it will be set to the number of lines for the smallest category

    seed: int (default 1)
        Random state for consistency.

    Returns
    ----------
    rebalanced_dataset : pandas.DataFrame
        A dataset with fewer lines than dataset, but with the same number of lines per category in categ_column
    """

    categs = dataset[categ_column].value_counts().to_dict()
    max_lines_by_categ = max_lines_by_categ if max_lines_by_categ else min(categs.values())

    return pd.concat([(dataset
                       .loc[dataset[categ_column] == categ, :]
                       .sample(max_lines_by_categ, random_state=seed))
                      for categ in list(categs.keys())])


@curry
def rebalance_by_continuous(dataset: pd.DataFrame, continuous_column: str, buckets: int, max_lines_by_categ: int = None,
                            by_quantile: bool = False, seed: int = 1) -> pd.DataFrame:
    """
    Resample dataset so that the result contains the same number of lines per bucket in a continuous column.

    Parameters
    ----------
    dataset: pandas.DataFrame
        A Pandas' DataFrame with an categ_column column

    continuous_column: str
        The name of the continuous column

    buckets: int
        The number of buckets to split the continuous column into

    max_lines_by_categ: int (default None)
        The maximum number of lines by category. If None it will be set to the number of lines for the smallest category

    by_quantile: bool (default False)
        If True, uses pd.qcut instead of pd.cut to get the buckets from the continuous column

    seed: int (default 1)
        Random state for consistency.

    Returns
    ----------
    rebalanced_dataset : pandas.DataFrame
        A dataset with fewer lines than dataset, but with the same number of lines per category in  categ_column
    """

    bin_fn = partial(pd.qcut, q=buckets, duplicates="drop") if by_quantile else partial(pd.cut, bins=buckets)

    return (dataset
            .assign(bins=bin_fn(dataset[continuous_column]))
            .pipe(rebalance_by_categorical(categ_column="bins",
                                           max_lines_by_categ=max_lines_by_categ,
                                           seed=seed))
            .drop(columns=["bins"]))
