import pandas as pd
from typing import List
from toolz import curry


@curry
def quantile_partitioner(series: pd.Series, segments: int) -> List:
    """
    Returns the bins using a quantile-based approach.

    Parameters
    ----------
    series : Pandas' Series
        A Pandas' Series to partition by quantiles.

    segments : int
        The number of bins to be computed.


    Returns
    ----------
    bins: Array
        The bins to partition the series argument.
    """

    _, bins = pd.qcut(series, q=segments, retbins=True)

    return bins
