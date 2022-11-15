import pandas as pd
from typing import List
from toolz import curry

@curry
def quantile_partitioner(series: pd.Series, segments: int = 10) -> List:
    
    _, bins = pd.qcut(series, q=segments, retbins=True)
    
    return bins
