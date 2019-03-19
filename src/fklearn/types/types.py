from datetime import datetime
from typing import Any, Callable, Dict, List, Tuple, Union

import pandas as pd

# Date type (used to filter datetime columns in dataframes)
DateType = Union[pd.Period, datetime, str]

# Log types
LogType = Dict[str, Any]
LogListType = List[LogType]
ListLogListType = List[LogListType]

# Learner types
PredictFnType = Callable[[pd.DataFrame], pd.DataFrame]
LearnerLogType = Dict[str, LogType]
LearnerReturnType = Tuple[PredictFnType, pd.DataFrame, LearnerLogType]

UncurriedLearnerFnType = Callable[..., LearnerReturnType]
LearnerFnType = Callable[[pd.DataFrame], LearnerReturnType]

# Evaluator types
EvalReturnType = Dict[str, Union[float, Dict]]
UncurriedEvalFnType = Callable[..., EvalReturnType]
EvalFnType = Callable[[pd.DataFrame], EvalReturnType]

# Splitter types
FoldType = List[Tuple[pd.Index, List[pd.Index]]]
SplitterReturnType = Tuple[FoldType, LogListType]
SplitterFnType = Callable[[pd.DataFrame], SplitterReturnType]

# Validator types
ValidatorReturnType = Dict[str, Union[LogType, LogListType]]

# Extractor types
ExtractorFnType = Callable[[str], float]
