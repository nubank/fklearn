from typing import Any, Dict, List


class MultipleTreatmentsError(Exception):
    def __init__(
        self,
        msg: str = "Data contains multiple treatments.",
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(msg, *args, **kwargs)


class MissingControlError(Exception):
    def __init__(
        self,
        msg: str = "Data does not contain the specified control.",
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(msg, *args, **kwargs)


class MissingTreatmentError(Exception):
    def __init__(
        self,
        msg: str = "Data does not contain the specified treatment.",
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(msg, *args, **kwargs)


class MissingColumnsError(Exception):
    """Raised when required DataFrame columns are missing."""

    def __init__(
        self,
        missing_columns: List[str],
        available_columns: List[str],
        msg: str = None,
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        if msg is None:
            msg = f"Missing required columns: {missing_columns}. Available: {available_columns}"
        self.missing_columns = missing_columns
        self.available_columns = available_columns
        super().__init__(msg, *args, **kwargs)


class EmptyDataFrameError(Exception):
    """Raised when DataFrame is empty but non-empty is required."""

    def __init__(
        self,
        msg: str = "DataFrame is empty but non-empty data is required.",
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(msg, *args, **kwargs)


class InvalidParameterRangeError(Exception):
    """Raised when numeric parameter is outside valid range."""

    def __init__(
        self,
        param_name: str,
        value: Any,
        min_value: Any = None,
        max_value: Any = None,
        msg: str = None,
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        if msg is None:
            bounds = []
            if min_value is not None:
                bounds.append(f">= {min_value}")
            if max_value is not None:
                bounds.append(f"<= {max_value}")
            bounds_str = " and ".join(bounds) if bounds else "within valid range"
            msg = f"Parameter '{param_name}' has value {value}, but must be {bounds_str}."
        self.param_name = param_name
        self.value = value
        self.min_value = min_value
        self.max_value = max_value
        super().__init__(msg, *args, **kwargs)


class InvalidParameterValueError(Exception):
    """Raised when parameter value is not in allowed set."""

    def __init__(
        self,
        param_name: str,
        value: Any,
        allowed_values: List[Any],
        msg: str = None,
        *args: List[Any],
        **kwargs: Dict[str, Any]
    ) -> None:
        if msg is None:
            msg = f"Parameter '{param_name}' has value '{value}', but must be one of: {allowed_values}."
        self.param_name = param_name
        self.value = value
        self.allowed_values = allowed_values
        super().__init__(msg, *args, **kwargs)
