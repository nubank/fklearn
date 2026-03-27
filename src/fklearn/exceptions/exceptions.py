from typing import Any, Dict, List


class MultipleTreatmentsError(Exception):
    def __init__(
        self, msg: str = "Data contains multiple treatments.", *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(msg, *args, **kwargs)


class MissingControlError(Exception):
    def __init__(
        self, msg: str = "Data does not contain the specified control.", *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(msg, *args, **kwargs)


class MissingTreatmentError(Exception):
    def __init__(
        self, msg: str = "Data does not contain the specified treatment.", *args: List[Any], **kwargs: Dict[str, Any]
    ) -> None:
        super().__init__(msg, *args, **kwargs)
