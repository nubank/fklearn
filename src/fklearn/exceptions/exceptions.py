class MultipleTreatmentsError(Exception):
    def __init__(self, msg: str ="Data contains multiple treatments.", *args, **kwargs):
        super().__init__(msg, *args, **kwargs)


class MissingControlError(Exception):
    def __init__(
        self, msg: str ="Data does not contain the specified control.", *args, **kwargs
    ):
        super().__init__(msg, *args, **kwargs)


class MissingTreatmentError(Exception):
    def __init__(
        self, msg: str ="Data does not contain the specified treatment.", *args, **kwargs
    ):
        super().__init__(msg, *args, **kwargs)
