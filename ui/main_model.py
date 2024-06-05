"""MVC Main Window Model for ML Face Recognition Project"""


class MainModel:
    def __init__(self):
        self._labels = ""

    @property
    def labels(self):
        return self._labels

    @labels.setter
    def labels(self, value):
        if value is None or not isinstance(value, list):
            raise ValueError("Value must be a list that is not None")
        self._labels = value
