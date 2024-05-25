"""MVC Controller for ML Face Recognition Project"""

import UiModel
import UiView

class Controller:
    def __init__(self, master):
        self.model = UiModel()
        self.view = UiView(master, self)

    def set_data(self, data):
        self.model.set_data(data)
        print("Data updated:", self.model.data)
