"""MVC Main Window Controller for ML Face Recognition Project"""

from ui.main_model import MainModel
from ui.main_view import MainView

class MainController:
    def __init__(self, labels):
        self.model = MainModel()
        self.model.labels = labels
        self.main_view = MainView(self)
        

    def set_data(self, data):
        self.model.set_data(data)
        print("Data updated:", self.model.data)

    def mainloop(self):
        self.main_view.mainloop()

    def set_state(self, state):
        """Set the state of the UI to the specified condition"""
        pass

   #def set_ui_state(window, state, face_name=None, certainty=None):
   #     pass
