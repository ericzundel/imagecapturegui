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
        """State is one of "WAITING", "CAPTURING", or "NAMING"""
        if (state is "WAITING"):
            self.main_view.show_back_button(False)
            self.main_view.set_status_label("Waiting for Capture")
        elif (state is "CAPTURING"):
            self.main_view.show_back_button(True)
            self.main_view.set_status_label("Capturing image...")
        elif (state is "NAMING"):
            self.main_view.show_back_button(True)
            self.main_view.set_status_label("Name the Image")
        else:
            raise Exception("Invalid State passed: %s" % state)

    def set_state_capturing_cb(self):
        print("Capturing")
        self.set_state("CAPTURING")

    def set_state_waiting_cb(self):
        print("Waiting")
        self.set_state("WAITING")

    def set_state_naming_cb(self):
        self.set_state("NAMING")

    def close_window(self):
        self.main_view.close_window()
