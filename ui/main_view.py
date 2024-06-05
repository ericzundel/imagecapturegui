"""MVC Main Window View for ML Face Recognition Project"""

import tkinter as tk


class MainView:
    def __init__(self, controller):
        """ Create a UI with two columns"""

        self.controller = controller

        self.root = tk.Tk()
        self._min_height = 400
        self._min_width = 400
        self.root.geometry("400x400")
        self.root.minsize = (self._min_width, self._min_height)

        self.build_ui()

    def build_ui(self):
        self.root.title("ML/Engineering Concepts")

        # top has a pane with a back button and a status message

        self.status_pane = tk.Frame(
            self.root, highlightbackground="blue", highlightthickness=2)
        self.status_pane.pack(side=tk.TOP, fill=tk.BOTH)
        self.back_button = tk.Button(self.status_pane, text="Back")
        self.show_back_button(True)

        self.status_label = tk.Label(self.status_pane,
                                     text="Waiting for Capture")
        self.status_label.pack(fill=tk.Y)

        # Below that is an area where the sub-views live

        self.subview_pane = tk.Frame(
            self.root, highlightbackground="red", highlightthickness=2)
        self.subview_pane.pack(side=tk.TOP, fill=tk.BOTH)

        self.debug_set_state_capture_button = tk.Button(self.subview_pane,
                                                        text="Toggle State to Capturing",
                                                        command=self.controller.set_state_capturing_cb)
        self.debug_set_state_capture_button.pack()

        self.debug_set_state_waiting_button = tk.Button(self.subview_pane,
                                                        text="Toggle State to Waiting",
                                                        command=self.controller.set_state_waiting_cb)

        self.debug_set_state_waiting_button.pack()
        self.debug_set_state_naming_button = tk.Button(self.subview_pane,
                                                       text="Toggle State to Naming",
                                                       command=self.controller.set_state_naming_cb)
        self.debug_set_state_naming_button.pack()

        # Right column with list and button
        # self.right_frame = tk.Frame(root)
        # self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        # self.listbox = tk.Listbox(self.right_frame)
        # self.listbox.pack(side=tk.TOP, pady=5)

    def show_back_button(self, visible):
        if (visible):
            self.back_button.pack(side=tk.LEFT, padx=2, pady=2)
        else:
            self.back_button.pack_forget()

    def set_status_label(self, value):
        self.status_label.config(text=value)

    def mainloop(self):
        self.root.mainloop()

    def close_window(self):
        self.root.close()
