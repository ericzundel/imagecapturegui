"""MVC View for ML Face Recognition Project"""

import tkinter as tk


class View:
    def __init__(self, master, controller):
        self.controller = controller
        self.entry = tk.Entry(master)
        self.entry.pack()
        self.button = tk.Button(master, text="Update", command=self.update_data)
        self.button.pack()

    def update_data(self):
        data = self.entry.get()
        self.controller.set_data(data)
