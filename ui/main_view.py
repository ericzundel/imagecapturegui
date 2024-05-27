"""MVC Main Window View for ML Face Recognition Project"""

import tkinter as tk


class MainView:
    def __init__(self, controller):
        """ Create a UI with two columns"""

        self.controller = controller

        self.root = tk.Tk()
        self.root.title("ML/Engineering Concepts")

        # Left column with buttons
        self.back_button = tk.Button(self.root, text="Back")
        self.back_button.pack(side=tk.LEFT, padx=2, pady=2)

        self.pane = tk.Frame(self.root, highlightbackground="blue", highlightthickness=2)
        self.pane.pack(fill=tk.BOTH)

        # Right column with list and button
        #self.right_frame = tk.Frame(root)
        #self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        #self.listbox = tk.Listbox(self.right_frame)
        #self.listbox.pack(side=tk.TOP, pady=5)

    def update_name_selection():
        # This returns a tuple containing the indices (= the position)
        # of the items selected by the user.
        indices = listbox.curselection()
        listbox_item = None            
        if len(indices > 0):
            self.listbox_item = listbox.get(i)
            controller.set_selected_name(listbox_item)

    def mainloop(self):
        self.root.mainloop()
        
