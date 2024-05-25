"""MVC View for ML Face Recognition Project"""

import tkinter as tk


class View:
    def __init__(self, root, controller):
        """ Create a UI with two columns"""

        
        self.controller = controller

        self.root = root
        self.root.title("Two Column Layout")

        # Left column with buttons
        self.left_frame = tk.Frame(root)
        self.left_frame.pack(side=tk.LEFT, padx=10, pady=10)

        self.button1 = tk.Button(self.left_frame, text="Button 1")
        self.button1.pack(side=tk.TOP, pady=5)

        self.button2 = tk.Button(self.left_frame, text="Button 2")
        self.button2.pack(side=tk.TOP, pady=5)

        # Right column with list and button
        self.right_frame = tk.Frame(root)
        self.right_frame.pack(side=tk.RIGHT, padx=10, pady=10)

        self.listbox = tk.Listbox(self.right_frame)
        self.listbox.pack(side=tk.TOP, pady=5)

    def update_name_selection():
            # This returns a tuple containing the indices (= the position)
            # of the items selected by the user.
            indices = listbox.curselection()
            listbox_item = None            
            if len(indices > 0):
                self.listbox_item = listbox.get(i)
            controller.set_selected_name(listbox_item)
        
