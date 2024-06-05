
import tkinter as tk


class TwoColumnGUI:
    def __init__(self, master):
        self.master = master
        self.master.title("Two Column GUI")

        # Create the left column widgets
        self.label1 = tk.Label(self.master, text="Label 1:")
        self.label1.grid(row=0, column=0, padx=10, pady=10, sticky=tk.W)

        self.button1 = tk.Button(self.master, text="Button 1")
        self.button1.grid(row=0, column=1, padx=10, pady=10, sticky=tk.W)

        # Create the right column widgets
        self.label2 = tk.Label(self.master, text="Label 2:")
        self.label2.grid(row=1, column=0, padx=10, pady=10, sticky=tk.W)

        self.button2 = tk.Button(self.master, text="Button 2")
        self.button2.grid(row=1, column=1, padx=10, pady=10, sticky=tk.W)


def main():
    root = tk.Tk()
    gui = TwoColumnGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
