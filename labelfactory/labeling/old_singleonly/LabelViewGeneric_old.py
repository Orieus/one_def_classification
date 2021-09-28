# -*- coding: utf-8 -*-
import sys
if sys.version_info.major == 3:
    import tkinter as tk
else:
    import Tkinter as tk


class LabelViewGeneric(tk.Frame):
    """Encapsulates of all the GUI logic.

    Represents a window with labels and buttons

    Attributes:
        master: where to open the Frame, by deafult root window
        main_label: Main Label indicating the task.
        url_label : Url Label we are considering now.
        yes       : Button to indicate that the answer to main_label is "yes"
                    for url_label
        no        : Button to indicate that the answer to main_label is "no"
                    for url_label
        error     : Button to indicate that webpage for url_label is not
                    correct
    """

    def __init__(self, categories, master=None):

        tk.Frame.__init__(self, master)
        master.wm_attributes("-topmost", 1)
        self.grid()

        self.createWidgets(categories)

    def update_guilabel(self, new_label):
        """ Updates the Url label to show.

           If url is None then the window is destroyed. Nothing left to label

           Args:
                new_label: url to put as label in the window.
                If None the GUI is destroyed
        """

        self.url_label.configure(text=new_label)
        self.master.update()

        # This is the old version. Not needed because new_label is not None
        # if new_label:
        #     self.url_label.configure(text=new_label)
        #     self.master.update()
        # else:
        #     self.master.destroy()

    def start_gui(self, new_label):
        """ Starts the GUI, puting the initial Url to show.

           If url is None then the window is destroyed. Nothing left to label

           Args:
                new_label: url to put as label in the window.
                If None the GUI is not started
        """

        if new_label:
            self.url_label.configure(text=new_label)
            self.mainloop()
        else:
            self.master.destroy()

    def createWidgets(self, categories):

        """ Create the labeling window with one button per category.
        """

        # Maybe I should set this as a configurable parameter.
        n_cols = 3

        # Create the main label
        self.main_label = tk.Label(self,
                                   text=" Etiquete la página según su tarea ")
        self.main_label.grid(row=0, column=0, columnspan=n_cols,
                             sticky=tk.E+tk.W)

        # Create the url label but without the text yet.
        self.url_label = tk.Label(self)
        self.url_label.grid(row=1, column=0, columnspan=n_cols,
                            sticky=tk.E+tk.W)

        # Create the collection of buttons
        r = 2
        c = 0
        for class_name in categories:

            # This is just an assignment of a button at object 'self', at
            # the attribute with the name contained in class_name.
            setattr(self, class_name, tk.Button(self, text=class_name,
                                                bg='magenta',
                                                activebackground='red',
                                                activeforeground='green',
                                                disabledforeground='cyan'))
            getattr(self, class_name).grid(row=r, column=c)
            c = (c + 1) % 3
            r = r + (c == 0)

        # self.other = tk.Button(self)
        # self.other["text"] = "Other"
        # self.other.grid(row=r, column=c)
        # c = (c + 1) % 3
        # r = r + (c == 0)

        self.error = tk.Button(self)
        self.error["text"] = "Error"
        self.error.grid(row=r, column=c)
