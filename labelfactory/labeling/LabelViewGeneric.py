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

    def __init__(self, categories, master=None, cat_model='single',
                 parent_cat={}, datatype='url', text2label=None):

        tk.Frame.__init__(self, master)
        if datatype == 'url':
            master.wm_attributes("-topmost", 1)

        self.grid()

        # Initializations
        self.mybuttons = None
        self.url_label = None
        self.main_label = None
        self.cat_model = cat_model
        self.parent_cat = parent_cat
        self.datatype = datatype
        self.text2label = text2label

        self.createWidgets(categories)

    def update_guilabel(self, new_label):
        """ Updates the Url label to show.

           Args:
                new_label: url to put as label in the window.
                tex2label: A method to capture the text to be shown in the
                           labelling window
        """

        if self.cat_model == 'single':
            self.url_label.configure(text=new_label)
        else:
            self.url_label['state'] = 'normal'
            self.url_label.delete(1.0, tk.END)
            if self.datatype == 'txt':
                if self.text2label is None:
                    # The content of "new_label" is the text to show.
                    self.url_label.insert(tk.END, new_label)
                else:
                    # The content of "new_label" is the text to show.
                    self.url_label.insert(tk.END, self.text2label(new_label))
            self.url_label['state'] = 'disabled'

        self.master.update()

        # This is the old version. Not needed because new_label is not None
        # if new_label:
        #     self.url_label.configure(text=new_label)
        #     self.master.update()
        # else:
        #     self.master.destroy()

    def start_gui(self, new_label):
        """ Starts the GUI, puting the initial Url to show.

           Args:
                new_label: url to put as label in the window.
                           If None the GUI is destroyed
                tex2label: A method to capture the text to be shown in the
                           labelling window
    """

        if new_label:
            if self.cat_model == 'single':
                self.url_label.configure(text=new_label)
            else:
                self.url_label['state'] = 'normal'
                self.url_label.delete(1.0, tk.END)
                if self.datatype == 'txt':
                    if self.text2label is None:
                        # The content of "new_label" is the text to show.
                        self.url_label.insert(tk.END, new_label)
                    else:
                        # The content of "new_label" is the text to show.
                        self.url_label.insert(
                            tk.END, self.text2label(new_label))
                self.url_label['state'] = 'disabled'

            self.mainloop()
        else:
            self.master.destroy()

    def createWidgets(self, categories):

        """ Create the labeling window with one button per category.
        """

        # The configurations is arbitrary:
        n_cat = len(categories)

        if n_cat < 10:

            # Maybe I should set this as a configurable parameter.
            n_cols = 3

            # Create the main label
            self.main_label = tk.Label(
                self, text=" Select the correct labels ")
            self.main_label.grid(
                row=0, column=0, columnspan=n_cols, sticky=tk.E+tk.W)

            # Create the url label but without the text yet.
            r = 1
            if self.cat_model == 'single':
                self.url_label = tk.Label(self)
                self.url_label.grid(
                    row=r, column=0, columnspan=n_cols, sticky=tk.E+tk.W)
                r += 1

            # Create the collection of buttons
            c = 0
            self.mybuttons = {}
            for class_name in categories:

                # This is just an assignment of a button at object 'self', at
                # the attribute with the name contained in class_name.
                self.mybuttons[class_name] = tk.Button(
                    self, text=class_name, bg='magenta',
                    activebackground='red', activeforeground='green',
                    disabledforeground='cyan', relief=tk.RAISED)
                self.mybuttons[class_name].grid(row=r, column=c)
                c = (c + 1) % 3
                r = r + (c == 0)

            # Error button
            self.mybuttons['error'] = tk.Button(self, relief=tk.RAISED)
            self.mybuttons['error']["text"] = "Error"
            self.mybuttons['error'].grid(row=r, column=c)

            if self.cat_model == 'multi':

                # Add one more buttom to finish labeling
                c = (c + 1) % 3
                r = r + (c == 0)
                self.mybuttons['end'] = tk.Button(self, text="END",
                                                  relief=tk.RAISED)
                self.mybuttons['end'].grid(row=r, column=c, sticky=tk.W)

                # Create the main label
                self.url_label = tk.Text(
                    self, height=30, width=120, wrap=tk.WORD, relief=tk.SUNKEN,
                    bg='white')
                self.url_label.grid(
                    row=r+1, column=0, rowspan=40, columnspan=120)

        else:

            # Get list of root categories:
            root_cats = [c for c, p in self.parent_cat.items() if p is None]

            # Get subcats for each root category
            subcats = {}
            for rc in root_cats:
                subcats[rc] = [c for c, p in self.parent_cat.items()
                               if p == rc]

            # maxumum number of columns:
            n_cols = max([len(s) for p, s in subcats.items()])

            # Create the main label
            self.main_label = tk.Label(
                self, text=" Select the correct labels ")
            self.main_label.grid(
                row=0, column=0, columnspan=n_cols, sticky=tk.E+tk.W)

            # Create the url label but without the text yet.
            r = 1
            if self.cat_model == 'single':
                self.url_label = tk.Label(self)
                self.url_label.grid(
                    row=r, column=0, columnspan=n_cols, sticky=tk.E+tk.W)
                r += 1

            # Create the collection of buttons
            self.mybuttons = {}
            for rc, sc in subcats.items():

                c = 0
                # Create button for the root class
                self.mybuttons[rc] = tk.Button(self, text=rc, relief=tk.RAISED)
                self.mybuttons[rc].grid(row=r, column=c, sticky=tk.W)

                for class_name in sc:

                    c += 1
                    # Create button for the subcategories of the root class
                    self.mybuttons[class_name] = tk.Button(
                        self, text=class_name, relief=tk.RAISED)
                    self.mybuttons[class_name].grid(row=r, column=c,
                                                    sticky=tk.W)

                r += 1

            # Error button
            self.mybuttons['error'] = tk.Button(self, text="Error",
                                                relief=tk.RAISED)
            self.mybuttons['error'].grid(row=r, column=c, sticky=tk.W)

            if self.cat_model == 'multi':

                # Add one more buttom to finish labeling
                c += 1
                self.mybuttons['end'] = tk.Button(self, text="END",
                                                  relief=tk.RAISED)
                self.mybuttons['end'].grid(row=r, column=c, sticky=tk.W)

                # Create the main label
                self.url_label = tk.Text(
                    self, height=30, width=120, wrap=tk.WORD, relief=tk.SUNKEN,
                    bg='white')
                self.url_label.grid(
                    row=r+1, column=0, rowspan=40, columnspan=120)
 
