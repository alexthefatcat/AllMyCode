# -*- coding: utf-8 -*-

"""Created on Fri Apr 26 12:10:35 2019@author: milroa1"""

"""


    A frame is a widget that displays just as a simple rectangle. Frames are primarily used as a container for other widgets,
             which are under the control of a geometry manager such as grid.
    
    A label is a widget that displays text or images, typically that the user will just view but not otherwise interact with. 
            Labels are used for such things as identifying controls or other parts of the user interface, providing textual feedback or results, etc.
    
    A button, unlike a frame or label, is very much designed for the user to interact with, and in particular, press to perform some action.
             Like labels, they can display text or images, but also have a whole range of new options used to control their behavior.
    
            A checkbutton is like a regular button, except that not only can the user press it, which will invoke a command callback,
                    but it also holds a binary value of some kind (i.e. a toggle). Checkbuttons are used all the time when a user is asked to choose between e.g. two different values for an option.
    
            A radiobutton lets you choose between one of a number of mutually exclusive choices; unlike a checkbutton, it is not 
                    limited to just two choices. Radiobuttons are always used together in a set, and are a good option when the number of choices is fairly small, 
    
            An entry presents the user with a single line text field that they can use to type in a string value. These can be just 
                    about anything: their name, a city, a password, social security number, and so on.
    
            A combobox combines an entry with a list of choices available to the user. This lets them either choose from a set of values
                    you've provided (e.g. typical settings), but also put in their own value (e.g. for less common cases you don't want to include in the list).


A text widget provides users with an area so that they can enter multiple lines of text. Text widgets are part of the classic Tk widgets, not the themed Tk widgets.
A scrollbar helps the user to see all parts of another widget, whose content is typically much larger than what can be shown in the available screen space. 
A progressbar widget provides a way to give feedback to the user about the progress of a lengthy operation.
A spinbox widget allows users to choose numbers (or in fact, items from an arbitrary list).

A treeview widget can display and allow browsing through a hierarchy of items, and can show one or more attributes of each item as columns to the right of the tree.


#######################################################################################################################################################################
frame = ttk.Frame(parent)
frame['padding'] = (5,10)
frame['borderwidth'] = 2
frame['relief'] = 'sunken'


label = ttk.Label(parent, text='Full name:')
resultsContents = StringVar()
label['textvariable'] = resultsContents
resultsContents.set('New value to display')
image = PhotoImage(file='myimage.gif')
label['image'] = image

button = ttk.Button(parent, text='Okay', command=submitForm)
button.state(['disabled'])            # set the disabled flag, disabling the button
button.state(['!disabled'])   

check = ttk.Checkbutton(parent, text='Use Metric', 
	    command=metricChanged, variable=measureSystem,
	    onvalue='metric', offvalue='imperial')

phone = StringVar()
home = ttk.Radiobutton(parent, text='Home', variable=phone, value='home')
office = ttk.Radiobutton(parent, text='Office', variable=phone, value='office')
cell = ttk.Radiobutton(parent, text='Mobile', variable=phone, value='cell')

username = StringVar()
name = ttk.Entry(parent, textvariable=username)

countryvar = StringVar()
country = ttk.Combobox(parent, textvariable=countryvar)

country.bind('<<ComboboxSelected>>', function)
country['values'] = ('USA', 'Canada', 'Australia')
"""

#https://tkdocs.com/tutorial/grid.html
#%%################################################################################################################################################################


import tkinter as tk
from tkinter import Label,Frame,Button,Entry,Checkbutton,Menu,Canvas,PhotoImage
from tkinter import BOTTOM,LEFT,X,Y,E,TOP,SUNKEN,W
from PIL import ImageTk, Image
from tkinter import filedialog
import os

def tk_img(filepath):
    if filepath.endswith(".jpg") or filepath.endswith(".png"):
        return ImageTk.PhotoImage(Image.open(filepath))
    return PhotoImage(filepath)# gifs I think work with this

def printbutton(*args):
    """    frame.bind("<Button-1>", printbutton("left"))    """
    def print_temp(event):
        print(*args)
    return print_temp

#%%############################################################################################
##################################################################################
# Most Basic 0

window = tk.Tk() # create window

window.mainloop()#keep window open 

##################################################################################
## Baisc Window 1

window = tk.Tk()
window.title("Example 1")

label = Label(window, text="This is label")
label.pack()# just packs text it in there
window.mainloop() # puts it inf loop so window continously displays
##################################################################################
##  Window 2

window = tk.Tk() # create window
window.title("Example 2")

topframe = Frame(window)
topframe.pack()
bottomframe = Frame(window)
bottomframe.pack(side=BOTTOM)
buttons={}
for i, clr in enumerate(["red","red","red","purple"]):
    buttons[i] =  Button(topframe if clr =="purple" else bottomframe, text=f"Button-{i}", fg=clr)
      
for i,b in buttons.items():
    if i<3:
       b.pack(side=LEFT)
    else:
       b.pack(side=BOTTOM)

window.mainloop()#keep window open
##################################################################################
## Basic Window 3

window = tk.Tk() # create window
window.title("Example 3")

one =Label(window, text="One", bg="red",fg="white")
one.pack()
two =Label(window, text="Two", bg="green",fg="black")
two.pack(fill=X)
three =Label(window, text="Three", bg="blue",fg="white")
three.pack(side=LEFT,fill=Y)
window.mainloop() 
##################################################################################
## Basic Window 4

window = tk.Tk() # create window
window.title("Example 4")

label_1 = Label(window,text="Name")
label_2 = Label(window,text="Password")
entry_1 = Entry(window)
entry_2 = Entry(window)

label_1.grid(row=0,sticky=E)
label_2.grid(row=1,sticky=E)

entry_1.grid(row=0,column=1)
entry_2.grid(row=1,column=1)

c_button = Checkbutton(window,text="Keep me logged in ")
c_button.grid(columnspan=2)

window.mainloop() 
##################################################################################
# binding fuction 

window = tk.Tk() # create window
window.title("Example 5")
def printdog(event):
    print("Dog!!")
    
button_1 = Button(window, text="bbbuton", )   
button_1.bind("<Button-1>",printdog)#left mouse button
button_1.pack()
window.mainloop() 


##################################################################################



def leftclick(event):
    print("Left")
def rightclick(event):
    print("Right")


window = tk.Tk()
window.title("Example 6")
frame = Frame(window, width=300, height=250)
frame.bind("<Button-1>", leftclick)
frame.bind("<Button-3>",printbutton("Right")) # custom code at top written for this
frame.pack()

window.mainloop() 
















##################################################################################

class BuckysBall:
    def __init__(self,master):
        frame = Frame(master)
        frame.pack()
        
        self.printbutton = Button(frame, text = "Print-Message", command=self.printMessage)
        self.printbutton.pack(side=LEFT)
        
        self.quitButton = Button(frame, text="Quit", command=frame.quit) # quit wasnt working
        self.quitButton.pack(side=LEFT)
        
    def printMessage(self):
        print("WOW this worked")
window = tk.Tk()
b = BuckysBall(window)
window.mainloop() 

##################################################################################

# Maybe below would be better sytax
#with tkwindow() as window:    
#    with Menu(window) as menu:
#        window.config(menu=menu)
#        with Menu(menu, tearoff=0) as submenu:# this gets ride of the "----" at top
#            menu.add_cascade(label="File",menu=submenu)
#            submenu.add_command(label="New Project",command=donothing)
#            submenu.add_command(label="2nd one",command=donothing)
#            submenu.add_separator()
#            submenu.add_command(label="Exit",command=donothing)
#        with Menu(menu) as editmenu:# this gets ride of the "----" at top    
#            menu.add_cascade(label="Other",menu=editmenu)
#            editmenu.add_command(label="Redo",command=donothing)
#    with Frame(window, bg="blue",pack={"side":TOP,"fill":X}) as toolbar:
#        insertButt = Button(toolbar,text="Insert Image",command=donothing,pack={"side":LEFT, "padx":2,"pady":2} )   
#        printButt  = Button(toolbar,text="Insert Image",command=donothing,pack={"side":LEFT, "padx":2,"pady":2} )   
#
#




def donothing():
    print("ok ok I wont")


window = tk.Tk()

menu = Menu(window)
window.config(menu=menu)
submenu = Menu(menu, tearoff=0)# this gets ride of the "----" at top
menu.add_cascade(label="File",menu=submenu)
submenu.add_command(label="New Project",command=donothing)
submenu.add_command(label="2nd one",command=donothing)
submenu.add_separator()
submenu.add_command(label="Exit",command=donothing)

editmenu= Menu(menu)
menu.add_cascade(label="Other",menu=editmenu)
editmenu.add_command(label="Redo",command=donothing)

toolbar = Frame(window, bg="blue")

insertButt = Button(toolbar,text="Insert Image",command=donothing)
insertButt.pack(side=LEFT, padx=2,pady=2)
printButt = Button(toolbar,text="Print",command=donothing)
printButt.pack(side=LEFT, padx=2,pady=2)
toolbar.pack(side=TOP,fill=X)

##status bar at the bottom
status = Label(window,text="Blank",bd=1,relief=SUNKEN, anchor=W)
status.pack(side=BOTTOM,fill=X)

window.mainloop()


##################################################################################

# message boxs

import tkinter.messagebox as msgbox
window = tk.Tk()
#tk.messagebox.showinfo("WIINDOW TITLE","DOGS AND CATS")
msgbox.showinfo("info name","This is a Test")
ans = msgbox.askquestion("Q1","WHAT IS A COW")
if ans =="yes":
    print("cowcowo cow")
 
window.mainloop()

##################################################################################

window = tk.Tk()

canvas = Canvas(window,width=200,height=100)
canvas.pack()
blackline = canvas.create_line( 0,  0,200,50)
redline   = canvas.create_line( 0,100,200,50,fill="red")
greenbox  = canvas.create_rectangle(25,25,120,60,fill="green")
canvas.delete(redline)
window.mainloop()
##################################################################################

path = r"C:\Users\milroa1\Downloads\New2_7\_106598514_gettyimages-175035140.jpg"
path = r"C:\Users\milroa1\Downloads\DKL.png"



#This creates the main window of an application
window = tk.Tk()

img = tk_img(path)
label_img = tk.Label(window, image = img)
label_img.pack(side = "bottom", fill = "both", expand = "yes")

window.mainloop()

##################################################################################


def ImgGallery(*args):
    class imgbutton:
        def __init__(self,parent,files):
            self.files,self.count = files, 0
            self.nfiles = len(self.files)
            self.img = tk_img(self.files[self.count])
            self.label_img = tk.Label(parent, image = self.img)
            
            self.label_img.bind("<Button-1>", self.shift_image( -1))
            self.label_img.bind("<Button-3>", self.shift_image( +1)) 
            self.label_img.bind("<Left>"    , self.shift_image( -1)) # "<KeyPress>" 
            self.label_img.bind("<Right>"   , self.shift_image( +1)) # "<KeyPress>"  
            self.label_img.bind("<Up>"      , self.shift_image(+10)) 
            self.label_img.bind("<Down>"    , self.shift_image(-10)) 
            self.label_img.bind("<p>"       , self.printfilepath)
            
            self.label_img.focus_force()#focus_set()           
            self.label_img.pack(side="bottom", fill="both", expand="yes")
        def printfilepath(self,*args):
            print(self.files[self.count])
        
        def shift_image(self,shift):
            def mover(args):
                self.count =(self.count+shift) % self.nfiles
                self.draw_img()
            return mover
    
        def draw_img(self):    
            self.img = tk_img(self.files[self.count])
            self.label_img.configure(image=self.img)
            self.label_img.image = self.img
                         
    window = tk.Tk()
    if len(args)==0:
       folder = filedialog.askdirectory() 
    else:
       folder = args[0]  
    files = [folder+"/"+f for f in os.listdir(folder)  if (os.path.isfile(folder+"/"+f) and (f.endswith(".jpg") or f.endswith(".png")))]
    frame = Frame(window, width=300, height =20).place(x=700, y=0)
    imgbutton(frame, files)
    window.mainloop()

ImgGallery()


#%%
################################################
## Dynamic Content

counter = 0 
def counter_label(label):
  counter = 0
  def count():
    global counter
    counter += 1
    label.config(text=str(counter))
    label.after(1000, count)
  count()
 
 
root = tk.Tk()
root.title("Counting Seconds")
label = tk.Label(root, fg="dark green")
label.pack()
counter_label(label)
button = tk.Button(root, text='Stop', width=25, command=root.destroy)
button.pack()
root.mainloop()
#####################################################



window = tk.Tk()
window.title("Example 6")
frame = Frame(window, width=300, height=250)
frame.bind("<Button-1>", printbutton("left"))
frame.bind("<Button-3>", printbutton("right"))
frame.pack()

window.mainloop() 











window = tk.Tk() # create window

window.mainloop()#keep window open 

## Baisc Window 1
window = tk.Tk()
window.title("Example 1")

label = Label(window, text="This is label")
label.pack()# just packs text it in there
window.mainloop() # puts it inf loop so window continously displays

##  Window 2
window = tk.Tk() # create window
window.title("Example 2")

topframe = Frame(window)
topframe.pack()
bottomframe = Frame(window)
bottomframe.pack(side=BOTTOM)
buttons={}
for i, clr in enumerate(["red","red","red","purple"]):
    buttons[i] =  Button(topframe if clr =="purple" else bottomframe, text=f"Button-{i}", fg=clr)

b[0].pack(side=LEFT)
b[1].pack(side=LEFT)
b[2].pack(side=LEFT) 
b[3].pack(side=BOTTOM)   
 


window.mainloop()#keep window open

## Basic Window 3
##################################################################################
window = tk.Tk() # create window
window.title("Example 3")

one =Label(window, text="One", bg="red",fg="white")
one.bind("<Button-3>")
one.pack()
two =Label(window, text="Two", bg="green",fg="black")
two.pack(fill=X)
three =Label(window, text="Three", bg="blue",fg="white")
three.pack(side=LEFT,fill=Y)
window.mainloop() 
##################################################################################

## Basic Window 4
window = tk.Tk() # create window
window.title("Example 4")

label_1 = Label(window,text="Name")
label_2 = Label(window,text="Password")
entry_1 = Entry(window)
entry_2 = Entry(window)

label_1.grid(row=0,sticky=E)
label_2.grid(row=1,sticky=E)

entry_1.grid(row=0,column=1)
entry_2.grid(row=1,column=1)

c_button = Checkbutton(window,text="Keep me logged in ")
c_button.grid(columnspan=2)

window.mainloop() 





##################################################################################




from tkinter import *
from tkinter import ttk

def calculate(*args):
    try:
        value = float(feet.get())
        meters.set(round((0.3048 * float(value) ),6))
    except ValueError:
        pass
    
root = Tk()
root.title("Feet to Meters")

mainframe = ttk.Frame(root, padding="3 3 12 12")
mainframe.grid(column=0, row=0, sticky=(N, W, E, S))
root.columnconfigure(0, weight=1)
root.rowconfigure(0, weight=1)

feet, meters   = StringVar() , StringVar()

feet_entry = ttk.Entry(mainframe, width=7, textvariable=feet)
feet_entry.grid(column=2, row=1, sticky=(W, E))

ttk.Label( mainframe, textvariable=meters).grid(column=2, row=2, sticky=(W, E))
ttk.Button(mainframe, text="Calculate", command=calculate).grid(column=3, row=3, sticky=W)

ttk.Label(mainframe, text="feet"            ).grid(column=3, row=1, sticky=W)
ttk.Label(mainframe, text="is equivalent to").grid(column=1, row=2, sticky=E)
ttk.Label(mainframe, text="meters"          ).grid(column=3, row=2, sticky=W)

for child in mainframe.winfo_children(): 
    child.grid_configure(padx=5, pady=5)

feet_entry.focus()
root.bind('<Return>', calculate)

root.mainloop()


#%%%%%%%%%%%%%



########################################################################
expand_top_left_box = True

from tkinter import *
from tkinter import ttk

root = Tk()

content = ttk.Frame(root, padding=(3,3,12,12))
frame = ttk.Frame(content, borderwidth=5, relief="sunken", width=200, height=100)
namelbl = ttk.Label(content, text="Name")
name = ttk.Entry(content)

onevar   = BooleanVar()
twovar   = BooleanVar()
threevar = BooleanVar()

onevar.set(True)
twovar.set(False)
threevar.set(True)

one = ttk.Checkbutton(content, text="One", variable=onevar, onvalue=True)
two = ttk.Checkbutton(content, text="Two", variable=twovar, onvalue=True)
three = ttk.Checkbutton(content, text="Three", variable=threevar, onvalue=True)
ok = ttk.Button(content, text="Okay")
cancel = ttk.Button(content, text="Cancel")

content.grid(column=0, row=0, sticky=(N, S, E, W))
frame.grid(  column=0, row=0, columnspan=3, rowspan=2, sticky=(N, S, E, W))
namelbl.grid(column=3, row=0, columnspan=2, sticky=(N, W), padx=5)
name.grid(   column=3, row=1, columnspan=2, sticky=(N, E, W), pady=5, padx=5)
one.grid(    column=0, row=3)
two.grid(    column=1, row=3)
three.grid(  column=2, row=3)
ok.grid(     column=3, row=3)
cancel.grid( column=4, row=3)

if expand_top_left_box:
    root.columnconfigure(0, weight=1)
    root.rowconfigure(0, weight=1)
    content.columnconfigure(0, weight=3)
    content.columnconfigure(1, weight=3)
    content.columnconfigure(2, weight=3)
    content.columnconfigure(3, weight=1)
    content.columnconfigure(4, weight=1)
    content.rowconfigure(1, weight=1)

root.mainloop()
#######################################################################
#                    Dynamic Buttons
#######################################################################







import tkinter as tk                
from tkinter import font  as tkfont 

class SampleApp(tk.Tk):

    def __init__(self, *args, **kwargs):
        tk.Tk.__init__(self, *args, **kwargs)

        self.title_font = tkfont.Font(family='Helvetica', size=18, weight="bold", slant="italic")

        # the container is where we'll stack a bunch of frames
        # on top of each other, then the one we want visible
        # will be raised above the others
        container = tk.Frame(self)
        container.pack(side="top", fill="both", expand=True)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        for F in (StartPage, PageOne, PageTwo):
            page_name = F.__name__
            frame = F(parent=container, controller=self)
            self.frames[page_name] = frame

            # put all of the pages in the same location;
            # the one on the top of the stacking order
            # will be the one that is visible.
            frame.grid(row=0, column=0, sticky="nsew")

        self.show_frame("StartPage")

    def show_frame(self, page_name):
        '''Show a frame for the given page name'''
        frame = self.frames[page_name]
        frame.tkraise()


class StartPage(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is the start page", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)

        button1 = tk.Button(self, text="Go to Page One", command=lambda: controller.show_frame("PageOne"))
        button2 = tk.Button(self, text="Go to Page Two",
                            command=lambda: controller.show_frame("PageTwo"))
        button1.pack()
        button2.pack()


class PageOne(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is page 1", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Go to the start page", command=lambda: controller.show_frame("StartPage"))
        button.pack()


class PageTwo(tk.Frame):

    def __init__(self, parent, controller):
        tk.Frame.__init__(self, parent)
        self.controller = controller
        label = tk.Label(self, text="This is page 2", font=controller.title_font)
        label.pack(side="top", fill="x", pady=10)
        button = tk.Button(self, text="Go to the start page",
                           command=lambda: controller.show_frame("StartPage"))
        button.pack()


if __name__ == "__main__":
    app = SampleApp()
    app.mainloop()





##############################################################################################################

from tkinter import *


def raise_frame(frame):
    frame.tkraise()

root = Tk()

f1 = Frame(root)
f2 = Frame(root)
f3 = Frame(root)
f4 = Frame(root)

for frame in (f1, f2, f3, f4):
    frame.grid(row=0, column=0, sticky='news')

Button(f1, text='Go to frame 2', command=lambda:raise_frame(f2)).pack()
Label(f1, text='FRAME 1').pack()

Label(f2, text='FRAME 2').pack()
Button(f2, text='Go to frame 3', command=lambda:raise_frame(f3)).pack()

Label(f3, text='FRAME 3').pack(side='left')
Button(f3, text='Go to frame 4', command=lambda:raise_frame(f4)).pack(side='left')

Label(f4, text='FRAME 4').pack()
Button(f4, text='Goto to frame 1', command=lambda:raise_frame(f1)).pack()

raise_frame(f1)
root.mainloop()
##############################################################################################################





from tkinter import *


def raise_frame(frame):
    frame.tkraise()

root = Tk()

f1 = Frame(root)
f2 = Frame(root)

for frame in (f1, f2):
    frame.grid(row=0, column=0, sticky='news')

Label(f1, text='FRAME 1').pack()
Button(f1, text='Add extra button', command=lambda:raise_frame(f2)).pack()

Label(f2, text='FRAME 1').pack()
Button(f2, text='Add extra button', command=lambda:raise_frame(f1)).pack()
Button(f2, text='Nothing').pack()

raise_frame(f1)
root.mainloop()












###################################################
#  Dynamic Buttons Class add button
###################################################
from tkinter import *
def raise_frame(frame):
    frame.tkraise()

class GUI_Dynamic_Buttons:
    def __init__(self,root,extra_button=False):
        self.root = root
        self.extra_button = extra_button
        #create a frame and display
        self.create_frame() 
        
    def create_frame(self):
       # create a frame(based on settings) and display 
       self.frame = Frame(self.root)
       self.frame.grid(row=0, column=0, sticky='news')
       Label(self.frame, text='FRAME Dynamic Buttons').pack()
       # button to change settings and call this function(so redraw frame)
       Button(self.frame, text='Add extra button', command= self.switch_button_over).pack()
       
       if self.extra_button:
           Button(self.frame, text='Nothing').pack()
       #draw_frame    
       raise_frame(self.frame)  

    def switch_button_over(self,*args):
       self.extra_button = not(self.extra_button)
       self.create_frame() 

root = Tk()
f1 = GUI_Dynamic_Buttons(root)
root.mainloop()

###################################################
#  Dynamic Buttons Fake Pages
###################################################

from tkinter import *
def raise_frame(frame):
    frame.tkraise()
# create settings
# create a frame from settings
# create a button in the frame
# this button calls on a function that changes settings are recreates frame
# rais frame 

class GUI_Dynamic_Buttons:
    def __init__(self,root,pages=5):
        self.root = root
        self.page      = 1
        self.last_page = pages
        #create a frame and display
        self.create_frame() 
        
    def next_page(self,*args):
        next_page = (((self.page)%self.last_page)+1)
        if "string" in args:
            return next_page
        self.page = next_page
        self.create_frame() 
        
    def create_frame(self):
       # create a frame(based on settings) and display 
       self.frame = Frame(self.root)
       self.frame.grid(row=0, column=0, sticky='news')
       Label(self.frame, text=f'Page {self.page}').pack()
       next_page_no = self.next_page("string")
       # button to change settings and call this function(so redraw frame)
       Button(self.frame, text=f'Next Page {next_page_no}', command= self.next_page).pack()
       if self.page==3:
           Button(self.frame, text='Nothing').pack()
       #draw_frame    
       raise_frame(self.frame)  

root = Tk()
f1 = GUI_Dynamic_Buttons(root)
root.mainloop()
######################################################################################################














window = tk.Tk()

class snake:
    def __init__(self):
        self.speed=1
        self.food =0
        self.length=7
        self.location=(4,4)
        self.up=1
        self.down=1
        self.space=[[0 for i in range(15)] for i in range(15)]
        
        
canvas = Canvas(window,width=150,height=150)
canvas.pack()




greenbox  = canvas.create_rectangle(25,25,120,60,fill="black")
window.mainloop()




import tkinter as tk 
import random

# Создаем окно
root = tk.Tk()
# Устанавливаем название окна
root.title("PythonicWay Snake")
 
# Запускаем окно
root.mainloop()




# ширина экрана
WIDTH = 800
# высота экрана
HEIGHT = 600
# Размер сегмента змейки
SEG_SIZE = 20
# Переменная отвечающая за состояние игры
IN_GAME = True




# создаем экземпляр класса Canvas (его мы еще будем использовать) и заливаем все зеленым цветом
c = tk.Canvas(root, width=WIDTH, height=HEIGHT, bg="#003300")
c.grid()
# Наводим фокус на Canvas, чтобы мы могли ловить нажатия клавиш
c.focus_set()



#%%%
# sendex

from tkinter import *
class Window(Frame):
    def __init__(self,master=None):
        Frame.__init__(self,master)
        self.master = master
        self.init_window()
    def init_window(self):
        self.master.title("GUI")
        self.pack(fill=BOTH,expand=1)
        quitButton = Button(self, text="QUIT-ME")
        quitButton.place(x=0,y=0)
        
root = Tk()
root.geometry("200x300")

app = Window(root)
root.mainloop()











