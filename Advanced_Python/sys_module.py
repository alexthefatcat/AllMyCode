# -*- coding: utf-8 -*-
"""Created on Tue Nov 26 10:00:13 2019@author: Alexm"""



"""

In the following code in the command line 
"cmd>" will be whats input into the command line

# no arguments given
cmd> python Sys_modlue.py

cmd>python Sys_modlue.py "new_argument"
   
cmd>python Sys_modlue.py "mulitply" 5 6

"""   

import sys

ORGINAL_FILENAME = "Sys_modlue.py"

#%%#########    Erorr Message       ########
sys.stderr.write("RED ERROR MESSAGE EXAMPLE 1 (stderr)\n")
sys.stderr.flush()
sys.stdout.write("RED ERROR MESSAGE EXAMPLE 2 (stdout)\n")

#%%#########   Filename and Arguments Given to Function

def get_filename_and_input_arguments():
    data_in          = sys.argv
    current_filename = data_in[0]
    args             = data_in[1:] if len(data_in)>1 else [] 
    return current_filename, args
    
current_filename, args = get_filename_and_input_arguments()

#%%############################################################
if current_filename != ORGINAL_FILENAME:
    print("The filename has been changed from",ORGINAL_FILENAME," to ",current_filename)
else:
    print("The filename has not been changed recently")

print("These are the arguments given to the script ",args)

#%%#########    Example of Using Args To Multiply
def multiply_if_possible_and_print_result(args):
    if len(args)==3:
       if args[0]=="mulitply":
           try:
               out = float(args[1])*float(args[2])
               print(args[1]," * ",args[2],"=",out)
           except:
               pass
           
multiply_if_possible_and_print_result(args)
   
################################################################
# import module in other folder
# some_file.py
import sys
# insert at 1, 0 is the script path (or '' in REPL)
sys.path.insert(1, '/path/to/application/app/folder')

import file



