# -*- coding: utf-8 -*-
"""Created on Tue Dec 17 09:33:22 2019@author: Alexm"""

"""


Summary / Key Points
   import statements search through the list of paths in sys.path
   sys.path always includes the path of the script invoked on the command line and is agnostic to the working directory on the command line.
   importing a package is conceptually the same as importing that package’s __init__.py file
Basic Definitions
   module:          any *.py file. Its name is the file name.
   built-in module: a “module” (written in C) that is compiled into the Python interpreter, and therefore does not have a *.py file.
   package:         any folder containing a file named __init__.py in it. Its name is the name of the folder.
        >  in Python 3.3 and above, any folder (even without a __init__.py file) is considered a package
   object: in Python, almost everything is an object - functions, classes, variables, etc.

When a module named spam is imported, the interpreter first searches for a built-in module with that name. If not found, it then searches for a file named spam.py in a list of directories given by the variable sys.path. sys.path is initialized from these locations:

    The directory containing the input script (or the current directory when no file is specified).
    PYTHONPATH (a list of directory names, with the same syntax as the shell variable PATH).
    The installation-dependent default.
    
    Order it searches for a file
        1) first searches through the list of built-in modules (sys , math, itertools, time ...)
        2) Then directory of the current script. 
        3) Python’s standard library (not built-ins)  
    
    # buitlin modules
    sys.builtin_module_names
    
    
    
"""

absolute imports:
  import other
  import packA.a2
  import packA.subA.sa1
explicit relative imports:
  import other
  from . import a2
  from .subA import sa1
#%%
from foo import *
import module
from module import foo
import mod.b as b
import os.path as p

# if filename has space in it
foo_bar = __import__("foo bar")

#import numpy.zeros  #  does not work

from numpy import zeros as z
from os import path as p2

python main.py

# running python script from other ones

import sys
print(sys.path)

import sys
sys.path.insert(0, "/path/to/your/package_or_module")

import sys
sys.path.insert(0, "/home/myname/pythonfiles")

#%%
#good
import file

# bad
exec('file.py')

# very bad
os.system('python file.py')

#%%

#"$ program a.txt b.txt"
cmd = ['/Users/me/src/program', 'a.txt', 'b.txt']
process = subprocess.Popen(cmd, stdout=subprocess.PIPE)
process.wait()
for line in process.stdout:
    print(line)


cmd = "C:\\FIOCheck\\xutil.exe  -i get phy " +HBEA + ">C:\\FIOCheck\\HBEAResult.txt"
print cmd
os.system(cmd)


#%%
# Lists
list_a = [1,4,9,16,25,36,49,50]
list_1 = [e for e in list_a if e>12] # filter
# if the list output is different size than input if is at the end
list_2 = [e/2 if e>12 else 2*e for e in list_a]


list_aa  = [[n*m for n in range(1,11)] for m in range(1,11)]
list_aa2 = [inner+1 for outer in list_aa for inner in outer]
list_aa3 = [[inner+1 for inner in outer] for outer in list_aa]




















