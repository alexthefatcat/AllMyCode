# -*- coding: utf-8 -*-
"""Created on Thu Dec 12 09:16:59 2019@author: Alexm"""

#%%          Just Iterate over the first loop
####################################################################
items = [1,2,3,4,5,6,7,8,9,0]

def process(*args):
    pass
# - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
# before    
for n in items:
    process(n)
    
# after  add break if True 
for n in items:
    break
if True:
    process(n)

# or   ,   use a break and indent shift
for n in items: # <<< prefered
    break
process(n)
####################################################################

def unwrap_dict_to_globals(locs):
    for k,v in locs.items():
        globals()[k] = v

def foo():
    def bar():
        n=34
        m=62
        unwrap_dict_to_globals(locals())
    bar()
foo()

####################################################################

# / code / #

assert False, "Temporary break the code here"

####################################################################

ENABLE_BREAKPOINTS = True
JUST_BREAKPOINTS   = [1]
__breakpoint_count = 0

def counter(varname = "__breakpoint_count"):
    globals()[varname] = globals().get(varname,0)+1
    return globals()[varname]

def breakpoint_input(msg,enable_breakpoints=True):
    extra=""
    points = globals().get("JUST_BREAKPOINTS",None)
    if points is not None:
        count = counter()
        if count not in points:
            return
        extra = f" #(breakpoint {count})"
    if enable_breakpoints:
        print("\n",msg,end=extra)
        res = None
        while res not in "y Y n Y".split():
           res = input("     >> Breakpoint does user proceed (y/[n])?")
        if res in "nN":
            assert False, "User chose to break"

# leaner easer to understand
def breakpoint_input2(msg,enable_breakpoints=True):
    if enable_breakpoints:
        print("\n",msg,end="")
        res = None
        while res not in "y Y n Y".split():
           res = input("     >> Breakpoint does user proceed (y/[n])?")
        if res in "nN":
            assert False, "User chose to break"


def breakpoint_input3(msg):
    if globals().get("ENABLE_BREAKPOINTS",None) is True:
        print("\n",msg,end="")
        res = None
        while res not in "y Y n Y".split():
           res = input("     >> Breakpoint does user proceed (y/[n])?")
        if res in "nN":
            assert False, "User chose to break"

i = 0
breakpoint_input("Start Section 1",ENABLE_BREAKPOINTS)
print("section 1")
i +=1
breakpoint_input("Start section 2",ENABLE_BREAKPOINTS)
print("section 2")
i +=1

print("a")

####################################################################



















