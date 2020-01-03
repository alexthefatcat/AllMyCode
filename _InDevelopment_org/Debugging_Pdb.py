# -*- coding: utf-8 -*-
"""Created on Thu May 16 13:37:17 2019@author: milroa1"""


#pdb trace logger
################################################################
#                 set_trace()                                  #
################################################################
#import pdb; pdb.set_trace()
#


import logging

logging.basicConfig(level=logging.DEBUG)
# Save to File
#logging.basicConfig(filename="test.log",level=logging.DEBUG,format="%(asctime)s":%(levelname)s:%(message)s")


#
#DEBUG	Designates fine-grained informational events that are most useful to debug an application.
#INFO	Designates informational messages that highlight the progress of the application at coarse-grained level.
#WARN	Designates potentially harmful situations.
#ERROR	Designates error events that might still allow the application to continue running.
#FATAL	Designates very severe error events that will presumably lead the application to abort.
#OFF	The highest possible rank and is intended to turn off logging.
#TRACE

def add(x,y): 
    p=9
    return x+y
def subtract(x,y):
    return x-y
def multiply(x,y):
    logging.debug(f"multiply")
    return x*y
def divide(x,y):
    return x/y
    
num1 = 10
num2 = 7
add_result = add(num1,num2)
logging.debug(f"add({num1},{num2})")
#subtract

def main():
    r=3
    pp=100
    pp=add(r,pp)
    def g(x):
        x=multiply(x,x)
        return x
    pppp=g(pp)
    out=add(pppp,pp)
    return out

main()


















#Here is the auxiliary module:

import logging

module_logger = logging.getLogger('spam_application.auxiliary')
class auxiliary_module:
    # create logger  
    class Auxiliary:
        def __init__(self):
            self.logger = logging.getLogger('spam_application.auxiliary.Auxiliary')
            self.logger.info('creating an instance of Auxiliary')
    
        def do_something(self):
            self.logger.info('doing something')
            a = 1 + 1
            self.logger.info('done doing something')

    def some_function():
        module_logger.info('received a call to "some_function"')
########################################################################################################## 
#import logging


# create logger with 'spam_application'
logger = logging.getLogger('spam_application')
logger.setLevel(logging.DEBUG)
# create file handler which logs even debug messages
"fh = logging.FileHandler('spam.log')               "  
"fh.setLevel(logging.DEBUG)                         "
# create console handler with a higher log level
ch = logging.StreamHandler()
ch.setLevel(logging.ERROR)
# create formatter and add it to the handlers
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
"fh.setFormatter(formatter)"
ch.setFormatter(formatter)
# add the handlers to the logger
"logger.addHandler(fh)"
logger.addHandler(ch)

logger.info('creating an instance of auxiliary_module.Auxiliary')
a = auxiliary_module.Auxiliary()
logger.info('created an instance of auxiliary_module.Auxiliary')
logger.info('calling auxiliary_module.Auxiliary.do_something')
a.do_something()
logger.info('finished auxiliary_module.Auxiliary.do_something')
logger.info('calling auxiliary_module.some_function()')
auxiliary_module.some_function()
logger.info('done with auxiliary_module.some_function()')
##########################################################################################################
   
"""   
The output looks like this:

2005-03-23 23:47:11,663 - spam_application - INFO -   creating an instance of auxiliary_module.Auxiliary
2005-03-23 23:47:11,665 - spam_application.auxiliary.Auxiliary - INFO -   creating an instance of Auxiliary
2005-03-23 23:47:11,665 - spam_application - INFO -   created an instance of auxiliary_module.Auxiliary
2005-03-23 23:47:11,668 - spam_application - INFO -   calling auxiliary_module.Auxiliary.do_something
2005-03-23 23:47:11,668 - spam_application.auxiliary.Auxiliary - INFO -   doing something
2005-03-23 23:47:11,669 - spam_application.auxiliary.Auxiliary - INFO -   done doing something
2005-03-23 23:47:11,670 - spam_application - INFO -   finished auxiliary_module.Auxiliary.do_something
2005-03-23 23:47:11,671 - spam_application - INFO -   calling auxiliary_module.some_function()
2005-03-23 23:47:11,672 - spam_application.auxiliary - INFO -   received a call to 'some_function'
2005-03-23 23:47:11,673 - spam_application - INFO -   done with auxiliary_module.some_function()
"""


def trace_calls(frame, event, arg):
    if frame.f_code.co_name == "sample":
        print(frame.f_code)
        
        
        
        

import sys





def trace_calls(frame, event, arg):
       if frame.f_code.co_name == "sample":
           print(frame.f_code)
           return trace_lines
       return
   
def trace_lines(frame, event, arg):
    if frame.f_code.co_name == "sample":
        #return
        print(frame.f_code.co_name,frame.f_lineno,event,arg)

sys.settrace(trace_calls)
sample(45,2)
sys.settrace(trace_lines)
sample(45,2)

j=[]
def trace_all(frame, event, arg):
    if event == "call":
#    if frame.f_code.co_name == "sample":
#        #return

        j = j+[str(frame.f_code.co_name,frame.f_lineno,event,arg)]
        
        
        
        
def sample(a, b):
        if a>0:
           nnn= sample(a-1,99)
           
        x = a + b
        y = x * 2
        print('Sample: ' + str(y))        
        
import sys        
        
class trace_class:
    def __init__(self):
        self.lines = []
        self.selectedlines=[]
        self.stop = False
    def tracer(self,frame, event, *arg):
       if not self.stop :
           func_name = frame.f_code.co_name
           self.line = ",,".join([str(n) for n in [func_name,frame.f_lineno,event,arg]])
           self.lines = self.lines + [self.line]
           if event=="call":
               self.selectedlines = self.selectedlines +[self.line]
               
           
           
        
trace_obj = trace_class()
sys.settrace(trace_obj.tracer)
sample(45,2)
trace_obj.stop=True

SelectedTracedLines = trace_obj.selectedlines
TracedLines = trace_obj.lines
c=0
for a,b in zip(TracedLines,SelectedTracedLines):
    c=c+1
    if a!=b:
        print(a,b)





TracedLines[-10:]
SelectedTracedLines[-10:]






n= ['co_argcount',
 'co_cellvars',
 'co_code',
 'co_consts',
 'co_filename',
 'co_firstlineno',
 'co_flags',
 'co_freevars',
 'co_kwonlyargcount',
 'co_lnotab',
 'co_name',
 'co_names',
 'co_nlocals',
 'co_stacksize',
 'co_varnames']

for nn in n:
    print(nn,":",getattr(j.f_code,nn))



























####################################################################################

#Pdf Python Debuging

#import pdb;pdb.set_trace() #put this is code into where to start
#
#pdb.set_trace() # like breakpoints
#
#(Pdb) next # next line
#(Pbd) step # next line steps into a function
#(Pbd) print(val) # will print val even in function
#(Pbd) continue # contiues untill the next set_trace()
#
#Others 
#(Pdb) l
#previous and next 3 lines of code
#(Pbd) cl # clears all break points
#
pdb.set_trace()
autoencoder.fit(quick_train_in, quick_train, batch_size = Config["batch_size"], epochs =1)


string = 'autoencoder.fit(quick_train_in, quick_train, batch_size = Config["batch_size"], epochs =1)'
import trace
tracer = trace.Trace( ignoredirs=[sys.prefix, sys.exec_prefix], trace=0, count=1)
tracer.run(string)
################################################################
import sys,trace
# create a Trace object, telling it what to ignore, and whether to
# do tracing or line-counting or both.
tracer = trace.Trace( ignoredirs=[sys.prefix, sys.exec_prefix], trace=0, count=1)
# run the new command using the given tracer
tracer.run('main()')
# make a report, placing output in the current directory
r = tracer.results()
t=r.write_results(show_missing=True, coverdir=".")
################################################################
f=str(r)

r.calledfuncs

import sys

def traceit(frame, event, arg):
    if event == "line":
        lineno = frame.f_lineno
        print("line", lineno)
    return traceit

def main():
    pdb.set_trace()
    print("In main")
    for i in range(5):
        print( i, i*3)
    print("Done.")
    

sys.settrace(traceit)
main()

import sys
m=pdb.Pdb()
m.run("main()")
m("s")
####################################################################################














#
#
#
#import sys
#import trace
#
## create a Trace object, telling it what to ignore, and whether to
## do tracing or line-counting or both.
#tracer = trace.Trace( trace=1, count=1,countfuncs=1,countcallers=1)# ignoredirs=[sys.prefix, sys.exec_prefix],
#
## run the new command using the given tracer
#tracer.run('main()')
#
## make a report, placing output in the current directory
#r = tracer.results()
##r.write_results(show_missing=True, coverdir=".")
#
#r.write_results()
#
#
#
