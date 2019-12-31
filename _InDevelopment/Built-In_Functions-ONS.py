# -*- coding: utf-8 -*-
"""Created on Sat Aug  5 13:27:16 2017@author: Alex"""

#%%    ALL THE FUNCTIONS
"""
abs()           dict()          help()          min()           setattr()
all()           dir()           hex()           next()          slice()
any()           divmod()        id()            object()        sorted()
ascii()         enumerate()     input()         oct()           staticmethod()
bin()           eval()          int()           open()          str()
bool()          exec()          isinstance()    ord()           sum()
bytearray()     filter()        issubclass()    pow()           super()
bytes()         float()         iter()          print()         tuple()
callable()      format()        len()           property()      type()
chr()           frozenset()     list()          range()         vars()
classmethod()   getattr()       locals()        repr()          zip()
compile()       globals()       map()           reversed()      __import__()
complex()       hasattr()       max()           round()      
delattr()       hash()          memoryview()    set()           """

also #%% creates sperator # comment and  ; like a new line  /n  new ine in text;spyder runfile


#%%    Basic Ones
#numeric
len(),min(),max(),abs(),all(),any(),sum(),
round(),complex(),divmod(),pow()
#list related  & coment about generator () []
list(),enumerate(),range(),zip(),reversed(),sorted()#sorted sorts it and if in zip just first value
iter()# generator
#other basics
help(),print(),dir(),type(),input()# s = input('--> ')
open() #open(file, mode='r', buffering=-1, encoding=None, errors=None, newline=None, closefd=True)
#types
str(),int(),float(),bool()#bool converts to True or False, False only 0 so 0.1 is True
#data structures or data containers
set(),dict(),tuple(),#list already mentioed should be here,set type doesn't have a literal like [] and {} for lists and dicts.
#if a list is in a tuple, the list can be changed

"""** key optinal input, exists in [max,min,sorted]   """ 


#%%  iterable   p=range(5);m=next(p)
next()


#lambsa operations
map(),filter()
##################################
eval(),exec()

locals(),globals()


"""             
ascii()        staticmethod()  bytearray() 
super()         bytes()         memoryview()           
format()        property()
frozenset()     compile()       __import__()
"""



#%% 
chr(),ord()#chr(65)=>"A", ord("A") =>65
hex() #hex(255)=>'0xff'
oct()
bin()

#%% Class related #############################################################

setattr(),delattr(),hasattr(),getattr()
isinstance(), issubclass()

#special method repr(object),used for debugging shows info can be used to recreate object
repr(object):

# Return the __dict__ attribute for a module, class, instance, or any other 
# object with a __dict__ attribute.#class related
vars([object])

# It is normally used to let the user define their own slice that can be later
#  applied on data, without the need of dealing with many different cases.
#a[slice(1,4)]=a[1:4]# rare to use
slice()

# Return the hash value of the object (if it has one).Hash values 
# are integers, and to do with dictaries
hash(object)

# Return the identity of an object.This is an integer which is guaranteed to be unique and constant for this object during its lifetime.
# Two objects with non-overlapping lifetimes may have the same id() value.
# CPython implementation detail: This is the address of the object. 
# return object’s memory address
id(object)

# Return a new featureless object. object is a base for all classes. 
# It has the methods that are common to all instances of Python classes. This function does not accept any arguments.
 object()

#    Return True if the object argument appears callable, False if not.
#  If this returns true, it is still possible that a call fails, but if it is false, calling object will never succeed. 
#  Note that classes are callable (calling a class returns a new instance); instances are callable if their class has a __call__() method.

 callable(object)

#  Return a class method for function.
#   A class method receives the class as implicit first argument, just like an instance method receives the instance.
#  To declare a class method, use this idiom:
#    class C:
#        @classmethod
#        def f(cls, arg1, arg2, ...): 
            
 classmethod(function)


#%%##############################################
#################################################
#  Keywords

and       del       from      not       while
as        elif      global    or        with
assert    else      if        pass      yield
break     except    import    print
class     exec      in        raise
continue  finally   is        return 
def       for       lambda    try
################################################
################################################

[[  /   //   %  ],  [  ==  !=  ],  [+  -  *  %  **]  ]


#%%  list sub operations
x.append(5)#add 5 to list
j=x.pop(1)#removes indexed 1 and j equals this
x.insert(1,7)#inset 7 postion 1 and shift others to the right
x.reverse()# reverse
x.count(no2bcounted)
x.index(no2find)
del(x[1])


#%%    set1 operation set 
[&, |, ^, ]- and, or, xor
[-,<=,=>]  - in set1 not in set 2, set2 contains set1,opposite


[/   //   % ==  +   -   **]
#%% Dictionary Basics

d={"pork":3,"beef":2}
d["beef"]# = 3
del d["beef"]
d.clear()
d.keys()
d.values()
for k,v in x.items():
    pass

for i,(k,v) in enumerate(d.items()):
    print(i,k,v)
del i,k,v  
    
#%% String Basics
    
"1-2-3".split("-")         #=['1', '2', '3']
"-".join(['1', '2', '3'])  #= '1-2-3'
s.strip() #remove white space
s.startswith('other'), s.endswith('other')#does the string begin and end with other
s.replace('old', 'new')  
s.lower(), s.upper() #change string to upper class and lower class
 
#%% Formatting strings old and new ways
# s- strings, d - intergers, f - fixed point , e - expotinal

print("%s %s" % ("one","two"))#old
print("{} {}".format ("one","two"))#new
#> "one two"
print("%d %d" % (1,2))#old
print("{} {}".format (1,2))#new
 #>1 2
print('{1} {0}'.format('one','two'))
#> "one two"
 
## Padding and Aligning
# align right
print("%10s" % ("test",))#old
print("{:>10}".format("test"))#new
#>"      test"
#align left
print("%-10s" % ("test",))#old
print("{:10}".format("test"))#new

#>"test      
'{:_<10}'.format('test')
#>"test______"
'{:^10}'.format('test')
#>"   test   "
'{:^6}'.format('zip')
#>" zip  "

##Tuncating and Truncating and padding combined
'%.5s' % ('xylophone',)#old
'{:.5}'.format('xylophone')#New
#>"xylop"

'%-10.5s' % ('xylophone',)#Old
'{:10.5}'.format('xylophone')#New
#> "xylop     "
'{:6.6}'.format('xylophone')#New
#> 'xyl   '

#Numbers ignore the old
'{:d}'.format(42)
#> "42"
'{:f}'.format(3.141592653589793)
#> "3.141593"
'{:4d}'.format(42)
#> "  42"

'{:06.2f}'.format(3.141592653589793)
#> "003.14"
'{:04d}'.format(42)
#> "0042
'{:+d}'.format(42)
#> "+42
'{: d}'.format((- 23))
#> "-23
'{: d}'.format(42)
#> " 42"
'{:=+5d}'.format(23)
#> "+  23"
data = {'first': 'Hodor', 'last': 'Hodor!'}
'{first} {last}'.format(**data)
#> "Hodor Hodor!"

data = [4, 8, 15, 16, 23, 42]
'{d[4]} {d[5]}'.format(d=data)
#> "23 42"

from datetime import datetime
'{:%Y-%m-%d %H:%M}'.format(datetime(2001, 2, 3, 4, 5))
#> "2001-02-03 04:05"

'{:{align}{width}}'.format('test', align='^', width='10')
#>    test   "


'12'.zfill(5)
'00012'

## python f-strings new in 3.6 
name,age="bob",20
print(f'He said his name is {name} and he is {age} years old.')



#%%  Example of reading in a text file
    
    infile="in.txt";outfile="out.txt"

with open(infile) as file:#  'r' for reading, 'w' for writing 'a' for appending ##buffer byte size
   #allcontent=file.read(chunksize)#have this in while statment
    line65=file.readline(65)

   for line in file:
      print(line[:-1])

# run a scripy
exec(open(r"B:\Pricestats\Data\WS_Split_into_Seperate_Item_Groups(Part_1).py").read(), globals())

#%% Extract numbers out of list of strings
import  re
list_strs_with_nos_in=["jgfaiejgieqa","3434","asfasfdas45432.5656loo67"]
nos_extracted=[[word_number for word_number in re.findall("[-+]?[.]?[\d]+(?:,\d\d\d)*[\.]?\d*(?:[eE][-+]?\d+)?", line)] for line in list_strs_with_nos_in]

#%%  Useful Modules
import time
time.sleep(60)#sleep for a minute
time_1=time.time()
time_2=time.time()
time_taken=time_2-time_1
#%% Useful built in moduls
datetime
time
itertools
csv


#%%
numpy;
scipy;
matplotlib;
skikitlearn;
nltk;
pandas
tensorflow
cython
pyqt?, wxPython
pyautogui
selenium
beautiful soup
SQLAlchemy
pywin32??
Ipython
################################################################################################
################################################################################################
#%% Useful Snipets




#change for loop number to contast string, and only print out at every 40th line
 str_no=str(page_count);str_no="0000"+str_no;str_no=str_no[-4:]    
        if page_count%40==0: print("Image Number "+str_no)



#%% Pop up message code has finished + and beep noise
import win32api;win32api.MessageBox(0, 'Code has Finished', 'Code has Finished', 0x00001000)
import winsound;winsound.Beep(440,500)#duration_ms,freq_hz

#%%finding what lines of code take the longest to run

all string things like %s
; in python
: *args and **kwargs

3rd party

    collections -- specifically namedtuples
    csv -- always use this to read/write CSV files, don't try and roll your own methods, it'll end in tears
    datetime
    math -- try and use these functions rather than the global ones, as they're faster when you import them into the global namespace
    re -- regular expressions
    string -- I rarely see this used, but it's very handy
    tempfile -- always use this to create temporary files
    unittest
namedtuples 
math, sys, re, os, os.path, logging
math, decimal, datetime, time, re
random
urllib
itertools and functools
glob, fnmatch and shutil.
pickle
