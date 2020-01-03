# -*- coding: utf-8 -*-
"""Created on Mon Sep 18 15:01:33 2017@author: milroa1"""
#http://www.cs.put.poznan.pl/csobaniec/software/python/py-qrc.html
#%% Importing Numpy in a Special Way
from numpy import *
all_numpy=all
all = __builtins__.all
#sum max min
## remove numpy from workspace
##not yet working
#import numpy
#for n in dir(numpy):
#    print(n)
#    if n in dir():
#       exec("del "+str(n))
#del numpy
#all = __builtins__.all

#%%################################################################################################################################      
"""                 ***                                Useful Functions                                   ***                   """       
################################################################################################################################### 


def mean(x):                           return(sum(x)/len(x))   
def reorder(lis_,order):               return([ lis_[i] for i in order])  
def fill(x,val=0):                     return([val for n in x])  
def find_nearist(list_,val):           return(min(myList, key=lambda x:abs(x-myNumber)))     
def find_nearist_idx(myList,myNumber): return(  tuple(reversed(  min(list(enumerate(myList)), key=lambda x:abs(x[1]-myNumber)) ) )   ) 
def common_elements(lll,ll):           return([l for l in ll if l in lll])
def standard_deviation(l,sample=0):    return(sum([((n-sum(l)/len(l))**2)/(len(l)-sample) for n in l]))
def round2(x):                         return(int(x+(x>0)-.5))
def split(lis_,lam):                   return( [[x for x in lis_ if bool(lam(x))==split_] for split_ in [True,False]]  )
def unique(lis):                       return([n for i,n in enumerate(lis) if lis.index(n)==i])
def relu(x):                           return(max([x,0]))
def maxidx(i):                         return(i.index(max(i))) 
def primes(n):                         return([x for x in range(2, n) if not 0 in map(lambda z : x % z, range(2, int(x**0.5+1)))] )
def split2(str_in, deli, m="%"):       return([n for n in (m+deli+m).join(str_in.split(d)).replace(m+m,m).split(m) if len(n)>0 ]    )
def mode(List):                        return(max(set(List), key=List.count))
def s(str_,len_=30):                   return(  str(str_).ljust(len_)[:len_]  )
def inv_dict(dict_):                   return({v: k for k, v in dict_.items()})
    
ll =[87,87,88,23,65,867,12,3,9]
v,vv = find_nearist_idx(ll,4)   
lll = [811,88,239,65,3,13] 
common_elements(lll,ll)
m=[1,2,3,4,5,6,7]   
even,odd = split(m,lambda x:x in[0,2,4,6,8,10])
#no nans
p=list(filter(None,[5,6,7,8,8,None]))
    







#%% Check if an element is in both lists
nnn,kk=[1,2,3,4,5,6],[4,5]
for n in [n for n in nnn if n in kk]: # for n in both( nnn and kk)
    print(n)

#%% Run code in Stages like this 
    # So when a program reads in a script the #'s are replaced with these
    run_in_stages
    #<<>>
    for n_n_n in [[0,1,2,3,4,5,6],range(6)][0]:
    #<0>
    if n_n_n in [0]:
    #<1>
    if n_n_n in [1]:
    #<2>
    #<3>
    #>><<
    shelve
    
    "or maybe"
    
    def sections_runner(str_):
        """
        run_up_to_6
        run_from_7_to_8
        save_all
        """
    """   
    sect["descp"] = "run_up_to_6"
    sect["Load"] , sect["Run" ] , sect["Save"]  =     sections_runner(sect["descp"])  
    """
    
    Config={}
    sect={"Numb" : str(15) }
    sect["All" ] = list(range(int(sect["Numb"]))) 
    sect["Load"] , sect["Run" ] , sect["Save"]  =   [ 6  ] , [ 7, 8] ,  [ 9 ] # save>load+save
    sect["Run_Save"]=sect["Run" ]+sect["Save"] 
    sect["Skip"] = [n for n in sect["All" ] if not n in sect["Load"]+sect["Run_Save"] ]
    Config["Sections"]=sect
    
    ##############################################################################################
    
    
    
      
    #%%################################################################################################################################      
    """                                                      Section 0                                                             """       
    ###################################################################################################################################  
    Section = 0
    
    
    if Section in Config["Sections"]["Run_Save"]:
        
        
      ################ main code starts here  
      #
      #
      #
      #
      #
      pass # run code
    
    
    
    if Section in Config["Sections"]["Save"]:
      pass # save
    if Section in Config["Sections"]["Load"]:
        pass # load  
    #%%################################################################################################################################      
    """                                                      Section 1                                                             """       
    ###################################################################################################################################  
    Section = 1
    
    if Section in Config["Sections"]["Run_Save"]:
        
        
      ################ main code starts here  
      #
      #
      #
      #
      #
      pass # run code
    
    
    
    if Section in Config["Sections"]["Save"]:
      pass # save
    if Section in Config["Sections"]["Load"]:
        pass # load
    #%%################################################################################################################################      
    """                                                      Section 2                                                             """       
    ###################################################################################################################################  
    Section = 2
    
    if Section in Config["Sections"]["Run_Save"]:
        
        
      ################ main code starts here  
      #
      #
      #
      #
      #
      pass # run code
    
    
    
    if Section in Config["Sections"]["Save"]:
      pass # save
    if Section in Config["Sections"]["Load"]:
        pass # load
    ###################################################################################################################################

#%%
understand
a[:]=b
b=a[:]
b=a 
shallow copy
delete is different
l.copy() for a list may be more readable than l[:]
if b=a
and a[:] is used then b will change as well

b = a[:] #deep copying the list a and assigning it to b   
#%%
def filter(txt, oldfile, newfile):
    '''\
    Read a list of names from a file line by line into an output file.
    If a line begins with a particular name, insert a string of text
    after the name before appending the line to the output file.
    '''

    with open(newfile, 'w') as outfile, open(oldfile, 'r', encoding='utf-8') as infile:
        for line in infile:
            if line.startswith(txt):
                line = line[0:len(txt)] + ' - Truly a great person!\n'
            outfile.write(line)

# input the name you want to check against
text = input('Please enter the name of a great person: ')    
letsgo = filter(text,'Spanish', 'Spanish2')

#%%    understand r+ w+ writing and saving

#%% Strings

print("Today is not THUrsday".lower()) #> today is not thursday
print("Today is not THUrsday".upper()) #> TODAY IS NOT THURSDAY
print("Today is not THUrsday".title()) #> Today Is Not Thursday


os.path.join(r"C:\mypath", "subfolder")

p = r"\\nsdata7\HOUSINFL\SUB-GROUPS CODE QA\Weights"
pp=p.split("\\")#>['', '', 'nsdata7', 'HOUSINFL', 'SUB-GROUPS CODE QA', 'Weights']
p="\\".join(pp)

.split()     #default " "
.splitlines()#splits by "n\"

#read more:-
#https://stackoverflow.com/questions/647769/why-cant-pythons-raw-string-literals-end-with-a-single-backslash


# string format with {}
names = "{}, {} and {}".format(  'John',  'Bill',  'Sean')
names = "{1}, {0} and {2}".format(  'Bill',  'John',  'Sean')
names = "{j}, {b} and {s}".format(j='John',b='Bill',s='Sean')

# formatting integers
"Binary representation of {0} is {0:b}".format(12)       #> 'Binary representation of 12 is 1100'
# formatting floats
"Exponent representation: {0:e}".format(1566.345)        #> 'Exponent representation: 1.566345e+03'
# round off
"One third is: {0:.3f}".format(1/3)                      #> 'One third is: 0.333'
# string alignment
"|{:<10}|{:^10}|{:>10}|".format('butter','bread','ham')  #> '|butter    |  bread   |       ham|'

"20{:02d}".format(1) #> 2001
#d	Decimal integer
#e	Exponential notation. (lowercase e)
#f	Displays fixed point number (Default: 6)
#g	General format. Rounds number to p significant digits. (Default precision: 6)

#always 2 digits
"20"+"{:02d}".format(901)[-2:] #> 2001
"20{:02d}".format(901%100) #> 2001

## Location and Replace
'Happy New Year'.find('ew')  #>7
'Happy New Year'.replace('Happy','Brilliant')  #>'Brilliant New Year'


# old sprintf() like 
print( '%s %d %s'%('python',56,'fun')) % substion check


String_Ecaspe_Characters_dict={
 "\n"        : "*new line*",                   
 "\\"        : r"\ ",                          
 "\'"        : r"'",                          
 "\""        : r'"',                          
 "\a"        : "*beep sound*",                
 "\b"        : "*backspace*",              
 "\r"        : "*restart line*",
 "\t"        : "*Tab to the right*" }      
 
#-----  ones which are used less or don't work
#"\newline"  : "Backslash and newline ignored"#
#"\v"        : "ASCII Vertical Tab"          #
#"\ooo"      : "Character with octal value ooo"#
#"\xHH"      : "Character with hexadecimal value HH"
#"\f"        : "ASCII Formfeed"              #
  
print("-"*80)
count=0
for i,(key, value) in enumerate(String_Ecaspe_Characters_dict.items()):
#for i,key, value in enumerate(zip(String_Ecaspe_Characters_dict.keys,String_Ecaspe_Characters_dict.values):  
    print(str(i)+" : "+value+"\n  A--"+key+"--A  "+"  "+str(count))
    count=count+1

#Raw String to ignore escape sequence
# print(r"\")doenst work print("\\") and print(r"\ ") does

### for more advanced formatinh use regex







#%%  Reorder Pandas Rows
cols = df.columns.tolist() #>[0L, 1L, 2L, 3L, 4L, 'mean']
#insert switch delete 
#reorder the cols to your liking
df=df[cols]

#blank dataframe
blank_df = pd.DataFrame()

# mapping over dataframe
mapping={"one":[1,2,3],"two":[89,45]}
for key, value in dict.items():
      df[key] = df[value].sum(axis = 1) 

temp['year'] = "20{:02d}".format(i)
#%%   Memoize:this takes an inefficient algorithm and caches the values it has already produced.

class memoize:
    def __init__(self, function):
        self.function = function
        self.memoized = {}

    def __call__(self, *args):
        try:
            return self.memoized[args]
        except KeyError:
            self.memoized[args] = self.function(*args)
            return self.memoized[args]
 # # # # ## # # # # # 
@memoize
def fibonacci(n):
    if n in (0, 1): return n
    return fibonacci(n - 1) + fibonacci(n - 2)

#%%  Create a repeatable list of random intergers
import random
def repeatable_random_intergers_1000(nos,seed=321):
    random.seed(321)
    return [random.randint(1,1000) for x in range(nos)]
print(repeatable_random_intergers_1000(10))
#%% nested lists
d_10 =  [ []  for x in range(10)]
d_10_10=[d_10 for x in range(10)]
#%%  xor  apperenlty ^ does this 
def xor(a,b):
    return bool(a) != bool(b)# not(bool(a) == bool(b))
#%% both read and write to a file at the same time

""" Here's how you read a file, and then write to it 
   (overwriting any existing data), without closing and reopening:"""

with open(filename, "r+") as f:
    data = f.read()
    f.seek(0)
    f.write(output)
    f.truncate()
    
#%% count and create the counter if it does not exist
for _ in range(893):    
    count = count+1 if "count" in dir() else 0 
    
if "count" in dir(): print("###>>>",count);del count
#%% print for loop if 100 values have passed
for n in range(1000):
    if(n%100)==0:print("The line number is "+str(n))

#%%    Returns the current line number in our program
import inspect
def lineno():
    return inspect.currentframe().f_back.f_lineno  
print(str(lineno()),"###>>>")
#%% FORMULA evaluation
from numpy.random import randn
from pandas import DataFrame
df = DataFrame(randn(10, 2), columns=list('ab'))
df.eval('c = a + b',inplace=True)
#maybe import excel_pandas as epd
# epd.epd2pd()
# edf.refresh()#>when values have changed input the new ones
#
#
#%% changing a list reference and copying
org_list = ['y', 'c', 'gdp', 'cap']

copy_list = org_list

copy_list.append('hum')

print(copy_list)
print(org_list)

copy_list = org_list[:]

copy_list.append('hum')

print(copy_list)
print(org_list)

a=[0]
b1,b3,b5=a,a,a
b2=a[:]
b4=b3
b3=[5]
b5=b5
a[0]=+1
print(   a,      b1,       b2,      b3,      b4,      b5 )
#>      [1]      [1]      [0]      [5]      [1]      [1] 
print(a is a, b1 is a, b2 is a, b3 is a, b4 is a, b5 is a )
#>     True     True    False    False    True     True
#can't do this with ints as they are immutable
# but can with lists
#%%    *zip
""" zip and *
    zip wants a bunch of arguments to zip together. 
    The * in a function call "unpacks" a list (or other iterable),
    making each of its elements a separate argument.
    So without the *, you're doing zip( [[1,2,3],[4,5,6]] ).
    With the *, you're doing zip([1,2,3], [4,5,6])                    """

m=list(zip(*[[1,2,3],[4,5,6],[7,8,9]]))
#=>    [(3, 6, 9),
#       (2, 5, 8),
#       (1, 4, 7)]



#%% cool fractal traingle

import numpy as np, matplotlib.pyplot as plt
from random import random

##options
info={"img_size":4000,"no_points":3,"vertical_shift":0,"out_of_bounds":["skip","overlap"][0]}
info["options"]=[["show_image","add_points_2_image"],["show_image"]][1]

#created from options
info["center"]=[int(info["img_size"]/2),int(info["img_size"]/2)]
info["radius"]=info["center"][0]-10
info["points"]=list(range(info["no_points"]))+["center"]###

##calcualate points
for n in range(info["no_points"]):
   nn=((n/info["no_points"])+0.5)*3.14*2  
   info[n]=[int(round(info["center"][0]+nnn*info["radius"])) for nnn in [np.cos(nn),np.sin(nn)]]
del n,nn
#starting postion
info["p_org"]=info[0]

# create a blank image
img=np.zeros([info["img_size"],info["img_size"]])
p=info["p_org"]

# fill in the image
for _ in range(1000000):
    p_incr=np.floor(info["no_points"]*random())
    p=[int(round((foo1+foo2)/2)) for foo1,foo2 in zip(p,info[p_incr])]
    #img[p[0]+info["vertical_shift"],p[1]]=1
    ##new
    if info["out_of_bounds"]=="skip":
        
           if all(  [info["img_size"]>(p[0]+info["vertical_shift"])>=0,  info["img_size"]>(p[1])>=1 ] ):
               img[p[0]+info["vertical_shift"],p[1]]=1
 
    if info["out_of_bounds"]=="overlap":
        
         img[p[0]+info["vertical_shift"] % info["img_size"] ,p[1] % info["img_size"]]=1
    ##new     
del p, p_incr

#put points on imgage
if "add_points_2_image" in info["options"]:###
    for n in info["points"]:###
        img[  info[n][0]:(info[n][0]+10)  ,  info[n][1]:(info[n][1]+10)  ] = 1###
    del n###
#show image
if "show_image" in info["options"]:
    %matplotlib qt
    plt.imshow(img,cmap='Greys',  interpolation='nearest')
    if 0:
        %matplotlib inline
        import scipy
        scipy.misc.imsave('fractal.png', img)
#%%
all(  [ 4000>(5900)>=0,  4000>(674)>=0 ] )
#%% 
options{"Print_2":True}


if options["print_2"]:print()
   
   
result = {  'a': lambda x: x * 5,
            'b': lambda x: x + 7,
            'c': lambda x: x - 2}["a"](5)

def f(x):return {'a': 1,
                 'b': 2 }.get(x, 9) 

#%% Splitting List

mylist   = [0,1,2,3,4,5,6,7,8,9]
goodvals = [2,4,10,10,6,9]

good = [x for x in mylist if (x     in goodvals)]
bad  = [x for x in mylist if x not in goodvals]
# same as 
good, bad = [[x for x in mylist if (x in goodvals)==split] for split in [True,False]]

# good=>[2, 4, 6, 9]
# bad=> [0, 1, 3, 5, 7, 8]

#%% old style of rounding
def round2(x):
    return(int(x+(x>0)-.5))
    
#%%  #how to transpose an nested list or permute  ??
for x,n in enumerate(nest_list):
    for y,m in enumerate(n):
         
#%%         extend array in a dimension 
img3D =  np.tile(img, (1,1,10))   
#%%  one line for loop that produces flat list a lot bigger than the input

[m+n for n in [1,2,3,4,5,6] for m in [1000, -1000] ]
#>[1001, -999, 1002, -998, 1003, -997, 1004, -996, 1005, -995, 1006, -994]
[[1000+n, -1000+n] for n in [1,2,3,4,5,6] ]
#>[[1001, -999], [1002, -998], [1003, -997], [1004, -996], [1005, -995], [1006, -994]]

#%%
example="L:\\Branch folders\Big_Data\\R&D_System\\Prelim_work\\PinkBlue\\Pink and Blue 2010-11.xls"
example_out = year_str_replace(example,2017)
#>     "L:\\Branch folders\\Big_Data\\R&D_System\\Prelim_work\\PinkBlue\\Pink and Blue 2017-18.xls"
def year_str_replace(string, year, year_base=2010, year_range=2 ):
    #string,year_base,year,year_range =    'hello 42 I\'m a 32 string 30 2000 2008 2000afs 2010-11',2010, 2014, 2        
    import re       
    #re.findall(r'\d+', 'hello 42 I\'m a 32 string 30')
    year_shift = list(range((-year_range),(year_range+1))  )   
    
    year, year_base = int(year), int(year_base)  
    if year_base<1000: year_base = year_base +2000
    if year     <1000: year      = year      +2000  
           
    years       = [m for n in year_shift for m in [str(year     +n),str(year      +n)[-2:]]  ] # years      = [[str(year     +n),str(year     +n)[-2:]] for n in year_shift] 
    year_bases  = [m for n in year_shift for m in [str(year_base+n),str(year_base +n)[-2:]]  ] # year_bases = [[str(year_base+n),str(year_base+n)[-2:]] for n in year_shift]
    
    p = re.compile(r'\d+')
    for m in p.finditer(string):
        for i, n in enumerate(year_bases):
            if m.group() == n:                                                                 #  if m.group() in n:
#                #print( m.start(), m.group())
                string= string[:m.start()] + years[i] + string[m.start()+len(m.group()):]      #    string= string[:m.start()] + years[i][len(m.group())<3]+  string[m.start()+len(m.group()):]       
    return(string)



#%%  Dealing with objects


Alpha="ABCDEFGHIJKLMNOPQRSTUVWXYZ"
excel=[letter for letter in Alpha]
excel=excel+[letter1+letter2 for letter1 in Alpha for letter2 in Alpha]

#or
Alpha=list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
excel=Alpha+[letter1+letter2 for letter1 in Alpha for letter2 in Alpha]


#%% print a dataframe so its copy and pastable in code, as well from nested list straight to dataframe

def print_df(df2print):
    cols, inds = df2print.columns.tolist(),df2print.index.tolist()
    print("[",["***"]+cols,",")
    for ind in inds:
      temp= df2print.loc[ind,:].tolist()
      if not ind == inds[-1]:
          print(" ",[ind]+temp,",")
      else :
          print(" ",[ind]+temp,"]")#end on this   


def nested_list_2_df(nested_list):
   return(pd.DataFrame(nested_list[1:],columns=nested_list[0]).set_index(nested_list[0][0]))

def df_2_nested_list(df,no_row_or_col=False):
    if no_row_or_col:
        nested_list=df.values.tolist()
    else :        
        nested_list=[['***']+list(df.columns)]
        for val,ind in zip(df.values.tolist(),df.index.tolist()):
           nested_list.append([ind]+val)
    return(nested_list)



print_df(CIVIL_df)
#>
data = [
  ['***', "c_Gov't", "c_RC'S", 'c_H. Ed.', 'c_Business', 'c_PNP', 'c_Total', "c_O'seas", "c_Gov & RC's"] ,
  ["r_Gov't", '_', '_', '_', '_', '+q0303', '_', '_', '_'] ,
  ["r_RC'S", '_', '_', '_', '_', '+q0327', '_', '_', '_'] ,
  ["r_HEFC'S", '_', '_', '_', '_', '_', '_', '_', '_'] ,
  ['r_H. Ed.', '_', '_', '_', '_', '+q0307', '_', '_', '_'] ,
  ['r_Business', '_', '_', '_', '_', '+q0305', '_', '_', '_'] ,
  ['r_PNP', '_', '_', '_', '_', '+q0301', '_', '_', '_'] ,
  ["r_O'seas", '_', '_', '_', '_', '+q0309+q0311+q0313+q0315+q0317+q0319+q0321+q0323', '_', '_', '_'] ,
  ['r_Total', '_', '_', '_', '_', '_', '_', '_', '_'] ,
  ['r_GOVERD TOTAL', '_', '_', '_', '_', '_', '_', '_', '_'] ]

data_df = nested_list_2_df(data)
data_2 = df_2_nested_list(data_df)

#%%
for r_idx, row in enumerate(["A","B","C","D"],7):# the second input starts from this number
    print(r_idx,row)
#> 7 A
#> 8 B
#> 9 C
#> 10 D

#%%
Options={"print":True, "del":False, "Stages_to_Run":[list(range(5))],"Run_Save_Load":[["Run","Save","Load"][1]] }#maybe call config   
if Options["print"]: print(Options)
year=2017

if Options["print"]: print("The year selected is ",year)
if Options[ "del" ]: del year:
    
if 3 in Options["Stages_to_Run"]:
    if Options["print"]: print(" Stage 3 is running")

if "Run" in Options["Run_Save_Load"]:
    if Options["print"]: print(" The scripts main code is running")

    
#%%  Some More Pandas and Numpy

import pandas as pd, numpy as np
df = pd.DataFrame(np.random.rand(10, 3), columns=list('abc'))

df2=df.copy()
df2[df2>0.4]=-df2[df2>0.4]
df[(df.a < df.b) & (df.b < df.c)]
#eqivelent_to - 
df.query('(a < b) & (b < c)')



df=round(15*df)
#find what groupno the column "a"belongs to and input that
group_1=[1,2,3,4,5,6]
group_2=[7,8,9,10,15]
group_3=[67]
for i,group in enumerate([group_1,group_2,group_3],1):
    df.loc[df["a"].isin(group),"a-group"]=i
    
df.head()       # first five rows
df.tail()       # last five rows
df.sample(5)    # random sample of rows
df.shape        # number of rows/columns in a tuple
df.describe()   # calculates measures of central tendency
df.info()       # memory footprint and datatypes

df.sort_values('price', axis=0, ascending=False)
np.where(df['price']>=15.00, True, False)

df['mean'] = df.mean(axis=1)
df['std' ] = df.std(axis=1)




pd.concat([df_1, df_2], axis=1)
merged_df = df_1.merge(df_2, how='left', on='order_id')

df.isnull().sum().sum()

bins = [0, 5, 15, 30]
names = ['Cheap', 'Normal', 'Expensive']

df['price_point'] = pd.cut(df.price, bins, labels=names)


df.values
CategoricalIndex
RangeIndex
SparseArray'
crosstab
'eval'
melt'
pivot_table
'scatter_matrix
'value_counts
'unique'

unuqe 
dupicates

stack 
filter
select

dropna fillna
replace
#analyze smaller chunks
chunksize = 500
chunks = []
for chunk in pd.read_csv('pizza.csv', chunksize=chunksize):
    # Do stuff...
    chunks.append(chunk)

df = pd.concat(chunks, axis=0)


df.loc[df['column_name'].isin(some_values)]
itterrows
iteritems


merge join append concat
print("121/10=",121/10,",  121//10=",121//10,",  121%10=",121%10,",  121**10=", 121**10)
#        121/10= 12.1   ,   121//10= 12       ,   121%10= 1       ,   672749994932560009201

#isin
#where pandas

my_list.index(a)
my_list.count(a)
my_list.append('!')
my_list.remove('!')
del(my_list[0:1])
my_list.reverse()
my_list.extend('!')
my_list.pop(-1)
my_list.insert(0,'!')
my_list.sort()

my_string.upper()
my_string.lower()
my_string.count('w')
my_string.replace('e', 'i')
my_string.strip()

fill_value

#copy
add dir to the basics


#%% reverse going throug list

a1, a2, a3 = [11,111,11,1] , [12,13,14,15] , [1000,2000,3000,40]

for i, (b1,b2,b3) in reversed(list(enumerate(zip(a1,a2,a3),1))):
    print( i, b1,b2,b3)

for i, e in reversed(list(enumerate([11,111,1111,111111]))):
    print( i, e)

#%%
## Warning Message
import warnings
warnings.warn("Warning...........Message")

## Error Message
raise NameError('HiThere')

#%%
if 0:## this prints out the code below
        objectname="font"
        objectname_2=objectname+"={ "
        objectname_2_blank=" "*len(objectname_2)
        
        methods=["name", "size", "bold", "italic", "vertAlign", "underline", "strike", "color" ]
        first_last=[0]+[1 for n in methods[:-2]]+[2]
        mx=max(len(n) for n in methods)

        for method, first_last_ in zip(methods, first_last):             #^#
            left_side  = ("'"+method+"'").ljust(mx+2)
            right_side = "cell." + objectname + "." + method.ljust(mx)
            
            statement = left_side + ":" + right_side
        
            if   first_last_==0:   print(        objectname_2  +  statement  +  ",")
            if   first_last_==1:   print(  objectname_2_blank  +  statement  +  ",") 
            if   first_last_==2:   print(  objectname_2_blank  +  statement  +  "}")
 #  font={           'name':cell.font.name     ,
 # objectname_2   +  statement              + ",")

if 0:#printed code below
        
        font={ 'name'     :cell.font.name     ,
               'size'     :cell.font.size     ,
               'bold'     :cell.font.bold     ,
               'italic'   :cell.font.italic   ,
               'vertAlign':cell.font.vertAlign,
               'underline':cell.font.underline,
               'strike'   :cell.font.strike   ,
               'color'    :cell.font.color    }
 
#%%

methods=["name", "size", "bold", "italic", "vertAlign", "underline", "strike", "color" ]
#first_last=[1 for n in methods]
#first_last[ 0],first_last[-1]=0,2
first_last=[0]+[1 for n in methods[:-2]]+[2]

for i, (m,method) in enumerate(zip(first_last,methods)):
    print(i,m,method)

              
#%% Trasposing Nested List

a=[1,2,3]
b=[4,5,6]
c=[7,8,9]
d=[0,10,11]
A=[a,b,c,d]

#for aa,bb,cc,dd in zip(*A):
#    print(aa,bb,cc,dd)
B=[list(aa) for aa in zip(*A)]

######################################

p1=[1,2,3,4,5,6,7,8,9,10]
p2=[1,2,3,4]
p3=[1,2]
p4=[1,2,3,4,5,6,7]
P=[p1,p2,p3,p4]
P2=P.copy()

P2=[p+[None for _ in range(  max(len(n) for n in P)  -len(p))]   for p in P2]
P3=[list(aa) for aa in zip(*P2)]
## below is the same but more clearly
mx=max(len(n) for n in P2)
P3=[]
for p in P2:
    P3.append( p + [None for _ in range( mx-len(p))]   )    
P3=[list(aa) for aa in zip(*P3)]    
## or use itertools 
 zip_longest()
'  --  Transposing  --  '
#Iguess numpy


#%%
row = ["1", "bob", "developer", "python"]
>>> print(','.join(str(x) for x in row))
1,bob,developer,python
#%%
sum(i for i in range(1300) if i % 3 == 0 or i % 5 == 0)
#%%
portfolio = [
  {'name':'GOOG', 'shares': 50},
  {'name':'YHOO', 'shares': 75},
  {'name':'AOL', 'shares': 20},
  {'name':'SCOX', 'shares': 65}
]
min_shares = min(s['shares'] for s in portfolio)


##slow
s = "line1\n"
s += "line2\n"
print(s)
##fast
lines = ["line1"]
lines.append("line2")
print("\n".join(lines))


df[(df.A < 0.5) & (df.B > 0.5)]
# Alternative, using query which depends on numexpr
df.query('A < 0.5 & B > 0.5')

#%% This doesnt work process bar

import time
import sys
for progress in range(100):
  time.sleep(0.1)
  sys.stdout.write("Download progress: %d%%   \r" % (progress) ) 
  sys.stdout.flush()
  

#%%
#Append to a list in a dictionary or create a nre one
A={1:"1"}
A.setdefault(3, ["default","elements"]).append(1)


#%%  Table programming maybe use dictaries inested# decsion trees
A= [[[1,2],[3,4]],
    [[5,6],[7,8]]]
    
                   #            #            #            #   
_=[         print(n3)  for n1 in A for n2 in n1 for n3 in n2]
_= [ print(i1,i2,i3,n3)  for i1,n1 in enumerate(A) for i2,n2 in enumerate(n1) for i3,n3 in enumerate(n2)]
A2= [ n3 for i1,n1 in enumerate(A) for i2,n2 in enumerate(n1) for i3,n3 in enumerate(n2)]

A2=  [[[  n3 +3*(i1==1)+ 2*(i2>0) +100*(i1==0 and i2==0 and i3 ==0)    for i3,n3 in enumerate(n2)      ]for i2,n2 in enumerate(n1) ]  for i1,n1 in enumerate(A)] 


##nested dicts

#create
dict_={}
for n1 in ["one","two"]:
    dict_[n1]={}
    for n2 in ["name","address","postcode"]:
        dict_[n1][n2]={}
        for n3 in ["a","b","c"]:
           dict_[n1][n2][n3]=34

m1=["one","two"]
m2=["name","address","postcode"]
m3=["a","b","c"]
for n1 in m1:
    for n2 in m2:
        for n3 in m3:
            print(n1 )
            if all([n1 in dict_,n2 in dict_[n1],n3 in dict_[n1][n2]]):  
               #dict_[n1][n2][n3]
             if n2=="name":
                 dict_[n1][n2][n3]=67
            
#%% find the index in a list
indexs=[i for i,x in enumerate(testlist) if x == 1]

#%% find the index in a string
indexs=[i for i,x in enumerate(testlist) if x == 1]

from itertools import accumulate
str_match="12"
test="12   02rffq9f92342k12dsofkewof1212asdfasd"

test_split=test.split(str_match)
test_split2=[len(n)+len(str_match)*i for i,n in enumerate(test_split)]
test_split3=list(accumulate(test_split2))

test_join= "".join(test_split)
for n in test_split3[:-1]:
    test_join=test_join[:n]+str_match+test_join[n:]

#%%
import re

dict={}
for pattern,string_in in zip(["ds","#","?"],["dskui#guiui#gigds##tygj ds    ds"]*3):
    dict[pattern] = [m.start(0) for m in re.finditer(pattern, string_in)] if string_in.count(pattern) else []






#%%

def mo(m):

    for n in range(5):
        print(n)
        if n==m:
            print("break")
            break
    else:
        print("nobreak")

mo(2)
mo(15)

#>#    0,1,2,break,0,1,2,3,4,nobreak


#%%



"Shortest Code"

#[x for _,x in sorted(zip(Y,X))]

"Example:"

X = ["a", "b", "c", "d", "e", "f", "g", "h", "i"]
Y = [ 0,   1,   1,    0,   1,   2,   2,   0,   1]

Z = [x for _,x in sorted(zip(Y,X))]
print(Z)  
#> ["a", "d", "h", "b", "c", "e", "i", "f", "g"]



ind=[i[0] for i in sorted(enumerate(X), key=lambda x:x[1])]



Styles_count_ordered_dict=dict([ (x[0],i) for i,x in enumerate(sorted(enumerate(Styles_count), key=lambda x:x[1], reverse=True))])

Styles_count_ordered=[ i[0] for i in reversed(sorted(enumerate(Styles_count), key=lambda x:x[1]))]
Styles_count_ordered_dict=dict([ (b, a) for a, b in enumerate(Styles_count_ordered)])




#%%
a={"1":1}
b={"2":2}
c={**a,**b}



#%%  Sting Convolution
string_test="therse example string is a ** and * whatever"
v=2
for i,n in enumerate(string_test[v:-v],v):
    mini=string_test[(i-v):(i+1+v)]
    print(mini)

#%%  change the number of a character in a string
string_test=">>## 10 stars **********,  1 star *, two star **, thress star *** and the famous 4,n5 **** *****##"#try removing last 2 hashs
string_test_org=string_test
#> I want "therse is a *** and ** whateever"
#> as well as the reverse
#find singlur ones and insert them
indexs1=[i for i,x in enumerate(string_test[1:-1],1) if x == "*" and not string_test[i-1]=="*"]
indexs2=[i for i,x in enumerate(string_test[1:-1],1) if x == "*" and not string_test[i+1]=="*"]

sta_end=zip(reversed(indexs1),reversed(indexs2))
del indexs1,indexs2
for sta, end in sta_end: 
    
    length=1+end-sta   
    no=2*length+2   
    no=no if no >= 0 else 0
    string_test=string_test[:sta]+"*"*no+string_test[1+end:]  
    
del sta,end      
    
#%%






#%%
'Useful itertools'


#accumalte



a = numpy.matrix([[1, 2, 3, 4], [5, 6, 7, 8]])
b = numpy.reshape(a, -1)




#%%reads in a list and prints a nice version of if
def print_list(lst):
    lst=regions_perc22
    
    lst2=[[None for elm in col] for col in lst]
    
    col_lens = [0 for elm in lst[0]]  
    for i1,col in enumerate(lst):
        for i2,elm in enumerate(col):
            #print(elm,col)
            if type(elm) is str:
                elm_2="'"+elm+"'"
            else :
                elm_2=str(elm)
               
            col_lens[i2]=max(col_lens[i2],len(elm_2))
            lst2[i1][i2]=elm_2
            
    lst3=[[ elm.ljust(col_lens[i]) for i, elm in enumerate(col)] for col in lst2]
    out="\n".join([  " ["+", ".join(lst3_)+"]," for lst3_ in lst3 ])
    out="["+out[1:-1]+"]"
    print(out)
#%%
list_example=[1,2,3,4,5,6,7,8,9,10]
fist_last = [0]+[1 for n in list_example[:-2]]+[2]
def First_Last(list_):return( [0]+[1 for n in list_[1:-1]]+[2]):
[n,len(list_)-n for n in list_]   

def enumerate2(list_,start=0):
    n = start
    for elem in list_:
        yield [n,n-len(list_)], elem
        n += 1
        
list_=[  87, 87, 88, 23, 65, 867, 12, 3, 9  ]
for ii,n in enumerate2(list_):
    if    0 in ii:
        print("Start",n)
    #elif  0 in [sum(ii)]:#this needs work
    #    print("Middle",n)      
    elif -1 in ii:
        print("Finish",n)       
    else:
        print("*** ",n) 



for exam,fl in zip(list_example,fist_last):
    #do stuff
    pass

    
#%% Extra String Methods Useful
##rstrip() ljust
#'ljust','rjust',,'lstrip''rstrip'


'casefold',
'center',
'count',
'endswith',
'find',
'isdigit',
'isnumeric',
'replace',
'maketrans',
'partition',
'rfind',
'rindex',
'rpartition',
'rsplit',
'translate',
'zfill',
'encode',
#%%
df.rename(columns={ df.columns[1]: "whatever" })

#%% Factorial
def fact(x):
    v=1
    for n in range(x):
        v=v*(n+1)
    return(int(v))

def perm(n,k):return(int(fact(n)/fact(n-k)))
def comb(n,k):return(int(fact(n)/(fact(k)*fact(n-k))))    
#so 10 coins 3 blue 7 red how many differnet combinations
print(comb(10,3))
# how many possible different states if blue and red varied
print(sum([comb(10,n) for n in range(0,11)]))




#%%##################################################################
"""    Example of using Dictionaries as if elif else    """

name="John"
#####################################################################
if   name ==  "John"  : print(1, "This is John, he is an artist"  )
elif name ==   "Ted"  : print(1,"This is Ted, he is an engineer"  )
elif name =="Kennedy" : print(1,"This is Kennedy, he is a teacher")
else                  : print(1, "None- .........................")
##   By using a dictionary, we can write the same code like this:  ##
#####################################################################                
print(2,{"Josh"  : "This is John, he is an artist"          ,
         "Ted"   : "This is Ted, he is an engineer"         ,   
         "Kenedy": "This is Kennedy, he is a teacher"       }
      .get(name  , "None- this can act as the else statemnt"))
######################################################################
## no else statment possible though                
print(3,{"Josh"  : "This is John, he is an artist",
         "Ted"   : "This is Ted, he is an engineer",   
         "Kenedy": "This is Kennedy, he is a teacher"}[name])
#####################################################################
## lambdas can exist in dictionaries         
print(3,list(map({"Josh"  : lambda x:x**2}["Josh"],[5])))
#####################################################################

# Example of how to vectorize code in numpy 
import numpy as np
def muldiv(a,b):
    if b<1:      out=a*b  
    else:        out=a/b  
    return(out)
    
vmuldiv = np.vectorize(muldiv)


#%% if statment in neural netowrls if_a_b

#Its still not perfect but if in this case a<50 b =False wont work a>True or if b near zero

def relu(x,w=1,b=0):return(max([w*(x+b),0])):
    
def nn_3_3(x1,x2,x3, w11=1,w12=0,w13=0, w21=1,w22=0,w23=0, w31=0,w32=0,w33=1, b1=0,b2=0,b3=0):
    return( max([(w11*x1)+(w12*x2)+(w13*x3)+b1,0]) ,  max([(w21*x1)+(w22*x2)+(w23*x3)+b2,0]) , max([(w31*x1)+(w32*x2)+(w33*x3)+b3,0]) )

In1=0,In2=0,
layer_1=[In1,0,In2]
##greater than 0 implimented here  layer2[0+1] 
layer_2= nn_3_3(layer_1[0],layer_1[1],layer_1[2],w11=10,w22=0,w21=10,b2=0.1)      
##if a_b
layer_3= nn_3_3(layer_2[0],layer_2[1],layer_2[2],w11=100,w12=100,w22=100,w21=100,w23=1,b1=-20,b2=-20)     
layer_4= nn_3_3(layer_3[0],layer_3[1],layer_3[2],w33=0, w12=1,w22=0,w23=1)    
#layer_4 gives output carrier wave    
#-------------------------------------------------------------------------------------------------------------------- 
#if_greater than 0 is 0 below 0;1 above 1      
def if_greater_than_0(x,w1=10,b1=0,b2=0.1):
   return(  relu(x,w=w1,b=b2) -  relu(x,w=w1,b=b1)  )  
    
if_greater_than_0_test_m5_to_5=[if_greater_than_0((n/10)-5) for n in range(100)]

def if_a_b(a,b):
   return( relu( (100*if_greater_than_0(b)) +a, b=-50 ) - relu(  (100*if_greater_than_0(b)), b=-50)  )  

if_a_b_test_m5_to_5  =[[  if_a_b((m-50)/10,(n-50)/10)  for m in range(100)  ] for n in range(100) ]

if 0:
    if_a_b__ =[[  (m-50)/10  for m in range(100)  ] for n in range(100) ]
    if_a_b___=[[  (n-50)/10  for m in range(100)  ] for n in range(100) ]
#multiplication can be achived by exp(in(a)*in(b))
########################################################################################################################


def desc(df,axis_=0): 
        if axis_==0: basic = pd.DataFrame(index=df.columns) 
        if axis_==1: basic = pd.DataFrame(index=df.index  )                                    
        
        basic["max" ] = df.max(axis=axis_)   
        basic["min" ] = df.min(axis=axis_) 
        basic["mean"] = df.mean(axis=axis_)
        basic["isna"] = df.isnull().sum()
        basic["len" ] = df.count(axis=axis_)
        #basic["std" ] = (((df-basic["mean"])^2).sum(axis=0))^0.5
        basic["Typ" ] = df.dtypes.apply(lambda x: str(x))
        return(basic)
                        
           
## Weieghted linear regression  in numpy                       
def WLS(df, x_cols, y_cols, w_col):
        # this maybe need wieghted mean rather than mean to be subtracted
        def unbias(arr):
            return(arr-np.mean(arr,axis=0))
            
        df = df.copy()
        X = df[x_cols]
        y = df[y_cols]
        weight_col = df[w_col]
        #########################################            
        X = unbias(  X.as_matrix())
        y = unbias(  y.as_matrix().reshape(-1,1)  )
        W = weight_col.as_matrix().reshape(-1,1)                                    
        #W, X, y = W[:40,:], X[:40,:], y[:40,:]   
        #      form1             form2  
        # inv(Xt * W * X)  * (Xt  *  W  *  y)                                    
        Xt = X.transpose()
        WX,    Wy   =    W * X,    W * y

        form_1 = np.matmul( Xt, WX )
        form_1 = np.matrix( form_1 ).I
        
        form_2 = np.matmul( Xt, Wy )
        form_2 = np.matrix( form_2 )    

        formula = np.matmul( form_1, form_2 )
                 
        return(  pd.DataFrame(data = formula ,index=x_cols )         )




temp={str(n+5):n for n in range(15)}



def print_tab(nested_list,print_=False) :  
    nested_list_max = [len(str(n)) for n in max(nested_list, key=lambda x: len(str(x[0])))]
    out=[ [m.ljust( nested_list_max[i])[: nested_list_max[i]] for i,m in enumerate(n) ] for n in nested_list]
    if print_: [print("".join(key)) for key in out]
    return(out )

pm["n_keys_print2"]=print_tab(pm["n_keys_print"])
m=print_tab(pm["n_keys_print"],print_=True)




###############################################################
"""                 Easy Nested dicts                       """
###############################################################
from collections import defaultdict

def default_to_regular(d):
    if isinstance(d, defaultdict):
        d = {k: default_to_regular(v) for k, v in d.items()}
    return d

defaultdictnested = lambda: defaultdict( defaultdictnested  ) 
###############################################################
root = defaultdictnested ()

root['menu']['id'   ] = 'file'
root['menu']['value'] = 'File'
root['menu']['value2']["myname"]["is"] = 'spartucus'

root2=default_to_regular(root)

###############################################################









x = (classA if y == 1 else classB)(param1, param2)



multiStr = "select * from multi_row \
where row_id < 5"
print(multiStr)

# select * from multi_row where row_id < 5

multiStr = """select * from multi_row 
where row_id < 5"""
print(multiStr)

#select * from multi_row 
#where row_id < 5

import numpy
print(numpy)
<module 'numpy' from 'C:\\Python36\\lib\\site-packages\\numpy\\__init__.py'>


9+10
print(_)
#>>> 19
"""The “_” references to the output of the last executed expression."""


testDict = {i: i * i for i in range(10)} 
testSet = {i * 2 for i in xrange(10)}



n=10
testDict = **{i: i * i for i in range(n)} ,{n-i:i for i in range(n)}


if   n in [1]:
elif n in [2]:
elif n in [3]:
..





import sys

print("Current Python version: ", sys.version)



*testDict)
test(**testDict)




stdcalc = {	'sum': lambda x, y: x + y,'subtract': lambda x, y: x - y}

print(stdcalc['sum'](9,3))
print(stdcalc['subtract'](9,3))


test = [1,2,3,4,2,2,3,1,4,4,4]
print(max(set(test), key=test.count))



##memory use of an objecy
import sys
x=1
print(sys.getsizeof(x))

dict (zip(t1,t2))


print("http://www.google.com".startswith(("http://", "https://")))
print("http://www.google.co.uk".endswith((".com", ".co.uk")))







import itertools
test = [[-1, -2], [30, 40], [25, 35]]
print(list(itertools.chain.from_iterable(test)))

#-> [-1, -2, 30, 40, 25, 35]




a, b = 1, 2
a, b = b, a
a, *b, c = [1, 2, 3, 4, 5]
a[1:-1] = []
#>>> a =[1, 5]



zip(*z)#[[a]+[b]]


transpose nested array


invert
dict(zip(m.values(), m.keys()))  {v: k for k, v in m.items()}




#count histogramal info in a list
B = collections.Counter([2, 2, 3])
>>> A
Counter({2: 2, 1: 1})






sorted(names, key=lambda name: name.split()[-1].lower())
key 


import os
files_in_folder = os.listdir('dirname')
if any(filename.endswith('.py') for filename in files_in_folder ):

print(*row, sep=',')



sum(i for i in range(1300) if i % 3 == 0 or i % 5 == 0)

#generator

lines = (line.strip() for line in f)


And concatenating strings is inefficient:

s = "line1\n"
s += "line2\n"
s += "line3\n"
print(s)
Better build up a list and join when printing:

lines = []
lines.append("line1")
lines.append("line2")
lines.append("line3")
print("\n".join(lines))


portfolio = [   {'name':'GOOG', 'shares': 50},
  		{'name':'YHOO', 'shares': 75},
  		{'name':'AOL', 'shares': 20},
  		{'name':'SCOX', 'shares': 65}]

min_shares = min(s['shares'] for s in portfolio)






Calcs_df.columns = [ {0:"Months",1:"Changed",2:str(c)+"_23"}.get(i,c) for i,c in enumerate(Calcs_df.columns) ]






######################################
"       Are Files Identical          "
######################################

filepath1=r"\\NDATA12\milroa1$\Desktop"
filepath2 = filepath1

Config={"limit":5}

import os, filecmp, pandas as pd

filepath=filepath1

def create_quick_fileinfo_df(filepath,filepath2=None):
    files_1=os.listdir(filepath)
    Files=pd.DataFrame(data=files_1,columns=["Files"])
    Files["Folder"]=1
    if True:#filepath2 in [None]:
        files_2=os.listdir(filepath)
        Files2=pd.DataFrame(data=files_2,columns=["Files"])
        Files2["Folder"]=2
        Files = pd.concat([Files, Files2])
    Files.set_index('Folder', append=True, inplace=True)
    Files=Files.reorder_levels([1,0])

    Files["Filepaths" ] = Files["Files"].apply(lambda x:filepath+"\\"+x)
    Files["Size"      ] = Files["Filepaths"].apply(lambda x: os.path.getsize(x) )
    
    df=Files.groupby("Size")
    df1=pd.DataFrame(df.groups[0])
    
    
    dict_unique_val = { n:i  for i,n in enumerate(sorted(list(Files["Size"].unique())))}
    Files["Size_G"] = Files["Size"].apply(lambda x: dict_unique_val[x] )
    
    return(Files, dict(Files["Size_G"].value_counts()))

def compare_filenames(f1,f2):
    def create_grid_compare(df1,df2):
        return(pd.DataFrame(columns=df1["Filepaths"],index=df2["Filepaths"]).fillna(0))
    
    grid=create_grid_compare(f1,f2)
    if "limit" in Config:
       grid=grid.iloc[:Config["limit"], :Config["limit"]]
       
    grid = grid.apply(lambda x: x.index, axis=1 ).applymap(lambda x: [x] ) + grid.apply(lambda x: x.index, axis=0 ).applymap(lambda x: [x] )
    #filecmp.cmp(f1, f2, shallow=False)
    grid=grid.applymap(lambda x:filecmp.cmp(*x, shallow=False))
    return(grid)


Files1, File_dict1 = create_quick_fileinfo_df(filepath1)
Files2, File_dict2 = create_quick_fileinfo_df(filepath1)

for k, v in File_dict1.items():
    print("")
   if v>1:
       f1 = Files1[Files1["Size_G"]==k][["Filepaths"]]
       comparison_grid_df = grid=create_grid_compare(f1, f1)

################################
df=pd.DataFrame(columns=list("ABCD"),data=[[1,2,3,4],[5,6,7,8]])
# Both of these work
a=df["A"][1]
b=df.loc[1,"A"]
d=df.loc[[0,1],["A","B"]]
e=[  [df[c][r] for c in ["A","B"] ] for r in [0,1]  ]
c=df[["A","B"]][0,1]
################################
for i in range(1_020):
    print(i,10_20,500, i%500, i//500, i/500,  i/500 - i//500)

import time   
t0 = time.time()  

for i in range(3):
    if i in [0]:
       _ = [m for m in range(10_000)  ]
    if i in [1]:
       _ = [m for m in range(100_000) ]
    if i in [2]:
       _ = [m for m in range(1_000_000)]
       
    t1 = time.time()   
    print(i, t1-t0 )
    t0 = t1 
    
#######
#reverse a list with correct index
L=list("ABCDEFGHIJKLMNO")
for i, n in reversed(list(enumerate(L))):
    print(i,":",n)   
for i, n in enumerate(L):
    print(i,":",n)
        












#%%  not finished yet
####
import numpy as np
from scipy.stats import maxwell

#def gaussian(x, mu, sig):
#    return np.exp(-np.power(x - mu, 2.) / (2 * np.power(sig, 2.))) 
 
def gaussian_pdf(x, mu=0, sig=1):
    x=np.array(x)
    con_1 = 2 * np.power(sig, 2.)
    return np.exp(-np.power(x - mu, 2.) / con_1 )


def max_bol_pdf(x, sig=1):
    x=np.array(x)   
    con_1 = 2 * np.power(sig, 2.)
    con_2 =     np.power(sig, 3.)
    x_2 = np.power(x , 2.)
    return np.exp(- x_2 / con_1 )  *  x_2/con_2


m=[n/100 for n in range(1000)]

m1 = gaussian_pdf(m)
m2 = max_bol_pdf(m)


def gaussian_cdf(x, mu=0, sig=1):
    m   = [n/1000 for n in range(10000)]
    m1  = gaussian_pdf(m)
    m1c = np.cumsum(m1)
    m1c = m1c/m1c[-1]*10000
    m1c = np.round_(m1c)
    m_blank =[-1 for m in range(10001)]
    m_blank2=[ 0 for m in range(10001)]
    for i,n in enumerate(m1c):
        m_blank[ int(n)]=i
        m_blank2[int(n)]=1

    m_blank3 = [[ 0 for m in range(10001)] for _ in "----"]
    
    c1,i1,c2,i2=-1,0,1,10000
    
    for i, n in enumerate(m_blank2):
        if n ==1:
            c1,i1 = m_blank[i], i
        m_blank3[0][i], m_blank3[1][i] = c1 , i1        
    
    for i, n in reversed(list(enumerate(m_blank2))):
        if n ==1:
            c2,i2 = m_blank[i], i
        m_blank3[2][i],m_blank3[3][i] = c2 , i2  
    
    #some how find the inverse of this




def max_bol_cdf(x, sig=1):
    x=np.array(x)   
    con_1 = 2 * np.power(sig, 2.)
    con_2 =     np.power(sig, 3.)
    x_2 = np.power(x , 2.)
    return np.exp(- x_2 / con_1 )  *  x_2/con_2





def gauss(n):
    return(np.random.normal(size=n))

def gauss_iq(n,iq=100):
    return(iq+15*gauss(n))

start_men   = gauss_iq(100_000)
start_women = gauss_iq(100_000)

#salary

#def


b=np.random.uniform(0,5,400)

r = maxwell.rvs(size=100000)

# x()    __call__




# try   yeild finally 
#%%
# Basic OOP

class Polynomial:
    def __init__(self, *coeffs):
        self.coeffs = coeffs
    def __repr__(self):
        return "Polynomial(*{!r})".format(self.coeffs)
    def __add__(self,other):
        return Polynomial(*(x+y for x,y in zip(self.coeffs, other.coeffs)))
    def __len__(self):
        return len(self.coeffs)
    def __call__(self,x):
        self.x = x
        if type(x) in [int, float]:
            self.y =  sum( n *(x**i) for i, n  in reversed(list(enumerate( self.coeffs  )))   )   
        else :
            self.y = [sum( n *(x_**i) for i, n  in reversed(list(enumerate( self.coeffs  )))   ) for x_ in x ]
        return self.y
    def __getitem__(self,n):
        return(self.coeffs[n])

class turd: pass

p1 = Polynomial(1, 2,3) #  1*x2 + 2*x +3
p2 = Polynomial(0,-2,0)
p3 = p1 + p2

values = p3[:]

p1(1)
p1__x_y = [(x/10, p1(x/10)) for x in range(-500, 500)  ]


############################################################### 
1 ) Python_Basics_1        # for people who dont know how to program
2 ) Python_Basics_2        # for people new to python but can program
3 ) Python_Intermidate
4 ) Python_Useful_Modules
5 ) Python_Pandas           # basic conecpts
6 ) Python_Pandas_Advanced  # advanced as well as premier league example??
7 ) Python_Useful_Functions_and_Snippets
8 ) Python_Advanced
9 ) Python_Tricks_and_Tips
10) Python_Expert
###############################################################
#%%
#     Hieratchial_Indexing     #

London:     £ 727 1   10_236_000  9_750_500
Bristol:    £ 547 8   646_000      706_600
Leeds:      £ 533 4   1_893_000    761_500
Birmingham: £ 527 2   2_512_000  2_453_700
Glasgow:    £ 526 5   1_220_000  1_057_600
Liverpool:  £ 512 6   875_000      793_100
Manchester: £ 512 3   2_639_000  1_903_100
Newcastle:  £ 501 10  793_000      837_500
Sheffield:  £ 474 7   706_000      818_800
 
--National Average: £539





df = pd.DataFrame(index=[['a', 'a', 'b', 'b'], [1, 2, 1, 2]],columns=list('YZ')
pop.index.names = ['state', 'year']



                         "Metropol_area", "Municipal boundaries", "MB_size",  "Urban_Area",  "UA_size"  ,"UA_world", "Prim_UA", "City1"     ,"City1pop","City2","City2pop"
data =  [                        
 [   "London"                   , 13_709_000 , "Greater_London"      , 8_173_941 , ""            , 10_236_000 ,   32,    9_750_500 , "","","",""],
 [   "Birmingham-Wolverhampton" ,  3_683_000 , "West_Midlands"       , 2_736_460 , ""            ,  2_512_000 ,  183,    2_453_700 , "Birmingham", 1_073_045, "Wolverhampton", 249_470],
 [   "Manchester"               ,  2_556_000 , "Greater_Manchester"  , 2_682_528 , ""            ,  2_639_000 ,  170,    1_903_100 , "Manchester",   503_127, "",""],
 [   "Leeds-Bradford"           ,  2_302_000 , "West Yorkshire"      , 2_226_058 , ""            ,  1_893_000 ,  259,      761_500 , "Leeds"     ,   751_485, "Bradford"     , 522_452],
 [   "Liverpool-Birkenhead"     ,  2_241_000 , "Merseyside"          , 1_381_189 , "Liverpool"   ,    875_000 ,  570,      793_100 , "Liverpool" ,   466_415, "",""],
 [   "Newcastle-Sunderland"     ,  1_599_000 , "Tyne_&_Wear)"        , 1_104_825 , "Newcastle"   ,    793_000 ,  619,      837_500 , "Newcastle" ,   280_177, "Sunderland"   , 275_506],
 [   "Sheffield"                ,  1_569_000 , "South_Yorkshire"     , 1_343_601 , ""            ,    706_000 ,  701,      818_800 , "Sheffield" ,   552_698, "",""],
 [   "Southampton-Portsmouth"   ,  1_547_000 , ""                    , ""        , ""            ,    883_000 ,  565,         ""   , "","","",""],
 [   "Nottingham-Derby"         ,  1_543_000 , ""                    , ""        , "Nottingham"  ,    755_000 ,  650,         ""   , "","","",""],
 [   "Glasgow"                  ,  1_395_000 , ""                    , ""        , ""            ,  1_220_000 ,  390,    1_057_600 , "","","",""]]

"""
Urban_Area
11 (776) – Bristol – 646,000
12 (824) – Belfast – 600,000
13 (942) – Leicester – 534,000

Primary urban areas
Bristol,    706_600
Belfast,    675_600

"cities" they contain
Bristol – 428,234"""


import pandas as pd
df = pd.DataFrame(data=data)



## create a multi-index array
ii = pd.DataFrame(index=[["a","a","b"],[1,2,2]] , columns= [["d","e","e"],[4,4,5]]  ).fillna("")
ii.loc[("a",1),"d"]=5
iii = ii.loc["a",:]




###########################################################################
"Using a key with a list using functions: 'sorted, max, min, (abs, lambda)'"
x=[ -5, 1, 0, 6, -90 ]
print("x                              =", x                               )
print("max(x)                         =", max(x)                          )
print("max(x,key=abs)                 =", max(x, key=abs)                 )
print("sorted(x)                      =", sorted(x)                       )
print("sorted(x,key=lambda x:abs(x-2))=", sorted(x, key=lambda x:abs(x-2)))
##########################################################################


def index_of_same_element_in_list(A):              
    return(  [[i for i,a in enumerate(A) if a == c].index(ii) for ii,c in enumerate(A)])  
A  = [list(n) for n in "dadafe dfawefa ddffd adfwee yhtyhty pujoyuj fgeragp qefmagvr".split(" ")] 
R  = [index_of_same_element_in_list(n) for n in A]
RR = [(i,nn,[ii for ii,nnn in enumerate(n) if nnn==nn]) for i,(m,n) in enumerate(zip(R,A)) for mm,nn in zip(m,n) if mm==2]
    


def convert_nested_dict_2_list(dict_in):
    for k in dict_in:
           dict_in[k]= convert_nested_dict_2_list(dict_in[k]) if type(dict_in[k]) is dict else dict_in[k]
    return( list(dict_in.values())  )

def unnest_list(list_in):
    a=[]
    for l in list_in:
         a.extend(unnest_list(l) if type(l) is list else [l])
    return( a )

y1 = {1:12,3:{45:56,67:1,77:1,78:{12:12,90:{}}}}
y2 = convert_nested_dict_2_list(y1)
y3 = unnest_list(y2)







"""
if in cmd you type   :    python script.py  monkeys
in the python script(scipt.py) 
 sys.argv[0]="script.py"   sys.argv[1]="monkeys"
allows you to into variables from cmd to the tunning pytohn script"""


############################################################################
def bracket_info(str_):
    string_bracket_dict={"str":str_,"b_level":[0],"b_hit":[]}
    for m in str_:
        mm={"(":1,")":-1}.get(m,0)
        string_bracket_dict["b_hit"  ].append(mm)
        string_bracket_dict["b_level"].append(string_bracket_dict["b_level"][-1]+mm)
    string_bracket_dict["b_level"]=string_bracket_dict["b_level"][1:]
    string_bracket_dict["b_level2"]=[(m)*(n+(0>m)) for m,n in zip(string_bracket_dict["b_hit"  ],string_bracket_dict["b_level"])]
    return(string_bracket_dict)
def bracket_info_split(string_bracket_dict,i):
    string_bracket_dict["b_hit"   ]=string_bracket_dict["b_hit"   ][i:]
    string_bracket_dict["b_level" ]=string_bracket_dict["b_level" ][i:]
    string_bracket_dict["b_level2"]=string_bracket_dict["b_level2"][i:]   
    str_out,string_bracket_dict["str"]=string_bracket_dict["str"][:i],string_bracket_dict["str"][i:]
    print(string_bracket_dict["str"])
    return(str_out,string_bracket_dict)   
#Answer "-16763867.376068376"
string_="5+7+8-(1222+99+999*(5656)+(3334*3333+88/(222+12)))"
#['5+7+8-',['1222+99+999*',['5656'],'+',['3334*3333+88/',['222+12']]]]

#num="0123456789"
string_bracket_dict = bracket_info(string_)
z = []  
new_level = string_bracket_dict["b_level"][0]
f, string_bracket_dict = bracket_info_split(string_bracket_dict, string_bracket_dict["b_level"].index(new_level+1)  )
z.append(f,[string_bracket_dict])  
a=3
for _ in range(10):
   a=[a]
############################################################################   
def var_inserter(unravel,loc=None):
    if not loc is None:
        if loc in unravel:
            unravel=unravel[loc]
        else :
            unravek={}
    else :
        loc=""
    for key, value in unravel.items():
       exec(f"print(f'{loc}<key:{key} , value:{a}')")        
       exec(f"global {key};{key}={value}")
       exec(f"print(f'{loc}>key:{key} , value:{value}')")
temp = {"a":56}   
a=67
var_inserter(temp)
print(a)
temp = {1:{"a":56},5:{"bb":[23,44]}}  
var_inserter(temp,1)
var_inserter(temp,5)
############################################################################   
#################### Returning Functions ################################
def f(y):
   def g(x):  print(f'The sum of x: {x} and y: {y} is {x+y}')
   return g
for x in [10,100]:
    for y in [1,2]:
        myG = f(x)
        myG(y)
        f(x)(y)

############################################################################   

def add_new_blank_columns(df,columns,fill=""):
    temp = pd.DataFrame( columns=columns )
    df=df.join(temp )#.fillna(fill)
    df[columns]=fill
    return(df)

#import sys

def simple_extract_curly_brackets(_str_):
    out1="".join([aaa.split("{")[0] for aaa in _str_.split("}")])
    out2=[aaa.split("}")[0] for aaa in _str_.split("{")][1:]
    return( out1, out2 )
str_removed, str_inside_brackets = simple_extract_curly_brackets( "afadsgfadrfgi{ss}ifeeei{uiu}i" )

def example_args_nice(*args):        
    a1, a2, a3, a4 = (args + (None,)*(4))[:4]
    out=[a1,a2,a3]   
    if not a4 is None:
        out.extent(a4)
    return(out)




def extract_the_raw_header_data_from_orginal_df(file_path=None, first_columns=None, sheet_names=None, raw_data=None, reverse=False):
    if reverse :
        raw_data = "\n".join([b.replace("\t","  ",).rstrip().lstrip() for b in raw_data.split("\n")])
#        raw_out={k.replace(":","").replace("\n","") :[n.rstrip().lstrip().split(",") for n in v.split("\n")] for k,v in zip(raw_data.split("\n\n")[0::2],raw_data.split("\n\n")[1::2])}#         
        raw_out={k.replace(":","").replace("\n","") :[[nn.replace("\t","  ").rstrip().lstrip() for nn in n.split(",")] for n in v.split("\n")] for k,v in zip(raw_data.split("\n\n")[0::2],raw_data.split("\n\n")[1::2])}#   
    else :
             
        dfs = pd.ExcelFile( file_path )
        if sheet_names is None:
           sheet_names = list(dfs.sheet_names)
           print(sheet_names)
        
        def clean_temp(str_):
            str_=str(str_)
            return(str_.lstrip().rstrip().replace("- ","-"))
    
        starting_column_dict = {key:"" for key in range(first_columns)}
    
        df_strings=[]
        for sheet_name in sheet_names:    
            df_loop = dfs.parse(sheet_name, header=None, index=None).fillna("")
            df_loop = df_loop.loc[:first_columns, :]
            nested_list = [[values+clean_temp( df_loop.loc[ key, col_i ] ) for col_i in df_loop ]  for key, values in starting_column_dict.items() ]
            df_strings.append( "\n".join([",".join(n2) for n2 in nested_list ]))
        
        raw_out="\n\n".join([f"{sheet_name}:\n\n"+string for sheet_name,string in zip(sheet_names,df_strings)])
        print(raw_out)
        if False:
            print('raw_data = """\n'+raw_data+'"""')        
    return(raw_out)






def strlimit(obj,no=8):
    return str(obj).ljust(no)[:no]

def create_prints(number,cutoff):
    def empty(*args,**kwargs):
        pass
    def wrapper(cut, function):
        return function if cut else empty
    return (wrapper(n<=cutoff,print) for n in range(1,number+1))
        
print1, print2, print3, print4 = create_prints(4,2)

print1(111)
print2(222) 
print3(333)
print4(444)
   
