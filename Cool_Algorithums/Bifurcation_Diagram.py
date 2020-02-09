# -*- coding: utf-8 -*-
"""Created on Fri Feb  7 19:56:07 2020@author: Alexm """
import matplotlib.pyplot as plt 
#%matplotlib qt # seperate window
#%matplotlib inline # in console

def func(r,x):
    return r*(x)*(1-x)

def func2(r):
    return lambda x: func(r,x)
    
def repeat_func(start,func,repeated=2000,amount=256,roundingto=4):
    """ keep repeating 
       x = f(x)
       so effectivly f(f(f(f(f(x)))))... but 2000 times
       if it can reach a stable set it should of 
       then collect a collection of number after this
       defualt is 256
       then round them so numbers that are very very near are the same
       then get only the unique number sets of that collection
    """
    x = start
    for n in range(repeated):
        x = func(x)
        
    x_stable_collection = []
    for n in range(amount):
        x = func(x)
        x_stable_collection.append(x)
    x_stable_collection = [round(a,roundingto) for a in x_stable_collection]
    x_stable_collection = list(set(x_stable_collection)) # unique values   
    return x_stable_collection

def is_rational_number(n):
    return n not in [float("-inf"),float("inf"),float("nan")]

def range2(max_value,step_size=0.05,c=0):
    return [i*step_size for i in range(c,int(max_value/step_size))]

def loop_through_differnt_r_values_and_create_dic(max_value,step_size):
    results = {}
    #4 goes to infiite
    for r in range2(max_value=4.5, step_size=0.0001):
        results[r] = repeat_func(start,func2(r))
    ##############################################
    # remove irrational numbers
    results = {a:[bb for bb in b if is_rational_number(bb)] for a,b in results.items() }
    results = {a:b for a,b in results.items() if len(b)>0} # remove number lists with no numbers
    ##############################################
    return results

start     = 0.666
max_value = 4.5
step_size = 0.0001
print(f"Number of Iterations of loop is {int(max_value/step_size)}")

results = loop_through_differnt_r_values_and_create_dic(max_value,step_size)

results2 = [ (a,bb) for a,b in results.items() for bb in b ]
results3 = [ (a,len(b)) for a,b in results.items() ]
########################################################
# PLot the values at a differnent r # not great
x,y = zip(*results2)
plt.scatter(x,y,s=0.3)

#plot the number of values at a different r
x,y = zip(*results3)
plt.scatter(x,y,s=5)
del x,y


