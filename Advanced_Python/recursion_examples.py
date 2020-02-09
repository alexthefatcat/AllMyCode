# -*- coding: utf-8 -*-
"""Created on Fri Nov 15 10:16:00 2019@author: Alexm"""

def IterateThroughFunction(func,coun=32):
    for n in range(coun):
        c=0
        if n==coun:
            n=5
        print(f"   Fib fib({n}) = {func(n)} , calls = {c}")
    
    
#########################################################################################    
def fib(n):
    if n in [0,1]:
        return 1
    return fib(n-1)+fib(n-2)

print("Basic Slow One, Takes a Long Time")
IterateThroughFunction(fib)
#########################################################################################
#  memoization
fib_cach={}
def fib2(n):
    if n in fib_cach:
        return fib_cach[n]
    if n in [0,1]:
        return 1
    fib_cach[n] = fib2(n-1)+fib2(n-2)
    return fib_cach[n]

print("Faster Caches to dic")
IterateThroughFunction(fib2,1000)
#########################################################################################
#  memoization
from functools import lru_cache

@lru_cache(maxsize = 1000)
def fib(n):
    if n in [0,1]:
        return 1
    return fib(n-1)+fib(n-2)

print("Fasher uses a decorator to speed it up")
IterateThroughFunction(fib)
#########################################################################################
# Extract Nested Dict
# Loop through all files in folder
# that disc probelwem with three columns hanoi




#########################################################################################
def recursion_create_nested_list_with_function_applied_to_it(size,loc=None,dimno=None,func=None):
    if dimno is None:
       dimno,loc = 0, []
    if len(size)==dimno:
        return func(loc)
    return [recursion_create_nested_list_with_function_applied_to_it(size,dimno=dimno+1,loc=loc+[n],func=func) for n in range(size[dimno])]

func = lambda loc:sum([abs(n-3) for n in loc])

out = recursion_create_nested_list_with_function_applied_to_it(size=[6,6,6], func=func)




#########################################################################################

def add_recursion(a,b):
    if b==0:
        return a
    b,a = b-1,a+1 
    return add_recursion(a,b)
    
m = add_recursion(5,5) # 10 out

#########################################################################################

def unnest_nested_container(obj,unnested_list=None):
     if unnested_list is None:
        unnested_list = []

     print(obj,unnested_list)
     
     if hasattr(obj, '__iter__'):
         obj_index = range(len(obj)) if type(obj) is list else None
         obj_index = obj.keys()      if type(obj) is dict else obj_index  
         print(obj_index)
         for ind, obj_child in zip(obj_index, obj):
             print("#",ind)
             unnested_list = unnested_list +[ind]
             print("###",unnested_list)             
             val = unnest_nested_container(obj_child,unnested_list=unnested_list,unnested_list=unnested_list)
     else:
         
         return [unnested_list,obj]
         
             
dog = [[1,2,[2,3,[[4]]]],5,[[6]]]
val = unnest_nested_container(dog)          



# basic recursion
# hanoi
# binary split


#########################################################################################
def flatten(nlis,out=None):
    out = [] if out is None else out
    if type(nlis) is list:
       for n in nlis:
           out = flatten(n,out)
    else:
        return out+[nlis]
    return out

v = flatten([1,2,[3,[4,5],6,7,[8,9,[10]]],[11,[12,13]]])









