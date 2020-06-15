# -*- coding: utf-8 -*-
"""
Created on Fri Mar 13 00:15:01 2020@author: Alexm
"""

def binary_search(volunters, task, reverse=False, find=None, loc=None, sz=None):
    """
    if reverse is true
    """
    if sz is None:
        sz   = len(volunters)  
    if loc is None:
        loc  = sz-1 if reverse else 0
    if find is None:
        find = task(volunters[ loc ])

    if sz==1:
        return loc
    
    sz2a   = sz//2  

    loc2_ = loc-sz2a if reverse else loc+sz2a
    out   = task(volunters[ loc2_ ])
    
    loc2  = loc2_   if out==find else loc
    sz2   = sz-sz2a if out==find else sz2a

    return binary_search(volunters, task, reverse, find, loc2, sz2)






if __name__ == "__main__":
    
    volunters = [i for i in range(9601)]
    task      = lambda i: i <5000
    
#    volunters = [i for i in range(9)]
#    task      = lambda i: i <4
    
    task_outputs = [task(e) for e in volunters]
    
    # find the first output from task different to zeroth
    loc1 = binary_search(volunters,task)
    
    # find the last output from task different to last one  
    loc2 = binary_search(volunters,task,reverse=True)
    
    # find the first output that is True
    loc3 = binary_search(volunters,task,find=True)
    
    # find the last output that is False    
    loc4 = binary_search(volunters,task,reverse=True,find=False)    
    
    # Will Only ever see True but looking for False so wont move
    loc5 = binary_search(volunters,task,find=False)
    
    # Will Ony see True and looking for True so stays low
    loc6 = binary_search(volunters,task,reverse=True,find=True)    
        
    
    print([ (i,task(volunters[i])) for i in range(loc1-2,loc1+3) ])

    
    