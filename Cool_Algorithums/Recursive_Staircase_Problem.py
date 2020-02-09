# -*- coding: utf-8 -*-
"""
Created on Wed Jan 29 12:51:30 2020

@author: Alexm
"""

# Recursive Staircase Problem
"""
If a stairs is 5 steps long
and you can either take 1 stair or 2 stair steps
what are the possible roots

paths = find_steps_to_add_to(5,[1,2])

[1, 1, 1, 1, 1]
[2, 1, 1, 1]
[1, 2, 1, 1]
[1, 1, 2, 1]
[2, 2, 1]
[1, 1, 1, 2]
[2, 1, 2]
[1, 2, 2]

"""

# This is a recursive function(not not include paths just npaths)
# also ineffiecnt for 1,2 steps
def num_ways(n):
    if n in [0,1]: return 1
    return num_ways(n-1) + num_ways(n-2)

def num_ways2(n,mem=None):
    if mem is None:
        mem = {0:1,1:1}
    if n not in mem:
       mem[n] = num_ways2(n-1,mem) + num_ways2(n-2,mem) 
    return mem[n]

#def wrap_memorize(func,mem={0:1,1:1}):
    
    


def find_steps_to_add_to(amount,steps,nlimit=None,macro_states=False,print_=False,npaths=False):
    nlimit = nlimit if type(nlimit) is not int else [nlimit]
    def remove_if_incorrect(nlis,nlimit=nlimit,macro_states=macro_states,final=False):
        """
        This function either limits the total number of steps
        or remove entries if the order doenst matter i.e. if same macrostate
        """
        if macro_states:
           nlis = [sorted(a) for a in nlis] 
           nlis = [a for i,a in enumerate(nlis) if i==nlis.index(a)]
        if nlimit is None:
            return nlis
        if final:
            return [a for a in nlis if len(a) in nlimit]
        return [a for a in nlis if len(a)<=max(nlimit)]
    """
    for step n (say 55)
    subract possible steps from this step and 
    find the possible paths to reach them and add that possible step to it
    so for possible steps = [1,2]
      paths[55] = [i+[1] for i in paths[54]] + [i+[2] for i in paths[53]]
    each path in the dic is worked out 0 to the final amount
    """
    paths = {}
    for step_no in range(amount+1):
        if print_:
           print(step_no,amount+1)
        paths_for_this_step =[]
        for n in steps:
            n2 = step_no - n
            if n2>0:
                for temp_ in remove_if_incorrect(paths[n2]):
                    paths_for_this_step.append(temp_ + [n])
            elif n2==0:
                 paths_for_this_step.append([n])
        paths[step_no] = paths_for_this_step
    return  remove_if_incorrect(paths[amount],final=True)
            

if not __name__ == "__main__":
    paths = find_steps_to_add_to(100,[10,20],macro_states=True,nlimit=[8])
    
    """
    say you want £1.90
      how many ways to make this in english coins
      1,2,5,10,20,50,100,200
      only 20 coins or less
    
    """
    #coins = find_steps_to_add_to(190,[1,2,5,10,20,50,100,200],macro_states=True,nlimit=list(range(21)))
    
    print("Coin Paths")
    coins = find_steps_to_add_to(30,[1,2])
  # this takes a while
    print("N paths")
    print(num_ways(36))
    print("Memorize")
    print(num_ways2(59)) 
    print("End")
 








