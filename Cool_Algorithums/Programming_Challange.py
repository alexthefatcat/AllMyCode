# -*- coding: utf-8 -*-
"""Created on Sat Feb  8 17:55:39 2020 @author: Alexm"""

class create_top_scorer():
    def __init__(self):
        self.mem =[None,None]
    def __call__(self,score,solution):
        if score is not None:
            if self.mem[0] is None or self.mem[0]>score: 
               self.mem = [score,[i for i in solution]]
        return self.mem
    def top_score(self):
        return self.mem[0]
    def top_solution(self):
        return self.mem[1]
    
def scorer(temp):
    if None in temp:
        return None
    return max(temp)-min(temp)


def iterate_through_sorted_joined_list(v):
    # challagne find the numbers from the three differnt numbers
    # which are nearist
    v2   = sorted([(ll,i) for i,l in enumerate(v) for ll in l])

    top_scorer = create_top_scorer()
    solution = [None for i in range(len(v))]    
    for n,l in v2:
        solution[l] = n
        score     = scorer(solution)
        top_scorer(score,solution)
    top_solution  = top_scorer.top_solution()
    return top_solution
        

def iterate_through_all_lists_at_same_time(v): 
    """
    v_info has three lists to start 0s, first elem in list,last elem in list
    
    each iteration it finds the index of the value which is smallest(unless it has reached the end)
    and adds one to it
    as going through find the one with the top score
    
    
    ## try this thing working down instead
    ## should simplify thigns
    ## so indexs start at len -1        
    
    """
    top_solution2 = create_top_scorer()
    v_info = list(zip(*[[0,l[0],len(l)-1] for l in v])) # zero,first,last
    while True:
        # a is index , b is value, c is length of list
        # find minimum value not at the end
        minvinfo1 = min([b for a,b,c in zip(*v_info) if a!=c])
        # add 1 of the index of the smallest
        v_info[0] = [ a+ (b==minvinfo1 and a!=c) for a,b,c in zip(*v_info) ]
        # now using the index get the value of the one above it
        v_info[1] = [ v[i][a] for i,(a,b,c) in enumerate(zip(*v_info))]
        
        solution   = v_info[1]
        score      = scorer(solution)
        top_solution2(score,solution)
        if v_info[0]==list(v_info[2]):
            top_solution = top_solution2.top_solution()
            break
    return top_solution
    

if __name__ == "__main__":
        """
        find a number from each of the three lists which is nearist
        """
        v = [[0, 1, 4, 17, 20, 25, 31],[5, 6, 10],[0, 3, 7, 8, 12]]
        top_solution1 = iterate_through_sorted_joined_list(v)
        top_solution2 = iterate_through_all_lists_at_same_time(v)
    
    
    #functional rosnfom
    #add memory memoization
    
    
    
    
    
    
    
    
    