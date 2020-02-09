# -*- coding: utf-8 -*-
"""Created on Sat Jan 11 04:36:11 2020@author: Alexm"""

"""

Lottery_Lowest_Unique_Numbers_Wins

There was a competion
where the winner who was the person who picked the lowest number won
if you know the number of betters or votes whats the best probablity distrition to bet on each number

"""
import random
import numpy as np
from numpy import array

def cumsum(lis):
   out=[]
   current_val  = 0
   for n in lis:
       current_val+=n
       out.append(current_val)
   return out 

def normalize_so_sum_is_one(lis):
    s = sum(lis)
    return [n/s for n in lis]
 
def mean(lis):   
    return sum(lis)/len(lis)

def hist(lis):
    # histogram or value counts
    out ={}
    for l in sorted(lis):
        out[l] = out.get(l,0)+1
    return out

def wrap_functions(*func):
    main_func = func[0]
    def func_new(*args,**kwargs):
        out = main_func(*args,**kwargs)
        for wfunc in func[1:]:
            out = wfunc(out)
        return out
    return func_new
###############################################################################
def create_poly(args):
    # args= a,b,c => f(x): a*x^2 +b*x +c
    def poly(x):
        return sum([i2*(x**i1) for i1,i2 in enumerate(reversed(args))])
    return poly

def polynomial_fit(probs,terms=8,ignore_low=True):
    probs2 = probs
    if ignore_low:
        probs2=[p for p in probs if p>0.000001]
    x = array(list(range(len(probs2))))
    y = array(probs2)
    z = np.polyfit(x, y, terms)
    return z

def create_apporox_for_1000(n=1000):
    #print(polynomial_fit(probs))
    z = array([-2.75750341e-18,  1.47477128e-15, -3.10862432e-13,  3.28155254e-11, -1.86580500e-09,  6.22647201e-08, -1.55549672e-06,  1.68555685e-05,  7.22527318e-03])
    above_zero = lambda x:max([0,x])
    poly = create_poly(list(z))
    poly = wrap_functions(poly,above_zero)
    return [poly(i) for i in range(n)]

def tan_appoximation_of_probablity_descions(votes=1000):
    import math
    tanshift = votes**(2/3)
    maxval   = votes**(1/3)
    remove_after = int(math.tau*tanshift/4)
    
    pp = [ maxval-math.tan(i/tanshift) for i in range(votes)]
    
    for n in range(len(pp)):
        if n >remove_after or pp[n]<0:
           pp[n]=0
    pp = normalize_so_sum_is_one(pp)
    return pp

#%%###############################################################################
def update_probablities_based_on_lottery_winners(starting_probs,updates=[1.3,1.2,1.15,1.10,1.08,1.05,1.03,1.02]):
    probs = starting_probs
    for iupdate,update in enumerate(updates):
        print("#>>",update)
        winners=[]
        for effort in range(efforts):
            guess1 = sorted([random.random() for i in probs])
            iprobs = cumsum(probs)
            picks = [find_value(g, iprobs) for g in guess1]
            losers = [p for p in picks if picks.count(p)>1]
            winner___lowest = [p for p in range(votes) if p not in losers][0]
            winners.append(winner___lowest)
            probs = update_descion_probabilities(probs,winner___lowest,update,0)
        print("mean winner>>",mean(winners),"max",max(winners))
    return probs,iprobs,winners




if __name__ == "__main__":
    
    def find_value(v,cprobs):
        return len([n for n in cprobs if n<v])
    
    def update_descion_probabilities(probs,winner___lowest,update,add=0.00005):
        probs[winner___lowest] = ((probs[winner___lowest]+0.00002) *update)
        return normalize_so_sum_is_one(probs)  
    
    ########################################################
    Config = {"Stating_Position":("Good","Basic")[0]}
    
    votes   = 100 # number of betters that are going to place a bet
    efforts = 500 # times to repeat the algorthium and update
                  # this is reapeted for each of the updates
    
    if   Config["Stating_Position"]== "Basic":
       starting_probs = [1/votes for i in range(votes)]
       
    elif Config["Stating_Position"]== "Good":
        probs_approx = create_apporox_for_1000()
        starting_probs = probs_approx
        starting_probs = starting_probs[:votes]
    starting_probs = normalize_so_sum_is_one(starting_probs)

        
    
    updates = [3,2,1.6,1.4,1.2]
    updates = [1.3,1.2,1.15,1.10,1.08]
    updates = [1.05,1.03,1.02]
    
    updates = [1.3,1.2,1.15,1.10,1.08]+[1.05,1.03,1.02]
    

    # probs is best probability to bet to win,iprobs is this intergrated, 
    # winners is a list of all the winning results
    probs,iprobs,winners = update_probablities_based_on_lottery_winners(starting_probs,updates)
        
        
    iprobs2 = iprobs[:350]        
    print("This value should be low", probs[len(probs)//2]*1000)
            
    winners_hist = hist(winners)
    
    print("The highest winner has this probablity to betted on, and ites neiborus",[probs[max(winners)-1] , probs[max(winners)], probs[max(winners)+1] ])
    
    
    
    
    







if False:
    n = votes
    prob_mean = mean(probs[:100])
    p = [p>prob_mean/3 for p in probs]
    for n in range(157,180):
       probs[n] = prob_mean/4
    probs = normalize_so_sum_is_one(probs)

# newton approx// polynomial fit
 
    
    
if False:

    def inbetween(value,min_=None,max_=None):
        out = value
        if min_ is not None:
            out = max([min_,out])
        if min_ is not None:
            out = min([max_,out])
        return out
    def inbetween2(value,min_,max_):
        return min([max_,max([min_,value])])
    def inbetween3(value,min_,max_):
        return sorted([value,min_,max_])[1]
    inbetween2(5,1,4)
 







