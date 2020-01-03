# -*- coding: utf-8 -*-
"""Created on Mon Sep 18 08:29:58 2017@author: milroa1"""

import random
import matplotlib.pyplot as plt
import itertools as it

def intergrate(in_array):#spe_x= [0]+list(it.accumulate(acc_x))
    return [0]+list(it.accumulate(in_array))
    
    
def add_random(list_,mu=0,sigma=1,error=1):#e_acc_x= [n + er_a*random.gauss(mu, sigma) for n in acc_x] 
    return [n + error*random.gauss(mu, sigma) for n in list_]
    

    
no=200
#def loc_acc(n=200):
mu,sigma=0,1 
error_a_s_p=[0.05,0,0]
Options=["plot"]#maybe in future Options={plot:"True"} if Options["plot"]:
# i think the error will lead to random walk
# the error accuamaltion should be error_accumalted=error*(time^0.5)
# er*t^0.5 intergrated: =er*1.5*t^1.5;double_intergrated: =er*3.75*t^2.5;

time= list(range(no))
acc_x= [random.gauss(mu, sigma) for n in range(no)]
acc_y= [random.gauss(mu, sigma) for n in range(no)]

spe_x,spe_y= intergrate(acc_x),intergrate(acc_y)

pos_x,pos_y= intergrate(spe_x),intergrate(spe_y)

e_acc_x=add_random(acc_x,error=error_a_s_p[0])
e_acc_y=add_random(acc_y,error=error_a_s_p[0])

e_spe_x=add_random(spe_x,error=error_a_s_p[1])
e_spe_y=add_random(spe_y,error=error_a_s_p[1])

e_pos_x=add_random(pos_x,error=error_a_s_p[2])
e_pos_y=add_random(pos_y,error=error_a_s_p[2])

ea_pos_x=intergrate(intergrate(e_acc_x))
ea_pos_y=intergrate(intergrate(e_acc_y))


if "plot" in Options:
    plt.plot(ea_pos_x) 
    plt.plot(pos_x) 
    plt.show()

