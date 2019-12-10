# -*- coding: utf-8 -*-
"""
Created on Mon Nov 18 15:48:56 2019@author: Alexm"""

from random import random

norounds = 10000000

count=0
for rounds in range(norounds):
    if (random()**2)+(random()**2)<1:
        count +=1
        
pi_approx = count*4/norounds
error = pi_approx /(norounds**0.5)
print(f"Pi should be {pi_approx-error} to {pi_approx+error}")
