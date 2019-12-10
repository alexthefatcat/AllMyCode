# -*- coding: utf-8 -*-
"""Created on Fri Nov 15 09:07:54 2019@author: Alexm"""

from math import pi

def circle_area(r):
    if type(r) not in [float,int]:
        raise TypeError("The radius must be a non-negative real number")
    if r < 0:
        raise ValueError("The radiud can not be negative")
        
        
    return pi*(r**2)



  
    