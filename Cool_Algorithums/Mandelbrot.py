# -*- coding: utf-8 -*-
"""Created on Sat Feb  8 04:34:07 2020@author: Alexm"""
#  Mandelbrot Set

import numpy as np
import matplotlib.pyplot as plt

def mandelbrot(Re,Im,max_iter=100):
    """
    f(x) = x^2 +c
    x0=0
    xn = f(xn-1)
    if any xn here 1 to 100 is above 4 return that n
    """
    c = complex(Re,Im)
    z = 0.0j
    for i in range(max_iter):
        z = z*z + c
        if (z.real*z.real +z.imag*z.imag) >=4:
            return i
    return max_iter

def plot_image(results,dimensions =[-2,1,-1,1] ):
    plt.figure(dpi=100)
    plt.imshow(results.T,cmap="hot",interpolation="bilinear",extent=dimensions)
    plt.xlabel("Real")
    plt.ylabel("Imaginary")
    plt.show()    

def fill_array_with_function(func,x,y,columns =2000,rows =2000):
    results = np.zeros([rows,columns])
    for row_ind,xx in enumerate(np.linspace(x[0],x[-1],num=rows)):
        for col_ind,yy in enumerate(np.linspace(y[0],y[-1],num=columns)):
            results[row_ind,col_ind] = func(xx,yy)
    return results

xs,ys=[-2,1],[-1,1]
results = fill_array_with_function(mandelbrot,x=xs,y=ys)
plot_image(results,dimensions =xs+ys)










