# -*- coding: utf-8 -*-"""Created on Mon Sep 18 10:36:05 2017@author: milroa1"""

#%%################################################################################################################################      
"""                                           Draw a Interactive graph rotatable 3D Plot                                       """       
###################################################################################################################################

## drawing interactive graph

# Tools > preferences > IPython console > Graphics > Graphics backend > Backend: Automatic
# Then close and open Spyder. or %matplotlib auto

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

def muldiv(a,b):
    if b<1:      out=a*b  
    else:        out=a/b  
    return(out)
    
vmuldiv = np.vectorize(muldiv)

# Set up grid and test data
nx, ny = 400, 400
x =[(n-200)/80 for n in range(nx)]
y =[(n-200)/80 for n in range(ny)]
X, Y = np.meshgrid(y, x)
Z = vmuldiv(X, Y)

hf = plt.figure()
ha = hf.add_subplot(111, projection='3d')
  # `plot_surface` expects `x` and `y` data to be 2D
ha.plot_surface(X, Y, Z)
plt.draw()

#%%################################################################################################################################      
"""                                           Draw a Interactive graph rotatable 3D Plot                                       """       
###################################################################################################################################


#color plot
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm

x,y,temp = np.loadtxt('data.txt').T #Transposed for easier unpacking
nrows, ncols = 100, 100
grid = temp.reshape((nrows, ncols))

plt.imshow(grid, extent=(x.min(), x.max(), y.max(), y.min()), interpolation='nearest', cmap=cm.gist_rainbow)
plt.show()
###############################################################################








