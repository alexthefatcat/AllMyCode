# -*- coding: utf-8 -*-
"""Created on Wed May 15 13:22:54 2019@author: milroa1"""
###########################################################################
"""
Regressions Methods
    SVR:
    TreeRegression:             Like normal descion trees but instead of a classification can give
                                rational number uses things MSE for loss,(not smooth)
    Elastic net regularization: Regression where it has both L1 and L2 terms

    Linear(normally OSL),Logestic(more classification),Multivariabel Regression   
    Neural Nets
"""
###########################################################################
print(__doc__)

# Import the necessary modules and libraries
import numpy as np
from sklearn.tree import DecisionTreeRegressor
from sklearn.svm import SVR
import matplotlib.pyplot as plt
from matplotlib import style
style.use("ggplot")

def plotter(plots,scatters,title="Decision Tree Regression"):
# Plot the results
    plt.figure()
    for point, clr in zip(plots,["darkorange","cornflowerblue","yellow"]):
        X,y,label = point
        plt.scatter(X, y, s=20, edgecolor="black", c=clr, label=label)  
    for i,(tree_, clr) in enumerate(zip(scatters,["cornflowerblue","yellowgreen"])):  
        if i==0:
           X_test, tree, label = tree_
        else:   
           tree, label = tree_  
        plt.plot(X_test, tree, color= clr, label=label, linewidth=2)
    plt.xlabel("Data")
    plt.ylabel("Target")
    plt.title(title)
    plt.legend()
    plt.show()

#%%#####################################################################################
# Create a random dataset
rng = np.random.RandomState(1)
X = np.sort(5 * rng.rand(80, 1), axis=0)
y = np.sin(X).ravel()
y[::5] += 3 * (0.5 - rng.rand(16))
#%%#####################################################################################

# Fit regression model
regr_1 = DecisionTreeRegressor(max_depth=2)
regr_2 = DecisionTreeRegressor(max_depth=5)
# min_samples_leaf is also a very slightly useful parameter 
# The minimum number of samples required to be at a leaf node:
regr_1.fit(X, y)
regr_2.fit(X, y)

# Predict
X_test = np.arange(0.0, 5.0, 0.01)[:, np.newaxis]
y_1 = regr_1.predict(X_test)
y_2 = regr_2.predict(X_test)

plotter([(X,y,"data")],[(X_test,y_1,"max_depth=2"),(y_2,"max_depth=5")])

#%%#####################################################################################
#SupportVectorRegression

#SVM:  Tries to find a function that creates a hyperplane(at 0) which sepreates the two classes
#SVR:  Find a function, f(x), with at most e-deviation, from the target y (e being ideally maximum error)
# so we do not care about prediction errors as long as there below w
# loss max([0,d-e])  # is d distance from hyperplane: 
    
clf  = SVR(gamma=1, C=1.0, epsilon=0.2)#gamma='scale'
clf2 = SVR(kernel="linear", C=1.0)#gamma='scale'

clf.fit(X, y) 
Ypred = clf.predict(X_test)

plotter([(X,y,"data")],[(X_test,Ypred,"SVC")],"SVC")

#%%#####################################################################################










