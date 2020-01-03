# -*- coding: utf-8 -*-"""Created on Tue Feb 26 14:07:502019@author: milroa1"""

import pandas as pd
from sklearn import svm
from sklearn.svm import SVC
from sklearn.datasets import load_iris
data = load_iris()

import numpy as np
import matplotlib.pyplot as plt

def remove_cm_from_list(lis):
    return [n.replace(" (cm)","") for n in lis]

df = pd.DataFrame(data=data["data"], columns=remove_cm_from_list(data["feature_names"]))
df["target"] = data["target"]
df["target_names"] = df["target"].map({i:n for i,n in enumerate(data["target_names"])})
# pd.Series(["a", "b", "c", "a"], dtype="category")
cols = ['sepal length', 'sepal width', 'petal length', 'petal width'] # <   list(df.columns)
cols2 = cols[:2]

    

#######################################################################################
#        Support-vector machines are supervised learning models with associated learning algorithms that analyze data used for classification and regression analysis. 
#        Given a set of training examples, each marked as belonging to one or the other of two categories, an SVM training algorithm builds a model that assigns new
#         examples to one category or the other, making it a non-probabilistic binary linear classifier (although methods such as Platt scaling exist to use SVM in
#        a probabilistic classification setting). An SVM model is a representation of the examples as points in space, mapped so that the examples of the separate
#        categories are divided by a clear gap that is as wide as possible. New examples are then mapped into that same space and predicted to belong to a category
#        based on which side of the gap they fall.
#        
#        In addition to performing linear classification, SVMs can efficiently perform a non-linear classification using what is called the kernel trick, implicitly
#         mapping their inputs into high-dimensional feature spaces.
#        
#        When data is unlabelled, supervised learning is not possible, and an unsupervised learning approach is required, which attempts to find natural clustering
#         of the data to groups, and then map new data to these formed groups. The support-vector clustering[2] algorithm, created by Hava Siegelmann and Vladimir
#         Vapnik, applies the statistics of support vectors, developed in the support vector machines algorithm, to categorize unlabeled data, and is one of the most
#         widely used clustering algorithms in industrial applications.
#######################################################################################
"""SVM uses support vectors to create a hyperplane that best seperates the data by maximizing the marging so for linear ones this is quite easy
Nonlinear SVM use the kernal trick where they dont have to transform the data points into another space to calulate the margin but by the dot product of points and transforming them"""
# Hard-margin
# Soft-margin: To extend SVM to cases in which the data are not linearly separable, we introduce the hinge loss function.
# adds an extra thing to miminize where the algorithum is punhed by how fat the wronly labelled point is form the hyperplane

# rbf gaussian keranl
"""The effectiveness of SVM depends on the selection of kernel, the kernels parameters, and soft margin parameter C.
 A common choice is a Gaussian kernel, which has a single parameter gamma . The best combination of C gamma, 
 gamma  is often selected by a grid search with exponentially growing sequences of C."""
 
#Potential drawbacks of the SVM include the following aspects:

"""Requires full labeling of input data
Uncalibrated class membership probabilities -- SVM stems from Vapniks theory which avoids estimating probabilities on finite data
The SVM is only directly applicable for two-class tasks. Therefore, algorithms that reduce the multi-class task to several binary problems have to be applied; see the multi-class SVM section.
Parameters of a solved model are difficult to interpret."""

#Multiclass SVM
#Multiclass SVM aims to assign labels to instances by using support-vector machines, where the labels are drawn from a finite set of several elements.
#
#The dominant approach for doing so is to reduce the single multiclass problem into multiple binary classification problems

#Regression
#
#Support-vector regression (prediction) with different thresholds ε. As ε increases, the prediction becomes less sensitive to errors.
#A version of SVM for regression was proposed in 1996 by Vladimir N. Vapnik.

# Technically speaking, large gamma leads to high bias and low variance models, and vice-versa.

#Scale your data
#
#This can become an issue with SVMs. According to A Practical Guide to Support Vector Classification
#Because kernel values usually depend on the inner products of feature vectors, e.g. the linear kernel and the polynomial kernel, large attribute values might cause numerical problems.
#
#  Three main ones:-   linear     poly      rbf





#######################################################################################
def create_array_of_all_points_grid(df_, columns):
    """Return  a long array of cordinated on a grid so that it can be coloured,
    
    all_points, shape = create_array_of_all_points_grid(train, cols2)
    """
    def min_max_of_df(df_temp, columns):
        out = []
        for column in columns:
           out = out +[ min(df_temp[column]),max(df_temp[column])]
        return out
    x_min, x_max, y_min, y_max = min_max_of_df(df_, columns)
    XX, YY = np.mgrid[x_min:x_max:200j, y_min:y_max:200j] 
    return np.c_[XX.ravel(), YY.ravel()] , XX.shape

def colour_the_plot_descion(Z, all_points, shape, clf):
    # Put the result into a color plot
    Z, XX, YY = Z.reshape(shape), all_points[:,0].reshape(shape), all_points[:,1].reshape(shape)  
 
    plt.pcolormesh(XX, YY, Z > 0, cmap=plt.cm.Paired)
    plt.contour(XX, YY, Z, colors=['k', 'k', 'k'], linestyles=['--', '-', '--'], levels=[-.5, 0, .5])

def plot_points(df_,fig_num, cols2, target, add_circle_around_points=True,create_plt=True):
    """
    This plots the point of dataframe
    df, fignum,cols_X,col_Y
    """
    if create_plt:
        plt.figure(fig_num)    
    plt.clf()
    plt.title(kernel)
    plt.scatter(df_[cols2[0]], df_[cols2[1]], c=df_["target"], cmap=plt.cm.Paired, zorder=10, edgecolor='k', s=20)
    if add_circle_around_points:
        # Circle out the test data        
        plt.scatter(df_[cols2[0]], df_[cols2[1]], facecolors='none',  zorder=10, edgecolor='k', s=80)
    plt.axis('tight')
#######################################################################################

train, test = df.iloc[:80], df.iloc[80:]


# fit the model
for fig_num, kernel in enumerate(('linear', 'rbf', 'poly')):
    clf = svm.SVC(kernel=kernel, gamma=10)
    clf.fit(train[cols2], train["target"])
    
    plot_points(train, fig_num, cols2, "target")
    
    if fig_num ==0:
       all_points, shape = create_array_of_all_points_grid(train, cols2)   
       
    Z = clf.decision_function(all_points)

    colour_the_plot_descion(Z, all_points, shape, clf)
    #plot_points(test, fig_num, cols2, "target",False,False)

plt.show()








#Optimal hyperplane for linearly separable patterns
#– Extend to patterns that are not linearly
#separable by transformations of original data to
#map into new space – the Kernel function
#
#Support vectors are the data points that lie closest
#to the decision surface (or hyperplane)
#• They are the data points most difficult to classify
#• They have direct bearing on the optimum location
#of the decision surface






















