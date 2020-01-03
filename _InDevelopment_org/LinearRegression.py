# -*- coding: utf-8 -*-"""Created on Thu Apr 11 14:28:32 2019@author: milroa1"""

"""   Simple linear regression is what you can use when you have one independent variable and one dependent variable. 
      Multiple linear regression is what you can use when you have a bunch of different independent variables!
        
        Multiple regression analysis has three main uses.
        
          > You can look at the strength of the effect of the independent variables on the dependent variable.
          > You can use it to ask how much the dependent variable will change if the independent variables are changed.
          > You can also use it to predict trends and future values.
        
        There are some assumptions that absolutely have to be true:
        
          > There is a linear relationship between the dependent variable and the independent variables.
          > The independent variables aren’t too highly correlated with each other.
          > Your observations for the dependent variable are selected independently and at random.
          > Regression residuals are normally distributed.
"""
# Importing the libraries
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

dataset = pd.read_csv('startups.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 4].values

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
labelencoder = LabelEncoder()
X[:, 3] = labelencoder.fit_transform(X[:, 3])
onehotencoder = OneHotEncoder(categorical_features = [3])
X = onehotencoder.fit_transform(X).toarray()

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

regressor = LinearRegression()
regressor.fit(X_train, y_train)

y_pred = regressor.predict(X_test)

#ols plus others
#some cooed std err >[t]  95 cpmfoem rsqaured