# -*- coding: utf-8 -*-
"""Created on Tue Apr  9 16:01:55 2019@author: milroa1"""

"     Data Analysis Guide      "
#Original data is separated by delimiter “ ; “ 
import pandas as pd

## maybe write a class of this, which prints out the code as well

df = pd.read_csv(filepath)


#%%
df.head()
df.tail()
df.shape
#%%
df.info # check what types as well as nan coun

df.describe
#%%
## look at the target column

df["quality"].unique()
df["quality"].value_counts()
#%%
## show what variables are correlated
df.corr(annot=True)

boxplot = df.boxplot(column=['Col1', 'Col2', 'Col3'])



