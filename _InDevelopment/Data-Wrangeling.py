# -*- coding: utf-8 -*-"""Created on Wed Aug 15 14:46:20 2018@author: milroa1"""
#%%######################################################################################
"""                      Data Wrangaleing                                          """
#########################################################################################


## countnans, mixedtpyes, parse html pdf

import pandas as pd
data = pd.read_csv('data/TB_notifications_2016-12-18.csv')

#for chunk in pd.read_csv(file, chunksize=500):
#    # process the chunk

def value_counts_special(df):#min([4,x.value_counts().shape()[0] ])
    print(df.apply(lambda x: [(x.value_counts().index[n], x.value_counts().iloc[0] )  for n in range(4)] ))

def quick_convert_string_to_numbes(p):
    p1="".join([n if n in ".0123456789" else " " for n in p])
    return [n for n in p1.split(" ") if not n in ""]



data.info

data.shape  #: have a look at the size of it and the ifist 10 sets of values

data.columns
data.index
data.head()

if False:
    
    data.describe(include="all")
    df.dtypes
    data.nunique(axis=0)   
    pd.isnull(data).any() # maybe dropna
    
    data.sample()
    
    value_counts_special(df)
    
    if False:
        data["col_1"].value_counts()
        #one hot encodeing
        df=pd.get_dummies(df, columns=["col_1"]).head()
    

    #data["colname"].unique()

    
#only columns that begin with new
data.filter(regex='^new').head()
data.filter(like='514').head()
data.filter(like='514').max()
data.filter(like='514').apply(lambda x: x + 1).max()

new = data.filter(regex='^new.*(f|m).*[0-9]$')
crit = new.columns.map(lambda x: x.endswith('04') | x.endswith('514'))
new.columns[~crit]

df = pd.concat([data[['country', 'g_whoregion', 'year']], data[new.columns[~crit]]], axis=1)

import re
df.rename(columns=lambda x: re.sub('newrel', 'new_rel', x), inplace=True)
melted = pd.melt(df, id_vars=['country', 'g_whoregion', 'year'], var_name='key', value_name='cases')
melted.head()


melted[['new', 'type', 'sexage']] = melted['key'].apply(lambda x: pd.Series(x.split('_')))
melted.head()
melted.drop('new', axis=1, inplace=True)

melted[['sex'      ]] = melted['sexage'].apply(lambda x: pd.Series(x[0]))# "m34" > "m"
melted[['age_range']] = melted['sexage'].apply(lambda x: pd.Series(x[1:]))

melted.to_csv('data/tidy_who.csv', index=False)
melted['cases'].dtypes#>dtype('float64')
melted.groupby('type').size()

import matplotlibmelted.groupby(('year', 'sex'))['cases'].sum().unstack().plot()

#pivoted_df = df.pivot(index='Date', columns='Code', values='VWAP')

data['column_numerical'].plot()
data['column_numerical'].hist()

