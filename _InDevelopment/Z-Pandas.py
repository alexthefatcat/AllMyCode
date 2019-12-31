# -*- coding: utf-8 -*-
"""Created on Mon Oct  2 10:32:40 2017@author: milroa1"""
section=[5.4,5.3]
# generally i see theres four levels of coding
# knowing if its possible, knowing roughly how to do it, knowing nearly excatly how to do it,know it
# with the first three there is cast difference in time to google the subject

# should probably put isnumeric
# where
# create random one
# merge,concat,join.. another one??



#**************************************************************
"""                     Advanced                          """
#**************************************************************

# -load in an example example set of sata Iris data set
#- categorical data
#- create data, look at memory footprint, is column right to covnvert, convert column to categorical
#- check if it contains mixed types and change accordingly
#- read in large not in memory but blocks
#- downcast value to improve memory footprint
#- maybe write a function called compress using downcasting and catogrizing
#- dealing with multiple datatypes in a column


#-HDF5 with Pandas

#%% Load Iris data good example
from sklearn.datasets import load_iris
import pandas as pd

data = load_iris()
df = pd.DataFrame(data.data, columns=data.feature_names)
df.head()

#%% Quick Create Blank DataFrame
    
examp_df = pd.DataFrame(index=range(10),columns=list("ABCDEFGHIJK")).fillna("0")


#%% map using dict
dict_map={'sepal length (cm)':"sepal_l",'sepal width (cm)':"width"}
#df2=df.map(dict_map)
df['sepal length (cm)']=df['sepal length (cm)'].map(dict_map)

#%% Section 5.4 
#improve memory use of Pandas
# -Downcasting numeric columns to more efficient types.
# -Converting string columns to the categorical type.

import pandas as pd
df = pd.DataFrame({"A":["a","b","c","a"]})

df["B"] = df["A"].astype('category')


#%% Print out the memory of the DataFrame,find if column is suitable for category,change to category


import numpy as np,pandas as pd
dtypes = ['int64', 'float64', 'datetime64[ns]', 'timedelta64[ns]','complex128', 'object', 'bool']
data = dict([ (t, np.random.randint(100, size=5000).astype(t)) for t in dtypes])
df = pd.DataFrame(data)
###################################################################
# print out memory info
df.info()#not accurate
df.info(memory_usage='deep')
df.memory_usage()
print(df.memory_usage().sum())#total
df.describe()
####################################################################
print("Number of unique values in the object column: ",df['object'].nunique()     )
print("The name and count of the values\n",           df['object'].value_counts())
print("the number of rows in the dataframe: ",len(df))
print(df.dtypes)

#print out the dtype strings I think fall under object
print(str(df['object'     ].dtype))
print(str(df[df.columns[1]].dtype))
####################################################################
# change the column to a category
df['object_cat'] = df['object'].astype('category')

df.memory_usage()






#%%
if 5.4 in section:
  pass
  #print animated image moving graph




# apperently to check the variables type is correct use in python
use isinstance(object, classinfo) and type(object).

# only read in part of the file pandas


#%% large data shavnge to a SQL format and allows pandas to access the files using SQL comands
import pandas as pd
from sqlalchemy import create_engine

print pd.read_csv(file, nrows=5)

csv_database = create_engine('sqlite:///csv_database.db')

chunksize = 100000
i,j=0,1

for df_chunk in pd.read_csv(file, chunksize=chunksize, iterator=True):
      df_chunk = df_chunk.rename(columns={c: c.replace(' ', '') for c in df_chunk.columns}) 
      df_chunk.index += j
      i+=1
      df_chunk.to_sql('table', csv_database, if_exists='append')
      j = df_chunk.index[-1] + 1


df = pd.read_sql_query('SELECT * FROM table', csv_database)

df = pd.read_sql_query('SELECT COl1, COL2 FROM table where COL1 = SOMEVALUE', csv_database)
#%% Using Pandas with Big Data using Sql
import pandas as pd
import numpy as np
import sqlite3

# Look at the first few rows of the CSV file
pd.read_csv("data.csv", nrows=2).head()
# Peek at the middle of the CSV file
pd.read_csv("data.csv", nrows=2, skiprows=7, header=None).head()

#read in by chunk
for chunk in pd.read_csv("data.csv", chunksize=4):
    print(chunk.loc[0, "last_name"])


df = pd.read_csv("data.csv", nrows=5)

df.head()
df.dtypes
df.memory_usage()


# Cast during the csv read
df = pd.read_csv("data.csv", nrows=5, dtype={"active": np.int8})  

# ...or cast after reading 
df["age"] = df["age"].astype(np.int16)
df["id" ] = df["id" ].astype(np.int16)

df.dtypes
df.memory_usage()


df.fillna(value={"sex": "NAN"}, inplace=True)
df["sex"].unique()


#sqlite3

import sqlite3
connex = sqlite3.connect("all_your_base.db")  # Opens file if exists, else creates file
cur = connex.cursor()  # This object lets us actually send messages to our DB and receive results


for chunk in pd.read_csv("data.csv", chunksize=4):
    chunk.to_sql(name="data", con=connex, if_exists="append", index=False)  #"name" is name of table 
    print(chunk.iloc[0, 1])


sql = "CREATE TABLE person (id INTEGER PRIMARY KEY, first_name TEXT, last_name TEXT, age INTEGER);"

sql = "SELECT DISTINCT sex FROM data;"  # A SQL command to get unique values from the sex column 
cur.execute(sql)  # The cursor gives this command to our DB, the results are now available in cur

cur.fetchone()  # Pop the first result of the query like `next()`

sql = "SELECT * FROM data WHERE id IN " + str_matching + ";"
print(sql)

df = pd.read_sql_query(sql, connex)
df.head()

#%%
chunk_info=[]
file_name="data.csv"
column_names=pd.read_csv(file_name, chunksize=1)
no_columns=len(column_names)

df_i=      {file_name     : "data.csv"}
df_i.append{column_names  : pd.read_csv(df_i[file_name], chunksize=100000)}
df_i.append{no_columns    : len(df_i[column_names])}


#for chunk in pd.read_csv(file_name, chunksize=100000):#iterator
#    chunk_info.append(
#          chunk[column_names[0]].agg([np.sum, np.mean, np.std, len]))

for chunk in pd.read_csv(df_i[file_name], chunksize=100000):# iterator=True,
    chunk_info.append(
          chunk[df_i[column_names][0]].agg([np.sum, np.mean, np.std, len]))


chunk_info_df=pd.concat(chunk_info)


#%%
def foo(bar=[]):        # bar is optional and defaults to [] if not specified
    bar.append("baz")    # but this line could be problematic, as we'll see...
    return bar

def fooo(bar=[]):        # bar is optional and defaults to [] if not specified
    bar=bar+["baz"]    # but this line could be problematic, as we'll see...
    return bar
    
print(foo())           #>["baz"]
print(foo())           #>["baz", "baz"]
print(foo(),foo())   #>['baz', 'baz', 'baz', 'baz'] ['baz', 'baz', 'baz', 'baz']
print(fooo(),fooo()) #>['baz'] ['baz']


#%%

odd        = lambda x : bool(x % 2)
numbers    = [n for n in range(10)]
numbers[:] = [n for n in numbers if not odd(n)]  # ahh, the beauty of it all
#>numbers is [0, 2, 4, 6, 8]

#%%
 datza.filter(regex = "WTar").sum(axis = 1)
 
 
 
 
data_new = data[data["a"]>1 & data["b"] ]














#%%   Pivot Table Crosstab












#%% types of columns numeric stuff

import pandas as pd
import numpy as np

mixed_col_df = pd.DataFrame(dict(col1=[1, '1', False, np.nan, ['hello']], col2=[2, 3.14, 'hello', (1, 2, 3), True]))
mixed_col_df = pd.concat([mixed_col_df for _ in range(2)], ignore_index=True)


df.col1.apply(type).value_counts()


#  <class 'bool'>     2
#  <class 'int'>      2
#  <class 'float'>    2
#  <class 'list'>     2
#  <class 'str'>      2
#  Name: col1, dtype: int64


df._get_numeric_data()
df.select_dtypes(exclude=['object'])

#isnumeric python pandas type


#%%########################################################################################################################
"By “group by” we are referring to a process involving one or more of the following steps"
###########################################################################################################################

    "Splitting the data into groups based on some criteria"
    "Applying a function to each group independently"
    "Combining the results into a data structure"
        # Of these, the split step is the most straightforward. In fact, in many situations you may wish to split the data set into groups and
        # do something with those groups yourself. In the apply step, we might wish to one of the following:

"Aggregation: computing a summary statistic (or statistics) about each group. Some examples:"

    #  Compute group sums or means
    #  Compute group sizes / counts

"Transformation: perform some group-specific computations and return a like-indexed. Some examples:"

    #  Standardizing data (zscore) within group
    #  Filling NAs within groups with a value derived from each group

"Filtration: discard some groups, according to a group-wise computation that evaluates True or False. Some examples:"

    #  Discarding data that belongs to groups with only a few members
    #  Filtering out data based on the group sum or mean

#  Some combination of the above: GroupBy will examine the results of the apply step and try to return a sensibly combined result if it doesn’t fit into either of the above two categories

#%%
#reading a csv with a list in 
# use this like eval but safer
import ast
x=ast.literal_eval('0,1,2')
print(x)
# (0, 1, 2)
[int(x) for x in "0,1,2".split(",")]


#%%
vars(df)
# shows the data stored in it
# i think this shows the attributes of an object
#%%





df = df[df["one"] & df["two"]]
df = df.T
melt pivottab crosstable


VALUE#>attributr
Value#>method
value#>inplace_method

#survey.index=[ survey['Ref'],  survey[ 'Q Code'] ]
#survey.loc[1000,114]#[1000,114],]
#unpivot, pivot,multiindex,multi columns


## read in a text file and convert
## script changer
## learn short cuts
## learn multiindex and multi columns
## learn merge join lookup

## capitalize upper lower title


#these where the scripts open 16_11_2017
"""AM_2-ons; Built in Functions;Ecxcel_2_python,Useful-Snippets;getd_proto"""


isin where


#%%
import sys 
import os
sys.path.append(os.path.abspath("/home/el/foo4/stuff"))
from riaa import *
watchout()



#%%

lines = ["bannana","77a","a55a66","aa44","111a222a","a3a33a", "333"]
d = "a"

lines_1,lines_2,lines_3 = [], [], []

for line in lines:
   lines_1.append( line.split(d)) 
   ## these ones keep the deliminator in the splited list 
   lines_2.append(  [m+n for n,m in zip( line.split(d), ([""]+[d]*line.count(d)+[""])[:-1] ) if n+m] )
   lines_3.append(  [n+m for n,m in zip( line.split(d), ([""]+[d]*line.count(d)+[""])[1: ] ) if n+m] )
   lines_2.append(  [m+n for n,m in zip( line.split(d),  [""]+[d]*line.count(d) ) if m+n] )# the same

#%%
     def df_drop_columns_index(dataframe):
         dataframe_out=pd.DataFrame()
         
         for i in range(len(dataframe.columns)+1):
                dataframe_out.loc[:,i] = [dataframe.index, dataframe.iloc[:,i-1].values][i>0]
         temp = dataframe_out.copy()    
         for i in range(len(dataframe_out.index)+1):
                dataframe_out.loc[i,:] = [[""]+list(dataframe.columns), temp.iloc[i-1,:].values  ][i>0]
                
         return(dataframe_out) 
         
     df_a=df_drop_columns_index(df)
     
#%%
#Use Multiple names for bol operation   not equal =!
     

DB    = df[df['Train'] == 'London']
notDB = df[df['Train'] != 'London']

DB2    = df[ df['Train'].isin(['London', 'SNCF'])]
notDB2 = df[~df['Train'].isin(['London', 'SNCF'])]
#or
df[(df['Train'] != 'London']) & (df['Train'] != 'SNCF')]
#where
# what about not "in" ??

#%%  Get Categoriacl data in pandas 
import pandas as pd
examp_df = pd.DataFrame(index=range(10),columns=list("ABCDEFGHIJK")).fillna("0")
examp_df["A"]=["GRE","RED","BLU","YEL","RED","RED","YEL","GRE","RED","GRE"]
examp_catego_df=pd.get_dummies(examp_df)

#%%  Basic Stats describe correlation covariance

"""describe:The describe method provides quick stats on all suitable columns."""

df.describe()

#       float_col   int_col
#count    4.00000  5.000000
#mean     2.65000  3.200000
#std      4.96689  3.701351
#min      0.10000 -1.000000
#25%      0.17500  1.000000
#50%      0.20000  2.000000
#75%      2.67500  6.000000
#max     10.10000  8.000000

"""covariance:  The cov method provides the covariance between suitable columns."""

df.cov()#>
#           float_col    int_col
#float_col  24.670000  12.483333
#int_col    12.483333  13.700000

"""correlation:The corr method provides the correlation between suitable columns."""

df.corr()#>
#           float_col   int_col
#float_col   1.000000  0.760678
#int_col     0.760678  1.000000

#%%  











