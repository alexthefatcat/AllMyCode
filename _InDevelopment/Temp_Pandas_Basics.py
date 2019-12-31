# -*- coding: utf-8 -*-
"""Created on Wed Apr 10 10:16:48 2019@author: milroa1"""


import pandas as pd
import numpy  as np
#####################################################
df = pd.DataFrame(np.arange(12).reshape(4,-1),"i1 i2 i3 i4".split(),"c1 c2 c3".split())
#####################################################
df2 = df[["c1","c2"]]
df2 = df.loc[:, ["c1","c2"]]#same

df2 = df.loc[["i1","i2"]]
df2 = df.loc[["i1","i2"],:]#same

df2 = df.loc[["i1","i2"], ["c1","c2"]]

s = df["c1"]
s = df.loc["i1"]

s = df.loc[["i1","i2"],"c1"]
s = df["c1"][["i1","i2"]] #same bad

e = df.loc["i1","c1"]
e = df.loc["i1"]["c1"]# bad
e = df.at["i1","c1"]#same but quicker

# use iloc for location instead of loc iat replaces iat
# loc can take wither name(string) list of names(strings) bolean series
#mix

df2 = df.loc[["i1","i2"], df.columns[[0,2]]]
df2 = df.loc[["i1","i2"]].iloc[:,[0,2]]

df3=df.copy()
df3.loc[["i1","i2"], df.columns[[0,2]]] = 0
df3.loc[["i1","i2"]].iloc[:, [0,2]]     = 1 #doesnt work
#####################################################
"""              Boolean Selection                          """

bool_mask_inds_series = (df['c1'] %2)==0
df.loc[bool_mask_inds_series, ["c1","c2"]] 
df.iloc[bool_mask_inds_series.values, [2, 4]] 

df=df[df>5]#repalaces values below 5 with NaN
#####################################################


def colselect(df,*args,drop=None,Columns=False):
    columns     = list(df.columns)
    def filter_function(columns,func):
        return list(filter(func, columns))

    def get_columns(columns,args):
        for arg in args:
            if type(arg) is str:
                if ":" in arg:
                    string = arg.split(":")[1]
                if   arg.startswith("endswith"):
                    columns = filter_function(columns,lambda c: str(c).endswith(string))
                elif arg.startswith("startswith"):
                    columns = filter_function(columns,lambda c: str(c).startswith(string))
                elif arg.startswith("in"):
                    columns = filter_function(columns,lambda c: string in str(c))
                continue    
            else:
                pass
        return columns
    
    columns_out = get_columns(columns,args)
    if not drop is None:
        drop = drop if type(drop) is list else [drop]
        columns_drop = get_columns(columns,drop)
        columns_out = [c for c in columns_out if c not in columns_drop]
    if Columns:
       return columns_out        
    return df[columns_out]



dfi = colselect(df,drop="endswith:1")




"""  select is being depreciated

df.select(lambda x: x in ['bar', 'baz'])

df2 = df.select(lambda col: col.endswith('2'), axis=1)
select # .loc[labels.map(crit)]

"""


#####################################################
