# -*- coding: utf-8 -*-"""Created on Mon Jun 25 09:50:38 2018@author: milroa1"""

#  multiindexing *> top
#  types in columns
#  chunking
#  merging,concat,join, appending,,add?
#  casting *>bottem // mutliple datatypes
#  value counts
#  advanced groupby*
#  time data
#  insert *>bottem
#  dropping
#  ufuncs
#  differece unstack melt pivottables
#  catecorigcal
#  binning or cut


#%% #################################################################################
"""                               Multi Index                                     """
#####################################################################################
import pandas as pd,numpy as np

col1=["ONE', 'ONE', 'TWO', 'TWO', 'THR', 'THR']
col2=['_1_', '_2_', '_1_', '_2_', '_1_', '_2_']
ind1=["A"  ,  "A" ,  "B" ,  "B" ,  "C" ,  "C" ]
ind2=["a"  ,  "b" ,  "c" ,  "a" ,  "b" ,  "c" ]

df = pd.DataFrame(np.round(np.random.randn(6,6), decimals=2), index=[ind1,ind2], columns=[col1,col2])
## WAYS OF ACCESSING THE DATA
df_a = df.loc[ ("A","a" ), :]
df_b = df.loc[ ["A"]     , :]
df_c = df.loc[ "A":"B"   , :]
df_d = df.loc[ "A"       , "ONE"].loc["a","_1_"]


print(df)

#            lv3       ONE         TWO         THR      
#            lv4       _1_   _2_   _3_   _1_   _2_   _3_
#            lv1 lv2                                    
#            A   a    1.60  0.35  0.56 -0.69 -0.11  0.97
#                b    1.45 -1.74  0.62 -0.58 -0.15  0.79
#            B   c   -0.17  1.39 -0.69  0.08  1.28 -1.08
#                a    0.39  0.63  0.25  0.78  1.50 -0.79
#            C   b    1.04  0.07  0.68  0.80 -0.21 -0.11
#                c   -0.32 -0.24  0.56  0.88 -0.61 -0.09

"swap the column levels and then reorder"
df.columns = df.columns.swaplevel(0, 1)
df=df.sortlevel(0, axis=1,)
print(df)

#            lv4       _1_         _2_         _3_      
#            lv3       ONE   TWO   ONE   THR   THR   TWO
#            lv1 lv2                                    
#            A   a    0.48 -0.20 -0.93  0.07 -0.23  0.51
#                b   -0.49 -0.36  0.05 -0.27  0.83 -0.78
#            B   c   -2.57  0.92 -1.35  0.99 -0.12 -0.47
#                a   -1.45 -1.34 -1.25 -0.52  0.85 -0.06
#            C   b   -0.00  0.90 -1.64 -1.34  0.09  2.56
#                c   -0.03  0.02 -0.46 -0.17  0.12 -1.87

df['_1_', 'FOUR'] = list("GHIJKL")
df = df.sort_index(axis=1)
print(df)

#    lv4      _1_                     _2_            
#    lv3     FOUR   ONE   THR   TWO   ONE   THR   TWO
#    lv1 lv2                                         
#    A   a      G  0.14  1.51  0.21  1.23  1.20 -0.57
#        b      H -1.32  1.33  1.07  0.77  1.87 -0.71
#    B   c      I -0.83  0.02 -0.31 -1.68  1.27  1.00
#        a      J -1.04 -1.11  0.01  0.96 -0.76 -0.38
#    C   b      K  0.05 -0.36 -0.81  1.24 -0.96  0.37
#        c      L  2.70  0.41 -0.24  2.58  0.44 -1.25




# use a column to add to the multiindex
df.columns=list("1234567")
df["orderby"]=list("aabbcc")
df = df.set_index("orderby", append=True)
print(df)


#                         1     2     3     4     5     6     7
#        lv1 lv2 orderby                                       
#        A   a   a        G  0.74 -0.64 -1.17  1.46 -0.98  1.25
#            b   a        H  0.46  0.38 -0.74  0.83  0.62 -0.06
#        B   c   b        I -0.58 -0.19 -0.15 -0.87  0.54 -0.10
#            a   b        J  0.26 -1.07 -0.28 -0.71  0.70  0.55
#        C   b   c        K  0.16 -1.31  0.31 -0.54  1.06  0.15
#            c   c        L  0.28  1.20 -0.49 -0.95 -1.07 -0.45

#df.index=df.index.droplevel(2)

df= df.reset_index()

#          lv2 orderby  1     2     3     4     5     6     7
#        0   a       a  G -0.04  0.82  0.02  0.24 -0.40  0.44
#        1   b       a  H  0.01 -0.12  1.11  1.08  2.04 -0.05
#        2   c       b  I  0.82  0.20  0.71  0.69 -0.74 -0.41
#        3   a       b  J  0.84 -0.03 -0.30 -0.21  0.97  0.13
#        4   b       c  K  0.58 -0.62 -2.87 -0.11 -2.17  0.34
#        5   c       c  L  1.12 -0.61  0.02  1.39  1.36 -0.74






###Stack and Unstack


df_stack=df.stack()
print(df_stack)

#        lv4            _1_   _2_
#        lv1 lv2 lv3             
#        A   a   ONE  -1.74 -0.88
#                THR   0.04  1.70
#                TWO  -0.39  0.62
#                FOUR     G   NaN
#            b   ONE  -0.64 -0.54
#                THR    0.5  0.09
#                TWO    0.1 -1.63
#                FOUR     H   NaN
#        B   c   ONE   0.58  1.50
#                THR   0.18  1.57
#                TWO  -0.26 -0.45
#                FOUR     I   NaN
#            a   ONE   0.27 -0.54
#                THR  -0.23 -0.73
#                TWO   0.13 -0.37
#                FOUR     J   NaN
#        C   b   ONE   1.65 -0.51
#                THR  -0.92  0.57
#                TWO  -0.74  0.51
#                FOUR     K   NaN
#            c   ONE   0.43  0.01
#                THR   1.87 -0.96
#                TWO  -0.77  1.20
#                FOUR     L   NaN

df2=df_stack.unstack(level=-1)
print(df2)

#        lv4       _1_                    _2_                 
#        lv3       ONE   THR   TWO FOUR   ONE   THR   TWO FOUR
#        lv1 lv2                                              
#        A   a   -1.74  0.04 -0.39    G -0.88  1.70  0.62  NaN
#            b   -0.64   0.5   0.1    H -0.54  0.09 -1.63  NaN
#        B   a    0.27 -0.23  0.13    J -0.54 -0.73 -0.37  NaN
#            c    0.58  0.18 -0.26    I  1.50  1.57 -0.45  NaN
#        C   b    1.65 -0.92 -0.74    K -0.51  0.57  0.51  NaN
#            c    0.43  1.87 -0.77    L  0.01 -0.96  1.20  NaN

df3=df2.unstack(level=-1)
print(df3)

#        lv4                                                   
#        lv3   THR               TWO             FOUR          
#        lv2     a     b     c     a     b     c    a   b   c  
#        lv1                                                   
#        A    0.13 -0.98   NaN -0.26 -0.90   NaN  NaN NaN NaN  
#        B   -0.24   NaN  0.66 -0.01   NaN -1.01  NaN NaN NaN  
#        C     NaN  0.25  0.00   NaN -0.51  0.22  NaN NaN NaN  












###########  Multi Index Using groupby and add one to the index ###########################
"""        Use group by to create an index         """


import pandas as pd,numpy as np


df_org = pd.DataFrame({'A':['a1','a1','a2','a3'],'B':['b1','b2','b3','b4'],'Vals':np.random.randn(4)})

#            A   B      Vals
#        0  a1  b1  0.137785
#        1  a1  b2  1.248855
#        2  a2  b3 -1.415857
#        3  a3  b4  1.503459

df=df_org.groupby(['A', 'B']).sum()

#                   Vals
#        A  B           
#        a1 b1  0.137785
#           b2  1.248855
#        a2 b3 -1.415857
#        a3 b4  1.503459

df['Firstlevel'] = 'Foo'
df.set_index('Firstlevel', append=True, inplace=True)
df.reorder_levels(['Firstlevel', 'A', 'B'])

#                              Vals
#        Firstlevel A  B           
#        Foo        a1 b1  1.055604
#                      b2  0.455587
#                   a2 b3  0.229820
#                   a3 b4  1.212974

############################################################
df=pd.concat([df_org], keys=['Foo'], names=['Firstlevel'],axis=1)

#        Firstlevel Foo              
#                     A   B      Vals
#        0           a1  b1  0.305621
#        1           a1  b2  0.385369
#        2           a2  b3  0.632955
#        3           a3  b4 -0.228578

df=pd.concat([df_org], keys=['Foo'], names=['Firstlevel'],axis=0)

#                       A   B      Vals
#        Firstlevel                    
#        Foo        0  a1  b1  0.305621
#                   1  a1  b2  0.385369
#                   2  a2  b3  0.632955
#                   3  a3  b4 -0.228578

###droping multi index and adding to the dataframe










############   Inding multiindexs  ############################

import pandas as pd,numpy as np
## LOAD IN DATA
index   = [np.array(list("AABBCC") ),np.array(list("abcabc") )]
columns = [np.array(['ONE', 'ONE', 'TWO', 'TWO', 'THR', 'THR']),np.array(['_1_', '_2_', '_3_', '_1_', '_2_', '_3_'])]
df = pd.DataFrame(np.round(np.random.randn(6,6), decimals=2), index=index, columns=columns)
df.index.names, df.columns.names=["lv1","lv2"], ["lv3","lv4"]
del index,columns

print(df)
#            lv3       ONE         TWO         THR      
#            lv4       _1_   _2_   _3_   _1_   _2_   _3_
#            lv1 lv2                                    
#            A   a    1.60  0.35  0.56 -0.69 -0.11  0.97
#                b    1.45 -1.74  0.62 -0.58 -0.15  0.79
#            B   c   -0.17  1.39 -0.69  0.08  1.28 -1.08
#                a    0.39  0.63  0.25  0.78  1.50 -0.79
#            C   b    1.04  0.07  0.68  0.80 -0.21 -0.11
#                c   -0.32 -0.24  0.56  0.88 -0.61 -0.09

#COLUMNS
print(    df["ONE"],                  df["ONE","_1_"] ,            df["ONE"]["_1_"])

#        lv4       _1_   _2_
#        lv1 lv2                    
#        A   a    0.69  0.97          A    a      0.69            A    a      0.69
#            b   -0.58  1.92               b     -0.58                 b     -0.58
#        B   c   -1.35  0.80          B    c     -1.35            B    c     -1.35
#            a   -0.69 -0.60               a     -0.69                 a     -0.69
#        C   b    0.77 -0.70          C    b      0.77            C    b      0.77
#            c   -1.55  0.92               c     -1.55                 c     -1.55


df.loc[( "A", "a" ),("ONE","_1_")]
df.loc[( "A", "a" ),("ONE")]
df.loc[( ["A","B"] ),("ONE")]
df.loc[:,("ONE")]


"""       crosstab is basically the same as pivot                """
"""       if multi index is wanted use pivottable not pivot      """
"""       melt is the opposite of pivot                          """






##why in multiindex are there missing entries
##reset index


# use pivot for single indexs
df.pivot(index='date', columns='variable', values='value')
pandas.pivot_table(...)
# pivot table
table = pd.pivot_table(df, index=['item_id', 'hour', 'date'], columns='when', values='quantity')


#inserting
#########################################################################

#insert a new column "new_column_name_5" as 6th column in dataframe (with value "")
def insert_row(df,loc,name,value):
    temp=df.T
    temp.insert(loc,name,value)
    return(temp.T)

df.insert(15,"new_column_name_51","")
df = insert_row(df,5,"new_column_name_5","")
############################################################################

# casting data type
data_df['grade'] = pd.to_numeric(data_df['grade'], errors='coerce').fillna(0).astype(int)
df_types=df.dtypes
df.dtypes.nunique()





# We"ve all been there, trying to convert a column to a numeric type, but somewhere in the depths of it is a letter and you can"t set the dtype.
df["col"]=df["col"].astype(str).convert_objects(convert_numeric=True)

# Getting rid of that damned scientific notation. Just run this code in any cell.
pd.set_option("display.float_format", lambda x: "%.3f" % x)

# When trying to select using multiple criteria from a data-frame. You can also use | for an or criteria.
newdf = df[(df["column_one"]>2017) & (df["column_two"]==8)]

# Renaming just a few of the column names
df = df.rename(columns = { "col1 old name":"col1 new name", "col2 old name":"col2 new name", "col3 old name":"col3 new name",})

# Looping through a data frame.
for index, row in df.iterrows():
    print (index, row["col_x"])

# Moving a column to the front of a data-frame.
cols = list(df)
cols.insert(0, cols.pop(cols.index("col_x")))
df = df.ix[:, cols]

# Dropping rows with nulls in a specific column.
df=df.dropna(how="all",subset=["col_x"])

# Finding duplicates based on a specific column.
duplicates = df[df.duplicated(["col1", "col2", "col3"], keep=False)]

# Referencing a range of columns easily.
df[list(df.columns[2:12]

# Converting a dictionary to a data-frame.
df = pd.DataFrame(list(dictiname.items()), columns = ["column1", "column2"])


def eda_helper(df):
    dict_list = []
    for col in df.columns:
        data = df[col]
        dict_ = {}
        # The null count for a column. Columns with no nulls are generally more interesting
        dict_.update({"null_count" : data.isnull().sum()})
        # Counting the unique values in a column
        # This is useful for seeing how interesting the column might be as a feature
        dict_.update({"unique_count" : len(data.unique())})
        # Finding the types of data in the column
        # This is useful for finding out potential problems with a column having strings and ints
        dict_.update({"data_type" : set([type(d).__name__ for d in data])})
        #dict_.update({"score" : match[1]})
        dict_list.append(dict_)
    eda_df = pd.DataFrame(dict_list)
    eda_df.index = df.columns
        
    return eda_df

#  isnumeric  astype    value_counts


## cumative

#%%###############################################################################
"""                             GROUP-BY                                       """
##################################################################################
import numpy as np, pandas as pd

columns = ["name",        "day", "no"]
data   = [["Jack",     "Monday",   10],
          ["Jack",   "Thursday",   20],
          ["Jack",    "Tuesday",   10],
          ["Jill",     "Monday",   40],
          ["Jill",  "Wednesday",  110],
          ["Jack",  "Wednesday",   50],
          ["Jim" ,     "Monday",  np.nan] ]

df = pd.DataFrame(data=data, columns=columns)

#getting stats out not to put back in the dataframe, that is what transform can do
df_agg1 = df.groupby('name').agg({'no':['sum', 'min'],'day_' : ['min', 'max']})
df_agg2 = df.groupby('name').agg('sum')
#----------------------------------------------------------------------------
df['no_cumulative'] = df.groupby(['name'])['no'].apply(lambda x: x.cumsum())
df['index_'       ] = df.groupby('name').cumcount()
#---------------------------------------------------------------------------- 
cats = [ 'Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']

df["day_"] = df["day"].apply(lambda x:["Tuesday","Monday","Wednesday","Thursday"].index(x))
df['day' ] = df['day'].astype(pd.api.types.CategoricalDtype(categories = cats ))
df = df.sort_values(["name","day"])
#---------------------------------------------------------------------------- 
df["no"                ] = df.groupby("day" ).transform(lambda x: x.fillna(x.mean()))
df["group_name_sum_no" ] = df.groupby('name')["no"].transform('sum')
df["group_name_mean_no"] = df.groupby('name')["no"].transform(lambda x: x.mean())
#---------------------------------------------------------------------------- 
# remove groupby sum is below 200; remove items in groups below a certain value
df1 = df[df.groupby("name")["no"].transform('sum') > 45]
df2 = df.groupby('name').filter(lambda g: g["no"].sum() > 45)
df3 = df.groupby('name').filter(lambda g: g["no"].count() > 2)

#%%###############################################################################
    #Splitting:   -    the data into groups based on some criteria
    #Applying:    -    a function to each group independently
    #Combining:   -   the results into a data structure
    
        #Of these, the split step is the most straightforward. In fact, in many situations you may wish to split the data set into groups and do something with those groups yourself. In the apply step, we might wish to one of the following:

"Aggregation: computing a summary statistic (or statistics) about each group. Some examples:"
    #Compute group sums or means
    #Compute group sizes / counts
    
"Transformation: perform some group-specific computations and return a like-indexed. Some examples:"
    #*Standardizing data (zscore) within group
    #*Filling NAs within groups with a value derived from each group
    
"Filtration: discard some groups, according to a group-wise computation that evaluates True or False. Some examples:"

    #*Discarding data that belongs to groups with only a few members
    #*Filtering out data based on the group sum or mean
    
        #Some combination of the above: GroupBy will examine the results of the apply step and try to return a sensibly combined result if it doesn"t fit into either of the above two categories
""" if we want to get a single value for each group -> use aggregate()
    if we want to get a subset of the input rows -> use filter()
    if we want to get a new value for each input row -> use transform()"""
###################################################################################################################################

def group_dataframe(df, grouped, column_different=False):
    dfg = df.groupby(grouped)
    df["group_no"   ] = dfg.ngroup()
    df["group_count"] = dfg[[n for n in df.columns if not n in grouped][0]].transform(lambda x:len(x))#["Filepath"]
    if not column_different ==False:
       df["group_diff"] = dfg[ column_different ].transform(lambda x:x.nunique())                       
    return(df) 

def simple_extract_curly_brackets(_str_):
    out1="".join([aaa.split("{")[0] for aaa in _str_.split("}")])
    out2=[aaa.split("}")[0] for aaa in _str_.split("{")][1:]
    return( out1, out2 )

def add_new_blank_columns(df,columns,fill=""):
    temp = pd.DataFrame( columns=columns )
    df=df.join(temp )#.fillna(fill)
    df[columns]=fill
    return(df)

###################################################################################################################################
#  value counts
df_vc   = df.iloc[:,0].value_counts()
df_bins = df.iloc[:,0].value_counts(bins=20)

pd.read_csv('../data/df1.csv', index_col=0)

df.sample(10)
df.describe()
df['g_whoregion'].unique()
df['country'].nunique()

df['country'].head(3)#.tail()
df.country[1000:1003]
#df.iloc[-1,0]
df.iloc[:-5,:]
df[cond & (df.country == 'Argentina') & (df.type == 'rel') & (df.sex == 'm')]
#where
df[df["country"].isin(['Greece', 'Italy'])]
df.loc[:3, lambda x: ['country', 'g_whoregion']]
great = df.loc[lambda x: x.cases > 100000, :]
df["cases"].loc[lambda x: x > 100000]
great.where(great["country"] == 'India')
great.mask(great.country == 'India')

df1.reset_index()

assert df['year'].dtype == 'int'
assert df['year'].max() <= 2017
assert df['cases'].min() == 0

df_sub['cases'].isnull().value_counts()
df_sub.fillna('NA')
df_sub.dropna()

#%%#  Categorical 

# how to modify categorical data
files1_df["Filetype"]=files1_df["Filetype"].astype('category')
files1_df["Filetype"].cat.categories
files1_df["Filetype"]=files1_df["Filetype"].cat.rename_categories({".jpg":"fuck-you"})
files1_df["Filetype"]=files1_df["Filetype"].cat.add_categories("junk")
files1_df["Filetype"].iloc[3]="junk"



#%%
"""         Ufuncs Pandas, shift, time stuff                         """
#  Pandas has an apply function which let you apply just about any function on all the values in a column. Note that apply is just a little bit faster
#  than a python for loop! That’s why it is most recommended using pandas builtin ufuncs for applying preprocessing tasks on columns (if a suitable
#  ufunc is available for your task).
#
#   .diff, .shift, .cumsum, .cumcount, .str commands (works on strings), .dt commands 
######################################################################################################
import pandas as pd
df = pd.DataFrame([["Chandler Bing","party","2017–08–04 08:00:00",51], ["Chandler Bing","party","2017–08–04 13:00:00",60], ["Chandler Bing","party","2017–08–04 15:00:00",59], ["Harry Kane","football","2017–08–04 13:00:00",80], ["Harry Kane","party","2017–08–04 11:00:00",90], ["Harry Kane","party","2017–08–04 07:00:00",68], ["John Doe","beach","2017–08–04 07:00:00",63], ["John Doe","beach","2017–08–04 12:00:00",61], 
                   ["John Doe","beach","2017–08–04 14:00:00",65], ["Joey Tribbiani","party","2017–08–04 09:00:00",54], ["Joey Tribbiani","party","2017–08–04 10:00:00",67], ["Joey Tribbiani","football","2017–08–04 08:00:00",84], ["Monica Geller","travel","2017–08–04 07:00:00",90], ["Monica Geller","travel","2017–08–04 08:00:00",96], ["Monica Geller","travel","2017–08–04 09:00:00",74],
                   ["Phoebe Buffey","travel","2017–08–04 10:00:00",52], ["Phoebe Buffey","travel","2017–08–04 12:00:00",84], ["Phoebe Buffey","football","2017–08–04 15:00:00",58], ["Ross Geller","party","2017–08–04 09:00:00",96], ["Ross Geller","party","2017–08–04 11:00:00",81], ["Ross Geller","travel","2017–08–04 14:00:00",60]],
                   columns=["name","activity","timestamp","money_spent"])
df["timestamp"] = pd.to_datetime(df["timestamp"])
######################################################################################################

df["name"] = df.name.str.split(" ", expand=True)# also .str.replace and a suitable regex.

df.groupby('name')['activity'].value_counts()

df.groupby('name')['activity'].value_counts().unstack().fillna(0)

df = df.sort_values(by=['name','timestamp'])
df['time_diff'] = df.groupby('name')['timestamp'].diff()

df = df.sort_values(by=['name','timestamp'])
df['time_diff'] = df['timestamp'].diff()
df.loc[df.name != df.name.shift(), 'time_diff'] = None

df['time_diff'] = df.time_diff.dt.total_seconds()

df["row_duration"] = df.time_diff.shift(-1)

df = df.sort_values(by=['name','timestamp'])
df2 = df[df.groupby("name").cumcount()==[1,2][0]]

df = df.sort_values(by=["name","timestamp"])
df['money_spent_so_far'] = df.groupby("name")['money_spent'].cumsum()

df['activity_change'] = (df.activity!=df.activity.shift()) | (df.name!=df.name.shift())

df['activity_num'] = df.groupby('name')['activity_change'].cumsum()

activity_duration = df.groupby(['name','activity_num','activity'])['activity_duration'].sum()

activity_duration = activity_duration.dt.total_seconds()

activity_duration = activity_duration.reset_index().groupby('name').max()

#%%

difference = set(count_df.columns) - set(tfidf_df.columns)


psf_detail_df.columns = pd.MultiIndex.from_tuples([n if n[0] not in ["Table_2"] else (n[0], list(psf_detail_df["Table_2"].columns).index(n[1])) for n in psf_detail_df.columns.tolist() ])
pp = psf_detail_df.stack(0)

# pivot and sacking


def make_elements_in_a_list_unique(list_,string_=True):
    """
    a=make_elements_in_a_list_unique("m,d,de,,,,,m,m,n,d,dddddd,m(2),m(88),n(jj,n(jj,((0((0())(),((0((0())(),".split(","))
    ",".join(a)#>
    'm,d,de,,(1),(2),(3),m(1),m(3),n,d(1),dddddd,m(2),m(88),n(jj,n(jj(1),((0((0())(),((0((0())()(1),(4)'

    """
    if string_:
          list_=[str(n) for n in list_]    
    out = []
    for i,n in enumerate(list_):
        # if the n element is not the first one it will be modified# +(1)  or if (1) 
        # or if a number in brackets is already on the end example:  r(2) then> r(3)
        def is_n_unique(n,list_,out):
           Flag = False
           if n in list_:
              if list_.index(n)==i:
                 Flag =True
           else :
              if not n in out: 
                 Flag=True
           return(Flag)  
        def incr_string(no):
            return(str(int(no)+1))
        def check_that_you_can_increment_bracket_number_by_one_and_do_so(n):
            m = n.split("(") 
            Flag=False
            if len(m) > 1:
                if len(m[-1])>1:
                    if m[-1][-1]==")":
                         no = m[-1][:-1]
                         if no.isdigit():
                             n = "".join(m[:-1]+["(",str(int(no)+1),")"])
                             #n = "(".join(m[:-1]+[incr_string(no),")"])
                             Flag=True
            return(n,Flag) 
        ######################################################    
        while True:
            if is_n_unique(n,list_,out):
               out.append(n)
               break
            n, Flag = check_that_you_can_increment_bracket_number_by_one_and_do_so(n)
            if not Flag:
                 n= n+"(1)"                
    return(out)  

def turn_dataframe_to_multiindex_one(df):
    index, columns = list(df.index),list(df.columns)
    if type(index[0])   is tuple:
        df.index   = pd.MultiIndex.from_tuples( index  )
    if type(columns[0]) is tuple:
        df.columns = pd.MultiIndex.from_tuples( columns)
    return(df)






def tsplit(s, sep):
    stack = [s]
    for char in sep:
        pieces = []
        for substr in stack:
            pieces.extend(substr.split(char))
        stack = pieces
    return stack





#pivottable stack unstack multiindex
#merge join append concat





