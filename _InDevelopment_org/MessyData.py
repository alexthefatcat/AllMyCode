# -*- coding: utf-8 -*-
"""Created on Tue Apr  9 15:04:37 2019@author: milroa1"""


class MessyData():
    def find_column_with_the_words(df,words,axis=0,column_count=False):
        if type(words) is str:
            words = words.split()
        def wordsinelement(ele):
            if type(ele) is str:
                return any([w in ele.upper() for w in words ])
            return False 
        count_df = df.applymap(wordsinelement).sum(axis=axis)
        count_df = count_df[count_df>0].sort_values(ascending=False)
        if not column_count:
            return count_df.index[0]
        return count_df.index[0], count_df 
    def find_common_words_in_Series(Series,min_words=5):    
        word_collection=[]
        for ele in Series:
            if type(ele) is str:
                word_collection.extend(ele.split(" "))
        word_collection = [word.upper() for word in word_collection if word.upper() not in ["","AND","THE","OR","&","FOR","OF"]]   
        word_hist={}
        for word in word_collection:
          word_hist[word] = word_hist.get(word,0)+1 
        clip = sorted(list(word_hist.values()),reverse=True)[min([min_words,len(word_hist)])]
        words = [k for k,v in word_hist.items() if v>clip]  
        return words      
    def col_select(df,*args,drop=None,Columns=False):
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
                        #elif type
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

if __name__ == "__main__":
    # From the pnp data
    words = MessyData.find_common_words_in_Series( df_pnp["_3"] ,20)
    words = "FOUNDATION CENTRE INSTITUTE ASSOCIATION TRUST"
    column_name = MessyData.find_column_with_the_words(df_pnp, words)


