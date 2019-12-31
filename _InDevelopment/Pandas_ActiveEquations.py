# -*- coding: utf-8 -*-
"""Created on Thu May 16 09:45:26 2019@author: milroa1"""
import pandas as pd

class FormulaPandas:
   """
   df.loc["a","b"] = '#.loc["a","c"]'
   also cells stat start with with "*" will be ignored
   
   big problem is the global process_df to get it to work
    as well as the replace "#" process_df
   """

   def count_strings(df):
       return df.applymap(lambda x:type(x) is str).sum().sum()

   def SpecialCondictions(e):
       "Cells that start with * ignored and # is replaced with the current dataframe"
       skip = e.startswith("*")
       
       if not skip:
          # e = e.replaces({})
           e = e.replace("#",'process_df') 
       return e, skip

   def tryeval(e,printer=False):
       """   Try to evaluate the String else keep the String"""
       out = e 
       if type(e) is str:
         e, skip = FormulaPandas.SpecialCondictions(e)# e,False to ignore these
         if not skip:       
             try :
                 out = eval(e)
                 if printer:
                    print("  ",e,">>",out)
             except:
                 out = e
       return out 

   def replace_strings_with_org(df,df_org):
       """   In case of situations where the It eval produces a string replace it with the orginal"""
       msk = df.applymap(lambda x:type(x) is not str) 
       #df[msk] = df_org[msk]
       df = (df[msk]).combine_first(df_org)

       return df

   def evalulation_loop(process_df_,printer=False):
      global process_df
      process_df = process_df_       
      org_df = process_df.copy()
      str_count = FormulaPandas.count_strings(process_df)
      count=0
      while True:
         count +=1
         if printer:
             print(f"###>>> Loop:{count}  <<<#### StringCount:{str_count}")
         process_df = process_df.applymap(lambda e:FormulaPandas.tryeval(e,printer))
         process_df = FormulaPandas.replace_strings_with_org(process_df, org_df)      
         str_count_ = FormulaPandas.count_strings(process_df)
         if str_count_ in[str_count, 0]:
             break
         else:
             str_count = str_count_
      return process_df 
  
   def remove_star(df):
       def remove_star_map(elem):
           if type(elem) is str:
               if elem.startswith("*"):
                   elem = elem[1:]
           return elem
       df = df.applymap(remove_star_map)  
       return df
       
   def replaces(string,dic):
        for k,v in dic.items():
            string= string.replace(k,v)
        return string
  
if __name__ == "__main__":    
    # So the idea is you give it a dataframe with equations and numbers and it returns them
    
    equations_df = pd.DataFrame(columns=list("abc"),data=[["* Start cell",343,"#.loc[1,'a']"],[56,"#.loc[0,'c']","#.loc[0,'b']"]],index=[0,1])
    values_df    = FormulaPandas.evalulation_loop(equations_df,True)
    values_df    = FormulaPandas.remove_star(values_df)












