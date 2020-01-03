# -*- coding: utf-8 -*-"""Created on Thu Dec  6 15:19:42 2018@author: milroa1"""

string ="""
def create_new_obr1_df(filename, sheet_name):    

    df = UF.pandas_read(filename, sheet_name)
    
    find_col = lambda elem:str(elem).ljust(4)[:4].isdigit()
    find_row = lambda e   :str(e   ).lower() in ['current budget']
    
    temp = UF.find_location_of_value_or_function_in_df(df, find_col, iloc=True)
    # count all number of elements in list, histograms
    counts = hist_count_or_list(temp)
    # the index number of the column is the extracted dataframe smallest one with more than 3 columns in it
    col_no = find_the_smallest_with_more_than_3_counts(counts)
    # selected dataframe out of the dataframe
    df = shrink_df(df, col_no=col_no)
                   
    temp = UF.find_location_of_value_or_function_in_df(df, find_row, iloc=True)
    row_no = temp[0][1]
    
    return shrink_df(df, row_no=row_no).fillna("")
OBR_forecast_df     = create_new_obr1_df( OBR_Files_and_Sheets["1"]["filepath"], OBR_Files_and_Sheets["1"]["sheetname"] )  
"""
####### What I Want The Output to Looklike  #############
string_out="""
#OBR_forecast_df     = create_new_obr1_df( OBR_Files_and_Sheets["1"]["filepath"], OBR_Files_and_Sheets["1"]["sheetname"] )
filename, sheet_name = OBR_Files_and_Sheets["1"]["filepath"], OBR_Files_and_Sheets["1"]["sheetname"]
#def create_new_obr1_df(filename, sheet_name):
if True:    

    df = UF.pandas_read(filename, sheet_name)
    
    find_col = lambda elem:str(elem).ljust(4)[:4].isdigit()
    find_row = lambda e   :str(e   ).lower() in ['current budget']
    
    temp = UF.find_location_of_value_or_function_in_df(df, find_col, iloc=True)
    # count all number of elements in list, histograms
    counts = hist_count_or_list(temp)
    # the index number of the column is the extracted dataframe smallest one with more than 3 columns in it
    col_no = find_the_smallest_with_more_than_3_counts(counts)
    # selected dataframe out of the dataframe
    df = shrink_df(df, col_no=col_no)
                   
    temp = UF.find_location_of_value_or_function_in_df(df, find_row, iloc=True)
    row_no = temp[0][1]
    
#    return shrink_df(df, row_no=row_no).fillna("")  
OBR_forecast_df = shrink_df(df, row_no=row_no).fillna("") 
"""
#%% 

def split_first_one(string,str_split,no=1):
    if no> len(string.split(str_split)):
       return None, None
    else :
       return str_split.join(string.split(str_split)[:no]),str_split.join(string.split(str_split)[no:])

def remove_empty_lines(string_list):
    return [line for line in string_list if line != ""]
        
def remove_tabs_and_spaces(string_list):
    return [line.replace("\t","   ").replace(" ","") for line in string_list ]

def tabs2spaces(string):
    return [line for line in string]

def remove_comments(string_list):
    return [line for line in string_list if not line.replace("\t","").replace(" ","").startswith("#")]

def indent_count(string_list):
    return [ l1.index(l2[0]) if len(l2)>0 else None for l1,l2 in zip(tabs2spaces(string_i),remove_tabs_and_spaces(string_i))]
#def get_func_info(string):
#    function_name= "def ".join(string.split("def ")[1:])

def get_function_info(func_str):
    start_temp,function_name = split_first_one(func_str,"def ")
    function_name,variables = split_first_one(function_name,"(") 
    variables, brackets   = split_first_one(variables,")")
    variables = variables.split(",")
    variables = remove_tabs_and_spaces(variables)
    return function_name,variables


string_i = string.splitlines()
string_i = remove_comments(string_i)
indents  = indent_count(string_i)


lists_defs=[]
for i, (line, ind) in enumerate(zip( tabs2spaces(string_i), indent_count(string_i) )):
    if ind is not None:
        if line[ind:].startswith("def "):
           name,variables = get_function_info(line[ind:])
           lists_defs.append([i,line[ind:],name,variables])











def unwrap_function(string):
   pass










