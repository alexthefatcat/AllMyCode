# -*- coding: utf-8 -*-
"""Created on Sat Jun 20 21:30:02 2020@author: Alexm"""

import pandas as pd
import time

def get_files_from_folder(folder=".",endswith=None,ignore_these=None,single=False):
    " Returns a list of files in this folder but ignore folders and files in them"
    from os import listdir
    from os.path import isfile, join
    files = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
    if endswith is not None:
        files = [f for f in files if f.endswith(endswith)]
    if ignore_these is not None:
        listify = lambda x: x if type(x) is list else [x]
        ignore_these = listify(ignore_these)
        files = [f for f in files if not any([ i in f for i in ignore_these])]
    if single:
        assert 1==len(files)
        return files[0]
    return files 


filename = get_files_from_folder("bittorent_in",".dat","ignore",True)

 


def string_io_pandas_example():
    import io
    output = io.StringIO()
    output.write('x,y\n')
    output.write('1,2\n')
    output.seek(0)
    df = pd.read_csv(output)
    
def lsplit_into_two(string):
    return ([""]+string.split(":",1))[-2:]


def read_dat_file_to_list(filename):
    with open(filename,'r', encoding="ISO-8859-1") as fileobj:#'utf-8'
         data = [line for line in fileobj.readlines()]
    data2 = "\n".join(data)
    *data2_start,data2 = data2.split(":",3)
    data2 = data2.split("webseedslee")
    data2 = [ lsplit_into_two(line) for line in data2]
    data3 = []
    for line,next_line in zip(data2,data2[1:]):
        if len(data3)==0:
            line_new = ":".join(line)
        else:
            line_new = line[1]
        data3.append(line_new+":"+next_line[0])
    return data3









def split_by_brackets(string,before="{",after="}"):
    """
    string0 = "adcrecr{first}sdcsds}dsd{sa{second}"
    lstring0 = split_by_brackets(string0)
    data = lstring0[1::2]
    # data => ['first', 'second']
    """
    parts1 = string.split(before)
    parts2 = [ part.split(after,1) for part in parts1]

    for i,line in  enumerate(parts2):
        if i == 0:
           out = [after.join(line)] 
        else:            
           out[-1] = out[-1]+before
           if len(line)==1 :
              out[-1] = out[-1]+line[0]
           else:
              out.extend([line[0], after+line[1]])
    return out    
 

 

def get_completedate(string,before="completei",after="e11",gettime=True):
#    global string2
#    string2=string
#    global data
    data = split_by_brackets(string,before,after)
    if len(data)>1:
       print(data) 
       out_int=time.ctime(int(data[1]))
       return out_int
    return None
 




data = read_dat_file_to_list(filename)

def quick_dataframe(data):
    
    completed_lis = []
    for string in data:
        date = get_completedate(string)
        completed_lis.append(date)
    df = pd.DataFrame(columns=["Name"],data = [line.split(".torrentd8:")[0] for line in data])
    df["Completed"] = completed_lis
    return df

df = quick_dataframe(data)
df["Completed2"] = pd.to_datetime(df["Completed"] )

df_recent = df[df["Name"].apply( lambda x: any([ n in x for n in ["2016","2017","2018","2019"]]))]

 
#







