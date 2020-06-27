# -*- coding: utf-8 -*-
"""Created on Sat Jun 20 21:30:02 2020@author: Alexm"""

import pandas as pd
import time








#%%------------------------------------------------------------------------------
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
        assert 1==len(files), f"{len(files)} where found not 1"
        return files[0]
    return files 


## example code move to useful
def string_io_pandas_example():
    """
    This an example of using string io
    """
    import io
    output = io.StringIO()
    output.write('x,y\n')
    output.write('1,2\n')
    output.seek(0)
    df = pd.read_csv(output)
    return df
    
#%%------------------------------------------------------------------------------
def lsplit_into_two(string):
    return ([""]+string.split(":",1))[-2:]

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

def read_raw_data(filename):
    with open(filename,'r', encoding="ISO-8859-1") as fileobj:#'utf-8'
         data = [line for line in fileobj.readlines()]  
    return data

#%%------------------------------------------------------------------------------




def convert_raw_data_to_list_with_each_torrents_data_as_a_element(raw_data0):
    """
    converts the raw file to a list with each element the string of data for a torrent
    """
    string_just_before_start_of_torrent = "webseedslee"
    data2 = "\n".join(raw_data0)
    *data2_start,data2 = data2.split(":",3)
    data2_start = ":".join(data2_start)
    data2 = data2.split(string_just_before_start_of_torrent)
    data2 = [ lsplit_into_two(line) for line in data2]
    
    data3 = []
    for line,next_line in zip(data2,data2[1:]):
        if len(data3)==0:
            line_new = ":".join(line)
        else:
            line_new = line[1]
        data3.append(line_new+string_just_before_start_of_torrent+next_line[0]+":")
    data3[-1] =data3[-1][:-1]
    data2_finish ="".join(data2[-1])
    assert "\n".join(raw_data0) == data2_start+":" + "".join(data3)+data2_finish,"Some data is lost "
        
    return data3,data2_start,data2_finish

def convert_torrent_list_to_df__complete_date(string,before="completei",after="e11",gettime=True):
    data = split_by_brackets(string,before,after)
    if len(data)>1:
       out_int = time.ctime(int(data[1]))
       return out_int
    return None
 
def convert_torrent_list_to_df(data):
    """
    This converts a list with each element for a string with all the info for a torrent
    Add more columns if needed
    """
    completed_lis = []
    for string in data:
        date = convert_torrent_list_to_df__complete_date(string)
        completed_lis.append(date)
    df = pd.DataFrame(columns=["Name"],data = [line.split(".torrentd8:")[0] for line in data])
    df["Completed"] = completed_lis
    df["Completed"] = pd.to_datetime(df["Completed"] )
    return df

def main(filename=None,filepath=None):
    if filepath is None:
        if filename is None:
            filename = "bittorent_in"
        filepath = get_files_from_folder( filename,".dat",["ignore","Notes.txt"],True)
        
    raw_data0 = read_raw_data(filepath)
    
    data__torrents1, *_extra = convert_raw_data_to_list_with_each_torrents_data_as_a_element(raw_data0)
    torrent_df = convert_torrent_list_to_df(data__torrents1)
    
    out = {"raw_data"    : raw_data0,
           "torrent_lis" : data__torrents1,
           "torrent_df"  : torrent_df,
           "_extra"      : _extra,
           "filepath"    : filepath}
    return out

#%%------------------------------------------------------------------------------





if __name__ == "__main__":
    
    # main is best used if importing functions using another script
    out = main(filename=None,filepath=None)
    
    filename = get_files_from_folder("bittorent_in",".dat",["ignore","Notes.txt"],True)
    raw_data0 = read_raw_data(filename)
    
    data__torrents1, *_extra = convert_raw_data_to_list_with_each_torrents_data_as_a_element(raw_data0)
    torrent_df = convert_torrent_list_to_df(data__torrents1)
    
    torrent_recent_df = torrent_df[torrent_df["Name"].apply( lambda x: any([ n in x for n in ["2016","2017","2018","2019"]]))]
    
     




