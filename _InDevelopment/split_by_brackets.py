# -*- coding: utf-8 -*-"""Created on Thu Aug  2 14:26:36 2018@author: milroa1"""
"""    code spits brackets as well as reads in chrome bookmarks and extracts all urls """
import re
#make_tree                           > HTML_2_nested_list
#reconstruct_string_from_nested_list > nested_list_2_HTML
#search_nested                       > search_nested_list
#modify                              > clean_nested_list
def make_tree(data, brackets="{}"):#,reverse=False
    """
    From a HTML string construct a nested list where "{" will start another list inside the current one 
    and "}"  will end.  you get nested list code blocks
    """
    
    items = re.findall("\\"+brackets[0]+"|\\"+brackets[1]+"|[^{}]+", data)
      
    def req(index):
        result = []
        item = items[index]
        while item != brackets[1]:
            if item == brackets[0]:
                subtree, index = req(index + 1)
                result.append(subtree)
            else:
                result.append(item)
            index += 1
            item = items[index]
        return result, index
    return req(1)[0]


def reconstruct_string_from_nested_list(nested_list,brackets="{}"):
    """
    This will do the opposite of the make_tree function
    From a nested list it will construct one large string(usually HTML data) where "{" will have been the start of a new list, 
    and "}"  will of been the end of that nested list.  
    """
    def search_nested1(listorg, searchout=[[]],loc=0):
        for i, elem in enumerate(listorg):
            if type(elem) is list:
                 loc1=loc+1
                 searchout =search_nested1(elem,searchout,loc1)
            elif type(elem) is str:
                  searchout.extend([[loc, elem],[]])
        return(searchout)   
    list_locs=search_nested1(nested_list)
    
    temp=[[-1]]+list_locs+[[-1]]
    for i in range(0,len(list_locs),2):
        list_locs[i] = brackets[ temp[i][0]> temp[i+2][0] ]        
    for i in range(1,len(list_locs),2):
             list_locs[i]=list_locs[i][1]    
    return("".join(list_locs))

def modify_nested(listorg):
    """Basic Function clean all nested lists so that ["',] are removed and spaces begging and ending are removed
    """
    listout=[]
    #listorg[i] = listorg[i]# for i,elem in enumerate
    for elem in listorg:
        if   type(elem) is list:
             listout.append(modify_nested(elem))
        elif type(elem) is str:
             
                 elem = elem.replace("'","").replace(",","").replace('"',"").rstrip().lstrip().rstrip(",").lstrip(",")
                 if not elem in [""]:
                     listout.append( elem )
    return(listout)   


def search_nested(listorg,function="url:", searchout=[],loc=[]):
    """
    searchs a nested list 
    in the format  search_nested(listorg,function="url:"
    where function can be a lambda that returns a false or negateive, or a list and of one element is in it keep it
    """
    function_ = function
    if type(function) is str:
        function_ =lambda x:function in x
    if type(function) is list:
        function_ =lambda x:any([m in x for m in function])
    
    for i, elem in enumerate(listorg):
        if type(elem) is list:
             loc1=loc+[i]
             searchout =search_nested(elem,function_,searchout,loc1)
        elif type(elem) is str:
            if function_(elem):
              searchout.append((loc,elem))
    return(searchout)   

#%%#################################################################################################################
file_path = r"H:\AM\Chrome-Bookmarks-Past-Copies\Bookmarks_example_split_by_brackes_example.txt"

with open(file_path,"r", encoding="utf8") as file:
#     for line in file:
#        text.append(line)   
    data = "".join( file.readlines() )
    data = data.replace("\n","").replace("\t","")
    
tree  = make_tree(data)
tree2 = modify_nested(tree)

data_reconstucted = reconstruct_string_from_nested_list(tree)

all_urls_and_loc = search_nested(tree2)
all_urls         = [n[1] for n in all_urls_and_loc]
all_urls         = [n.split("url:")[1].rstrip().lstrip() if len(n.split("url:")[1])>1 else ""   for n in all_urls ]






