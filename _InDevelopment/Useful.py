# -*- coding: utf-8 -*-
"""Created on Sat Sep 15 19:30:04 2018@author: Alex 
 
A Collection of Useful Objects and functions

import sys
sys.path.append(r'C:/Users/Alex/Desktop')
import Useful

"""

from urllib.request import Request as _Request
from urllib.request import urlopen as _urlopen
import requests                    as _requests
from PIL import Image              as _Image
from io import BytesIO             as _BytesIO
import re as _re
import requests as _req#, sys

############################################################################################################################################
#%%#############################################################################################################################################       
 

class Web():
    """
    A collection of useful functions that is used in websearching and scraping
    
    get_HTML_from_URL( url )
    save_image_url(url, filepath )
    get_all_images_jpgs_from_html(html,end = ".jpg",beg = "http", code_blocks = False)
    create_check(first_url)
    get_image_size(url)
    """
 
 
    
    def get_HTML_from_URL( url ,utf=False):
        
        """ get the HTML code from the url adreess  """
        if utf :
            return str(_urlopen(_Request( url, headers={'User-Agent': 'Mozilla'})).read().decode('utf-8'))
        return str(_urlopen(_Request( url, headers={'User-Agent': 'Mozilla'})).read()) 
    
    def save_image_url(url, filepath ):
        """ save the url of an image 
        save_image_url(url, filepath ) 
        this will overwrite a file alread there no folder exists check as well"""
        img_data = _requests.get(url).content
        with open(filepath, 'wb') as handler:
            handler.write(img_data)
            
    def get_all_images_jpgs_from_html(html,end = ".jpg",beg = "http", code_blocks = False):
        """searchs through a html with parts that start with 'http' and end in '.jpg'
        and extracts these as well as inbetween outputs a list
        
        get_all_images_jpgs_from_html(html,end = ".jpg",beg = "http", code_blocks = False)
        
        """
        list1 = html.split(end)
        if code_blocks :# this should give you the code block before the image
            return ([beg.join(ele.split(beg)[:-1]) + end for ele in list1], [beg + ele.split(beg)[-1] + end for ele in list1] ) 
        else :
            return [beg + ele.split(beg)[-1] + end for ele in list1]
    
    def HTML_2_dict_of_all_links(HTML, exclude_in=[".css","ajax"]):
        """
        Returns a dict in order of apperance key is links
        values are classes(html type) and order of appearance
        """
        class_split = HTML.split("class")
        count=[0 for n in range(len(class_split)+1)]
        for i,t in enumerate(class_split,1):
            count[i]=count[i-1]+len(t)+5
    
        
        class_urls = [c.split('href="')+[i] for i,c in zip(count, class_split) if len(c.split('href="'))>1]
        class_urls = [[c[0],c[1].split('"')[0],c[-1]] for c in class_urls]
        class_urls=  [n for n in class_urls if not n[1][0] =="#"]
        class_urls2= [n for n in class_urls if not any([ w in n[0] for w in exclude_in ]) ]
        all_links_dict={}
        for i,n in enumerate(class_urls2):
            prev= all_links_dict.get(n[1],{"class":[],"ind":[],"HTMLloc":[]})
            prev["class"  ]=prev["class"  ]+[n[ 0]]
            prev["ind"    ]=prev["ind"    ]+[ i   ]
            prev["HTMLloc"]=prev["HTMLloc"]+[n[-1]]
            all_links_dict[n[1]]= prev
        return all_links_dict

    def get_hd_src_url_from_facebook_site(url,filelocation,title=True,sitetitle="",save_bogus_txt_file=False,return_url=False):
        no_outputs= 2+return_url
        #url = sys.argv[-1]
        html    = _req.get(url)
        htmltxt = html.text
        try :
           video_url = _re.search('hd_src:"(.+?)"', htmltxt).group(1)
        except:
            try :
               video_url = _re.search('sd_src:"(.+?)"', htmltxt).group(1)
            except :
                return ("Failed to find hd or sd video", "NO TITLE","")[:no_outputs]
        if title:
            sitetitle = sitetitle + htmltxt.split('id="pageTitle">')[1].split("</title>")[0]
            for char in '"*?|/<>:.':
                sitetitle = sitetitle.replace(char,"")
            sitetitle = sitetitle.replace("\n","").replace("\t","")
        if "mp4" in video_url: 
            if title:
                if len(sitetitle)<130:
                    name = filelocation.replace("TITLE",sitetitle)+".mp4"
                    if not os.path.isfile(name):
                        if save_bogus_txt_file:
                             with open(name.replace(".mp4",".txt"),"w",encoding="utf-8") as f: 
                                 f.write("")
                        else:
                             Web.save_image_url(video_url,name )
                        return                ("Success",sitetitle,video_url)[:no_outputs]
                    return ("Failed File already exists",sitetitle,video_url)[:no_outputs]
                return          ("Failed Title to large",sitetitle,video_url)[:no_outputs]
        print(sitetitle[:min([300,len(sitetitle)]) ])
        return ("Failed Not mp4",sitetitle,video_url)[:no_outputs]


    def split_web_tags(HTML, tag, extract_tag=True):
        """ example for a HTML
            HTML_title = split_web_tags(HTML,["<head>","<title>"])[0]
        """
        if type(tag) is list:
            HTML_i = [HTML]
            for tag_i in tag:
                HTML_i = [flat for string in HTML_i for flat in Web.split_web_tags(string, tag_i, extract_tag) if flat != ""]
            return HTML_i
        tag_end = tag[0]+"/"+tag[1:]
        out = [z for ii in HTML.split( tag ) for z in ii.split(tag_end) ] 
        if extract_tag:
           return out[1::3]
        return out

        
    def create_check(first_url):
        """
        This is pretty basic, it returns a function that checks to see if another string(url)
        is similar to the one input here
        
        create_check(first_url)
        
        it checks to see whats after the com put before the last section is the same
        as well as the length of the url is within 1 of the input one
        
        code:
        middle = "/".join(first_url.split(".com/")[-1].split("/")[:-1])
        length = [len(first_url)+n for n in range(-1,2)]
        def check(str_):
            return (len(str_) in length) and (middle in str_)
        return check        
        
        """
        middle = "/".join(first_url.split(".com/")[-1].split("/")[:-1])
        length = [len(first_url)+n for n in range(-1,2)]
        def check(str_):
            return (len(str_) in length) and (middle in str_)
        return check

    def get_image_size(url):
        """ given an url to a image this returns the size of the image
        """        
        #data = urlopen(Request( url, headers={'User-Agent': 'Mozilla'})).read()
        data = _requests.get(url).content
        im = _Image.open(_BytesIO(data))    
        return im.size
    
    
    def parser_prefix_postfix(txt,pre,post,loc=None):
        """
        HTML="blah blah ( html5player.dog) several more lines( html5player.cat) 
        parser_prefix_postfix(HTML,"html5player.",")") 
        returns#=> ["dog","cat"]
        """
#        out = [ n.split(post)[0]  for n in txt.split(pre) if len(n.split(post))>1]
#        if len(out)>1:
#            return out[1:]
        nested = [p.split(post) for p in txt.split(pre)]
        if len(nested)>1:
           nested=[p[0] for p in nested[1:] if len(p)>1 ]
           if len(nested)>1:
               if loc is None:
                   return nested
               return nested[loc]                   
        return None    
    def split_into_threes(txt,split1):
      """txt= "dog  and name: gary and toad   name: bob fat dude   "
         out = split_into_threes(txt,"name:")
         out #=>   [['dog  and '         , 'name:', ' gary and toad   '],
                    [' gary and toad   ' , 'name:', ' bob fat dude   ' ]]
         out2=[n[-1] for n in out if ("dog" in n[ 0]) and ("gary" in n[-1])]
         out2#=>   [' gary and toad   ']
      """
      temp=txt.split(split1)
      if len(temp)>1:
         temp=[[bef,split1,aft] for bef,aft in zip(temp,temp[1:])]
      else:
         print("Not found in string")
      return temp


        
        
        
        
        
    def xvideo_save(url, path = None):
        """ Example 
            to save directly (wont override existing files) automatically gets right name
                               xvideo_save(xvideourl,r"C:\\Users\\Alex\\Desktop")
                               xvideo_save(xvideourl, True)#DEFAULT IS r"C:\\Users\\Alex\\Desktop"
            or to print out the info (source of the video):
     #xvideourl=    "https://vid2-l3.xvideos-cdn.com/videos/mp4/d/4/0/xvideos.com_d40425dfefbe92ce9b710b98a43e937a.mp4?e=1540656141&ri=1024&rs=85&h=c5102c48a0520e1e79683a992d3eaed5"
                                                          xvideo_save(xvideourl)            
         xvideo_save("https://www.xvideos.com/video31102559/intimo_de_yeimmy_rodriguez_vaza_na_net_-_link_-_adf.ly_1ofnev","C:\\Users\\Alex\\Desktop")                                                
                                                           """
            
        path = r"C:\Users\Alex\Desktop" if path is True else path    
        temp = Web.parser_prefix_postfix(HTML,"html5player.",")")    
        HTML5player_settings={block.split("(")[0]:block.split("(")[1] for block in temp}    
        name = HTML5player_settings["setVideoTitle"].replace("'","")    #name
        url  = HTML5player_settings["setVideoUrlHigh"].replace("'","")  #high url
        print(f"name:\n   {name} \nurl:\n{url}")
        name = name+"".join([k for k,v in {" - XVIDEOcom":"xvideos",".mp4":"mp4"}.items() if v in url])
        if type(path) is str:
            import os.path
            pathout = os.path.join(path,name)
            if not os.path.exists(pathout):
                Web.save_image_url(url, pathout  )      

    def regexp_basic_info():
        print('''              
        """
        abcdefghijklmnopqrstuvwxyz
        ABCDEFGHIJKLMNOPQRSTUVWXYZ
        1234567890
        
        Ha HaHa
        
        MetaCharacters (Need to be escaped):
        
        .[{()\^$|?*+
        
        coreyms.com
        
        321-555-4321
        123.555.1234
        
        Mr. Schafer
        Mr Smith
        Ms Davis
        Mrs Robinson
        Mr. T
        
        [1-7]    same as [1234567]
        [a-f]    same as [abcdef]
        [a-fA-C] same as [abcdefABC]
        [^A-C]^ isnt
        | Either
        ( group)
        
        Quantfiters:
        * 0 or more
        + 1 or more
        ? 0 or 1
        {3} exact number
        {3,5}
        
        
        allcharceter(nonnew line) "." for all characters
        for normal                "a" for "a"
        for metachars             "\."  for "."
        \d for digits ,                  \D for non digits
        \w for word charcters,           \W for non word charcters
        \s whitespace tab newline space, \S for non white search 
        \b word boundry                  \ non word boundry
        Ha$ end of string line
        """
        
        telphone number
        \d\d\d.\d\d\d.\d\d\d
        [] character set anything inthem is searched
        \d\d\d[-.]\d\d\d[-.]\d\d\d
        
        [89]00[-.]\d\d\d[-.]\d\d\d
        
        \d{3}.\d{3}.\d{4}
        
        
        Mr\.?\s[A-Z]\w*
        Mr. Schafer,Mr Smith,Mr. T
        
        Mr(r|s|rs)\.?\s[A-Z]\w*
        Mr. Schafer,Mr Smith,Ms Davis,Mrs Robinson,Mr. T
        
        [A-Za-Z0-9.-]+@[A-Za-Z-]+\.(com|edu|net)
        
        
        https?://(www\.)?(\w+)(\.\w+)
        ''')
    class curly_brackets_parser:
        """
        old name HTML_and_NList
        This section deals with converting string (HTMLdoesnt really work) into nested list
        by default it uses {} to create the nested list
        
        example:
        
        file_path = r"H:\AM\Chrome-Bookmarks-Past-Copies\Bookmarks_example_split_by_brackes_example.txt"

        with open(file_path,"r", encoding="utf8") as file:
        #     for line in file:
        #        text.append(line)   
            data = "".join( file.readlines() )
            data = data.replace("\n","").replace("\t","")
            
        tree  = str_2_NList(data)
        tree2 = modify_NList(tree)
        
        data_reconstucted = NList_2_str(tree)
        
        all_urls_and_loc = search_NList(tree2)
        all_urls         = [n[1] for n in all_urls_and_loc]
        all_urls         = [n.split("url:")[1].rstrip().lstrip() if len(n.split("url:")[1])>1 else ""   for n in all_urls ]

        """
        def str_2_NList(data, brackets="{}"):#,reverse=False
            """
            orginal called make_tree
            From a HTML string construct a nested list where "{" will start another list inside the current one 
            and "}"  will end.  you get nested list code blocks
            """
            
            items = _re.findall("\\"+brackets[0]+"|\\"+brackets[1]+"|[^{}]+", data)
              
#            def req(index):
#                result = []
#                item = items[index]
#                while item != brackets[1]:
#                    if item == brackets[0]:
#                        subtree, index = req(index + 1)
#                        result.append(subtree)
#                    else:
#                        result.append(item)
#                    index += 1
#                    item = items[index]
#                return result, index
#            return req(1)[0]

            def req(index=0,depth=0):
                result = []
                item = items[index]
                while item != brackets[1]:
                    if item == brackets[0]:
                        subtree, index = req(index + 1,depth-1)
                        result.append(subtree)
                    else:
                        result.append(item)
                    index += 1
                    if index>=len(items):
                        break
                    #print(index,depth,items[index][:(min(15,len(items[index])))])
                    item = items[index]
                return result, index  
            return req()[0]            


        def check_str_can_be_parsed(txt):
            depth,mindepth,maxdepth = 0,0,0
            for i,n in enumerate(txt):
                if n =="{":       depth=depth+1
                if n =="}":       depth=depth-1
                mindepth = min([depth,mindepth])
                maxdepth = max([depth,maxdepth])
            if (mindepth>-1)and(depth==0):
                 print("can be parsed by '{}' and max depth is :",maxdepth)
            else:
                 print(f"CANNOT BE PARSED !!, mindepth should be 0 or above {mindepth}, lcurly minus r curly{depth}, should be 0")    

        
        
        def NList_2_str(nested_list,brackets="{}"):
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
        
        def modify_NList(listorg):
            """Basic Function clean all nested lists so that ["',] are removed and spaces begging and ending are removed
            """
            listout=[]
            #listorg[i] = listorg[i]# for i,elem in enumerate
            for elem in listorg:
                if   type(elem) is list:
                     listout.append(Web.HTML_and_NList.modify_NList(elem))
                elif type(elem) is str:
                     
                         elem = elem.replace("'","").replace(",","").replace('"',"").rstrip().lstrip().rstrip(",").lstrip(",")
                         if not elem in [""]:
                             listout.append( elem )
            return(listout)   
        
        
        def search_NList(listorg,function="url:", searchout=[],loc=[]):
            """
            searchs a nested list 
            in the format  search_nested(listorg,function="url:"
            where function can be a lambda that returns a false or negateive, or a list and of one element is in it keep it
            """
            
            function_ = function
            if type(function) is str:
                searchout=[function]
                function_ =lambda x:function in x
            if type(function) is list:
                function_ =lambda x:any([m in x for m in function])
            
            for i, elem in enumerate(listorg):
                if type(elem) is list:
                     loc1=loc+[i]
                     searchout =Web.HTML_and_NList.search_NList(elem,function_,searchout,loc1)
                elif type(elem) is str:
                    if function_(elem):
                      searchout.append((loc+[i],elem))
            return(searchout)       
        
        
        
        
        
        
############################################################################################################################################
#%%#############################################################################################################################################       
    
    
    
    
    
    
    
    
    
    
class ListTools(): 
    def count_elements_in_nested_list(list_, count=0):
        for elem in list_:
            if type(elem) is list:
                count = ListTools.count_elements_in_nested_list(elem, count)
            else :
                count = count+1
        return count    

    def mean(x):                           return(sum(x)/len(x))   
    def reorder(lis_,order):               return([ lis_[i] for i in order])  
    def fill(x,val=0):                     return([val for n in x])  
    def find_nearist(list_,val):           return(min(list_, key=lambda x:abs(x-val)))     
    def find_nearist_idx(myList,myNumber): return(  tuple(reversed(  min(list(enumerate(myList)), key=lambda x:abs(x[1]-myNumber)) ) )   ) 
    def common_elements(lll,ll):           return([l for l in ll if l in lll])
    def standard_deviation(l,sample=0):    return(sum([((n-sum(l)/len(l))**2)/(len(l)-sample) for n in l]))
    def round2(x):                         return(int(x+(x>0)-.5))
    def split(lis_,lam):                   return( [[x for x in lis_ if bool(lam(x))==split_] for split_ in [True,False]]  )
    def unique(lis):                       return([n for i,n in enumerate(lis) if lis.index(n)==i])
    def relu(x):                           return(max([x,0]))
    def maxidx(i):                         return(i.index(max(i))) 
    def primes(n):                         return([x for x in range(2, n) if not 0 in map(lambda z : x % z, range(2, int(x**0.5+1)))] )
    def split2(str_in, deli, m="%"):       return([n for n in (m+deli+m).join(str_in.split(deli)).replace(m+m,m).split(m) if len(n)>0 ]    )
    def mode(List):                        return(max(set(List), key=List.count))
    def s(str_,len_=30):                   return(  str(str_).ljust(len_)[:len_]  )
    def inv_dict(dict_):                   return({v: k for k, v in dict_.items()})



############################################################################################################################################
#%%#############################################################################################################################################       
 


class nested_dict_default:
    """
    creates a nested default dict    so,...

    root = nested_dict_default.create ()

    root['menu']['id'   ] = 'file'
    root['menu']['value'] = 'File'
    root['menu']['value2']["myname"]["is"] = 'spartucus'
    
    root2=nested_dict_default.convert_2_regular(root)
    """
    def create():
        return lambda: defaultdict( defaultdictnested  ) 
    
    def convert_2_regular(d):
        if isinstance(d, defaultdict):
            d = {k: default_to_regular(v) for k, v in d.items()}
        return d
    
if False:
    ###############################################################
    """                 Easy Nested dicts                       """
    ###############################################################
    from collections import defaultdict
    
    def default_to_regular(d):
        if isinstance(d, defaultdict):
            d = {k: default_to_regular(v) for k, v in d.items()}
        return d
    
    defaultdictnested = lambda: defaultdict( defaultdictnested  ) 
    ###############################################################
    root = defaultdictnested ()
    
    root['menu']['id'   ] = 'file'
    root['menu']['value'] = 'File'
    root['menu']['value2']["myname"]["is"] = 'spartucus'
    
    root2=default_to_regular(root)
    ###############################################################


############################################################################################################################################
#%%#############################################################################################################################################       
 

    
def ask(question,ask=True,default=False,loop=True):
    """
    ask(question,ask=True,default=False)
    if ask is False then just output the default
    if ask is True then it will be the reponce
    if loop is true keeps asking until answered
    """
    out = default
    if ask:
        while True:
            response = input( question+" (y/n)").lstrip().rstrip().lower() 
            if response in ["n","no","false"]:
                out = False
                break
            if response in ["y","yes","true"]:
                out = True
                break
            if not loop:
                break
    return out    

############################################################################################################################################
#%%#############################################################################################################################################       
 
       
# where you can easily compare directories
# search for filenames, filetypes, filesizes, dates
#      
# add in future? date no so can order by date,   size order,     name order
# modification date in a cross-platform way is easy - just call os.path.getmtime(path)
# Windows, a file's ctime creation date. Python through os.path.getctime() or 
# why is it full of objects the dataframe
# categories will be good for some of the info
# compare files, compare,binareis
# have differnt modes to save
# save to text
# move and save file
# check in the same folder if multiple files are the same

#helper
# work on forign languages


r"""
    1) save and read csv data in
    2) change name before copying
    3) join data frames
    4) maybe compressed version so for folders unlikely to change hash the folder? and compare

    change file_save to
    from a given path of 
    C:\Users\milroa1\Downloads\delete__\New2_01_(274 Files)_\New2_01_250_Smallish\_99739658_mail.jpg
    
    to
    
    New2_01_(274 Files)_\New2_01_250_Smallish\_99739658_mail.jpg
    
    then do more duplicates
    
    difference
    
    
    
    
    then copy list format
    
    def create_move-copy_tupple_list(keep_old_dir_name=True,overright=[True,False,"rename","warning"][3]):
        # it will have to create folders
    
    df= "move_from",move_to
    
    
    
    then   copying
    
    
1) renamer if file exists
2) new_df create this
3) new_df_to_list_tupples
4) helper> runs through options ask user input and runs and prints out code
5) in mover of copier overwright if today
6) if tryin to overwright and file is identical
7) read/save in csv and text  
    
"""

from itertools import count 
import time, datetime
#from copy import copy        
import os
import pandas as pd  
import hashlib
import shutil
#import itertools
        
  
         


class FileInfo:
    """ Class to extract fileinfo
    """
    _ids = count(0)
#    """ This is a class that allows you to find info about files in a location
#    example :
#        dirloc1      = FileInfo(r"C:\Users\milroa1\Downloads")
#        file_info_df = dirloc1.get_file_info_for_each_of_the_files()
#        #file_info_df will contain all the info about the files
#    
#    """ 
#    newid = next(itertools.count())#itertools.count().next
    
    def __init__(self, directory,levels=["all","single"][0], complete=False): 
#        self.unique_id = FileInfo.newid() 
        self.Flag_run_in_old_mode_ = False
        self.Flag_new_df_exist     = False
        self.Flag_df_exist         = False
        self._id                   = next(self._ids)   
        
        self.mode                 = "file-read"
        
        self.directory            = directory
        self.dir_info             = [["\\".join(directory.split("\\")[:-1]), directory.split("\\")[-1], len(directory)+1 ]]        
        self.directory_name = directory.split("\\")[-1]              
        self.dirs_info = [{"dir_path":directory, "dir_name":self.directory_name, "scan_time":time.strftime('%y%m%d'), "lev":levels}]
        
#        directory_=directory
#        self.directory = [{ "Path"        : directory_, 
#                            "Name"        : directory.split("\\")[-1], 
#                            "Parents"     : "\\".join(directory_.split("\\")[:-1]), 
#                            "Path_length" : len(directory)+1,
#                            "scan_time"   : time.strftime('%y%m%d'), 
#                            "lev"         : levels}]        
 
        if   levels in ["all","All"]:
             self.data          = FileInfo.find_files_folders_paths_in_dir(self, self.directory)
        elif levels in ["single","Single","sing"]:
             self.data          = FileInfo.find_files_folders_paths_just_in_dir_no_subdir(self, self.directory)
        else :
            print("Error")
            
        self.files         = self.data["files"      ]
        self.folders       = self.data["folders"    ]
        self.folders_r     = self.data["folders_raw"]

        self.no_files      = len(self.files)
        self.Flag_print  = self.no_files>800
        self.hashed        = False
        self.Flag_filepaths_list_exist   = False
        
        if self.no_files==0:
             if os.path.exists(directory):
                 print("no files present in directory")
             else :
                 print("directory does not exist")
                 
        if self.Flag_print:
          print('There are '+str(self.no_files)+' in this dir')
        if self.Flag_df_exist:
            self.get_file_info_for_each_of_the_files(self)
            
            
        if complete==True :
            self.Flag_df_exist = True
            self.get_filepaths()             
            self.get_file_info_for_each_of_the_files()
            self.add_hash_column_on_df()
   
            
              
        
    ##fairly happy with
    def find_files_folders_paths_just_in_dir_no_subdir(self,directory):
        file_folders = {"files":[],"folders":[directory],"folders_raw":[self.directory_name]}
        
        for file_folder in os.listdir(directory):
            file_folderpath = os.path.join(directory, file_folder)
            
            if os.path.isdir(file_folderpath):
                file_folders["folders"    ].append(file_folderpath)
                file_folders["folders_raw"].append( os.path.join(self.directory_name,file_folder)  )
            else :
                file_folders["files"].append((0, file_folder))          

        return(file_folders)
    
    ##fairly happy with
    def find_files_folders_paths_in_dir(self,directory):
        file_folders = {"files":[],"folders":[directory],"folders_raw":[self.directory_name]}
        for root, folders, files in os.walk(directory):
            for folder in folders:
                folderpath = os.path.join(root, folder)
                file_folders["folders"    ].append(folderpath)
                file_folders["folders_raw"].append(  os.path.join(self.directory_name,folder)  )
            folder_index = file_folders["folders"].index(root)
            
            for filename in files:
                #filepath = os.path.join(root, filename)
                file_folders["files"].append((folder_index, filename))        
        return(file_folders) 
    
    ##fairly happy with
    def get_filepaths(self):  

        self.data["filepaths"] = [ os.path.join( self.data["folders"][file[0]],file[1] ) for file in self.data["files"]]
        self.Flag_filepaths_list_exist  = True
        return(self.data["filepaths"])

    ## needs some work  
    def get_file_info_for_each_of_the_files(self):
       self.get_filepaths()
       if self.Flag_run_in_old_mode_:
           column_names=['Filepath','File Name','Folder_Loc','Filetype','File_Save%','Date_Modifed','Date_Created','Size_MB','Date_Scanned','file_level_no','hms_Modifed','hms_Created','Extra',"root",'folders_raw','I']
       else :
           column_names=['Filepath','Filename','Filetype','Date_Modifed','Date_Created','Size_MB','3_Subfolder','2_root','1_root','Date_Scanned','file_level_no','hms_Modifed','hms_Created','Extra','I'] # new_Filepath
           
       self.df = pd.DataFrame(columns=column_names,index=range(self.no_files))

       self.df["Extra"       ] = pd.Categorical(self.df["Extra"       ],categories=["Fail-PermissionError","Fail-FileNotFoundError","Success"])
       self.df["Date_Scanned"] = pd.Categorical(self.df["Date_Scanned"],categories=[self.dirs_info[0]["scan_time"]])

       self.df["I"           ] = pd.Categorical(self.df["I"],categories=[self._id ])
       
       self.df["I"           ] = self._id
       self.df['Filepath'    ] = self.data["filepaths"]
       
       self.df.loc[1:,"Date_Scanned"] = self.dirs_info[0]["scan_time"]
       self.df.loc[0 ,"Date_Scanned"] = self.dirs_info[0]["scan_time"]  
       
       if self.Flag_run_in_old_mode_:
           self.df["root"        ] = pd.Categorical(self.df["root"        ],categories=[ self.directory])       
           self.df.loc[1:,'root'        ] = self.directory
           self.df.loc[0 ,'root'        ] = self.directory  
       else :
           self.df["1_root"        ] = pd.Categorical(self.df["1_root" ], categories=[ self.dir_info[-1][ 0]])     
           self.df["2_root"        ] = pd.Categorical(self.df["2_root" ], categories=[ self.dir_info[-1][ 1]])     
           
           self.df.loc[1:,'1_root'        ] = self.dir_info[-1][ 0]
           self.df.loc[0 ,'1_root'        ] = self.dir_info[-1][ 0]
           
           self.df.loc[1:,'2_root'        ] = self.dir_info[-1][ 1]
           self.df.loc[0 ,'2_root'        ] = self.dir_info[-1][ 1] 
           
       
       #self.df['folders_raw'] = self.data["folders_raw"]
       for lineno, filepath in enumerate(self.data["filepaths"]):#self.files): #       for lineno, filepath in enumerate(self.data["file_paths"]): 
           if self.Flag_print and (lineno % 100)==0:
              print('   '+str(lineno)+' out of '+str(self.no_files)+' done')
              
           try:
               #filepath2=os.path.join(self.folders[filepath[0]], filepath[1])
               statinfo = os.stat(filepath)
               if self.Flag_run_in_old_mode_:
                   self.df.loc[lineno,['File Name','Folder_Loc','Filetype','File_Save%']] = FileInfo.__old_filepath_info_for_a_file__( self,filepath )
               else :
                   self.df.loc[lineno,['Filename','3_Subfolder','Filetype']] = FileInfo.__filepath_info_for_a_file__( self, filepath )
               
               self.df.loc[lineno,['hms_Created','Date_Created','hms_Modifed','Date_Modifed'    ]] = FileInfo.__time_info_for_a_file__( self,    statinfo )
               self.df.loc[lineno, 'Size_MB'] = float(statinfo.st_size)/1048576# MB
               self.df.loc[lineno, 'Extra' ] = "Success"
       
           except PermissionError: 
               self.df.loc[lineno, 'Extra' ] = "Fail-PermissionError"  
           except FileNotFoundError:
               self.df.loc[lineno, 'Extra' ] = "Fail-FileNotFoundError"  
       self.Flag_df_exist = True
       ##try more of this 
       if self.Flag_run_in_old_mode_:
           self.df['file_level_no'] = self.df['File_Save%'].str.count("%")
           self.df['file_level_no'] = self.df['file_level_no']-3# may need to chane
       else :
           #p=self.df['3_Subfolder'].str.count("\\")
           p=self.df['3_Subfolder'].apply(lambda x:x.count("\\"))
           #self.df['file_level_no'] = self.df['3_Subfolder'].str.count("\\")+(self.df['3_Subfolder'].str.len()>0)
           self.df['file_level_no'] = p+(self.df['3_Subfolder'].str.len()>0)
           
       self.df['file_level_no'] = self.df['file_level_no' ].astype('category')   
       self.df["Filetype"     ] = self.df["Filetype"  ].astype('category')
       if self.Flag_run_in_old_mode_:
           self.df["Folder_Loc"   ] = self.df["Folder_Loc"].astype('category')
       else :
           self.df["3_Subfolder"   ] = self.df["3_Subfolder"].astype('category')
       
       
       return(self.df)
   
    def add_hash_column_on_df(self):
        self.df['Hashed'] = self.hash_files(list(self.df['Filepath']))
        self.hashed=True
        
    def group_dataframe(self,grouped, column_different=False, df=None):
        if df is None:
            df=self.df
            
        if len(column_different)==0:
            column_different =False
            
        if not column_different ==False:            
            df["temp"    ] = df.groupby(grouped + column_different).ngroup() 
            
        dfg = df.groupby(grouped)
        df["group_no"   ] = dfg.ngroup()
        #df["group_count"] = dfg[[n for n in df.columns if not n in grouped][0]].transform(lambda x:len(x))#["Filepath"]
        df["group_count"] = dfg[grouped[0]].transform(lambda x:len(x))
        
        if not column_different ==False: 
           df["group_diff"  ] = dfg["temp"].transform(lambda x:x.nunique())
           df.drop("temp", axis=1,inplace=True)
#        if not column_different ==False:
#           df["group_diff"] = dfg[ column_different ].transform(lambda x:x.nunique())  
#        if not column_different ==False:
#           df["group_diff"] = dfg[ column_different ].transform(lambda x:len(x.drop_duplicates()))             
        return(df)  




      
    def compare(self,self2):        
        grouped          = ["Size_MB", "Filetype"]
        column_different = ["Hashed"]
 
        if not self.Flag_df_exist:
           self.get_file_info_for_each_of_the_files
        if not self.Flag_df_exist:
           self2.get_file_info_for_each_of_the_files

        self.df_joined = self.df + self2.file_info_df
        
        
   #@staticmethod         
    def __time_info_for_a_file__(self,statinfo):
        temp=datetime.datetime.strptime(time.ctime(statinfo.st_ctime), "%a %b %d %H:%M:%S %Y")
        ctime2=int(temp.strftime('%y%m%d00%H%M%S'))
        ctime3=int(temp.strftime('%y%m%d'))
        temp=datetime.datetime.strptime(time.ctime(statinfo.st_mtime), "%a %b %d %H:%M:%S %Y")
        mtime2=int(temp.strftime('%y%m%d00%H%M%S'))
        mtime3=int(temp.strftime('%y%m%d'))
        return([ctime2,ctime3,mtime2,mtime3])  
    #@staticmethod
    def __old_filepath_info_for_a_file__(self,file):#['File Name','Folder_Loc','Filetype','File_Save%']
        file=file.replace("\\","/")
        last_bracket=file.rfind('/')
        file_save_name=file[3:].replace('/', '%').replace("\\", '%') 
        file_directory = len(file[3:]) - len(file[3:].replace('/', '').replace("\\", ''))
        return([file[1+last_bracket:],file[:last_bracket],file[file.rfind('.'):],file_save_name])
    
    
    #folderloc,filesavefile raw
    def __filepath_info_for_a_file__(self,file):#['File Name','Filetype','File_Save%']
       # file=file.replace("\\","/")
        last_bracket=file.rfind('\\')
        
        #1_root      = self.dir_info[-1][ 0],        2_root      = self.dir_info[-1][ 1],        3_subfolder = file[self.dir_info[-1][ 2]:]

        Filename       = file[1+last_bracket:]
        
        Subfolder_Name = file[self.dir_info[-1][ 2]:last_bracket] # 3_subfolder
        Filetype       = file[1+file.rfind('.'):]
        #print(Subfolder_Name,":",file,self.dir_info[-1][ 2],last_bracket,"file[self.dir_info[-1][ 2]:last_bracket]")
        #"""{1_root: C:\Users\milroa1, 2_foldername:download, 3_subfolder:delete__\New2_01_(274 Files)_\New2_01_250_Smallish}"""        
        return([Filename, Subfolder_Name, Filetype ])    
    

    
    
    
    def local_folder(self):
        temp = len(self.folders[0])+1
        self.local_folders=[n[temp:] for n in self.folders[1:]]
        return(self.local_folders)
    def __getitem__(self,str_):
        #print(str_)
        out=self.directory
        if str_ in ["files"]:
            out = self.files
        if str_ in ["folders"]:
            out = self.folders   
        if str_ in ["folders_raw"]:
            out = self.data["folders_raw"] 
        if str_ in ["df","dataframe","DataFrame","file_info_df","fileinfo_df"]:
            if not self.Flag_df_exist:
                self.get_file_info_for_each_of_the_files()
            out = self.df  
        if str_ in ["filepaths"]:
            if not "filepaths" in self.data:
               self.get_filepaths()
            out = self.data["filepaths"]  
        if str_ in ["all"]:
            out = {"files":self.files,"folders":self.folders ,"folders_raw":self.data["folders_raw"]}
            if not self.Flag_df_exist:
                    self.get_file_info_for_each_of_the_files()
            out["df"] = self.df  
            if not "filepaths" in self.data:
                   self.get_filepaths()
            out["filepaths"] = self.data["filepaths"]  
        return(out)
    def __setitem__(self,str_,value_):
        if str_ in ["new_df"]:
            self.new_df = value_
            self.Flag_new_df_exist = True
    
    
    ##fairly happy with    
    def __len__(self):
        return len(self.files)  
#    def __add__(self,other):
#        pass
#        self.directory = [self.org_directory, other.directory]
#        self.data    = {"files":self.data["files"  ]+other.data["files"  ], "folders": self.data["folders"] +other.data["folders"]}
#        self.files   = self.data["files"  ]
#        self.folders = self.data["folders"]        
    def __call__(self,x):
        pass
    def hash_files(self,file_paths, BUF_SIZE = 65536, HASH_MODE=["md5_","sha1"][0]):# maybe update for dataframes
        if type(file_paths) is str:
            file_paths=[file_paths]
        hash_list=[[] for n in file_paths]
    
        for i,file_path in enumerate(file_paths):
            if HASH_MODE in ["md5","md5_"]:     
                hasher = hashlib.md5()
            if HASH_MODE in ["sha1"]:     
                hasher = hashlib.sha1()  
            with open(file_path, 'rb') as f:
                while True:
                    data = f.read(BUF_SIZE)
                    if not data:
                        break
                    hasher.update(data)
            hash_list[i] = hasher.hexdigest()
        return( hash_list )
    def add_new_Filepath_2_df(self, newroot,dataframe=None, drop_main_folder_in_filepaths=False, drop_levels=False,replace_main_folder=None):
        
            r"""
            add_new_Filepath_2_df(self,dataframe, newroot, newfolder=None, same_folder_name=True, drop_levels=False)

            say old root to search is
                C:\Users\milroa1\Downloads
                for a file 
                C:\Users\milroa1\Downloads\aaa\b12.jpg
            #default    
            newroot     =  r"C:\Users\milroa1\Desktop"
            newFilepath => C:\Users\milroa1\Desktop\Downloads\aaa\b12.jpg
            
            if drop_main_folder_in_filepaths =True # Default False
            newFilepath => C:\Users\milroa1\Desktop\aaa\b12.jpg     
            
            drop_levels turns the / into %            
            so 
            newFilepath => C:\Users\milroa1\Desktop\aaa%b12.jpg 
            
            can put one owns dataframe
            by default it works on new_df but if doenst exist will work on main_df
            
            replace_main_folder="Desktop(1)"
            will thus change 
            
            """
            #pj=os.path.join
            def pj(*args):return("\\".join([n for n in args if not n in [""]]))
            if dataframe is None:

                if self.Flag_new_df_exist:
                   dataframe = self.new_df
                else :
                   dataframe = self.df
                    #dataframe['newFilepath'] = .apply(lambda x: os.path.join(x[0],x[1],x[2],x[3]), axis=1)
                    
            if not replace_main_folder is None:
                newroot="\\".join(newroot.split("\\")[:-1] +[replace_main_folder])
                
            if not drop_levels:
                if not drop_main_folder_in_filepaths:
                    dataframe['newFilepath'] = dataframe[["1_root","2_root","3_Subfolder","Filename"]].apply(lambda x: pj(newroot,x[1],  x[2],x[3])                  , axis=1)#Default
                else :
                    dataframe['newFilepath'] = dataframe[["1_root","2_root","3_Subfolder","Filename"]].apply(lambda x: pj(newroot,       x[2],x[3])                  , axis=1)
            else : 
                if not drop_main_folder_in_filepaths:    
                    dataframe['newFilepath'] = dataframe[["1_root","2_root","3_Subfolder","Filename"]].apply(lambda x: pj(newroot,x[1] ,(x[2]+x[3]).replace("\\","%")), axis=1)
                else :
                    dataframe['newFilepath'] = dataframe[["1_root","2_root","3_Subfolder","Filename"]].apply(lambda x: pj(newroot,      (x[2]+x[3]).replace("\\","%")), axis=1)
    def copy_or_move_list_of_tupples_from_to__folder_already_exist(self, operation, listt, settings="stop",print_=True ):
        """
            operation_list=

            #[("file_loc1","file_to1"),....]

        
        if no list is input default to the copymovelist in self
        
        operation = ["copy","move","delete","blank_txts"]
        if file already exists what to do ? ,(settings)
      i "stop"       : (default) ,stops function imiataly,  
      i "delete_org" : deletes the orginal file
        "rename_org" : renames the orginal so file > file(1) > file(2) 
        "move_org",  : moves the orginal
      i "skip",  : skips that file and carries on 
        "rename_new" : renames the new file in the format  file > file(1) > file(2) 
    
        """
        ## add one copy over if its today
        ## give different warning if both files seem identical
        for file_no,(from_, to_) in enumerate(listt):
            if os.path.exists(from_):#os.path.isfile(file_name)
                if operation=="blank_txts":
                    to_=to_.replace(".","_")+".txt"

                if os.path.exists(to_):
                    if print_:
                        print(f"*** Exists: from:{from_},  {to_}")
                    if settings in ["delete_org"]:
                        os.remove(to_)
                    if settings in ["stop"]:
                        print(f"Stopped: from:{from_},  {to_}")
                        break   
                    if operation in ["rename_new"]:
                       from_ = from_ + "update this"
                    if operation in ["rename_old"]:
                        to_new=to_+"update this"
                        shutil.move( to_, to_new)
                        
                if not os.path.exists(to_):# you should never try to save if file doesnt exist already
                
                    if operation=="move":
                       shutil.move( from_, to_)
                    if operation=="copy":
                       shutil.copy2(from_, to_)  
                    if operation=="blank_txts":
                       open(to_, 'x')
                       
                       
#    def convert_filepaths_to_all_parent_folders(self, filepaths, root=r"C:\Users\milroa1\Downloads") :   
#        folder_temp = [ root ] + filepaths
#        folders=[]
#        def remove_duplicates_in_sorted_list(dup_remove):
#            return([ dup_remove[0] ] + [ dup_remove[n] for n in range(1,len(dup_remove)) if not dup_remove[n-1]==dup_remove[n] ])
#        def parent_folders(fold):
#            return(sorted(["\\".join(n.split("\\")[:-1]) for n in folder_temp  ]))
#        while len(folder_temp)>1:
#             folder_temp = [ root ] + parent_folders(folder_temp)[1:]
#             folder_temp = remove_duplicates_in_sorted_list(folder_temp)
#             folders.extend(folder_temp)
#        return remove_duplicates_in_sorted_list(sorted(folders))
    
            
    def convert_filepaths_to_all_parent_folders(self,filepaths, root=None) :  

        if root is None:
           root = "".join([a  for a,b in zip(filepaths[0],filepaths[-1]) if a==b]).rstrip("\\")

        def remove_duplicates_in_sorted_list(dup_remove):
            return([ dup_remove[0] ] + [ dup_remove[n] for n in range(1,len(dup_remove)) if not dup_remove[n-1]==dup_remove[n] ])
        def parent_folders_list(fold):
            return(sorted(["\\".join(n.split("\\")[:-1]) for n in fold  ]))
        
        root_b = parent_folders_list([root])[0]
        folder_temp = [root_b] + filepaths
        folders=[]        
        count=0
        while len(folder_temp)>1:
             # we put the root in so that if appears else where we remove the duplicate and skip this one when saving
             folder_temp = [root_b] + parent_folders_list(folder_temp[1:])
             folder_temp = remove_duplicates_in_sorted_list(folder_temp)
             folders.extend(folder_temp[1:])
             count=count+1
             if count>50:
                 print("Obvius error count is over 50 in the loop to loop at parents")
                 break
        return remove_duplicates_in_sorted_list(sorted(folders))    
    
    def make_folder_list(self,folder_list):           
        for folderpath in sorted(folder_list):
            if not os.path.exists(folderpath):
                os.makedirs(folderpath)  
                
    def __add__(self, df2add):
        print( "This will only combine the dataframes, and does it inplace" )
        str_type = str(type( df2add )).split(".")[-1][:-2]
        if    str_type=="DataFrame":
            self.df=self.df.append(df2add)
        elif str_type=="FileInfo":
            self.df=self.df.append(df2add["df"])
        return self


    
#    def join(self,other):
#        pass   
#    def read_csv(self,options={}):
#        pass
#    def save_csv(self):
#        pass
#    def save_2_text(self):
#        pass    



    


############################################################################################################################################
#%%#############################################################################################################################################       
 


if __name__ == "__main__":
    if False:
    
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
        
        
############################################################################################################################################
#%%#############################################################################################################################################       
    if False:    
    
        Config={"1_try getting basic info from downloads"                      : False,
                "2_try the single setting so no subfolderfiles"                : False,
                "3_copy_files_in_one_folder_to_another"                        : False,
                "4_check if there are dupplicates,as well as identical files"  : False,
                "5_do all steps in one and try some groupbys and appending"    : False }
    
        def print_out_objects_methods(obj,_hidden=False):
            print("#"*40,f"\n Methods in Object     _hidden={_hidden} \n","#"*40)
            for n in dir(obj):
                if (not n[0]=="_")or(_hidden):
                   print(n)
            print("#"*40)
    
    
    
    
        if Config["1_try getting basic info from downloads"]:
            
            dirloc1      = FileInfo(r"C:\Users\milroa1\Downloads")    
            #dirloc1.add_hash_column_on_df()
            
            files1        = dirloc1["files"      ]
            folders1      = dirloc1["folders"    ]    
            files1_df     = dirloc1["df"         ]
            filepaths1    = dirloc1["filepaths"  ]
            foldersr1     = dirloc1["folders_raw"]
            file_all      = dirloc1["all"        ]
            # add extra col dataframe where to move/copy them to
            dirloc1.add_new_Filepath_2_df(r"D:")## add column where to move files  
            
            print(f"There are {len(dirloc1)} files in dirloc1")
            print(files1_df.dtypes)   
            print_out_objects_methods(dirloc1,_hidden=True)
    
        if Config["2_try the single setting so no subfolderfiles"]: # Single has been tested and works
        
            dirloc2       = FileInfo(r"C:\Users\milroa1\Downloads",levels="single")#  , dfed = True
            files2        = dirloc2["files"      ]
            folders2      = dirloc2["folders"    ]    
            files2_df     = dirloc2["df"         ]
            filepaths2    = dirloc2["filepaths"  ]
            foldersr2     = dirloc2["folders_raw"]
    
    
        if Config["3_copy_files_in_one_folder_to_another"]: 
    
            folderpath_from =  r"\\NDATA12\milroa1$\Desktop\test__a" 
            folderpath_to   =  r"\\NDATA12\milroa1$\Desktop\test__b" 
            dirloc3         = FileInfo(folderpath_from)#  , dfed = True
            filepaths3      = dirloc3["filepaths"]
            
            copy_list = [(n,n.replace("test__a","test__b")) for n in filepaths3]
    
            # for this test remove the folder if it exists(just for this test)
            if os.path.exists(folderpath_to):
               shutil.rmtree(folderpath_to)#file:os.remove,empty dir: os.rmdir
                     
            filepaths_to = [a[1] for a in copy_list]   
            folders_from_files = dirloc3.convert_filepaths_to_all_parent_folders(filepaths_to)
            dirloc3.make_folder_list(folders_from_files)       
            dirloc3.copy_or_move_list_of_tupples_from_to__folder_already_exist("copy",copy_list)
            
        if Config["4_check if there are dupplicates,as well as identical files"]: 
        
            dirloc4      = FileInfo(r"D:\New2_4")  #"hashed"
            files4       = dirloc4["files"]
            files4_df    = dirloc4["df"]      
            dirloc4.add_hash_column_on_df()
            # check if there any files where there are more than one file with the same ["Size_MB", "Filetype"]# column added onto end being group count
            files4_same_sz_and_type = dirloc4.group_dataframe(["Size_MB", "Filetype"])
            temp = files4_same_sz_and_type[files4_same_sz_and_type["group_count"]>1]
            
            files4_same_sz_and_type_unique_hash = dirloc4.group_dataframe(["Size_MB", "Filetype"],["Hashed"])
            files4_df_dupli = files4_same_sz_and_type_unique_hash[(files4_same_sz_and_type_unique_hash["group_diff"]==1)&(files4_same_sz_and_type["group_count"]>1)]
    
    
        if Config["5_do all steps in one and try some groupbys and appending"]: 
    
            dirloc_joined = FileInfo(r"D:\New2_4", complete=True) + FileInfo(r"D:\temp__", complete=True)
            df_joined = dirloc_joined.df
    
            print_out_objects_methods(dirloc_joined,_hidden=True)
            
            ############################################################################
            ############################################################################
    
            # These Different ['Filename' , '3_Subfolder', '2_root', '1_root'],[ 'Date_Modifed', 'Date_Created'], 'I'
            #'Size_MB','Filetype','Filename','Hashed'# useful to group
    
            def grouped_extract(df,dirloc,grouped,column_different,group_count,group_diff):
                dfg = dirloc.group_dataframe(grouped, column_different, df) 
                def temp(dfs, in_):
                   in_type=str(type(in_))[8:-2]
                   if in_type in ['int'     ]:
                       return dfs.isin([in_])
                   if in_type in ['function']:    
                       return dfs.apply(in_)
                   if in_type in ['list'    ]:    
                       return dfs.isin(in_)
                return dfg[(temp(dfg["group_count"],group_count)) & (temp(dfg["group_diff"],group_diff))]
     
     
    
            # find probably same files in different locations(not on different drives)
            df_dup1 = grouped_extract(df_joined, dirloc_joined, ["Size_MB", "Filetype"          ], ["Hashed"          ],lambda x:x>1, 1           )
            # find probably same files(although does not check hash) in different locations( on different drives)        
            df_dup2 = grouped_extract(df_joined, dirloc_joined, ["Size_MB", "Filetype"          ], ["I"               ],lambda x:x>1, lambda x:x>1)
            # find probably same files  in different locations(not neccarily on different drives)          
            df_dup3 = grouped_extract(df_joined, dirloc_joined, ["Size_MB", "Filetype", "Hashed"], ["1_root", "2_root"],lambda x:x>1, 1           )
            # find probably same files  in different locations( on different drives) 
            df_dup4 = grouped_extract(df_joined, dirloc_joined, ["Size_MB", "Filetype", "Hashed", '3_Subfolder'], ["I"],lambda x:x>1, 2           )   
            
            # create a dataframe matchs one file onto another
            _from = ["D:","temp__"]
            _to   = ["D:","New2_4"]
            df_dup4=df_dup4.sort_values("group_no")
            if list(df_dup4.iloc[0,:][["1_root","2_root"]])==_to:
                temp=df_dup4[["Filepath"]].iloc[1::2,:]
                df_dup4 = df_dup4.iloc[0::2,:]
                df_dup4["matched_Filepath"] = temp.values
                del temp
            
    #        df_dup4.loc[(df_dup4["1_root"]==_from[0])&(df_dup4["2_root"]==_from[1]), "matched_Filepath"] = df_dup4[(df_dup4["1_root"]==_to[0])&(df_dup4["2_root"]==_to[1]) ]["Filepath"].values
    #        df_dup4=df_dup4[(df_dup4["1_root"]==_from[0])&(df_dup4["2_root"]==_from[1]) ]
    #        
    #        
    #        
    #        
    #        
    #        lst=df_dup4.groupby("group_no").apply(lambda x:x[x[["1_root"]=_from[0]       )
    #        
    #        
    #        
            
            # then group these dataframes
            # copy or move same folders
            # search for like names
            # save/read a csv/text
            
            
    # save a txt file inplace od each file        
     
    
            # use this instead
            folders_from_files = dirloc3.convert_filepaths_to_all_parent_folders(filepaths_to)
            dirloc3.make_folder_list(folders_from_files) 
    
    
    
        if Config["4a)create-the-folders"]: 
            ignore_folders=[]
            folder_names_list = list(map(lambda x: Config["create_txt_file_for_each_file2"] +"\\" + x[0],folder_names_list))       
            for folder_n in folder_names_list:
                if Config["save_switch" ]:
                    if not os.path.exists(folder_n):
                       if len(folder_n)<245:
                           os.makedirs(folder_n)
                       else :
                           ignore_folders.append(folder_n)
            del folder_n
    ##################################################################
    
    
    Config={"File_number_limit_user_input":10_000}
        
    
    def user_input(str_1=None,str_2="Do you want to carry On"):
        print(str_1)
        input_ = input(str_2,"?(y/n) \n")
        return(input_ in ["y","yes","YES"])
        
            
        if Config["5a)create-the-files(txt)"]:    
            
            filepaths_s2 = list(map(lambda x: Config["create_txt_file_for_each_file2"] +"\\" + x,list(filepaths_s2)))  
            
            block_too_many_files, blocker_file_exists, blocker_file_exists2 = True, True, True
            
            if len(filepaths_s2)>Config["File_number_limit_user_input"]:
    
                block_too_many_files = user_input(str_1="warning there are "+str(len(filepaths_s2))+" files that are gonna be save!!")   
                
            if  block_too_many_files:
                
                for file in filepaths_s2:
                    
                    if blocker_file_exists:
                        if os.path.exists(file+".txt"):
                            print("warning there are already files in there with the same name, e.g " + file + ".txt")
                            blocker_file_exists2 = input("do you still want to carry on  y/n \n") in ["y","yes","YES"]
                            blocker_file_exists = False
                            
                        if blocker_file_exists2:
                           if Config["save_switch" ]:
                               if len(file)>245:
                                   file=file[:245]
                                   if not any([file in fi for fi in ignore_folders]):
                                      with open(file + ".txt", "w") as text_file:
                                           text_file.write(" ")                                  
                               else:
                                      with open(file + ".txt", "w") as text_file:
                                           text_file.write(" ")
                del file
                del block_too_many_files, blocker_file_exists, blocker_file_exists2
                print("All files saved")
                
                
                
            
    ######################################################################   
    ## add search and comparison, save to a csv,data info
           
            
    if 0: # These are some extra options    
        
        filetypes_counts=files1_df["Filetype"].value_counts()
        
        labels="0-10kb 10-100kb 100kb-1mb 1-10mb 10mb-100mb 100mb-1gb 1-10gb 10-100gb".split()
        bins=[0]+[10**n for n in range(-3,5)]
        memory_info=pd.cut(files1_df["Size_MB"],bins=bins,labels=labels).value_counts()
        del labels,bins
        
        # how to modify categorical data
        files1_df["Filetype"]=files1_df["Filetype"].astype('category')
        files1_df["Filetype"].cat.categories
        files1_df["Filetype"]=files1_df["Filetype"].cat.rename_categories({".jpg":"fuck-you"})
        files1_df["Filetype"]=files1_df["Filetype"].cat.add_categories("junk")
        files1_df["Filetype"].iloc[3]="junk"
                ## these are possible options as well:
        #
        #    rename_categories,    reorder_categories,    remove_categories
        #    remove_unused_categories,    set_categories
        
        #renmae category add addtional new ones
        
        
        #    """ This is a class that allows you to find info about files in a location
        #    example :
        #        dirloc1      = FileInfo(r"C:\Users\milroa1\Downloads")
        #        file_info_df = dirloc1.get_file_info_for_each_of_the_files()
        #        #file_info_df will contain all the info about the files
        #    
        #    """ 
        #    newid = next(itertools.count())#itertools.count().next
        
       
    
        
        
        
        
    folder2=r"C:\Users\Alex\Desktop\check if files are identical\samp"
        
    folder1=r"C:\Users\Alex\Desktop\FACEGIRLS\samp"
        
    dirloc_joined = FileInfo(folder1, complete=True) + FileInfo(folder2, complete=True)
    df_joined = dirloc_joined.df
    
    #print_out_objects_methods(dirloc_joined,_hidden=True)
    
    ############################################################################
    ############################################################################
    
    # These Different ['Filename' , '3_Subfolder', '2_root', '1_root'],[ 'Date_Modifed', 'Date_Created'], 'I'
    #'Size_MB','Filetype','Filename','Hashed'# useful to group
    
    def grouped_extract(df,dirloc,grouped,column_different,group_count,group_diff,both=False):
        global dog
        dfg = dirloc.group_dataframe(grouped, column_different, df) 
        dog=dfg
        def temp(dfs, in_):
           in_type=str(type(in_))[8:-2]
           if in_type in ['int'     ]:
               return dfs.isin([in_])
           if in_type in ['function']:    
               return dfs.apply(in_)
           if in_type in ['list'    ]:    
               return dfs.isin(in_)
        mask = (temp(dfg["group_count"],group_count)) & (temp(dfg["group_diff"],group_diff))
        if both:
             return dfg[mask],dfg[~mask]
        else:
            return dfg[mask]
     
     
    
    # find probably same files in different locations but differnet names as well (not on different drives)
    df_dup1 = grouped_extract(df_joined, dirloc_joined, ["Size_MB", "Filetype"          ], ["Hashed"          ],lambda x:x>1, 1           )
    
    # find probably same files(although does not check hash) in different locations( on different drives)        
    df_dup2 = grouped_extract(df_joined, dirloc_joined, ["Size_MB", "Filetype"          ], ["I"               ],lambda x:x>1, lambda x:x>1)
    # find probably same files  in different locations(not neccarily on different drives)          
    df_dup3 = grouped_extract(df_joined, dirloc_joined, ["Size_MB", "Filetype", "Hashed"], ["1_root", "2_root"],lambda x:x>1, 1           )
    # find probably same files  in different locations( on different drives) 
    df_dup4 = grouped_extract(df_joined, dirloc_joined, ["Size_MB", "Filetype", "Hashed", '3_Subfolder'], ["I"],lambda x:x>1, 2           )   
        
    df_dup5t,dup5f = grouped_extract(df_joined, dirloc_joined, ["Size_MB", "Filename", "Hashed"], ["I"],lambda x:x>1, 2    ,True       )   
     
    
    
    
    
    
    

    