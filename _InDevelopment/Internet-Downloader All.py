# -*- coding: utf-8 -*-
"""Created on Sun Oct  7 10:02:33 2018 @author: Alex """
import sys#, os
sys.path.append(r'C:/Users/Alex/Desktop')
from Useful import Web
import requests as req
 
def read_in_txt_file_utf(filepath):
    with open(filepath, 'r',encoding="utf-8") as f:
             txt = f.read()#.splitlines()    #txt = f.readlines()  #txt2 = txt.splitlines() 
    return txt
def read_in_HTML_file_or_url(path):
    if path.startswith("C:/"): HTML = read_in_txt_file_utf(path)
    else :                     HTML = req.get(path).text    
    return HTML
def squash_dict(dic,mode=True,i_limit=5 ):
    if mode:
        return { k:v for i,(k,v) in enumerate(dic.items()) if i<i_limit}
    else:
        return dic
def sitenamecorrect(url):
    return url.replace("&amp;","&") 

Config = {      "save"   : True,
                "squash" : False,
                "skip"   : True,
                "skipno" : 2,
                "folder" : "C:/Users/Alex/Desktop/img_dump",
                "path"   : r'C:/Users/Alex/Desktop/Alena (@_alena_alena_) • Instagram photos and videos.htm',
                "print"  : 2 }
#def print2(*args,**kwargs):   if Config["print"]=>2:   print(*args,**kwargs)


paths={"Home": "http://hq-pictures.com",
        "0"  : "http://hq-pictures.com/index.php?cat=189",
        "1"  : "http://hq-pictures.com/thumbnails.php?album=411",
        "1a" : "http://hq-pictures.com/thumbnails.php?album=411&page=2",
        "2"  : "http://hq-pictures.com/displayimage.php?album=411&pid=381453#top_display_media",
        "3"  : "http://hq-pictures.com/displayimage.php?pid=381453&fullsize=1"}

pathorder = [("0","1"),("1","1a"),("1","2"),("2","3")]

def pathend(word):     return word[1+len(paths["Home"]):]
def correctpath(word): return word if paths["Home"] in word else paths["Home"]+"//"+word
def correcturl(url):   return url.replace("&amp;","&")

def find_url_if_in_HTML(sites_dict, word): 
   for i in sites_dict:
       for w in [word, pathend(word)]:
           if w==i:
               return i 
           if w==correcturl(i):
               return i
   return None

def find_url_or_sta_end_strs_in_HTML(HTML,arg2,sta2="="):
    """First parameter is HTML code second is either a list with a start and end string before the url or
       the second paramter is the url and it finds a list of starting and end string"""
    HTML=correcturl(HTML)
    if type(arg2) is str:##its a url so find [before_str, after_str]
        HTMLsplit=HTML.split(arg2)
        if len(HTMLsplit)>1:
           return [ (a.split(sta2)[-1],b[0]) for a,b in zip(HTMLsplit,HTMLsplit[1:])]
    if type(arg2) in [ list, tuple  ]: ## its a [before_str, after_str] so find url
       HTMLsplit=HTML.split(arg2[0])
       if len(HTMLsplit)>1:
          return [ n.split(arg2[1])[0] for n in HTMLsplit[1:]]
    return None

def return_tripple_or_one(a,b,c,d):
    if d:  return a
    else:  return a,b,c
    

def find_similar_links(path_parrent, path_child=None, foundclass=None):#find class asscated with url or find sites
    mode =  {(0,1):"find_class",(1,0):"find_sites"}.get((path_child is None,foundclass is None),"Error")
    HTML       = correcturl(read_in_HTML_file_or_url(path_parrent))
    all_links  = Web.HTML_2_dict_of_all_links(HTML)
       
    found_link = find_url_if_in_HTML(all_links, path_child)
    
    if found_link is None:
#                path_child_try = "&".join([path_child.split("&")[0]])
#                found_link = find_url_if_in_HTML(all_links, path_child_try)        
            print("The path isnt found in the HTML") 
            for effort in [path_child, pathend(path_child)]:
                beg_fins   = find_url_or_sta_end_strs_in_HTML(HTML, effort)
                if beg_fins is not None: 
                    found_urls = [find_url_or_sta_end_strs_in_HTML(HTML, beg_fin) for beg_fin in beg_fins]
                    return return_tripple_or_one(found_urls, "no class pre-post str segemnt", effort,foundclass)
            raise Exception("No luck finding the url")  
             
    foundclassout = all_links[found_link]["class"] if foundclass is None else foundclass
    found_sites = [links_k for links_k,v in all_links.items() if v["class"]==foundclassout ]
    return return_tripple_or_one(found_sites, foundclassout, found_link, foundclass )



FoundClass,SimilarSites,FoundPath={},{},{}

for pathparent,pathchild in pathorder:
    path_parrent,path_child = paths[pathparent], paths[pathchild]
    FoundClass[pathchild],SimilarSites[pathchild],FoundPath[pathchild]=find_similar_links( paths[pathparent], paths[pathchild] )
del pathparent,pathchild




for pathparent,pathchild in pathorder:
    path_parrent,path_child = paths[pathparent], paths[pathchild]
    FoundClass[pathchild]
    find_similar_links(path_parrent,"Empty",foundclass=None)
    
del pathparent,pathchild

















path_parrent=path3
HTML = read_in_HTML_file_or_url(path3)
sta_end    = find_url_or_sta_end_strs_in_HTML(HTML, correcturl("displayimage.php?pid=381453&amp;fullsize=1"))[0]
sta_end    = find_url_or_sta_end_strs_in_HTML(HTML, pathend(path4))[0]
found_urls = find_url_or_sta_end_strs_in_HTML(HTML,sta_end)

arg2=sta_end


a=HTML.split(sta_end[0])[1:][0]
v=a.split(sta_end[1])[0]





#this would work with arrows (probably)
for n in range(30):
   similar_sites1a = correctpath(correcturl(similar_sites1a[-1]))
   similar_sites1a = find_similar_links(similar_sites1a, "empty", foundclass1a)
   if len(similar_sites1a)<2:
       break








similar_sites3, foundclass3, foundpath3 = find_similar_links(path3, path4)

HTML       = read_in_HTML_file_or_url(path_parrent)
all_links  = Web.HTML_2_dict_of_all_links(HTML) 





path_parrent,path_child,foundclass = path2,path3,None
path_parrent,path_child,foundclass = path3,path4,None



site = correctpath(correcturl(similar_sites2[0]))


HTML = read_in_HTML_file_or_url(path)
all_links            = Web.HTML_2_dict_of_all_links(HTML)
albulm_links = {k:v for k,v in all_links.items() if '="alblink"><a ' in v["class"]}
albulm_links =squash_dict(albulm_links,squash)
if skip:
    albulm_links={ k:v for i,(k,v) in enumerate(albulm_links.items()) if i>=skipno}
    
for k_path_2, v2 in albulm_links.items():
    endpage=""
    selected_links2={}
    print("Albulm:",k_path_2)
    for page in range(2,22):
        if page==20:
            print("Reach Max Page",page)
        HTML2   = req.get(site+k_path_2+endpage).text
        if "Critical error" in HTML2:
            break
        all_links2 = Web.HTML_2_dict_of_all_links(HTML2)
        selected_links2.update({sitenamecorrect(k):v for k,v in all_links2.items() if "thumbnail" in v["class"][0]})
        endpage=f"&page={page}"
    selec = selected_links2
    
    selected_links2 =squash_dict(selected_links2,squash)
    
    for i,(k_path_3, v3) in enumerate(selected_links2.items()):
        if i%20==0:
            print(f"      {i} / {len(selected_links2)}   : {k_path_3}")
        HTML3   = req.get(site+k_path_3).text  
        all_links3 =  Web.HTML_2_dict_of_all_links(HTML3)
  
        imgsrc = sitenamecorrect(HTML3.split("MM_openBrWindow('")[1].split("'")[0])   
        HTML4  = req.get(site+imgsrc).text
        imgsrc2= HTML4.split('window.close()"><img src="')[1].split('"')[0]
        name=imgsrc2.split("/")[-1]
        if save :
            save_to ="/".join([folder,name])
            #print("Saving",save_to) 
            Web.save_image_url(site+imgsrc2, save_to  )
  



