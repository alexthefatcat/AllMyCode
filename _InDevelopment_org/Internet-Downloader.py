# -*- coding: utf-8 -*-
"""Created on Sun Oct  7 10:02:33 2018 @author: Alex """
import sys, os
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

save = True
squash=False
skip =True
skipno=2
folder = "C:/Users/Alex/Desktop/img_dump"
path   = r'C:/Users/Alex/Desktop/Alena (@_alena_alena_) • Instagram photos and videos.htm'
path   = "http://hq-pictures.com/index.php?cat=189"
site   = "http://hq-pictures.com/"

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
  

    
    
    
    
    
    
    
    
    
    