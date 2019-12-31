# -*- coding: utf-8 -*-
"""Created on Mon Dec  3 20:23:37 2018 @author: Alex"""

import sys,os
sys.path.append(r'C:/Users/Alex/Desktop')
from Useful import Web

site           = "https://www.google.co.uk"
site           = "https://phys.org/news/2018-12-borophene-advances-d-materials-platform.html"
#maybe get title after <title> tag at the top for site name
#also create a list of images saved by location


def download_webpage(site, fold = r"C:\Users\Alex\Desktop\Sites_Downloaded_by_Python"):
    """
    Part 1: download the HTML of the site
            from the HTML get the title of the page
            create a nice name to save it site+ title
            create associated file and folder names and create them if they don't exist
    Part 2: get all the images from the page        
    Part 3: create a dict for each one with all the info    
    Part 4: go through the dict save images and change the html woth new location   
    Part 5: Save the HTML
    """
# Part 1:
    def make_nice_site_name(site,HTML_title):
        mainsitename = r"//".join(site.split("//")[1:])
        mainsitename = mainsitename.lstrip("www.")
        if "/" in mainsitename:
            mainsitename = "_".join(mainsitename.split("/")[:-1])
        mainsitename = mainsitename.replace(".","_")
        return mainsitename.capitalize() +"# "+HTML_title

    HTML  = Web.get_HTML_from_URL(site,True)
    #all_links_dict = Web.HTML_2_dict_of_all_links(HTML)
    HTML_title = Web.split_web_tags(HTML,["<head>","<title>"])[0]
    sitename = make_nice_site_name(site,HTML_title)
    html_loc       = fold+"/"+sitename+".htm"
    html_folder    = fold+"/"+sitename
    
    def make_folders_if_path_not_exist(folder_list):
        folder_list=[folder_list] if type(folder_list) is str else folder_list
        for folder in folder_list:
            if not os.path.exists(folder):
               os.makedirs(folder) 
               print(f"folder has been created at:{folder}")
               
    make_folders_if_path_not_exist( [fold, html_folder] ) 
    
# Part2:    
    HTMLsrcs = Web.parser_prefix_postfix(HTML,'src="','"')
    HTMLsrcs=[n for n in HTMLsrcs if any([n.endswith(post) for post in "svg png jpg".split()])]

# Part 3    
    def create_img_info_dict(HTMLsrcs,site):
            def html_friendly(string):
                return string.replace(" ","%20")              
            def get_full_url(url):
                if url.startswith("/"):
                   return main_site+url
                return url
            
            main_site = "/".join(site.split("/")[:3]) 
            img_info={}
            for img_url in HTMLsrcs:
                filename = img_url.split("/")[-1]
                img_info[img_url] = {"filepath": os.path.join(html_folder, filename),          "filename":filename,
                                     "full"    : get_full_url(img_url),          "html_friendly_filepath":html_friendly(sitename+"/"+filename)}
            return img_info            
    
    imgs_info = create_img_info_dict(HTMLsrcs, site)

    files_saved={}
# Part 4
    for img_url, info in imgs_info.items():
        Web.save_image_url(info["full"], info["filepath"] )
        HTML = HTML.replace('src="'+img_url, 'src="'+info["html_friendly_filepath"])
        files_saved[info["full"]] = info["filepath"]

# Part 5   
    with open(html_loc,"w",encoding="utf-8") as f: 
           f.write(HTML)
           
    #"GOOGLE%20-%20Google%20Search_files/googlelogo_color_92x30dp.png"
   
    
download_webpage("https://phys.org/news/2018-12-borophene-advances-d-materials-platform.html")    
    

#    def html_friendly(string):
#        return string.replace(" ","%20")    
#    files_saved={}    
#    for img_url in HTMLsrcs:
#        filename = img_url.split("/")[-1]
#        fileloc  = os.path.join(html_folder, filename)
#        
#        temp = "/".join(site.split("/")[:3])   
#
#        img_url2=img_url
#        if img_url2.startswith("/"):
#           img_url2 =temp+img_url2
#        Web.save_image_url(img_url2, fileloc )
#        filepath2="/"+img_url
#        HTML = HTML.replace('src="'+img_url, 'src="'+html_friendly(sitename)+'/'+filename) 
#        files_saved[img_url2] = fileloc