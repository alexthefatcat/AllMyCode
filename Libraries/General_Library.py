# -*- coding: utf-8 -*-
"""Created on Tue Dec 10 16:23:44 2019 @author: Alexm"""

#%%###########################################################################################################
def KeepAlphaNumbericWords2(string):
    "Only keep alpha Nummber and spaces but remove multiple spaces"
    first =  ''.join( l if l in " abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789./+=?!'@#*" else " " for l in string).strip()
    return  ' '.join([ l for l in first.split(" ") if l !=""])

def KeepAlphaNumbericWords(string):
    "Only keep alpha Nummber and spaces but remove multiple spaces"
    first =  ''.join( l if l.isalnum() else " " for l in string).strip()
    return  ' '.join([ l for l in first.split(" ") if l !=""])

def ExtractNumbersFromString(string):
    string1="".join([s  if s.isdigit() else " " for s in string ])
    return [int(n) for n in string1.split(" ") if n != ""]  

def Index2KeyValue(dic,ind=0):
    for i,(k,v) in enumerate(dic.items()):
        if i==ind:
            return k,v
        
def lstrippattern(string,pattern,changed=False):
    processed = False
    if string.startswith(pattern):
        processed = True
        string = string[len(pattern):]
    if changed:
        return string,processed          
    return string

def rstrippattern(string,pattern,changed=False):
    processed = False    
    if string.endswith(pattern):
        processed = True
        string = string[:-len(pattern)]
    if changed:
        return string,processed   
    return string             

#%%###########################################################################################################

def SaveNestedList2CSV(nlist,filename):
    with open(filename, 'w', newline='') as csvfile:
        csvsaver = csv.writer(csvfile, delimiter=',',quotechar='"', quoting=csv.QUOTE_MINIMAL)
        csvsaver.writerow('SiteNo Site TextNo Text Xpath'.split())
        for l in nlist:
            csvsaver.writerow(l)

def MakeDirIfNotExist(foldpath):
    import os
    if not os.path.isdir(foldpath) :
        if os.path.isfile(foldpath):
           print(f"Warning a File Exists at :'{foldpath}'") 
        else:
           os.mkdir(foldpath)

def ReadTextFile(fp,print_=True,sep="\n"):
   "ReadTextFile(fp,print_=True)" 
   prefix = "" if "." in fp else ".txt"
   if print_:
      print(f"Reading in File: '{fp+prefix}'") 
   with open(fp, 'r', encoding="utf-8") as file:
        return file.read().split(sep) 

def SaveTextFile(fp,data,print_=True,sep="\n"):
    "SaveTextFile(fp,data,print_)"
    prefix = "" if "." in fp else ".txt"   
    if print_:
       print(f"Saving File: '{fp+prefix}'")
    with open(fp+prefix, 'w', encoding="utf-8") as file:
         file.write(sep.join(data)) 
        
def SafeFilepath(filepath):
    for badchar in '<>"|?*':
        if badchar in filepath:
            filepath = filepath.replace(badchar,"")
    return filepath  


def SaveRead_Dict_with_text_df(*args,fp="allhtmlvisibletext",sep="**^**",both=False,to_dic=False):
    #from Scrape_Library import ReadTextFile,SaveTextFile,cumsum
    fp_txt, fp_csv = fp+".txt",  fp+".csv"

    if   len(args)==0:#Read
        
        dfin = pd.read_csv(fp_csv, index_col=0)
        dfin = [n[1] for n in dfin.groupby("Home_")] 
        cutlocs = CumSum([len(n) for n in dfin])
        
        textin = ReadTextFile(fp_txt, sep=sep)
        textin = [textin[s:f] for s,f in zip([0]+cutlocs,cutlocs)]
        if to_dic:
            text_dic_____ = { list(v2["Home_"])[0] :[v1,v2]  for v1,v2 in zip(textin, dfin)}
            return text_dic_____ 
        return textin,dfin
    
    elif len(args)==1: #Save_dic
        text_dic___ = args[0]        
        textout, dfout = [],[]        
        for k,v in text_dic___.items():
            text_,df_ = v
            df_["Home_"] = k
            textout.append(text_)
            dfout.append(df_)
            
    dfout = pd.concat(dfout)
    textout = [tt for t in textout for tt in t]    
    SaveTextFile(fp_txt,textout, sep=sep)
    dfout.to_csv(fp_csv)
    if both:
        return SaveRead_Dict_with_text_df(both=both,to_dic=to_dic)


#%%###############################################################################################################
def CumSum(lis,c=0):
    p=[]
    for ele in lis:
        c+=ele
        p.append(c)
    return p  

def standard_length(obj,length=20,mode="r",char=" ",function=False):
    if function is True:
        return lambda x: standard_length(x,length=length,mode=mode,char=char )
    if mode=="r":
        return str(obj).rjust(length,char)[:length ]
    if mode=="l":
        return str(obj).ljust(length,char)[ length:]       



for n in "1 12 123 1234 12345 123456".split():
   v=standard_length(n,3,char="0")
   print(v)
   
num_sta_len = standard_length(None,5,char="0",function=True)
for n in "1 12 123 1234 12345 123456".split():
   v = num_sta_len(n)
   print(v)


