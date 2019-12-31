# -*- coding: utf-8 -*-  This is a copy and created on the 18/09/2017
"""Created on Mon Dec 12 10:01:29 2016,,,@author: milroa,,,\\NDATA12\milroa$\My Documents\AM.py                    """
# runfile('//NDATA12/milroa$/My Documents/AM.py', wdir='//NDATA12/milroa$/My Documents')
#  import AM.py as AM
#  create a module
#
def listall():
    print('files_in_dir(dir_loc,all_files_dir=[])')
    print('file_info_of_file_paths(all_files_dir)')
    print('listall')
    print('pd')
    
##install   
host = 'fa1rpwapxx272.ons.statistics.gov.uk'
url = 'http://'+host+':8081/artifactory/api/pypi/ons-simple-repo/simple'
pip.main(['install', 'fuzzywuzzy','-i', url,'--trusted-host', host])  
    
    
#def pd():
def pd_help():
    print(
  '''df = pd.read_csv(filename,encoding = "latin-1",header=None) #,index_col=0                        \n'
    'df.to_csv( filename2 , encoding="latin_1")                                                       \n'
     blank_new_df = pd.DataFrame(index=range(4),columns=['A'], dtype='float').fillna(0)
     blank_new_df = pd.DataFrame(index=list('FGHI'),columns=list('ABCDE'), dtype='float').fillna(0)  '''
    '                                                                                 \n'
  '''df["C"] = ""        del df['column']                                             \n
     new_df= pd.DataFrame(index=Index, columns=Columns)                               \n
     new_df=new_df.fillna(0)                                                          \n'''
     

  '''df=series.to_frame('new_col_name').transpose()'  #check if transpose is needed
     df=pd.DataFrame(index=s.index,data=s, columns=['col_name'])#series to datafrane
     df.merge(s.to_frame(), left_index=True, right_index=True)# merge sereis to a dataframe'''
                                                            
    'df.head(5) #df.tail(5) #as well works                                                                     \n'
    'df.sample(10).sort_index()                                                       \n'
    'df.describe() # basic stat summary of the dataframe                              \n'
    'df.shape the  size of dataframe len(df)                                          \n'
    'df.ix[:,3]                                                                       \n'
    'df.rename(columns={old_name: new_name})                                          \n'
    'df2=df.copy                                                                      \n'
    'df.columns=Columns                                                               \n'  
    'index=df.index.tolist()                                                          \n'
    '                                                                                 \n'
    'df.drop(col_name, axis=1, inplace=True)# drop_duplicates                         \n'
    'df=df.dropna(subset=[column_name])                                               \n'
    '                                                                                 \n'   
    'columns_unique=pd.unique(x[column])# from just the first one add to the end[0]   \n'
    'hist=df[column name].value_counts().to_frame().reset_index()    ?                \n'
    '                                                                                 \n'
    'df[column_new]=df[column].map(lambda x: x[:6]# .apply(lambda x:str(x))           \n'
    'df_sort=df.sort_values([column_new])                                             \n'
    '                                                                                 \n'   
    '''df['rowno'] = range(df3.shape[0])                                              \n'''    
    

    df['A'].isin([3, 6])#mask if equal to values
    result = pd.merge(df_org, df_dict, on=['col_key1', 'col_key2'])#merge a df_dictary

######## [] ix iloc loc ,masks ###########################
df=pd.DataFrame(index=list('abcd'),columns=list('ABCDE'))
df['A'] df[['A','B']] #
df[:1]  df['a':'b']   #can't do   df[1]
##### label based loc      ############
df.loc[:,['A','B']]
df.loc['a':'b',['A','B']]
df.loc['a',['A','B']]
##### location based iloc ###########
#df[row/index , column]
df.iloc[[1,2,4],[0,2]]
df.iloc[3:5,0:2]
df.iloc[3]
######## mask using . #####################
df[df.A > 0]
########in df -Numbers =>nan
df[df > 0]
#######  avoid using ix  ##########
###REORDER COLUMNS
#SLICES SELCTION USING FOR LOOPS
          
# i think with dataframes with 500000 cells no colour is shown in dataframe when dispalyed
                                    \n'
    '                                                                                 \n'
    'df[b] = df[a].apply(lambda col: str(col)#do stuff with col here change to string is outputted)                                    \n'
    'ApplyMap: Every element of a df function applied to                                               \n'
         'Map: It iterates over each element of a series allows function to be applied to              \n'
       'Apply: Applies a function along the selected axis of the DataFrame.                            \n'
              'df[[‘column1’,’column2’]].apply(sum) gives the sums                                     \n'
    '          df.groupby(col_name).apply(lambda L: fun(L) )                                           \n'
    '                                                                   is there a split function?     \n'
     df['col_1_str']=df['col_1'].apply(lambda x:str(x))# can do functions  '    
    '                                                                                                  \n'
    '''timefunc=lambda timein:datetime.datetime.strptime(timein[0:9], '%d%b%Y').strftime('%Y%m%d')     \n'''
    'df[col_new]=df[col_new].apply(timefunc)                                          \n'
    '''nowdate = time.strftime("%d%m%y")                                              \n''' 
    '                                                                                 \n'
    'months=pd.unique(data[col_name])[0]#col_name is month                            \n'
    'if   months==201406:        data[col_name]=0                                     \n'
    'elif months!=201406:        data[col_name]=1                                     \n'
    
    df[col_name_2]=df[col_name][[col_name]==pd.unique(data[col_name])[0]]
    '                                                                                 \n'    
    '                                                                                 \n'     
     '''df_unique=df['col_name'].unique()                                             \n'''
     '''df_unique_counts=df['col_name'].value_counts()                                \n'''
    '                                                                                 \n'    
     ''''panel.sum('major_axis')'''
    'max, min ,first,mean'
    'DataFrame.sort_index()'
 
  #group data frame by row find the column in group with maximum value make sure theres one and reset index
  """dfg=df.groupby("row").apply(lambda x:x[x["value"]==x["value"].max()].iloc[0]).reset_index(drop=True)"""
  ##filteration and trandformation what to dy

    df_info = df_1.groupby('product_name').agg({ 
'cluster_pred r=0.1, m=5':{'no_clusters':pd.Series.nunique,'cluster_min':'min','cluster_max':'max','item_count':'count'},
'confidence':{'confidence':'first'},
'cluster_par_1':{'price_min':'min','price_mean':'mean','price_std':'std'}  })
df_info.columns = df_info.columns.droplevel(0)
df_info=df_info.reset_index()
        
    .sort_index().to_csv(csv_out , encoding="latin_1",index = False) 
    # Grab DataFrame rows where column doesn't have certain values
valuelist = ['value1', 'value2', 'value3']
df = df[~df.column.isin(value_list)]


norow=len(df.index)#quick way to find the number of rows
                   #drop any na lines'
    'df[(df != 0).all(1)]         # no zeroes'
    'empty, any(), all(), and bool() to provide a way to summarize a boolean result.'
    '(df > 0).all()')
    
    

""" #            Groupbys                                      #
df_out=df.groupby('col_1').sum()
#df[df['col_1']==df['col_1'][0]] # equivelent to one frame
"#########df groupby and for loop equivlent############"
df_out=df.groupby('col_1').sum()#df[df['item_no']==df['item_no'][0]] # equivelent to one frame
dfg=list(df.groupby('col_1'))#store df groupby
"##  equivelent for loop  ##"
df_out=pd.DataFrame(columns=df.columns)
for col in df['item_no'].unique():
  df_out=df_out.append(df[df['item_no']==col].sum().to_frame(col).transpose())

### alterantive method of groupby
a=[]
 a.append(x.groupby('ons_item_no').apply(lambda L: geksj(L, 'idvar', 'ons_item_no','month','item_price_num')))
Results = np.concatenate(a, axis=0)
  
  groupby and reset it
  
  ##also if you imagine fataframe with all teams in the premerleague with ages
  ## add extra column being the different between players age and the mean for there team
  df3.groupby(['X']).get_group('A')
  
  df["B distance mean A"]=df.groupby("A",group_keys=False).apply(lambda a:0*a["B"]+a["B"].mean())
  
  apply diciortay to a dataframe
  dic=dict(zip(["a","b","c"],[1, 2, 3]))
  df[col1a]=df[col1]
  df.replace({"col1a": dic})
"######################################################""""
         # in format_files import fucntions from scripts is explored"
 
        #other fills

#groupby results and project them back onto the dataframe
df=df.join(df.groupby('row_to_groupby')['row_apply'].sum(), on='row_to_groupby', rsuffix='_sum')
df['maxval'] = df.groupby(by=['idn']).transform('max')

df.groupby(key_columns).size()  

    # maybe one to check all data type and right one, numpy extraction
    # join explaiones
    type() function
   
        'for row in df.itertuples():  
        
#if data frame contains multiple datatypes print them
def find_rows_multi_datatypes(df):    
    for n in df:
        if len(df[n].apply(type).value_counts())>1:
           print(df[n].apply(type).value_counts())
    return()
#find_rows_multi_datatypes(df_1) 










    
    
###########################
#files=files_in_dir("K:\WGSN data\Subsetted Data\monthly")
def files_in_dir(dir_loc,all_files_dir=[]):
    import os
    Folder_Info=list(os.walk(dir_loc))
    Folder_Info=Folder_Info[0]
    if len(Folder_Info[2])>0:all_files_dir=all_files_dir+[dir_loc+'/' + s for s in Folder_Info[2]]
    folders=Folder_Info[1]
    if '$RECYCLE.BIN' in folders: folders.remove('$RECYCLE.BIN')
    for folder in folders:
              all_files_dir=files_in_dir(dir_loc+'/'+folder,all_files_dir)   
    return(all_files_dir) 
###########################   
    

%%%yyddmm
def file_info_of_file_paths(all_files_dir):
    import os
    fileinfo=[['File_Path'],['File Name'],['Folder_Loc'],['Filetype'],['File_Save%'],['Date_Modifed'],['Date_Created'],['Size_MB'],['No'],['Date_Scanned'],['file_level_no'],['hms_Modifed'],['hms_Created']]
    fileinfo[0]=fileinfo[0]+all_files_dir
    flag_print=0
    import time,datetime
    nowdate = time.strftime("%d%m%y")
    if len(all_files_dir)>4000:
         print('There are '+str(len(all_files_dir))+' in this dir')
         flag_print=1
         incr=len(all_files_dir)//100
         incr=max(incr,4000)

    for lineno2, file in enumerate(all_files_dir):
        if (flag_print)and((lineno2 % incr)==0):print('   '+str(lineno2)+' out of '+str(len(all_files_dir))+' done')
        #lineno =lineno2+1;#file=fileinfo[lineno][0]
        statinfo = os.stat(file)
#        fileinfo[lineno][1]=datetime.fromtimestamp(path.getmtime(file))
        last_bracket=file.rfind('/')
        file_save_name=file[3:].replace('/', '%').replace("\\", '%')

        temp=datetime.datetime.strptime(time.ctime(statinfo.st_ctime), "%a %b %d %H:%M:%S %Y")
        ctime2=int(temp.strftime('%y%m%d00%H%M%S'))
        ctime3=int(temp.strftime('%y%m%d'))
        temp=datetime.datetime.strptime(time.ctime(statinfo.st_mtime), "%a %b %d %H:%M:%S %Y")
        mtime2=int(temp.strftime('%y%m%d00%H%M%S'))
        mtime3=int(temp.strftime('%y%m%d'))
              
        fileinfo[1 ].append(file[1+last_bracket:])
        fileinfo[2 ].append(file[:last_bracket])
        fileinfo[3 ].append(file[file.rfind('.'):])
        fileinfo[4 ].append(file_save_name)
        fileinfo[5 ].append(mtime3)#fileinfo[5 ].append(statinfo.st_mtime)#time of most recent modifcation access.
        fileinfo[6 ].append(ctime3)#fileinfo[6 ].append(statinfo.st_ctime)#was atime now creation date #time of most recent access.
        fileinfo[7 ].append(float(statinfo.st_size)/1048576)# MB
        fileinfo[8 ].append(lineno2+1)
        fileinfo[9 ].append(nowdate)
        fileinfo[10].append(len(file[3:]) - len(file[3:].replace('/', '').replace("\\", '')))#need directory to be added on
        fileinfo[11 ].append(mtime2)#fileinfo[5 ].append(statinfo.st_mtime)#time of most recent modifcation access.
        fileinfo[12 ].append(ctime2)#fileinfo[6 ].append(statinfo.st_ctime)#was atime now creation date #time of most recent access.
        #add these ones in future
        # date no so can order by date,   size order,     name order
        #modification date in a cross-platform way is easy - just call os.path.getmtime(path)
        #Windows, a file's ctime creation date. Python through os.path.getctime() or        
    return(fileinfo)  
    
    

def list2df(Folder_1_fileinfo):
    import pandas as pd
    Folder_1_fileinfo_df=pd.DataFrame(Folder_1_fileinfo).transpose()
    Folder_1_fileinfo_df.columns=Folder_1_fileinfo_df.ix[0,:]
    Folder_1_fileinfo_df=Folder_1_fileinfo_df.drop(0)
    Folder_1_fileinfo_df=Folder_1_fileinfo_df.reset_index()
    Folder_1_fileinfo_df=Folder_1_fileinfo_df.drop('index',axis=1) 
    return(Folder_1_fileinfo_df)


def file_info_dir_df(Folder_1_dir_org):
    import os
    Folder_1_all_files_list=files_in_dir(Folder_1_dir_org)        
    Folder_1_fileinfo=file_info_of_file_paths(Folder_1_all_files_list)
    ## create a data frame of the files in folder
    Folder_1_fileinfo_df=list2df(Folder_1_fileinfo)
    return(Folder_1_fileinfo_df)
    
#import os
#Folder_1_dir_org=r'B:\Pricestats' 
#Folder_1_all_files_list=files_in_dir(Folder_1_dir_org)        
#Folder_1_fileinfo=file_info_of_file_paths(Folder_1_all_files_list)
### create a data frame of the files in folder
#Folder_1_fileinfo_df=list2df(Folder_1_fileinfo)
   
#import datetime
#import pathlib 
################## CODE THAT ACTUALLY RUNS to create folder info ##########
###########################################################################
file_info_1=file_info_dir_df(r'B:\Pricestats')




###maybe change this too a function
## create a file saver

## ability to search files easy gui select folder to save to   
    
    ## save files to a text  
    ## put 2 lists in seperates them into:
    ## same
    ## new_files
    ## deleted_files
    ## changed_files
    ## moved_files,(maybe a bit complicated as has 2 directories)
    
    
    ## list for folders as well as 2 lists moved from t
    ## are dates modified dates or last saved
    ## if same name but different folder check dates size and some data to see if same fill
#  create a object how to do   
    
################################################### 
#A "progress bar" that looks like:
#|#############################---------------------|
#59 percent done
#
#Code:
#class ProgressBar():
#    def __init__(self, width=50):
#        self.pointer = 0
#        self.width = width
#
#    def __call__(self,x):
#         # x in percent
#         self.pointer = int(self.width*(x/100.0))
#         return "|" + "#"*self.pointer + "-"*(self.width-self.pointer)+\
#                "|\n %d percent done" % int(x) 
#
#Test function (for windows system, change "clear" into "CLS"):
#if __name__ == '__main__':
#    import time, os
#    pb = ProgressBar()
#    for i in range(101):
#        os.system('clear')
#        print pb(i)
#        time.sleep(0.1)
##################################################################










      


##########        open folder or file in dialog button or save      #############################
import Tkinter, tkFileDialog
dialog_mode=  ["None","folder","file","save"][1]

if   dialog_mode=="folder"##======== Select a directory:
        root = Tkinter.Tk()
        dirname = tkFileDialog.askdirectory(parent=root,initialdir="/",title='Please select a directory')
        if len(dirname ) > 0:    print("You chose %s" % dirname ):
            
elif   dialog_mode=="file"    # ======== Select a file for opening:                 
        root = Tkinter.Tk()
        file = tkFileDialog.askopenfile(parent=root,mode='rb',title='Choose a file')
        if file != None:
            data = file.read()
            file.close()
            print "I got %d bytes from this file." % len(data) 
            
elif   dialog_mode=="save"          # ======== "Save as" dialog:
        myFormats = [
            ('Windows Bitmap','*.bmp'),('Portable Network Graphics','*.png'),
            ('JPEG / JFIF','*.jpg'),    ('CompuServer GIF','*.gif'), ]        
        root = Tkinter.Tk()
        fileName = tkFileDialog.asksaveasfilename(parent=root,filetypes=myFormats ,title="Save the image as...")
        if len(fileName ) > 0:    print("Now saving under %s" % nomFichier)
############################################################################################        
        













     
    
    
    
###################### save_open_workspace ##########################################  
choice=['open','save','skip'][2] 
file_path=""  
#####################################################################################
#####################        SHELVE          ########################################
if choice in ['open','save']:# to shelve 
    print("Running Shelve, "+file_path)
    import shelve
######################    SAVE VARIABLES     #######################################
    if load_run_save=="save":# SAVE 
        print("SAVING")   
        variables_ignore=['np','pd','os','plt','shelve','geksj','Out','In','quit','exit','get_ipython','BeautifulSoup','DesiredCapabilities','date','webdriver','timedelta','requests',
                          'key','handler','driver','file','elm','element','ss',"time",'variables','my_shelf',"supermarketz","variables_ignore"]
        with shelve.open(file_path, "c") as shelf:                 
            keys=    [key for key in dir() if key[0]!='_' and key not in variables_ignore] 
            for key in keys:
                     print(key)
                     if str(type(globals()[n]))[7:-1] not in ["module"]#new
                         shelf[key]= globals()[key] 
#######################     OPEN VARIABLES    #######################################
    if load_run_save=="open":#   OPEN
        print("LOADING")      
        with shelve.open(file_path, "r") as shelf:                  
            for key in shelf.keys():
                     globals()[key] = shelf [key]  
#####################################################################################  


####*********************     Alternate   ***********************************####

###################### save_open_workspace ##########################################  
choice=['open','save','skip'][2] 
file_path=""  
#####################################################################################
#####################        SHELVE          ########################################
if choice in ['open','save']:# to shelve 
    print("Running Shelve, "+file_path)
    print(file_path,       "SAVING" if choice=="save" else "LOADING" if choice=="open")
    import shelve
    with shelve.open(file_path, "c" if choice=="save" else "r"       if choice=="open") as shelf: 
######################    SAVE VARIABLES     #######################################
    if load_run_save=="save":# SAVE 
            variables_ignore=['np','pd','os','plt','shelve','geksj','Out','In','quit','exit','get_ipython',
                              'key','ss','variables','my_shelf',"supermarketz","variables_ignore"]                
            keys=    [key for key in dir() if key[0]!='_' and key not in variables_ignore] 
            for key in keys:
                 shelf [key] = globals()[key] 
#######################     OPEN VARIABLES    #######################################
    if load_run_save=="open":#   OPEN
            for key in shelf.keys():
                 globals()[key] = shelf [key]  
#####################################################################################  

 






        
        
map with join
        df_1['monthday']=df_1['price_date'].apply(lambda x:int(datetime.strptime(x[:9],'%d%b%Y').strftime('%y%m%d')))





















    
    def exist(m):# m is the string of the variable
    out_exist=1
    try:
        p=eval(m)
    except:
        out_exist=0
    return (out_exist)
    
    
    
    
    ############################ Neural Network  ################
def signmoid(IBN):
    #XMATRIX
    x=SUM(x,1)+B1#SUM NE COLUMN
    out=1/(1/e**(-x))
    return(out)    
    
I=[1,2]#INPUT
S=[5,6]#SOLUTION
#INTIAL CONDITIONS
W1=[[1 3][4 5]]
B1=[-2,2.1]
O1= I*W1
OF1=signmoid(O1,B1)# THE FUNCTION SIGMOND APPLIES

W2=[[1 3][4 5]]
B2=[-2,2.1]
O2= OF1*W2   
OF2=signmoid(O2,B2) 
    
ET=((S-OF2)**2)/2
   
import numpy as np
    
def signmoid(IBN):
    #XMATRIX
    x=SUM(x,1)+B1#SUM NE COLUMN
    out=1/(1/e**(-x))
    return(out)   

Input=   np.random.rand(elements,rounds)    
Solution=np.random.rand(elements,rounds) 

for roundno in rounds:
size_of_layers=[3,4,5,5,2]    
max_layer_W=max(size_of_layers,elements)
max_layer=max(size_of_layers)
no_layers=len(size_of_layers)

W=np.random.rand(max_layer,max_layer,no_layers)#weights
B=np.random.rand(max_layer,no_layers)
Out=

for layer_no,layer_count enumerate in no_layers:
    
Wtemp=W(:size_of_layers(lauercount),:size_of_layers(lauercount)+1,layer_count)






###text to speach,speach to text,, ocr computer machine



##############################################
## some intresting code which may work or not search all files
if 0:
    import os,re, win32api
    
    def find_file(root_folder, rex):
        for root,dirs,files in os.walk(root_folder):
            for f in files:
                result = rex.search(f)
                    if result:
                        print os.path.join(root,f)
                        break                         #if you want to find only one
    def find_file_in_all_drives(file_name):
        #create a regular expression for the file
        rex = re.compile(file_name)
        for drive in win32api.GetLogicalDriveStrings().split('\000')[:-1]:
            find_file( drive, rex )
    
    
    find_file_in_all_drivers( 'myfile\.doc' )
############################list all avalible drives#############################################
import ctypes, itertools, os, string, platform

def get_available_drives():
    if 'Windows' not in platform.system():
        return []
    drive_bitmask = ctypes.cdll.kernel32.GetLogicalDrives()
    return list(itertools.compress(string.ascii_uppercase,
               map(lambda x:ord(x) - ord('0'), bin(drive_bitmask)[:1:-1])))
#########################################################################    
        
##  does a file and folder exist
if 0:
  import os
  print(os.path.isdir("/home/el"))# does folder exist?
  print(os.path.exists("/home/el/myfile.txt"))#does file exist?
  os.makedirs()##make dir
  os.listdir()# list files and folders in a diretory
  import re
  re.findall(r'\d+', 'hello 42 I\'m a 32 string 30')#find numbers in string
  >>>['42', '32', '30']
  def compress(data, selectors):
    # compress('ABCDEF', [1,0,1,0,1,1]) --> A C E F
    return (d for d, s in izip(data, selectors) if s)
    
    [a*b for a,b in zip(lista,listb)]##multipley 2 lists
    s[::-1] to flip a string
####################################
    #huffman encoding copressor
##  





######################  BACKUP   BACKUP  ################################
## 1) what is todays date
## 2)does todays folder exist folder exist if not make it
## 3)when was last scanned?
## 4)read in allfilesfolder from last time
## 5)find all files folder today
## 6)find difference between the two
## 7)transfer the files
## 8)save allfiles_list


## 1) what is todays date
if 1=1:
        import time
        nowdate = time.strftime("%d%m%y")
## 2)does todays folder exist folder exist and if not make one, find usbdirve
if 2==2:
    import os
    list_of_drives=get_available_drives()
    if type(list_of_drives) is list:list_of_drives.remove("C")
    if len(list_of_drives)==1:
        usb_dir=list_of_drives[1]+":/"+nowdate
        new_folder=usb_dir+":/"+nowdate
        if not os.path.isdir(new_folder):
            os.makedirs(new_folder) 
    elif:print("there is not one usb drive plugged in(mightbe 0 or 2 and above)")
## 3)when was last scanned?
if 3==3:
        import re,os
        list_filders=os.listdir(usb_dir)
        
        #############  removable test  #####################
        import re,os
        usb_dir="K:\WGSN data\Subsetted Data\monthly"
        list_filders_org=os.listdir(usb_dir)
        ##add some in the list_filders
        list_filders=list_filders_org
        ###########################
        
        
        
        
        
        #remove filenames with less than 6 digits
        #longer than 5 chars(1),first 6 chars ahve number(2),number 6 digits long(3),filder 6 or 10 digits longs(4)
        list_filders2=[x for x in list_filders  if len(x)>5]#(1)
        list_filders2=[x for x in list_filders2 if len(re.findall(r'\d+',x[:6]))>0]#(2)
        list_filders2=[x for x in list_filders2 if len(re.findall(r'\d+',x[:6])[0])==6]#(3)
        list_filders2=[x for x in list_filders2 if ((len(x)==6)or(len(x)==10))]#(4)
        ### so only things like  010112  or 0102014.txt  passes
        list_filders_no=[re.findall(r'\d+',x[:6])[0] for x in list_filders2]
        #oder needs to be changed as years months and days
        #  ''.join(a) was added as roginal split##orginal effort  list_filders_no2=[[x[xx] for xx in [4,5,2,3,0,1]] for x in list_filders_no]
        list_filders_no2=[''.join([x[xx] for xx in [4,5,2,3,0,1]]) for x in list_filders_no]
        ##create distary
        ## there should be no files with same number allacation
        import itertools
        file_dict={l1:l2 for l1,l2 in itertools.zip_longest(list_filders_no2,list_filders2,fillvalue=None)}
        
        list_filders_no2.sort(key=int)
        lastet_filder=file_dict[list_filders_no2[-2]]#apart from the one weve added today the lastest one
        
        if not(int(''.join([nowdate[xx] for xx in [4,5,2,3,0,1]]))==int(list_filders_no2[-1])):
          print("The most recent file or folder in the usb is not today")
          print("today(ddmmyy) is "+nowdate + " and most recent file is " + file_dict[list_filders_no2[-1]])

## 4)read in allfilesfolder from last time
if 4==4:
    if   len(lastet_filder)==6:#folder
       if os.path.isdir(usb_dir+lastet_filder):
         path_last_files_list=usb_dir+lastet_filder+"//"+lastet_filder+"csv"  
       else:print("The Folder is not found "+usb_dir+lastet_filder)   
    elif len(lastet_filder)==10:#file
      path_last_files_list=usb_dir+lastet_filder
    ########################################################
    if  (len(lastet_filder)==6)or(len(lastet_filder)==10):  
          if os.path.exists(path_last_files_list):
              if path_last_files_list[-4::]==".csv":
                  file_list_old=load_csv(path_last_files_list)
              else:print("not a csv file but a "+ path_last_files_list[-4::])
          else print("cant find the file") 
    else:print("Error latest filder doenst have 6 or 10 digits in its name")

 
 ## add some break print user_input=input() in
 ## some warning about space 


## 5)find all files folder today
file_info_1=file_info_dir_df(r'B:\Pricestats')

## 6)find difference between the two

if 6==6:   
        file_info_1=file_info_dir_df(r'K:\WGSN data\Subsetted Data\monthly')
        effort=2#[1 2 ]
        load_save="load_both"#["save","load","none","load_both"]
        
        import pandas as pd
        if    effort==1:csv_out=r'K:\WGSN data\Subsetted Data\monthly\fileinfo_1.csv'
        elif  effort==2:csv_out=r'K:\WGSN data\Subsetted Data\monthly\fileinfo_2.csv'
        
        if   load_save=="save":
                 file_info_1.to_csv(csv_out , encoding="latin_1",index = False) 
        elif load_save=="load":   
            if load_save=="save":
                 file_info_1=pd.read_csv(csv_out , encoding="latin_1",index = False) 
        elif load_save=="load_both":   
                 file_info_1=pd.read_csv(r'K:\WGSN data\Subsetted Data\monthly\fileinfo_1.csv' , encoding="latin_1")
                 file_info_2=pd.read_csv(r'K:\WGSN data\Subsetted Data\monthly\fileinfo_2.csv' , encoding="latin_1")
                 #file_info_1["Data_Scanned"]=[]
                 file_info_1["Date Scanned"]=170201
        del(effort,load_save,csv_out)
        
         ## changes      ##Date_Scanned    ## Date_Created  Data_Modified
         ##folder one showing number of subfolder folder sub files files memory
         
        file_info_all=file_info_1.append(file_info_2)
        #file_info_c=file_info_c.sort_values(["File_Save%,Date_Scanned"])
        file_info_all=file_info_all.sort_values(["File_Save%","Date Scanned"])
        
        old_date,new_date=file_info_all["Date Scanned"].min(),file_info_all["Date Scanned"].max() 
            
        file_info_all["File_State"]=file_info_all.duplicated(["File_Save%","Size_MB","Data_Modifed","Data_Created"],False)   
        file_info_all["File_State"][file_info_all["File_State"]==1]="Same"
        file_info_all["File_State"][file_info_all["File_State"]==0]="Different"
        
        ## add updated new
        mask_updated=(file_info_all.duplicated(["File_Save%","Data_Created"],False))*(file_info_all["File_State"]=="Different")
        file_info_all["File_State"][mask_updated]="Updated"
        mask_new=(file_info_all["File_State"]=="Different")*(file_info_all["Date Scanned"]==new_date)
        file_info_all["File_State"][mask_new]="new" 
        del(mask_updated,mask_new)
        save_files=file_info_all[(file_info_all["Date Scanned"]==new_date)*(file_info_all["File_State"].isin(["new","updated"]))]
         #   file_info_all[(file_info_all["File_State"][mask_new]="new")*file_info_all["File_State"].isin(["new","updated"])] 



    
## 7)transfer the files make sure today is empty of that its folder has only one file in it
if 7==7:    
    if not os.path.isdir(new_folder):"print the folder disapered",break
    save_files
    ##other checks
    
    
## 8)save allfiles_list   
    
    file_info_1
    file_info_1.to_csv(csv_out , encoding="latin_1",index = False) 
    
    
                    #  dfALL=(df if "dfALL" not in globals() else dfALL.append(df, ignore_index=True) )

    
  ## delete stuff in worspace  
    names=dir()
    names= [name for name in names if name not in ['In' ,'Out', 'quit', 'get_ipython', 'exit'] and not name[0]=="_"]
#deltable names
    
    
    
    
    
    
#split a long string by newline
import re
listofstrings=re.split(r'\n+', longstringwithspaces)
#split by new tab
 for lineno,line in enumerate(listofstrings): listofstrings[lineno]=re.split(r'\t+', line)


# create a n * m list
    table=[[[]]*n] * m ###
#table_org=tabel.copy() then operate on this
   
##going throught a table,table list in a list  
for line_no,line in enumerate(table):#table[:-1],
#for line_no in range(len(table)-1):
    if line_no>0:
        linep=table[line_no-1].copy()#previus line
        linec=table[line_no  ].copy()#current line
        #linen=table[line_no+1].copy()
        
    #### some operation on linec ###
        
        table[line_no]=linec
del line_no,line,linep, linec
    
# df= pd.DataFrame(data=table) 

#linec=[linep[i] if linec[i]==[] else linec[i] for i in range(len(linec))]  if else  for    
#line=[line_prev[nn] if n==[] else n for nn,n in enumerate(line)]      if else enum for
#[ x if x%2 else x*100 for x in range(1, 10) ]     if else for
#[ EXP for x in seq if COND ]                                 if for
#[ EXP for x in seq ]                                          for

    
    
    
##  search for files in a list , how to search prefix suffix contains lower letters
Option_filepath_search=10 
if Option_filepath_search==1:   
    files = [f for f in os.listdir('.') if re.match(r'[0-9]+.*\.jpg', f)]

if Option_filepath_search==2: 
     glob search for patterens in a file path
       #>>> glob.glob('145592*.jpg')

if Option_filepath_search==3:     
        import os, fnmatch
     fnmatch.filter(os.listdir('.'), '*.py')
    
if Option_filepath_search==4: 
    
    import os, fnmatch
    def find_files(directory, pattern):
        for root, dirs, files in os.walk(directory):
            for basename in files:
                if fnmatch.fnmatch(basename, pattern):
                    filename = os.path.join(root, basename)
                    yield filename
    
    for filename in find_files('src', '*.c'):
        print('Found C source:', filename )
        
    
    
    
    
   ### meshgrid
import numpy as np
nx, ny, nz = (10,12,14)
x, y, z =(list(range(n)) for n in [nx, ny, nz])
xv, yv, zv = np.meshgrid(x, y, z)

## find morphing in python
    
###################  introduce new hidden values to the workspace ######################################
_hiddencol_supermarket=["tesco","sainsbury","waitros"]
for _hidden_supermarket in _hiddencol_supermarket:
    #run_script()
#_hidden_supermarket,_hidden_i,_hidden_number="tesco","tyty",23231

## code injection
for hidden_name in [n for n in dir() if len(n)>8 and n[:8]== '_hidden_']:
    globals()[hidden_name[8:]]=globals()[hidden_name]
    del globals()[hidden_name],hidden_name
    
########################### clean workspace midthrough #################################
var_names_del=[n for n in dir() if n not in ['In','Out','exit','get_ipython','quit','var_names_del'] and n[0]!="_" ]
for name in var_names_del+["var_names_del"]: del globals()[name],name
    
##############################  time code ##############################################
import time
start_time = time.time()
## simple code to time ##
print("--- %s seconds ---" % (time.time() - start_time))  

###########################  matrix(or array mod) of zeroes  ###########################
#matrix_zeroes_10_20=[[0]*10]*20 
# these are not copies
matrix_zeroes_10_20 = [[0] * 10 for i in range(20)]
########################  turn a number 342 to "00342"  ################################
number,digits=345,5
numberstr="0"*(digits-len(str(number)))+str(number)
#  out-> "00345"  ##maybe add rounding

######################  User input  ####################################################
person = input('Enter your name: ')
print('Hello, your name is', person)
########################################################################################



## broadcasting
    
    
#    Weekday Time         
#Monday  Morning  1  3
#        Noon     2  1
#Tursday Morning  3  3
#        Evening  1  2
#
#In [40]: pd.DataFrame(df.to_records())
#Out[40]: 
#   Weekday     Time  0  1
#0   Monday  Morning  1  3
#1   Monday     Noon  2  1
#2  Tursday  Morning  3  3
#3  Tursday  Evening  1  2
    
    
#########################################################    
# files in dir()
  ## always appeared
    '_i','_ii','_iii'
    '_ih','_oh','_sh','_dh'
    '__builtin__','__builtins__','__name__'
    
    ['In','Out','exit','get_ipython','quit']
  # after a restart 
     '_', '__', '___',
     '__doc__',
     '__loader__
     '__package__',
     '__spec__',
  ## unque ones that appear with in3 in1 and in79
    ['_i1','_i3','_i79']
########################################################  
## CODE 2 

# 0. check the free space of the drive
# 1.check if folder exists, and if it does are files in it, if not create it
if 1==1:
    import os
    folder_path = "H://My Documents//tyt"
    print(folder_path)
    try:
        os.stat(folder_path)
        files_in_folder=len(os.listdir(folder_path))
        print("Folder exists")
        if len(files_in_folder)==0:
            print("Folder is empty though")
        else:
            print("There are "+str(files_in_folder)+" files in the Folder")
    except:
        os.mkdir(folder_path)       
        print("Created new folder")
        files_in_folder=-1
##########################################################################    
cell2run=[1,2,3,4,5,6,7,8,9,10]      
#%%cell 1
if      2 in cell2run:
    a=67
#%%cell 2
if      2 in cell2run:

#%%cell 3
if      3  in cell2run:
elif   -3  in cell2run:
elif   "3" in cell2run:   
#%%






# user input
# create dates folder "260517bpc"?
#two folders in it
# n> new
# u> updated
# as well as a csv with the differences between the updates
# csv all same files,
# updated, new, moved, deleted  
    
    
    

    
    
    
    
    
  



