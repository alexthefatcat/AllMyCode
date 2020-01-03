# -*- coding: utf-8 -*-"""reated on Wed Jul 25 09:48:47 2018@author: milroa1"""
######################################################################
       
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
    
    
    
    then 
    
    copying
    
    
    
    
    
    
    
    
    
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
from copy import copy        
import os
import pandas as pd  
import hashlib
import shutil        
  
         


class FileInfo:
    """ This is a class that allows you to find info about files in a location
    example :
        dirloc1      = FileInfo("C:\\Users\\milroa1\\Downloads")
        file_info_df = dirloc1.get_file_info_for_each_of_the_files()
        #file_info_df will contain all the info about the files    """ 
#    newid = next(itertools.count())#itertools.count().next

    _ids = count(0)


    
    def __init__(self, directory,levels=["all","single"][0], complete=False): 
#        self.unique_id = FileInfo.newid() 
        self.Flag_run_in_old_mode_ = False
        self.Flag_new_df_exist     = False
        self.Flag_df_exist         = False
        self._id = next(self._ids)   
        
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
   
    def add_hash_column_on_df(self,df=None):
        if df is None:
            self.df['Hashed'] = self.hash_files(list(self.df['Filepath']))
            self.hashed=True
        else:
            df['Hashed'] = self.hash_files(list(df['Filepath']))
            return df
        
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

    def _remove_non_identical_files_from_df(files__df,hashed=True):
        files_identical_df = files__df.groupby("Size_MB").filter(lambda x: len(x) > 1)
        files_identical_df = files_identical_df.sort_values("Size_MB")
        # remove some so less hashing
        groupbys = ["Size_MB", "Filetype"]
        if hashed:
           files_identical_df = dirloc4.add_hash_column_on_df(files_identical_df)
           groupbys = ["Size_MB", "Filetype","Hashed"]
        files_identical_df = files_identical_df.groupby(groupbys).filter(lambda x: len(x) > 1)
        files_identical_df["grn"]  = files_identical_df.groupby(groupbys).ngroup()       
        return files_identical_df
    
    def _is_folder_empty(fp):
        if os.path.isdir(fp):
          if len(os.listdir(fp))==0:
              return True
        return False
    def _get_dir_folder_filename_ext(fp):
        fpsplit = fp.split("\\")
        folddir = "\\".join(fpsplit[:-2])
        folder,filename= fpsplit[-2],fpsplit[-1]
        ext  =  ""
        if "." in filename:
           filename,ext = FileInfo._split_and_rejoin(filename,".")
           ext = "."+ext
        return folddir,folder,filename,ext

    def _split_and_rejoin(string,char="\\",ind=-1):
        string_split  = string.split(char)
        part_1,part_2 = string_split[:ind],string_split[ind:]
        part_1,part_2 = char.join(part_1), char.join(part_2)   
        return part_1,part_2  
       
    def _find_similar_new_filepath(filepath, mx=300):
        filepath_l, filename = FileInfo._split_and_rejoin(filepath)
        filetype  =  ""
        if "." in filename:
           filename,filetype = FileInfo._split_and_rejoin(filename,".")
           filetype = "."+filetype
        filepath_l2 = filepath_l+"\\"+filename
        for n in range(mx):
            nostr = f"({n})"
            if n == 0:
               nostr = ""
            filepath_temp = filepath_l2 + nostr + filetype
            if not os.path.exists(filepath_temp):
               return filepath_temp
           
    def _make_dir_only_if_not_exist(folderpath):
       if not os.path.exists(folderpath):
          os.makedirs(folderpath)


      
    def compare(self,self2):        
        grouped         =["Size_MB", "Filetype"]
        column_different=["Hashed"]
 
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





######################################################################
if __name__ == "__main__":  
######################################################################            
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
    
        actuallymovethem=True
        #dirloc4      = FileInfo(r"D:\New2_4")  #"hashed"
        filepath     = r"C:\Users\Alex\Downloads"
        dirloc4      = FileInfo(filepath,levels="single")
        files4       = dirloc4["files"]
        files4_df    = dirloc4["df"]      


        files4_df2 = FileInfo._remove_non_identical_files_from_df(files4_df)
        
        
        if actuallymovethem:
            filepath_duplicates = filepath + "_Duplicates"
            if FileInfo._is_folder_empty(filepath_duplicates):
               folder_to = filepath_duplicates
            else:
                folder_to = FileInfo._find_similar_new_filepath(filepath_duplicates)
                FileInfo._make_dir_only_if_not_exist(folder_to)
            
            file_from_to_list = [ [fp_from,filepath_duplicates+"\\"+ fp_from.split("\\")[-1] ]  for fp_from in files4_df2["Filepath"]]
            file_from_to_list = file_from_to_list[6:]
            dirloc4.copy_or_move_list_of_tupples_from_to__folder_already_exist("move", file_from_to_list, settings="stop")
            
    
            

        
        assert False, "Not sure about the rest of it"
        #dirloc4.add_hash_column_on_df()
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

    assert False,"Yeah stop here work on the rest later"

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
    
   
    
    
    
     

    
    
    
    
    
