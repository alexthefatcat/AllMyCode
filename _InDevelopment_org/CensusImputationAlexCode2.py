# -*- coding: utf-8 -*-
"""Created on Tue Mar  5 11:09:27 2019@author: milroa1"""
import pandas as pd
import numpy as np
import keras
import keras.backend as K
from   keras.models import  Model,Input#,Sequential
from   keras.layers import  Dense#, InputLayer #Input
# Why does it ask to save ?

Config = {"INFO"                    : {}    ,
          "Show_Info"               : ["Latents","Keras_Code","Neural_Net_Summary","CorrelationMatrix"], # ["Latents","Keras_Code","Neural_Net_Summary","CorrelationMatrix"]
          "Epochs"                  : 3    ,
          "batch_size"              : 50    ,
          "remove_fraction"         : 0.4,
          "encoder_dim"             : [87, 100, 50, 20, 12],#[87, 40, 20,  7],#[87, 58, 44, 17, 7] # org
          "train_mode"              : "keras_mse" ,#["keras_mse","keras_mae","custome_mse","Normalized_Impute_Loss","Impute_Loss"]
          "train_test_split_ratio"  : 0.5,        
          "resample"                : False ,
          "load_save_skip"          : "skip"} #load_save_skip


"""
####################################################################################################

                                Quick Overview

####################################################################################################
Functions
    Create_Keras_Net
    NormalizeLatentLoss
    Autoencoder_Predict_2_DataFrame
    DataFrame_Value_Counts
    add_one_hot_coded_and_NA_columns_to_df_and_remove_old

# Read in data

produce_losses_and_block_mask_from_NA_dict_and_df
Imputation_Loss_Function_Generator

# Construct The Neural Net

# Train the Net

# Test the Net

# Show Latent Info

# Predict Missing Columns

####################################################################################################
"""


     

import sys,ctypes,winsound 



class ProgressBar:
    """
    look at replacing the below with this
    
        Some Examples also maybe add percentage directly as input and clean(which remove it after completion)
        from time import sleep
        import sys
        do_stuff = lambda :sleep(1)

    
    for no,width in [(5,60),(3,120)]:#[(30,60),(80,60),(5,60),(105,60),(30,120)]:
        General.PrintHeader(f"Parms no:{no}, width:{width}")
        for i in range(no):
            ProgressBar.print(i,no,width)
            do_stuff()
        ProgressBar.print(no,no,width)
    
    for no,width in [(5,60),(3,120)]:#[(30,60),(80,60),(5,60),(105,60),(30,120)]:
        General.PrintHeader(f"Parms no:{no}, width:{width}")
        PB = ProgressBar(no,width)
        for i in range(no):
            do_stuff()
            PB()                                                                      """
    
    def __init__(self,no,width,msg=""):
        self.i     = 0
        self.no    = no
        self.width = width
        ProgressBar.print(self.i,self.no,self.width,msg)
    def update(self,msg=None):
        self.i += 1
        ProgressBar.print(self.i,self.no,self.width,msg)
    def __call__(self,msg=""):
        self.update(msg)
    def print(i,no,width,msg=""): 
        _print = f"\r[{(('='*int(i*(width/no)))+'>').ljust(width)[:width]}] {int((100/no)*i): <3}%"   
        if i==no:
            msg = " 'Done...'\n"
        sys.stdout.write(_print+msg)    
        sys.stdout.flush()        




class General:
    """
    These Functions are not crucical and just for displaying info
    """
    class PrintBar():
        from time import sleep
        import sys
        def __init__(self,iters,size=None,final_msg=None,prefix=None):
            self.round=0
            self.bar_charcters="=>."       
            
            self.step=1        
            self.rounds=iters
            self.nstep=iters
            if prefix is None:
               self.prefix = "" 
            else:
               self.prefix = prefix 
            if size is not None:
                self.nstep = size
                self.step = size/iters
    
            if final_msg is None:
                final_msg=""
            self.final_msg = final_msg 
            
        def sys_print(self,msg):
                sys.stdout.write('\r')
                sys.stdout.write(msg)
                sys.stdout.flush()      
                
        def update(self,msg=None,prefix=None):
            if msg is None:#suffix 
                msg=""
            if prefix is None:
                prefix="" 
            prefix_ = self.prefix + prefix
            if self.rounds>self.round:
                self.round +=1
                i  = round(self.step*(self.round-1))
                i_ = self.nstep
                perc = round((self.round/self.rounds)*100)
                
                chars = self.bar_charcters
                bar_string = f"[{(chars[0]*i+chars[1]).ljust(i_,chars[2])}]" 
                self.sys_print(f"{prefix_}{bar_string}{perc}% {msg}")         
            if self.rounds==self.round:
                self.sys_print(" "*100)
                self.sys_print("*** "+self.final_msg+" ***")       
                
    def WarningMessage():
        """Maybe replace this with a tkinter version
        """
        ctypes.windll.user32.MessageBoxW(0,"", "Warning Script Finished",  1)
        winsound.Beep(2500, 1000)
        
    def PrintHeader(msg=None, width=60):
        if msg is None:
            print(f"{'#'*width}\n\n")
        else:
            print(f"{'#'*width}\n##{msg.center(width-4)}##\n{'#'*width}")
        
    def RemoveVariables(startswith=None,endswith=None,protected=None,everything=False):
        if protected is None:
            protected = []
        protect_list = ['In','Out','get_ipython','exit','quit','RemoveVariables']+protected
        protect = lambda x:(x.startswith("_")) or (x in protect_list)
        for k, v in list(globals().items()):
            if not protect(k):
                if   startswith is not None:
                   if k.startswith(startswith):
                      del globals()[k]
                elif endswith is not None:
                   if k.endswith(endswith):
                      del globals()[k]
                if everything:
                    del globals()[k]
 

#%% Functions
class NeuralNetworkFuncs:
    """ Functions directly related to keras
    
        >>    CreateKerasNet(layers,af_dict=None,model=False,Print=False,**kwargs)
        >>    NormalizeLatentLoss(latents)  # NotWorking or Used
        >>    AutoencoderPredict2DataFrame(model,df)
        >>    ImputationLossFunctionGenerator(Losses_df,Mask_NA)   
    
    """
    
    
    def CreateKerasNet(layers,af_dict=None,model=False,Print=False,**kwargs):
        """     net = Create_Keras_Net([34,23,12])
                # Notes
                   net[0 ] # input 
                   net[-1] # output
                   
                   #>> LEARN THESE
                   # clone    
                   # target_model.set_weights(model.get_weights())     
                   # concat split  
        """
        if af_dict is None:
            af_dict={}
        af_dict["default"] = af_dict.get("default","relu")
        Print = kwargs.get("print",Print)
        
        printout = {True : print, False : lambda *args,**kwargs: None}[Print]
        
        def if_not_keras_layer(arg):
            return "tensorflow" not in str(type(arg))
    
        net={ "Order" : [0]}
    
        printout(f"{'#'*100}\n###   NeuralNetworkFuncs.CreateKerasNet({layers}, af_dict={af_dict}, model={model}, Print={Print})   ###\n{'#'*100}")
        if  if_not_keras_layer(layers[0]):
            net[0] = Input(shape=(layers[0],))
            printout(f"net={{0 : Input(shape=({layers[0]},))}}")
        else:
            net[0] = layers[0]
            printout(f"net={{0 : net2[0]}}")        
            
        for i,layer_i in enumerate(layers[1:],1):
            if  if_not_keras_layer(layers[0]):
                activation = af_dict.get(i,af_dict["default"]) 
                activation = af_dict.get(i-len(layers),activation) 
                net[ i ] = Dense(layer_i,activation=activation)(net[i-1])
                activation_ = " "+str(activation)+" " if activation is None else "'"+activation+"'"
                printout(f"net[   {i}   ] = Dense( {layer_i: <3d}, activation={str(activation_).rjust(7)})(net[{i-1}])")
            else :
                net[ i ] = layer_i(net[i-1])
                printout(f"net[ {i}  ] = net2[**](net[{i-1}])")
            net["Order"] += [i]
                
        net[-1] = net[i]
    
        printout(f"net[  -1   ] = net[{i}]") 
        printout(f"net['Order'] = {net['Order']} ")
        if not model:
           return net
        model_ = Model(inputs=net[0], outputs=net[-1])
        printout(f"model = Model(inputs=net[0], outputs=net[-1])")  
        printout("#"*100,"\n")
        return net, model_
    
    def NormalizeLatentLoss(latents):
         """ The latents will have a mean=0, std =1
             NormalizeLatentLoss(encoder_net[-1])
             ##  NOT WORKING PROPERLY                          """
         lat_mean  = K.mean(latents, axis=0)
         lat_var   = K.mean(K.square(latents-lat_mean), axis=0)
         norm_mean = K.mean(K.abs(lat_mean))
         norm_lat  = K.mean(K.abs(lat_var-1))
         return norm_mean + norm_lat
    
    def AutoencoderPredict2DataFrame(model,df):
        "The Output has to have the same numner of columns,reall this only used for Autoencoders"
        data = model.predict(df)
        return pd.DataFrame(columns=df.columns, index=df.index, data=data)
    
    def ImputationLossFunctionGenerator(Losses_df,Mask_NA):
        """    losses should be a DataFrame with the loss for each of the output to be trained on
                NA_block matrix        """     
        KMask_NA    = K.variable( value=Mask_NA.values               )
        kCE_mask    = K.variable( value=Losses_df.loc["CE"  ].values )
        kMAE_mask   = K.variable( value=Losses_df.loc["MAE" ].values )    
        kMSE_mask   = K.variable( value=Losses_df.loc["MSE" ].values )
        #kNA_present = K.variable( value=Losses_df.loc["NA_present"].values )
    
        def Loss_Function(yTrue,yPred):
            ce  = K.binary_crossentropy(yTrue, yPred)
            mse = K.square(yTrue - yPred)
            mae = abs(yTrue - yPred)        
            combined = (kCE_mask * ce) + (kMSE_mask * mse) + (kMAE_mask * mae )
            not_block = K.dot(yTrue, KMask_NA )# 1s if         
            combined = (1. -not_block) * combined        
            return K.mean(combined,axis=-1)
        return Loss_Function
    
 
    
class PandasFunctions :   
    def DataFrameValueCounts(df,mx=200):
        """ Create a dataframe of all the value counts of each of the columns this only works with categocial data        """
        
        df_out = pd.DataFrame()
        for col in df.columns:
           df_small =  df[col].value_counts().to_frame(col)#.astype(int)
           df_out = df_out.combine_first(df_small)
           if df_out.shape[0]>mx:
                 raise Exception(f"Error the DataFrame is getting to large it has an index size:{df_out.shape[0]}, which is above max:{mx},\
                                  think of using:  'hist_dict = {{col: df[col].value_counts() for col in df.columns}}'")
        df_out= df_out.fillna(0).astype("int64")       
        df_out.loc["unique"] = df_out.apply(lambda s:tuple(sorted(list((s[s>0]).index)))) #sorted(list(df_small.index))               
        df_out.loc["len"   ] = df_out.loc["unique"].apply(len)
        return df_out
    def CompressDataFrame(df, axis=[0,1] ):
        """ If columns or a Index only contains 0s its removed """
        def listify(args):
            return list(args) if type(args) in [list,tuple,range,set,dict] else [args]
        axis = listify(axis)
        if (0 in axis)and(1 in axis):
           return df.loc[(df != 0).any(axis=1),(df != 0).any(axis=0)]
        axis_keep = (df != 0).any(axis=axis[0])
        if 0 in axis:
            return df.loc[:,axis_keep]
        return df.loc[axis_keep] # 1    
    def ReplaceColumnValueWithMean(df,colname,value):
        "  Just Replace a Value in a selected columns with the mean value of the column"
        if type(colname) is list:
            for colname_ in colname:
                df = PandasFunctions.ReplaceColumnValueWithMean(df,colname_,value)
            return df
        df.loc[df[colname] == value,colname] = df.loc[df[colname] != value,colname].mean() # df[df[colname] != value][colname].mean()
        return df



def add_one_hot_coded_and_NA_columns_to_df_and_remove_old(census_df,census_value_counts_df,columns_one_category):
    
    # Function that can remove old column and replace it with a one hot encoded version
    def replace_columns_with_one_hot_coded_columns(df, col_name, value_counts_df,  missing_values = [-9], skip_if_2=True, remove_org_column=True):
        """ turn the column into one hot encodeded version and remove orginal add too the dataframe, missing val=-9    """
        if skip_if_2:
           if value_counts_df.loc["len",col_name]==2:
                return df
        one_hot = pd.get_dummies(df[col_name])
        cols = [n if n not in missing_values else "NA" for n in list(one_hot.columns)]
        one_hot.columns = ["_".join([col_name, str(col)]) for col in cols]
        if remove_org_column:
           df = df.drop(col_name,axis = 1)
        df = df.join(one_hot)
        return df
    
    print("Create DataFrame census_df")  
    
    columns = list(census_df.columns)
    pb = General.PrintBar(len(columns),40,final_msg=f" >   Census_df: COMPLETE ** All Columns have been Added to the DataFrame\n\n",prefix=" >   Census_df: ")  
    
    # loop though all the columns if the column is a contium like age it should be in the columns_one_catogory
    # unlike region where the no2 is as close to no3 as no6 
    # also add NA if there is any missing data
    for i, col in enumerate(columns):
        pb.update( f" {str(col).ljust(16)}: adding columns, {i+1}/{len(columns)}" )
        missing_value = -9
        if col in columns_one_category:
           if census_value_counts_df.loc[missing_value,col]>0: 
              #census_df = add_missing_column_to_non_one_hot(census_df, col)
              census_df[f"{col}_NA"] = (census_df[col]==missing_value).astype("int64")
        else:
              census_df = replace_columns_with_one_hot_coded_columns(census_df, col, census_value_counts_df)
    return census_df

def produce_losses_and_block_mask_from_NA_dict_and_df(df, NA_dict):
    dfcolumns = df.columns
    
    col_i_na_related = [v[0]+[k] for k,v in NA_dict.items()]
    col_i_na_related = sorted([ii  for i in col_i_na_related for ii in i])
    col_dic={}
    for col in dfcolumns:
        k       = col.split("_")[0]
        col_dic[k] = col_dic.get(k,0)+1
     
    Losses_df = pd.DataFrame(columns=dfcolumns,index=["CE","MSE","MAE","NA_present","One-Hot"]).fillna(0.)
    Mask_NA   = pd.DataFrame(columns=dfcolumns,index=dfcolumns).fillna(0.)
                                        
    for k,v in NA_dict.items():            
           for vi in v[0]:
                Mask_NA.iloc[k,vi]=1
    
    for i,col in enumerate(dfcolumns):
        if "_" in col:
           Losses_df.loc["CE" ,col] = 1 
        else:  
           Losses_df.loc["MSE",col] = 1 
        if i in col_i_na_related:
           Losses_df.loc["NA_present",col] = 1
        if (col_dic[col.split("_")[0]]-int(i in col_i_na_related))>1:
           Losses_df.loc["One-Hot",col] = 1
        
    return Losses_df, Mask_NA


def remove_data_for_training_from_dataframe(data_df,Mask_NA,missing_fraction=0.3):
    """
            #             Three Learning Serinos
            #----------------------------------------------------------------
            #         In            Actual    What_to_Learn(backprop)
            # 1)    Missing        Missing         Na(=1)
            # 2)    Missing         Known          Values
            # 3)     known          known         Na(=0) + Values
                             
            #  Health_NA   Health_1,   Health_2,   Health_3 one hot encoding Na if all them are 0 else known so na is 1   
    """

    def convert_series_to_diag_df(Series):
        return pd.DataFrame(columns=Series.index, index=Series.index, data=np.diag(Series.values))
    
    #removed_vals = (Mask_NA == 1).any(axis=0).astype(int)
    NAs          = (Mask_NA == 1).any(axis=1).astype(int)
    NA_diag  = convert_series_to_diag_df(NAs)

    NA_map  = PandasFunctions.CompressDataFrame(NA_diag,[1])
    Val_map = PandasFunctions.CompressDataFrame(Mask_NA,[1])
    #############################################################
    np.random.seed(0)
    remover = np.random.rand(data_df.shape[0], NA_map.shape[0])
    remover = np.where( missing_fraction>remover , 1, 0)
    #############################################################    
    NA_1s     = np.matmul(remover,  NA_map.values)
    Values_0s = np.matmul(remover, Val_map.values)
    #############################################################    
    data_missing_df = data_df.copy()
    matrix = data_missing_df.values
    matrix[NA_1s    ==1] = 1
    matrix[Values_0s==1] = 0
    data_missing_df[:] = matrix
    #############################################################
    return data_missing_df

class Show:
    def NeuralNetSummary(neuralnet_flag):
        if neuralnet_flag:
            General.PrintHeader("Neural Net Summaries")   
            print("  Encoder Summary\n")
            encoder.summary()
            print("  Decoder Summary\n")
            decoder.summary()
            print("  Autoencoder Summary\n")
            autoencoder.summary()
            General.PrintHeader()   
    def Latents(latents_flag):            
        if latents_flag:
            General.PrintHeader("Latent_info")  
            def mean_and_std_print(array, msg=""):
                print(f"{msg}  \nmean:{np.mean(array,axis=0)}, \nstand_dev:{np.std(array,axis=0)}")
            
            def histogram_plot_latents():
                from matplotlib import pyplot    
                bins = np.linspace(-4, 4, 150)
                for i in range(Test["Latents"].shape[1]):
                    lat_i = Test["Latents"][:,i]
                    pyplot.hist(lat_i, bins, alpha=0.4, label=str(i))
                pyplot.legend(loc='upper right')
                pyplot.show()
            # have a Look at the latents    
            histogram_plot_latents()   
            mean_and_std_print(Test["Latents"],"Latents Properties")
            print(f"{'- '*30}\n")  
            
    def CorrelationMatrix(correlation_flag,df=None):
        if correlation_flag:

            #%matplotlib inline
            if df is None:
               corr = census_df.corr()
            else:
               corr = df.corr()
            Show._HeatMap(corr)
            #g = sns.heatmap(corr,  vmax=.3, center=0, square=True, linewidths=.5,  fmt='.2f', cmap='coolwarm')

    def _HeatMap(df):
            import seaborn as sns
            import matplotlib.pyplot as plt        
            g = sns.heatmap(df,   square=True,   cmap='coolwarm')
            sns.despine()
            g.figure.set_size_inches(14,10)
            plt.show()

 
def ShowOtherCorrelatedMatrix(display=True):
    """This Function isnt essential but shows how the columns correlate with one another
    variables used:
        Config["NA_Columns"] , census_df
    """
    if display:
        na_columns     = [col+"_NA" for col in Config["NA_Columns"]]
        not_na_columns = [col for col in census_df if col not in na_columns ]
        print("   NA Correlations")
        Show.CorrelationMatrix(True,census_df[na_columns])
        print("   Non - NA Correlations")
        Show.CorrelationMatrix(True,census_df[not_na_columns])
        
        corr = census_df[not_na_columns].corr()
        
        corr2 = corr.copy()
        for col in corr2.columns:
            #corr_temp.loc[col,col]=0
            others = [c for c in corr2.columns if c.startswith(col[:3])]#.split("_")[0]
            #remove similar columns
            corr2.loc[col,others] = 0
       
        topcorrelatedvalues = 10
        val_to_clipby = sorted([abs(i) for ii in corr2.values.tolist() for i in ii ],reverse=True)[topcorrelatedvalues*2]
        corr_temp = corr2[abs(corr2)>val_to_clipby].fillna(0)
        corr_small = PandasFunctions.CompressDataFrame(corr_temp)    
        
        print(" Intresting  Non - NA Correlations")
        #Show._HeatMap( corr_small )
        Show._HeatMap(corr2.loc[corr_small.index, corr_small.columns])        

def QuickCreateEncoderLayerSizes(nocols, ratio=0.57, depth=5):
    """ If a layer size isnt specified this creates a simple one
    where the previous layer is multiplied by the ratio
    """
    out=[nocols]
    for i in range(depth-1):
        out.append(round(out[i]*ratio))
    return out 

            
################################################################################################################################
################################################################################################################################
################################################################################################################################

#%%   Read in the Data and Wrangle it

print("------------------------------------------")
print(" > Read in Data to a DataFrame")
filepath = "Census.train.csv"
columns_one_category = ['Age', 'Health', 'HoursWorked', 'SocialGrade']# of these ['Health', 'HoursWorked', 'SocialGrade'] have missing NA values
columns_2_remove     = ["PersonID"]

census_df = pd.read_csv(filepath)

if Config["resample"]:
   census_df = census_df.sample(frac=1.0)

census_df.drop(columns_2_remove, axis=1, inplace=True)#del census_df["PersonID"]
print(f"{columns_2_remove} has been deleted from Census\n")

Config["INFO"]["Orginal_Columns"] = list(census_df.columns) 

census_value_counts_df = PandasFunctions.DataFrameValueCounts(census_df)
census_df = add_one_hot_coded_and_NA_columns_to_df_and_remove_old(census_df, census_value_counts_df, columns_one_category)

print("\n> DataFrame has been Organized")

Show.CorrelationMatrix("CorrelationMatrix" in Config["Show_Info"])     

  
#%%
# if NA then dont touch the others in the dataframe
# create a function to rand only grab data and turn it numpy data
# if NA dont backprop on the others but always backprop on NA 

NA_dict = {i:col  for i,col in enumerate(list(census_df.columns)) if "NA" in col}
NA_dict = {i:([ii for ii,c in enumerate(list(census_df.columns)) if (col.replace("_NA","") in c) and (i!=ii)],col) for i,col in NA_dict.items()}

Config["NA_Columns"] = [v[-1].rstrip("_NA") for k,v in NA_dict.items()]

ShowOtherCorrelatedMatrix("CorrelationMatrix" in Config["Show_Info"]) 

# Mask_NA is a square dataframe, its values when multiploed by the input,tell you what columns NOT to backprop
# Example if Health_NA is 1 in the input we dont want to learn the Health value so it blocks the backprop
# Dont learn what you dont know
# This Mask is used in the custom backprop
Losses_df, Mask_NA   = produce_losses_and_block_mask_from_NA_dict_and_df(census_df, NA_dict)
Imputation_Loss_Func = NeuralNetworkFuncs.ImputationLossFunctionGenerator(Losses_df,Mask_NA)

## Foget This
census_df = PandasFunctions.ReplaceColumnValueWithMean(census_df,["HoursWorked","SocialGrade"],-9)

#%%#########################################################################################################

Config["INFO"]["No_Columns" ] = census_df.shape[1]
Config["INFO"]["No_Rows"    ] = census_df.shape[0]
Config["INFO"]["Split_Loc"  ] = round(census_df.shape[0]*Config["train_test_split_ratio"])

quick_train, quick_test = census_df.iloc[:Config["INFO"]["Split_Loc"]], census_df.iloc[Config["INFO"]["Split_Loc"]+1:]

#diff_test_train = quick_train.mean()-quick_test.mean()

quick_train_in = remove_data_for_training_from_dataframe(quick_train, Mask_NA, Config["remove_fraction"])    

#quick_train_in=quick_train
if Config["encoder_dim"] is None:
     Config["encoder_dim"] = QuickCreateEncoderLayerSizes(Config["INFO"]["No_Columns"])
    
    
Config["decoder_dim"] = list(reversed( Config["encoder_dim"] ))

print("Autoencoder Layer sizes(Enc,Dec):", Config["encoder_dim"] ,Config["decoder_dim"])

#%% Construct The Neural Net
 
encoder_net, encoder = NeuralNetworkFuncs.CreateKerasNet(Config["encoder_dim"] , {-1:None}, model=True, print="Keras_Code" in Config["Show_Info"])
decoder_net, decoder = NeuralNetworkFuncs.CreateKerasNet(Config["decoder_dim"] ,            model=True, print="Keras_Code" in Config["Show_Info"])

autoencoder_out = decoder(encoder(encoder_net[0]))
autoencoder     = Model(encoder_net[0], autoencoder_out)

Show.NeuralNetSummary("Neural_Net_Summary" in Config["Show_Info"])  




#%%   Set up Loss Function and Train the Net


print(f"  >>  Training Mode: '{Config['train_mode']}'")
# TRAIN VARAIBLES


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true), axis=-1)

Normalized_Impute_Loss = lambda *args:Imputation_Loss_Func(*args)+NeuralNetworkFuncs.NormalizeLatentLoss(encoder_net[-1])

loss_func={"keras_mse"              : 'mean_squared_error'  ,
           "keras_mae"              : 'mean_absolute_error' ,
           "custome_mse"            : mean_squared_error    ,
           "Normalized_Impute_Loss" : Normalized_Impute_Loss,
           "Impute_Loss"            : Imputation_Loss_Func  }[Config['train_mode']]


###################### TRAIN ######################################################
autoencoder.compile(loss=loss_func, optimizer = "adam", metrics=['accuracy'])
autoencoder.fit(quick_train_in, quick_train, batch_size = Config["batch_size"], epochs = Config["Epochs"])
###################################################################################   
score = autoencoder.evaluate(quick_test, quick_test, verbose=0)

print(f'Test loss:     { score[0]:.2f} ' , f'Test accuracy: {100*score[1]:.2f} %', sep="\n" )
del score,loss_func
 

Test = {}
Test["Predict"] = NeuralNetworkFuncs.AutoencoderPredict2DataFrame(autoencoder, quick_test)
Test["Error"  ] = quick_test - Test["Predict"] 
# remove small errors to make it easier to see larger ones
Test["Error"  ][abs(Test["Error"  ])<0.3]=0
Test["Latents"] = encoder.predict(quick_test)

Show.Latents("Latents" in Config["Show_Info"])   
 




#%%    Predict Missing Columns
def predict_missing_column(column2remove,data,model):
    """No continous values also must also have missing column
    """
    missing_columns = [col for col in data.columns if col.startswith(column2remove)]
    column_na = column2remove+"_NA"
    
    if column_na in missing_columns:
       missing_columns.remove(column_na)
       if len(missing_columns)>1:
           print(f"Column to Predict: '{column2remove}', No of related Columns: {len(missing_columns)}")
           data_actual = data[data[column_na]==0].copy()
           print(f"Number of Indexs removed because of missing data: {len(data_actual)-len(data)} / {len(data)}")
           data_in = data_actual.copy()
           data_in[missing_columns] = 0
           data_in[  column_na    ] = 1
           data_out = NeuralNetworkFuncs.AutoencoderPredict2DataFrame(model, data_in)
           data_out_small = data_out[missing_columns].copy()
           data_out_small["Actual" ] = data_actual[missing_columns].idxmax(axis=1)
           data_out_small["Predict"] = data_out[   missing_columns].idxmax(axis=1)
           values = PandasFunctions.DataFrameValueCounts(data_out_small[["Actual","Predict"]])
           acc     = sum(data_out_small["Actual" ] ==  data_out_small["Predict"] )/len( data_out_small["Predict"])
           Config["INFO"]["Accuracy"] =  Config["INFO"].get("Accuracy",[])+[acc]
           acc_max = max(values.loc[missing_columns ,"Actual"  ])/len( data_out_small["Predict"])
           print(f"Fraction of Predictions correct: {100*acc:.2f}%, if just using the mode: {100*acc_max:.2f}%\n")
           return data_out_small,values
       else:
            print(f"Column:{missing_columns} is Continous, so this isnt calculated yet ")  
    else :        
        print(f"No Missing Column in the DataFrame:{column2remove}")
    return None,None

 

General.PrintHeader("Remove Certain Columns and Predict Them")  
predictions_misssing_cols={}
for missing_column in Config["NA_Columns"]:
    predictions_misssing_cols[missing_column+"_results"],predictions_misssing_cols[missing_column+"_hist"] = predict_missing_column(missing_column, quick_test, autoencoder)
del missing_column
General.PrintHeader() 

#    
#religion_pred_test_df,religion_hist = predict_missing_column("Religion"   ,quick_test,autoencoder)
#bir_pred_test_df,bir_hist = predict_missing_column("BirthCountry",quick_test,autoencoder)


   
#%%        Previous Results 
print("#>>", Config["remove_fraction"],[round(n,3) for n in Config["INFO"]["Accuracy"]], Config["Epochs"], Config["encoder_dim"],Config["train_mode"]) 

#>> 0.3 [0.597, 0.887, 0.878] 15 [87, 65, 44, 17, 11]
#>> 0.3 [0.594, 0.895, 0.884] 100 [87, 65, 44, 17, 11]
#>> 0.3 [0.6, 0.893, 0.877] 20 [87, 100, 50, 20, 12]
#>> 0.3 [0.495, 0.894, 0.885] 20 [87, 100, 50, 20]
#>> 0.3 [0.55, 0.864, 0.814] 20 [87, 100, 50, 20, 15]
#>> 0.4 [0.591, 0.864, 0.865] 20 [87, 100, 50, 20, 12]
#>> 0.4 [0.6, 0.892, 0.88] 20 [87, 100, 50, 20, 12] # basically a repeat of the above one
#>> 0.4 [0.548, 0.864, 0.859] 20 [87, 100, 50, 20, 12] # same as above but used keras mae
#>> 0.4 [0.591, 0.864, 0.91] 20 [87, 100, 50, 20, 12] # repeat
#>> 0.4 [0.621, 0.913, 0.91] 40 [87, 100, 50, 20, 12] # "keras_mse" but 40 iterations
#>> 0.4 [0.603, 0.912, 0.911] 40 [87, 100, 50, 20, 12]
#>> 0.4 [0.621, 0.881, 0.861] 40 [87, 100] keras_mse
if "skip" not in Config["load_save_skip"]:
    save_name = "Impute_Autoencoder_"+"#".join([str(n) for n in ( Config["Epochs"], Config["encoder_dim"],Config["train_mode"])])
    
    if "save" in Config["load_save_skip"]:
        #save weigths
        autoencoder.save_weights(save_name+"_Wieghts_"+'.h5')
        #save model
        autoencoder.save(save_name+'.h5')
    if "load" in Config["load_save_skip"]:    
        #load weigths
        autoencoder.load_weights(save_name+'.h5')
        #load model
        autoencoder2 = keras.models.load_model(save_name+'.h5')

General.WarningMessage()
#correlation info



