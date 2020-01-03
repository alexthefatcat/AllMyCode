# -*- coding: utf-8 -*-
"""Created on Wed Mar 13 14:11:22 2019@author: milroa1"""
# Random Forrest

Config={"train_test_%_split":0.7}
Config["dataset"]="blob"#("blob","breast")[]


#%%   Load Data

from sklearn.datasets import load_breast_cancer, make_blobs
data_breast = load_breast_cancer()
data_blob   = make_blobs(n_samples=300, centers=4,random_state=0, cluster_std=1.0)

#%%    Import Modules

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score

import matplotlib.pyplot as plt
import numpy as np

import seaborn as sns
sns.set()

def print_header(msg,no_hashs=70):   
    msg2 ="###   "+msg+"   ###" 
    indent =" " * ((no_hashs-len(msg2))//2)
    hashs = "#"*no_hashs
    print(f"\n{hashs}\n{indent}{msg2}\n{hashs}\n")

#%%   Check data 

def check_missing_data(df):
    missing_col_info = df.isna().sum()
    missing_total    = missing_col_info.sum()
    if missing_total.sum()==0:
        print("Check:PASSED, No Nans in DataFrame")
    else:
       print(f"Check:FAILED, {missing_total} missing Nans detected") 
       # How to handle this
       
print_header(f"Start the Script, Dataset: {Config['dataset']}")

if   Config["dataset"]=="breast":
        #print(data_breast["DESCR"])
        df = pd.DataFrame(data=data_breast["data"],columns=data_breast["feature_names"])
        df["target"] = data_breast["target"]
        
elif Config["dataset"]=="blob":
        df = pd.DataFrame(data=data_blob[0],columns=["X1","X2"])
        df["Y"] = data_blob[1]


def dataframe_info(df):
    dfinfo                = df.describe()
    dfinfo.loc["Type"   ] = df.dtypes
    dfinfo.loc["Nunique"] = df.nunique(axis=0)
    return dfinfo



print_header("Info>>, and split into train and test data")
dfinfo = dataframe_info(df)
check_missing_data(df)      





#train_x, test_x, train_y, test_y = train_test_split(df.iloc[:,:-1], df.iloc[:,-1],train_size=Config["train_test_%_split"])
#print(f"Train_x Shape :: {train_x.shape}, Train_y Shape :: {train_y.shape}, Test_x Shape :: {test_x.shape}, Test_y Shape :: { test_y.shape}")

data = {"Train":{},"Test":{}}
data["Train"]["X"],data["Test"]["X"],data["Train"]["Y"],data["Test"]["Y"] = train_test_split(df.iloc[:,:-1], df.iloc[:,-1],train_size=Config["train_test_%_split"])
print("Size of train and test:>>",f'data["Train"]["X"] Shape :: {data["Train"]["X"].shape}',f'data["Train"]["Y"] Shape :: {data["Train"]["Y"].shape}' ,f'data["Test" ]["X"] Shape :: {data["Test" ]["X"].shape}',f'data["Test" ]["Y"] Shape :: {data["Test" ]["Y"].shape}',end="\n\n",sep="\n\t")



#%%            PLOTTING
################################################################################################################################

def print_some_prediction_info(yp,y,msg="",no=5,global_confusion_matrix=False):
    """
    Print out some info about the predctions
    """
    hist = y.value_counts()
    
    print("\n",msg)
    for i in range(no):
        if i==0:
            print(f"   Examples of the predications and the actual for {no} values")
        print(f"      Actual outcome :: {list(y)[i]} and Predicted outcome :: {yp[i]}")
    print(f"   Accuracy  :: { accuracy_score(y, yp)}")
    if len(hist)==2:
       print(f"   F1-Score  :: { f1_score(y, yp)}")  
    else:
       print(f"   F1-Score(macro)  :: { f1_score(y, yp,average='macro')}")     
    print(f"   Confusion matrix , {confusion_matrix(y, yp).tolist()}\n")
    if global_confusion_matrix:
        global con_mat
        con_mat = confusion_matrix(y, yp)
    

def title_and_axiss(plt,title=None,xlabel=None,ylabel=None):
    for meth,name in zip(["title","xlabel","ylabel"],[title,xlabel,ylabel]):
        if name is not None:
           getattr(plt,meth)(name) 

def plot_feature_importance(df,bar_width = 0.4 ):
    #modelname.feature_importance_
    data = df.values
    col  = df.index
    #plot
    fig, ax = plt.subplots() 
    # the width of the bars 
    ind = np.arange(len(data)) # the x locations for the groups
    ax.barh(ind, data, bar_width, color="green")
    ax.set_yticks(ind+bar_width/10)
    ax.set_yticklabels(col, minor=False)
    title_and_axiss(plt,"Feature importance in RandomForest Classifier", "Relative importance", "feature")
    plt.figure(figsize=(5,5))
    fig.set_size_inches(6.5, 4.6, forward=True)

#%%


#Random Forrest
rf = RandomForestClassifier()
"""
n_estimators             : The number of trees in the forest. (default=10)
criterion                : The function to measure the quality of a split(default="gini")
max_depth                : (int,None) The maximum depth of the tree, if None until finished (default=None)
min_samples_leaf         : The minimum number of samples required to be at a leaf node (default=1)
min_weight_fraction_leaf : The min weighted fraction of the sum total of weights required to be at a leaf node(default=0.)

Features which make predictions of the model better:
     > max_features
     > n_estimators
     > min_sample_leaf
Features which will make the model training easier:     
     > n_jobs      
     > random_state 
     > oob_score 

     
"""

# mod.fit(X,Y), Yp = mod.predict(X)
rf.fit(data["Train"]["X"], data["Train"]["Y"])
print(f' "Trained model :: ", {rf}')

print_header("Predict the Yp, and Feature Importance")

data["Train"]["Yp"] = rf.predict(data["Train"]["X"])
data["Test" ]["Yp"] = rf.predict(data["Test" ]["X"])


print_some_prediction_info(data["Train"]["Yp"], data["Train"]["Y"],"Train>>")      
print_some_prediction_info(data["Test" ]["Yp"], data["Test" ]["Y"],"Test>>" )    
yp = data["Train"]["Yp"]
y  = data["Train"]["Y"]

dfinfo.loc["Feature_Importance"]=0
dfinfo.loc["Feature_Importance"].iloc[:-1] = rf.feature_importances_
 
plot_feature_importance(dfinfo.loc["Feature_Importance"])


"""
Sometimes best to reduce the number of features(dimension reduction)

for random forst   could use
   random forest feature importances
   
   or more advanced PCA ICA

"""


print_header(" Remove columns based on Feature Importance")

#plt.scatter(df["X1"], df["X2"], c=df["Y"], s=50, cmap='rainbow')  # SCATTER graph of blob
plt.scatter(df.iloc[:,0], df.iloc[:,1], c=df.iloc[:,-1], s=50, cmap='rainbow')  # SCATTER graph of blob

per_cut_off = 0.005
selected_cols = list(dfinfo.columns[dfinfo.loc["Feature_Importance"]>per_cut_off])

if len(dfinfo.columns)>8:
    "There are more than 8 columns so were going select only the most useful"
    data["Test"]["X2"],data["Train"]["X2"] = data["Test"]["X"][selected_cols], data["Train"]["X"][selected_cols]
    rf.fit(data["Train"]["X2"],data["Train"]["Y"])
    data["Train"]["Yp2"] = rf.predict(data["Train"]["X2"])
    data["Test" ]["Yp2"] = rf.predict(data["Test"]["X2"])
        
    print_some_prediction_info(data["Test" ]["Yp2"], data["Test" ]["Y"],"Test2>>" )    
else:
    print("There are less than 8 columns, so need to remove columns")




#%%
# one more indepth analysis
# grid search for best parameters
# validation set
###compare to svm



#https://jakevdp.github.io/PythonDataScienceHandbook/05.08-random-forests.html
from sklearn.tree import DecisionTreeClassifier
tree = DecisionTreeClassifier().fit( data["Test"]["X"],  data["Test"]["Y"])


def visualize_classifier(model, X, y, ax=None, cmap='rainbow'):
    ax = ax or plt.gca()
    # Plot the training points
    ax.scatter(X.loc[:, 0], X.loc[:, 1], c=y, s=30, cmap=cmap,clim=(y.min(), y.max()), zorder=3)
    ax.axis('tight')
    ax.axis('off')
    xlim,ylim = ax.get_xlim(),ax.get_ylim()
    # fit the estimator
    model.fit(X, y)
    xx, yy = np.meshgrid(np.linspace(*xlim, num=200),np.linspace(*ylim, num=200))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()]).reshape(xx.shape)
    # Create a color plot with the results
    n_classes = len(np.unique(y))
    contours = ax.contourf(xx, yy, Z, alpha=0.3,levels=np.arange(n_classes + 1)-0.5, cmap=cmap, clim=(y.min(), y.max()),zorder=1)
    ax.set(xlim=xlim, ylim=ylim)




visualize_classifier(DecisionTreeClassifier(), data["Test"]["X"], data["Test"]["Y"])

model = DecisionTreeClassifier()
ax = None
cmap='rainbow'
X,y = data["Test"]["X"], data["Test"]["Y"]



