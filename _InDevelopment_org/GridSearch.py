# -*- coding: utf-8 -*-
"""Created on Mon May 13 13:13:41 2019@author: milroa1"""
#GridSearch all possible combination of the parameters
# Random Search random paremters(although this means no repeating of very similar huperparemters)

# Load the dataset
x, y = load_dataset()

# Create model for KerasClassifier
def create_model(hparams1=dvalue,hparams2=dvalue,hparamsn=dvalue):
    pass 
    # create model

model = KerasClassifier(build_fn=create_model) 

# Define the range
hparams1 = [2, 4]
hparams2 = ['elu', 'relu']
hparamsn = [1, 2, 3, 4]

# Prepare the Grid
param_grid = dict(hparams1=hparams1,hparams2=hparams2,hparamsn=hparamsn)
#{'hparams1': [2, 4], 'hparams2': ['elu', 'relu'], 'hparamsn': [1, 2, 3, 4]}

#%%##################################################################################################
# GridSearch in action
grid = GridSearchCV(estimator=model, 
                    param_grid=param_grid, 
                    n_jobs=, #Number of jobs to run in parallel.
                    cv=,# default 3-fold cross validation,
                    verbose=)#Controls the verbosity: the higher, the more messages
                    
grid_result = grid.fit(x, y)

# Show the results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means  = grid_result.cv_results_['mean_test_score']
stds   = grid_result.cv_results_['std_test_score']
params = grid_result.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean} ({stdev}) with: {param}")
    
    
    
    
    

#%%##################################################################################################
# RandomSearch in action!
n_iter_search = 16 # Number of parameter settings that are sampled.
random_search = RandomizedSearchCV(estimator=model, 
                                   param_distributions=param_dist,
                                   n_iter=n_iter_search,
                                   n_jobs=, 
								   cv=, 
								   verbose=)
random_search.fit(X, Y)

# Show the results
print(f"Best: {grid_result.best_score_} using {grid_result.best_params_}")
means  = random_search.cv_results_['mean_test_score']
stds   = random_search.cv_results_['std_test_score']
params = random_search.cv_results_['params']
for mean, stdev, param in zip(means, stds, params):
    print(f"{mean} ({stdev}) with: {param}")  
#%%##################################################################################################    
#Bayesian Optimization    exists as well
    
    
    







    
    