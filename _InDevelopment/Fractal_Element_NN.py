# -*- coding: utf-8 -*-"""Created on Mon Mar  5 11:59:40 2018@author: milroa1"""
#%%###################################################################################
""" Three Types Inputs of Outputs 

•  Up   (In & Out)
•  Side [Mode, Current-State, Level-Info]
•  Down (In & Out)

"""
#%%###################################################################################
""" Operations 

•  Initalize(only one layer is initalized)
   # In -Side[Level-Info]
   
•  Start_Rising(only one later is initalized)
   # In Down, -Side[Level-Info]
   
•  Start_Falling(only one later is initalized)
   # In Up, -Side[Level-Info]

•  Calculate(only one later is initalized)
   # In Up, Down -Side[Level-Info, Current-State] #bottem layer no Down, top layer no up

Example with [3*3 down, 1 up,5 layers, no crossover]
                                       _
layer 1: 1    Elements              __| |__
layer 2: 9    Elements           __|       |__
layer 3: 81   Elements        __|             |__    < (start here)
layer 4: 729  Elements     __|                   |__
layer 5: 6561 Elements    |_________________________|
      
Initalize(layer 3)
Start_Rising(layer 2)
Start_Rising(layer 1)
Calculate(layer 2)
Calculate(layer 3)
Start_Falling(layer 4) # but on only selected ones that are needed from layer 3 and same for layer 5
Start_Falling(layer 5)
Calculate(layer 4)
Calculate(layer 3)
Calculate(layer 2)
Calculate(layer 1)
Calculate(layer 2)
Calculate(layer 3)
Calculate(layer 4)
Calculate(layer 5)
Calculate(layer 4)
Calculate(layer 3)
Calculate(layer 2)
Calculate(layer 1)## decside info of the image


"""
#%%###################################################################################
"""    Things it can do
impaint(segment image)
autoencoder


"""
#%%###################################################################################




















