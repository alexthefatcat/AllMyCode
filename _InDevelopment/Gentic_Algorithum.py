# -*- coding: utf-8 -*-
"""Created on Thu Apr  4 15:26:10 2019@author: milroa1"""
from numpy.random import standard_normal
import numpy as np

"""     try gans and ga
        ga learning in the wirghts of a neural net
        ga incoperate assortive amting where mate choice is partly "gentic"
        also relatedness but they would have to interact
"""

Config={"rounds"            : 250,
        "breeding_fraction" : 0.8}

#%%
def Create_Population(nsamples=55,parms=12,mean=60,std=3):
      return (std)*standard_normal([nsamples,parms])+mean

def Get_Fitness(samples,target=-42.5):#EvaluationofIndviduals2//fitnesscalculation
    " Basically population has a higher fitness if there near 0, for this game"
    return ((population-target)**2).sum(axis=1)

def SortbyFitness(population,fitness):
    order = fitness.argsort()
    return population[order], fitness[order]  

def Parent_Selection(population,bred_frac=0.8):#Selection3 matingpool/parentsselection
    """   Top 20%: parents have double the children
       Bottom 20%: parents have no children
       The parents are randomly shuffeled and combined in a tupplw
    """
    pop_len = len(population)    
    bredders_loc = round(pop_len*bred_frac)   # top 0.8 the mating_pool, top 0.2 breeds twice
    new_population = np.concatenate((population[:bredders_loc] , population[:(pop_len-bredders_loc)]))
    parents1, parents2 = new_population.copy(), new_population.copy()
    np.random.shuffle(parents1),np.random.shuffle(parents2)    
    parents = (parents1, parents2)
    return parents

def Reproduce(parents):#Reproduction4 mating ->crossover,mutation  Mutation5  
    children = (parents[0]+parents[1])/2
    return children

def Mutate(population,std=3): 
    mutate_array = std*standard_normal(population.shape)
    return population+ mutate_array
#%% Basic Run
population = Create_Population()

for n in range(Config["rounds"]):
    fitness    = Get_Fitness(population)
    population, fitness = SortbyFitness(population,fitness)
    parents    = Parent_Selection(population,Config["breeding_fraction"])
    if n%((Config["rounds"])//10)==0:# prints out 11 but first and near last
       print(f"Round {n:4d} : All_Population_Mean: {np.mean(population):6.2f},  Parent_Mean: {np.mean(parents[0]):6.2f}, {'^':3s}^")   
    population = Reproduce(parents) # new geneartion
    population = Mutate(population) # mutate them





