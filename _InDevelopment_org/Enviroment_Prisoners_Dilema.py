# -*- coding: utf-8 -*-
"""Created on Tue May 21 14:38:23 2019@author: milroa1"""



class PrisonersDilema:
    def __init__(self,*args):
        if len(args)==0:
            args = [1,0,3,5]
        self.values = list(args)
        self.cheat_cheat = args[0] # 1
        self.cheat_coop  = args[1] # 1        
        self.coop_cheat  = args[2] # 5/0
        self.coop_coop   = args[3] # 3
        self.actions     = [0,0]
        self.last        = [0,0]
    def info(self,player):
        return self.values + self.last 
    
    def play(self,actionplayer0,actionplayer1):
        self.actions = [actionplayer0, actionplayer1]
        
    def reward(self,player):
        actions = self.actions
        self.last = self.actions
        self.actions = [0,0]              
        if   actions==[0,0]:
           return [self.cheat_cheat, self.cheat_cheat]
        elif actions==[1,1]:
           return [self.coop_coop, self.coop_coop]   
        elif actions==[0,1]:
           return [self.cheat_coop, self.coop_cheat]   
        elif actions==[1,0]: 
           return [self.coop_cheat, self.cheat_coop]   



def BuildModel():
    pass
def TrainModel():
    pass                
                
Model1 = BuildModel()
Model2 = BuildModel()

for n in range(200):
    Game = PrisonersDilema()
    Info = Game.info()
    Out1 = Model1(Info)
    Out2 = Model1(Info)
    Game.play(Out1,Out2)
    Reward1,Reward2 = Game.reward()
    TrainModel(Model1,Info,Reward1)
    TrainModel(Model2,Info,Reward2)
    
    


