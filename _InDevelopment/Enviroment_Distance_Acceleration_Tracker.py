# -*- coding: utf-8 -*-
"""Created on Tue May 14 16:17:36 2019@author: milroa1"""
""" This gives a class to a problem
  If you have shit location data but really good accelerator data 
  how do you combine both to get a more accurate location
"""

class AcceleratomerNNTest:
    """
    Envioment = AcceleratomerNNTest()
    """
    global np
    import numpy as np

    def __init__(self,timepoints = 1000, nobatchs = 100):
        
        self.timepoints = timepoints
        self.nobatchs   = nobatchs
        self.run()
    
    def quick_random_matrix(self,width=None,height=None):
        if width is None:
           width  = self.timepoints
        if height is None:           
           height = self.nobatchs
        return np.random.rand(width,height)-0.5
    # So its moving
    def BrownianMotion(self,actual):
        for n in range(1,(self.timepoints)):
            actual[n,:] += actual[n-1,:] 
        return actual
    def SimpleSmooth(self,actual,fact=0.6):
        actual2 =actual.copy()    
        fact2 = (1-fact)/2
        for n in range(1,((self.timepoints-1))):
            actual[n,:] = actual2[n-1,:]*fact2 + actual2[n,:] *fact + actual2[n+1,:]*fact2 
        return actual
    
    def SecondDerviative(self,actual):
        actual_d2 = np.zeros_like(actual)
        for n in range(1,(self.timepoints-1)):
        #    d2 = (a - (a-1)) - ((a+1) - a ) =>#    d2 = (2a - (a-1) -(a+1) )    
            actual_d2[n,:] =  2*actual[n,:] - actual[n+1,:] - actual[n-1,:]
        return actual_d2
    
    def basicplot(self):
        global b
        b=self.truepoints[:,1]
        %varexp --plot b   
        
    def run(self):
        self.truepoints    = self.quick_random_matrix()
        self.truepoints    = self.BrownianMotion(self.truepoints ) 
        self.truepoints    = self.SimpleSmooth(self.truepoints )
        self.noisydata_d0  = self.truepoints + 5*self.quick_random_matrix()
        self.truepoints_d2 = self.SecondDerviative(self.truepoints)
        self.noisydata_d2  = self.truepoints_d2  + 0.2 * self.quick_random_matrix()
        self.basicplot()
        # x=(noisydata_d0, noisydata_d2); Ytrue=(truepoints)

Envioment = AcceleratomerNNTest(10000)
dir(Envioment)






