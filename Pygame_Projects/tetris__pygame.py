# -*- coding: utf-8 -*-
"""Created on Mon Jan 27 03:39:50 2020 @author: Alexm"""

"""
create an array called tetris_space
game while loop
    if next_object = True
       location = top
       object = new random object
       nec
    every few loops subjectact 1 from the location of the object(falling)
    #check if move is legal if not nex
    check if user has pressed a arrow or rotate
    try to move object except down
    if it can move there move it
    c
    

    



"""
START_POSITION = 0,50
import numpy as np

tetris = np.zeros([100,200])

tetris[: ,-1] =1
tetris[: ,0 ] =1
tetris[-1,: ] =1

shape1 = [[0,0,0],[0,1,0],[1,1,1]]
shape2 = [[1,1],[1,1]]
shape3 = [[0,0,1],[0,0,1],[0,1,1]]

shape = shape1
sz = len(shape)
shape = np.array(shape)


def find_locations_to_fill(ix,iy,sz):
    return slice(ix,ix+sz),slice(iy,iy+sz)
    
def move_if_possible(tetris,shape,locs,move):
    shape,locs = move_shape(shape,locs,keys)
    new = tetris.copy()
    new = np.where(new==2, 0, new) 
    new = new[locs[0],locs[1]]+shape
    possible = np.count_nonzero(new==3)==0
    return new,shape,locs, possible

starting_posistion =START_POSITION
cycle =10
count = 0
while True:
    count+=1
    if count%cycle ==0:
        starting_posistion[0] -=1
        locs = find_locations_to_fill(starting_posistion[0],starting_posistion[1],sz )

        keys = getkeys()
        move = getmovekeys(keys,"floor")
        new,shape,locs, possible = move_if_possible(tetris,shape,locs,move)
        if not possible:
            tetris = np.where(tetris==3, 2, tetris) 
            break
        tetris = new
        
        move = getmovekeys(keys,"allowed")
        new,shape,locs, possible = move_if_possible(tetris,shape,locs,move)
        if possible:
           tetris = new 
       
 
 
#screen



M     = 96
lis   = [1000,12,28,233,2445]
def find_element_to_remove(lis,M)
    lis2  = [(e**2) for e in lis]
    slis2 = sum(lis2)
    lis3  = [((slis2-i2),i) for i2,i in zip(lis2,lis)]
    XiSum,Xi = max(lis3,key=lambda x:x[0]%M)


# moduls and primes
# 
    










