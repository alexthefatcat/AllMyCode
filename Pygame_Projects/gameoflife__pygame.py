# -*- coding: utf-8 -*-
"""Created on Tue Feb 18 21:58:19 2020@author: Alexm"""

import pygame
import numpy as np
import time 
from scipy import signal


block_size  =  4
time_sleep  = 0.001

nblocksx,nblocksy = 140 , 240
#nblocksx,nblocksy = 280 , 480
 
s_width     = (nblocksy+4) * block_size
s_height    = (nblocksx+4) * block_size
top_left_y  = 2*block_size
top_left_x  = 2*block_size

randomize = True

"""

    Any live cell with fewer than two live neighbours dies, as if by underpopulation.
    Any live cell with two or three live neighbours lives on to the next generation.
    Any live cell with more than three live neighbours dies, as if by overpopulation.
    Any dead cell with exactly three live neighbours becomes a live cell, as if by reproduction.

These rules, which compare the behavior of the automaton to real life, can be condensed into the following:

    Any live cell with two or three neighbors survives.
    Any dead cell with three live neighbors becomes a live cell.
    All other live cells die in the next generation. Similarly, all other dead cells stay dead.

Every cell interacts with its eight neighbours, which are the cells that are horizontally, vertically, or diagonally adjacent. 

"""



def create_grid(fract=0.65):
    grid = np.random.uniform(0,1,[nblocksx,nblocksy])
    grid = (grid>fract).astype("float")
    return grid

def process_grid(grid):
    convo = np.ones([3,3])
    convo[1,1] = 0
    grid_neigbors = signal.convolve2d(grid,convo,"same")
    grid_3 = (grid_neigbors==3).astype("float")
    grid_2 = np.multiply(grid_neigbors==2,grid)
    grid = ((grid_3 +grid_2)>=1).astype("float")
    return grid

def change_array_based_on_mouse_clocks(grid,mouse_loc):
    if mouse_loc is not None:
        y,x = mouse_loc
        xx = (x -top_left_y)//block_size
        yy = (y -top_left_x)//block_size 
        if grid.shape[0]>xx>=0 and grid.shape[1]>yy>=0:
           grid[xx,yy] =  1 - grid[xx,yy]
    mouse_loc = None       
    return grid ,mouse_loc 

def add_rare_block_to_grid(grid):
    if randomize:
       grid = ((grid + create_grid(fract=0.9999))>=1).astype("float")
    return grid

#%%######################################################################
def draw_grid(surface, grid):
    shape = grid.shape    
    sx,sy = top_left_x, top_left_y
    sy_ns = [ sy + i*block_size for i in range(shape[0]+1)]
    sx_ns = [ sx + j*block_size for j in range(shape[1]+1)]    
    max_sx,max_sy = max(sx_ns),max(sy_ns)
    for sy_n in sy_ns:
        pygame.draw.line(surface, (128,128,128), (sx   ,sy_n ),(max_sx, sy_n))
    for sx_n in sx_ns:
        pygame.draw.line(surface, (128,128,128), (sx_n , sy  ),(sx_n  , max_sy))
    pygame.draw.rect(surface, (255, 0, 0), (top_left_x, top_left_y, max_sx-sx, max_sy-sy), 3)

def draw_all_blocks(surface,grid):
    shape = grid.shape 
    for i in range(shape[0]):
        for j in range(shape[1]):
            color_pic = (255,255,255) if grid[i][j] else (0,4,0)
            pygame.draw.rect(surface, color_pic, (top_left_x + j*block_size, top_left_y + i*block_size, block_size, block_size), 0)

def draw_nparray(surface,grid):
    surface.fill((0,0,0))
    draw_all_blocks(surface,grid)
    draw_grid(surface, grid)
    pygame.display.update()
    
  
def main_game():
    win = pygame.display.set_mode((s_width, s_height))
    pygame.display.set_caption('Game of Life')
    
    grid = create_grid()
    pause , mouse_loc = False, None
    run = True
    while run:
        draw_nparray(win,grid)  
        if pause:
            grid,mouse_loc = change_array_based_on_mouse_clocks(grid,mouse_loc)
        else:
           grid = process_grid(grid)
           grid = add_rare_block_to_grid(grid)
           
        time.sleep(time_sleep)
        
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.display.quit()
                run = False
                break
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    pause = not pause
            if event.type == pygame.MOUSEBUTTONDOWN:
                mouse_loc = pygame.mouse.get_pos()
    

if  __name__ =="__main__":

    main_game()








