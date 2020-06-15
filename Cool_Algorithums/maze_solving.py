# -*- coding: utf-8 -*-
"""Created on Fri Apr  3 23:08:53 2020@author: Alexm"""

import numpy as np
from matplotlib.pyplot import plot, draw, show, imshow
from scipy.signal import convolve2d

CONFIG = {"RUN GRAPH SEARCH ALGORITHUMS": False,
          "RUN A STAR MAZE SEARCH"      : False }

# Depth First Search  # graph one done
#  LIFO > Backtracking can be recursive /stack
#  it goes deep and it explores,
#  ABC,ACB,BAC,BCA,CAB,CBA

# Breadth First Search # graph one done
#   FIFO > path route map / queue
#   Try short routes first 1 strp out 2, step out
#   A,B,C,AB,AC,BA,BC,CA,CB,ABC,ACB,BAC,BCA,CAB,CBA

# Dijkstra
#   rember here the in the graph nodes have different distnaces between them
#   can be used for graphs you start at the start are try moving to neibouring nodes
#   nodes with shortest distance so far are p

# A-Star   # have a version half working
#    you also take into account how far away it is from the end

"""
### originally going to do this
# Map Image # maybe
# 0 Wall Not Possible
# 1 Not-Visited, 
# 2 Next, 
# 3 Visitied -Not Possible, 
# 4 Visitied -Possible
# 5 Start and End

############################################################
Stuff to Do

# a star graph
# convert maze to 2d to graph, using joint parts using convoultion
# binary and back tracking
# flood filling

# some functions in future to pass to graph search algorithums??
def get_neigbours(point):
    pass
def islegal(point):
    pass
def isend(point):
    pass
 



# there should really be a sorted array type
    
animation matplot lib line 85
"""

#%% --------------------------------------------------------------------

def add(a,b):
    return tuple([a_+b_ for a_,b_ in zip(a,b)]) 

def find_start_and_end(maze):
    shape = maze.shape
    rows = [(0, None),(shape[0]-1, None ),(None , 0),(None , shape[1]-1)]
    find_loc = lambda r:list(np.where(r==1)[0])
    out = []
    for x,y in rows:
        if x is None:
            loc = find_loc(maze[y,:])
            out.extend([(y,n) for n in loc])
        else:
            loc = find_loc(maze[:,x])
            out.extend([(n,y) for n in loc])
    return out

 
def imshow2(*args,**kwargs):
    imshow(*args,**kwargs,cmap = 'gray')
    show()

def animate_paths(maze,paths,animate=False):
    """
    %matplotlib qt
    animate_paths(maze_draw, paths)
    """
    from matplotlib import pyplot as plt
    from matplotlib import animation
    
    maze_draw = maze.copy() 
    
    point_type = type(paths[0][0]) in [float, int]
    if animate:
        for point in paths:  
            maze_draw[point]=2
        imshow2(maze_draw)
        return         
    
    fig = plt.figure()
    im  = plt.imshow(maze_draw,cmap = 'gray', vmin=0, vmax=3)#<<

    def loop(i):
    #for i in range(5):   
        points = paths[i]
        maze_draw[maze_draw==3] = 2
        if point_type:
            points = [points]
        for p in points:
            maze_draw[p] = 3
        im.set_data(maze_draw) #<<
        return im   
        
    anim = animation.FuncAnimation(fig, loop, frames=len(paths), interval=50,repeat = False)
    return anim




#%% Load Data (Create the maze and graphs)
print("Load Data")

class maze_graph:
    def __init__(self,maze):
        self.maze = maze.copy()
        self.xmax,self.ymax = maze.shape
        self.moves = [(0,-1),(1,0),(0,1),(-1,0)]    
    def __getitem__(self,arg):
       x,y = arg
       out = []
       for xd,yd in self.moves:
           x_new,y_new = x+xd,y+yd
           if self.xmax>x_new>=0 and self.ymax>y_new>=0:
               if maze[x_new,y_new]==1:
                  out.append((x_new,y_new))
       return out
    
def maze_to_array(maze_string):
   maze_lines =  [ [0 if char=="#" else 1 for char in line] for line in maze_string.splitlines()]
   maze_lines = [ line for line in maze_lines if len(line)>10]
   return np.array(maze_lines)
 
def convert_graph_string_to_dict(graph_string):
    def split_and_join_by_truth(part,truth="-1234567890",pnumeric = None):
        out = []        
        for s in part:
            numeric = s in truth
            if pnumeric == numeric:
                out[-1] = out[-1] + s
            else:
                out.append(s)   
            pnumeric = numeric
        out = [int(e) if e.isnumeric() else e for e in out]
        return out
    
    parts = graph_string.split("/")        
    connections = {}
    for part in parts:
        part0 = split_and_join_by_truth(part)
        for n1,length,n2 in zip(part0[0::2],part0[1::2],part0[2::2]):
            n2a,n1a = (n2,n1) if length == "-" else ((length,n2),(length,n1))  
            connections[n1] = connections.get(n1,[]) +[n2a]
            connections[n2] = connections.get(n2,[]) +[n1a]
    return connections

maze_string = """
############### ###############
# #                 #         #
# # ########### ### # ##### # #
# # #     #     #   #     # # #
# # # ### # ####### ##### # # #
#   # #   #       # #   # # # #
# ### # ### ##### ### # # # # #
#   # # #       # #   #   # # #
### # # ######### # ####### # #
# # # #           #     # # # #
# # # ####### ######### # # ###
#   #     # # #   #       #   #
# ##### # # # # # # ######### #
# # # # # #   # #     #     # #
# # # # # ##### ####### ### # #
#   # # #       #     # #   # #
##### # ######### ### # # ### #
#     # #       # #   # #     #
# ##### # ##### # # ### ##### #
# #     #   #   # #     #   # #
# ######### # ### ####### # # #
# #         #   #   #     # # #
# # ########### ### # ##### # #
# #   #       #   # #     # # #
# ### # ##### ### # # ### ### #
#     # #   #   #   # # #     #
####### # # # # ##### # #######
# #     # # # #     # #     # #
# # ##### # # ####### # ### # #
#         # #           #     #
##################### #########
"""   
#%% ----------------------------------------------------------------------        
print("Map Original")
maze_original = maze_to_array(maze_string)
maze = maze_original.copy()

maze_info={}
maze_info["start"], maze_info["end"] = find_start_and_end(maze)

imshow2(maze)

###########################################
#                  D                      #
#                 /|                      #
#                E-H-I C                  # 
#                |   | |                  #
#                A-B-F-G-F                #
#                       \|                #
#                       J                 #
############################################
         
graph_string1 = "A-E-H-I-F-B-A/E-D-H/C-G-J-F-G"            
graph1 = convert_graph_string_to_dict(graph_string1) 
graph_string2 = "A6B2D1A/D1E2B/B5C5E"
graph2 = convert_graph_string_to_dict(graph_string2)
graph3 = maze_graph(maze)


del maze_string,graph_string1,graph_string2

#%%----------------------------------------------------------------
print("Load graph search algorithums")


def create_get_neigbours(graph):
   def get_neigbours(s):
       return graph[s]
   return get_neigbours

def create_is_end(finish):
    def is_end(arg):
        return arg == finish
    return is_end

def breadth_first_search_graph(graph,start,finish,verbose=False,outvisited=False):
#        BFS:
#        list nodes_to_visit = {root};
#        while( nodes_to_visit isn't empty ) {
#          currentnode = nodes_to_visit.take_first();
#          nodes_to_visit.append( currentnode.children );
#          //do something }        
   is_end        = create_is_end(finish) 
   get_neigbours = create_get_neigbours(graph)
   visited = [start]
   queue   = [(start,start)]
   routes  = {start:[start]}

   while queue:
      parent,node = queue.pop(0) 
      children = get_neigbours(node)
      queue.extend([ (node,child) for child in children if child not in visited])

      if node not in visited:
         visited.append(node)
         routes[node] = routes[parent] + [node]
         if verbose:
            print("    -", routes[node] )
         if is_end(node):
            print(f"\nRoute Found: '{routes[node]}'\n") 
            if outvisited:
                return visited
            return routes[node] 
        
def depth_first_search_graph(graph,start,finish,verbose=False,outvisited=False):
#        DFS:
#        list nodes_to_visit = {root};
#        while( nodes_to_visit isn't empty ) {
#          currentnode = nodes_to_visit.take_first();
#          nodes_to_visit.prepend( currentnode.children );
#          //do something }
   is_end = create_is_end(finish)
   get_neigbours = create_get_neigbours(graph)        
   visited = [start]
   stack   = [(start,start)]
   routes  = {start:[start]}
   
   while stack:
     parent,node = stack.pop(-1) 
     children = get_neigbours(node)
     stack.extend([ (node,child) for child in children if child not in visited])

     if node not in visited:
        visited.append(node)
        routes[node] =  routes[parent] + [node]
        if verbose:
            print("    -", routes[node] )
     if is_end(node):
        print(f"\nRoute Found: '{routes[node]}'\n") 
        if outvisited:
            return visited
        return routes[node] 

def dijkstra_search_graph(graph,start,finish,verbose=False,outvisited=False):

   is_end = create_is_end(finish) 
   get_neigbours = create_get_neigbours(graph)       
   visited = {start:0}
   slist   = [(start,0,start)]
   routes  = {start:[start]}
   
   while slist:
     slist = sorted(slist,key=lambda x:x[1])  
     parent,distance,node = slist.pop(0) 
     children = get_neigbours(node)
     slist.extend([ (node,distance1+distance,child) for distance1,child in children ])

     if node not in visited or visited[node] > distance:
        visited[node] = distance
        routes[node] =  routes[parent] + [node]
        if verbose:
            print("    -", routes[node] )
     if is_end(node):
        print(f"\nRoute Found: '{routes[node]}'\n")
        if outvisited:
            return visited
        return routes[node] 

def astar_distance_maze_2d(maze,start,end,allout=False):
    xs,ys = np.meshgrid(range(31),range(31))
    dist_start = abs(xs-start[1]) + abs(ys-start[0])
    dist_finis = abs(xs-end[1])   + abs(ys-end[0])
    dist       = dist_start + dist_finis
    dist1      = dist * maze
    if allout:
       return dist1,dist_start,dist_finis
    return dist1






















#%%-----------------------------------------------------------------------------
if CONFIG["RUN GRAPH SEARCH ALGORITHUMS"]:
    
    print("RUN GRAPH SEARCH ALGORITHUMS")
    
    __start,__finish =  'A', 'D'       
    print("  >> Depth First Search")           
    route1 = depth_first_search_graph(   graph1,__start,__finish,True)    
    
    print("  >> Breadth First Search")        
    route2 = breadth_first_search_graph( graph1,__start,__finish,True)
    
    print("  >> Dijkstra Search Graph")
    __start, __finish =  'A', 'C' 
    route3 = dijkstra_search_graph( graph2,__start,__finish,True)
 
    print("  >> Depth First Search, on a converted 2D Map")       
    __start, __finish = maze_info["start"], maze_info["end"]
    route4 = depth_first_search_graph( graph3, __start, __finish,outvisited=True)
    animate_paths(maze,route4,animate=True)
    

#%%-----------------------------------------------------------------------------
if CONFIG["RUN A STAR MAZE SEARCH"]:
    
    print("RUN A STAR MAZE SEARCH")

    def a_star_2d_maze(maze,dist1,start,finish):
        moves = [(0,-1),(1,0),(0,1),(-1,0)] 
        def argmin_except_zero(arr):
            return list(zip(*np.where( arr==np.min(arr[np.nonzero(arr)]))))
        
        def colour_in_neigbours(maze_draw,point):
            for x2,y2 in moves:
                new_point = (point[0]+x2,point[1]+y2)
                try:
                    if maze_draw[new_point]==1:
                       maze_draw[new_point] =2
                except:
                    pass
            return maze_draw 
        
        out = []
        maze_draw = maze.copy()
        for i in range(1000):
            if i==0:
                point = start
            else:
                temp  = dist1 * (maze_draw==2)
                point = argmin_except_zero(temp)[0]   
            maze_draw[point] = 3
            maze_draw = colour_in_neigbours(maze_draw,point)
            out.append([point])
            if point==finish:
                break
        return out
    
    dist_astar = astar_distance_maze_2d(maze,maze_info['start'],maze_info['end'])
    route5     = a_star_2d_maze(maze,dist_astar,maze_info['start'],maze_info['end'])
    animate_paths(maze,route5)


 









 
 
    


 


 

 













  




















 

 
    
#%-----------------------------------------------------------------------------
class maze2d_functions:
    """
    maze_splits_numbered = maze2d_functions.find_junctions_in_maze_and_show_them(maze)
    """

    def conv2neig(maze,center=False):
        neib_conv = [[0,1,0],[1,0,1],[0,1,0]]
        if center:
            neib_conv[1][1]=1
        return convolve2d(maze,neib_conv, mode='same', boundary='fill', fillvalue=0)
        
    def find_splits_and_number_them(maze):
        mazee = maze2d_functions.conv2neig(maze)
        splits = np.logical_and(mazee>2,maze).astype(int)
        out = np.where(splits==1)
        splits_nos = np.zeros_like(maze)
        for i,(x,y) in enumerate(zip(*out)):
            splits_nos[x,y] = i
        return splits_nos

    def find_junctions_in_maze_and_show_them(maze):
        maze_splits_numbered = maze2d_functions.find_splits_and_number_them(maze)
        imshow(maze_splits_numbered)  
        for n in range(13):
            maze2 = maze2d_functions.conv2neig(maze_splits_numbered,True)
            maze3 = np.logical_and(maze,maze2>0).astype(int)
            maze_splits_numbered = maze3
            imshow(maze3)
      
 
