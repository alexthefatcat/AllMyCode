# -*- coding: utf-8 -*-
"""Created on Fri Jan 31 20:09:43 2020
@author: Alexm
"""

# backtrack
# start at first paramater
# see if lowest value there is legal if not add one until legal
# then adjust next paramter
# if no value in the next paramter go back to the previous paramter and add one

def backtracker(board,values, board_exchange, find_locs_to_change, check_if_legal):
    board = [[a for a in b] for b in board] # prevent affefcting input
    locs_to_change = find_locs_to_change(board)
    cur_loc = 0    
    while len(locs_to_change)>cur_loc:
       ixiy   = locs_to_change[cur_loc]
       val    = board_exchange(board,ixiy)    
       valind = values.index(val)

       while True:
           if values[valind]==values[-1]:
              board   = board_exchange(board,ixiy, values[0])
              cur_loc -= 1
              break           
           valind += 1
           board = board_exchange(board,ixiy,values[valind])
           if check_if_legal(board):
              cur_loc += 1
              break
    return board
#[(0, 0), (1, 4), (2, 7), (3, 5), (4, 2), (5, 6), (6, 1), (7, 3)]
def copy(l):
    if type(l[0]) is list:
       return  [[a for a in b] for b in l] # deepcopy   
    return [a for a in l]

def create_recursize_backtrack(values, find_locs_to_change, board_exchange, check_if_legal):
    def ordered(board,values,fast=False):
        #print("board,values,fast=",board,",",values,",",fast)
        #assert False
        board_val_index = [values.index(p) for p in board]
        if fast:
           return all([p2>p1 or p2==0 for p1,p2 in zip(board_val_index,board_val_index[1:])])
        sboard_val_index = sorted(board_val_index)
        return sboard_val_index!=board_val_index, [values[i] for i in sboard_val_index]

    def recursive_backtrack(board,n=0,locs_to_change=None,all_solutions=None,sort=False):
        if all_solutions is True:
           all_solutions = []
        if   n == 0:
            locs_to_change = find_locs_to_change(board)
        elif n == len(locs_to_change):
            if all_solutions is None:
               return board # return final answer 
            else:
               sboard = copy(board)
               if sort:
                  if not ordered(board,values,fast=True):
                      return None 
                  print(f"{len(all_solutions)+1} Solutions Found")
               #print("sort=",sort)
               all_solutions.append(sboard)
               global mmm
               mmm = all_solutions
               return None
            
        board = copy(board) # deepcopy 
        
        for val in values[1:]:
           board = board_exchange(board, locs_to_change[n], val)
           ## having an option 
           ## in some sitatuoans the order of the values does not matter
           ## so skipping them if ordered it should spead up
#           if sort:
#               print(ordered(board,values,fast=True))
#               if not ordered(board,values,fast=True): 
#                  print(">>") 
#                  return None
             
           
           if check_if_legal(board):
               out = recursive_backtrack(board,n+1,locs_to_change,all_solutions=all_solutions,sort=sort) # try next space 
               if out is not None:
                   return out
        if n==0 and all_solutions not in [False, None]:
            return all_solutions
        if n==0:
           print("No Solution")           
        return  None # backtrack
    return recursive_backtrack

def create_suduko_solvers():   
    
    def check_if_legal(board,*_):
        """ This doesnt check if the numbers are within range, and 0s should always be legal  """
        numbers = list(range(1,10))
        board = [[ aa if type(aa) is int else aa[0] for aa in a] for a in board ]
        def block_check(block,numbers=numbers):
            return any([block.count(n)>1 for n in numbers])
        for row_col in board+list(zip(*board)):
            if block_check(row_col):
               return False
        for ix in [0,3,6]:
            for iy in [0,3,6]:
               box = [bb  for b in board[ix:ix+3] for bb in b[iy:iy+3]]
               if block_check(box):
                  return False 
        return True
    
    def board_exchange(board,ixiy,value=None): # exchange info with 
        if value is None:
            return board[ixiy[0]][ixiy[1]]
        board[ixiy[0]][ixiy[1]] = value
        return board 
    
    def find_locs_to_change(board):
        return [(ix,iy) for ix,row in enumerate(board) for iy,elem in enumerate(row) if elem==0]
    
    values         = [0,1,2,3,4,5,6,7,8,9]
 
    def suduko_solver__backtrack(board):
        return backtracker( board, values, board_exchange, find_locs_to_change, check_if_legal )    

    suduko_solver__recursivebacktrack = create_recursize_backtrack(values, find_locs_to_change, board_exchange, check_if_legal)

    return suduko_solver__recursivebacktrack, suduko_solver__backtrack

def create__queen_8_solver(num_queens=8,mode=1):
    def queen_legal(board,*_):
        queen_locs = [a for a in board if a !=-1 ]   
        lqueens  = len(queen_locs)
        if type(queen_locs[0]) is not tuple:
            queen_locs = [(i,e) for i,e in enumerate(queen_locs)]
        out  = [(ix,iy,ix-iy,ix+iy) for ix,iy in queen_locs  ]
        out2 = [len(set(n)) for n in zip(*out)]  
        for out3 in out2:
            if out3!=lqueens:
                return False      
        return True
    
    def find_locs_to_change(board):
        return list(range(num_queens))
    
    def board_exchange(board,ixiy,value=None): # exchange info with 
        if value is None:
            return board[ixiy]
        board[ixiy] = value
        return board 
    if mode==1:
       values = [-1]+[ (m,n) for n in range(num_queens) for m in range(num_queens)]
    if mode==2:
        
       values = [-1]+ list(range(num_queens))
    queen_8_solver = create_recursize_backtrack(values, find_locs_to_change, board_exchange, check_if_legal=queen_legal)
    return queen_8_solver

if __name__ == "__main__":

    board = [[0,0,0,0,0,2,0,0,0],
             [2,9,0,4,0,0,0,0,0],
             [0,0,4,6,5,0,0,0,0],
             [0,0,5,0,4,0,0,0,8],
             [8,0,0,0,0,5,6,0,0],
             [4,0,2,0,0,0,1,9,0],
             [0,5,0,3,0,4,9,0,0],
             [0,0,6,0,0,0,0,1,0],
             [0,0,0,8,6,0,0,4,0]]
    
    print("Create Sudukuo Solver")
    suduko_solver__recursivebacktrack, suduko_solver__backtrack = create_suduko_solvers()
    
    print("Solving Suduko ...")
    #board_solved1 = suduko_solver__recursivebacktrack(board)
    #board_solved2 = suduko_solver__backtrack(board)
    
    print("*Suduko Solved!!\n\n")
    
    board__solved = [[5,6,3,1,7,2,4,8,9],
                     [2,9,1,4,3,8,7,5,6],
                     [7,8,4,6,5,9,3,2,1],
                     [6,1,5,9,4,3,2,7,8],
                     [8,7,9,2,1,5,6,3,4],
                     [4,3,2,7,8,6,1,9,5],
                     [1,5,8,3,2,4,9,6,7],
                     [3,4,6,5,9,7,8,1,2],
                     [9,2,7,8,6,1,5,4,3]]    

    print("Create N Queen Solver")
    num_queens = 8
    queen_8_solver2 = create__queen_8_solver(num_queens,mode=2)
    
    board          = [-1 for _ in range(num_queens)] # these are the positions
    
    print("Solving N Queen ...")
#    queen_postionss = queen_8_solver2(board,all_solutions =True,sort=False)
#    queen_postionss = [[ie for ie in enumerate(s)] for s in queen_postionss]
#    queen_postions  = queen_postionss[0]
    
    print("Second Solver")
    queen_8_solver1 = create__queen_8_solver(num_queens,mode=1)
    queen_postionss = queen_8_solver1(board,all_solutions =True,sort=True)    
    queen_postionss = [[ie for ie in enumerate(s)] for s in queen_postionss]
    queen_postions  = queen_postionss[0]    
    print("*N Queen Solved!!\n\n")
    
    def get_chess_board(positions):
        if type(positions[0]) is list:
            positions = positions[0]
        mx = max([n[0] for n in positions])+1
        chess_board = [[0 for n in range(mx)] for m in range(mx)]
        for v in positions:
            chess_board[v[0]][v[1]] = 1
        return chess_board
    
    chess_board = get_chess_board(queen_postions)
    chess_board_solved = [[1, 0, 0, 0, 0, 0, 0, 0], 
                          [0, 0, 0, 0, 0, 0, 1, 0], 
                          [0, 0, 0, 0, 1, 0, 0, 0], 
                          [0, 0, 0, 0, 0, 0, 0, 1], 
                          [0, 1, 0, 0, 0, 0, 0, 0], 
                          [0, 0, 0, 1, 0, 0, 0, 0], 
                          [0, 0, 0, 0, 0, 1, 0, 0], 
                          [0, 0, 1, 0, 0, 0, 0, 0]]
 
 



