# -*- coding: utf-8 -*-
"""Spyder EditorThis is a temporary script file."""

def print_move(fr,to):
   print(f"mode from {fr} to {to}")
moves=[]
def towers(n,fr,to,spare,moves):

   if n==1:
      print_move(fr,to)
      
      moves = moves.append([fr,to])
   else:
      #if you want to move 4 piled discs,
         #move top 3 to spare
            # this is recursive and runs the same function
         #move bottem to where you want to go
         #move top 3 to where you want to go
            # this is recursive and runs the same function
      towers(n-1,fr ,spare,to  ,moves )
      towers(1  ,fr ,to   ,spare,moves)
      towers(n-1,spare,to  ,fr ,moves)



class tower_img:
      def  __init__(self,n):
          self.n = n
          self.tower=[   list(reversed(range(1,n+1))), [], [] ]
      def move(self,f,t):
           print(self.tower)
           ff=int(f[-1]) -1
           tt=int(t[-1]) -1
           temp = self.tower[ff][-1]
           self.tower[ff] = self.tower[ff][:-1]
           self.tower[tt] = self.tower[tt]+ [temp]
           
      def print_state(self):
           n = self.n
           h=n*[0]
           discs = [ [(self.tower[n]+h)[m] for n in [0,1,2]] for m in range(n)]
           def draw(v,n):
              im = " "*(n-v)+"-"*v +"|"+"-"*v +" "*(n-v)
              return im
           out = "\n".join([ " ".join([draw(t,self.n) for t in ts   ]) for ts in reversed(discs) ])
           print(out)
           
towers(4,"t1","t2","t3",moves)           
v = tower_img(4)
for m in moves:
    v.print_state()    
    v.move(*m)
    print("+++++++++++++++++++++++++++++++")
v.print_state()    
########################################################
           
           
# add pygame extension
           
           
           
           
           
           
           
           