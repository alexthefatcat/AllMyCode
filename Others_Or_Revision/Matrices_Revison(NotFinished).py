# -*- coding: utf-8 -*-
"""
Created on Fri Jan 31 16:23:47 2020

@author: Alexm

Matrix and Linear Algerbra
"""

"""
>= means the output is



Matrix_A = [[ A, B],
            [ C, D] ]

Matrix_B = [[ E, F],
            [ G, H] ]

Matrix_A+Matrix_B >= [[ A+E, B+F],
                      [ C+G, D+H]

2 * Matrix_ A = [[ 2*A, 2*B],
                 [ 2*C, 2*D] ]     

Matrix_A*Matrix_B >= [[ (A*E+B*G), (A*F+B*H)],
                      [ (C*E+D*G), (C*F+D*H)]

            [[     E    ,     F    ],
             [     G    ,     H    ]]
            - - - - - - - - - - - - -
[[ A, B], | [[ (A*E+B*G), (A*F+B*H)],
 [ C, D]] |  [ (C*E+D*G), (C*F+D*H)]


 
                      




Matrix_rot = [[ 0, -1],
              [ 1,  0]]

Matrix_sca [[1,0],
            [0,1]]
# if you want to rotate Matrix_A

# Matrix_rot * Matrix_A => rotated matrix A

# Multipling by most matrixs changes most vectors
# direction

# If you multiply a vactor by a matrix and
# it scale is only changed it is known as a
# Eigevector and the scale factor is the
# Eigenvalue
##not every matrix has an eigenvalue

3x - 2y =1
-x +4y =3
# solve
mat=[[3  -2]
     [-1, 4]]

mat * [[x],[y]] = [[1],[3]]

inv_mat * [[1],[3]] = [[1],[1]]


Uses
   Electronic Circuits
   Markov Models
   Computer Graphics
   Imaging Processing# Convolutions
   Networks and Graphs
   Machine Learning + Neural Networks
   Quantum Mechanics
   
Markov Matrix Model Example
   markov(often probalistis,columns some to 1 no negative values)
    Humans 150, Zombies 150
    After a Hour
    0.2 Humans > Zombie # 0.8 stay same
    0.1 Zombie > Human  # 0.9 stay same
call first mat conversion, 2nd populations
[[0.8,0.1]
 [0.2,0.9]]    * [[150],[150]]
= [[135],[165]
basically a matrix of probablity of the next state


The other main type is Elemnt wise multiplication(Hadamard product)



The Eigenvector of this shows when the population is stable

Networks 
Dating Example
<-> both like
man1<->wom4
man1<->wom5
man1<->wom6
man2<->wom4
man2<->wom5
man3<->wom6
matrix of connection ->
#ordered persons 1-6
mat_connections>=
    0 0 0 1 1 1
    0 0 0 1 1 1
    0 0 0 0 0 1
    1 1 0 0 0 0
    1 1 0 0 0 0
    1 0 1 0 0 0 
#In graph theory and computer science, an adjacency matrix is a square matrix used to represent a finite graph.     
    

how many connections of lenght 2 connect persons 1 and 2
mat_connections ^2 # 
3 2 1 0 0 0 
2 2 0 0 0 0 
1 0 1 0 0 0
0 0 0 2 2 1
0 0 0 2 2 1
0 0 0 1 1 2
persons 1 and 2 have 2 conections
men1<->wom4<->men2
men1<->wom5<->men2
if you want all the conections of path length 3
mat_connections ^ 3(path lenght)





trace
   sum of the a11 a22 a33 


# determinates and eginevlaue caluation, Inverse



"""

 