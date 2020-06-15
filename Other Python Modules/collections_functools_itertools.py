# -*- coding: utf-8 -*-
"""
Created on Mon May  4 22:13:53 2020@author: Alexm


Most useful stuff from collections functools and itertools



"""
def equals(*args):
    if 2>=len(args):
       a = args[0]
       b = True if len(args)==1 else args[1]
       if a!=b:
          raise "Error dont equal"
          
E = equals

#%%############################################################################
#                       Most Useful
###############################################################################
from collections import Counter, namedtuple


# Counter like value counts
# namedtuple good output of a function
#
# count like while but counts
# product can combine mulit nested forloops
# zip_longest like zip but doesnt stop on shorted
# accumalate like intergram , but can put in function as 2nd arg
# 
# partial partially fill some values in a function
# reduce how previous element and next element in a list interact often int out
# lru_cache context manager if output has been calculated previously use that






out = dict(Counter([1,1,1,1,3,4,5,6,6,6]))
E(out == {1: 4, 3: 1, 4: 1, 5: 1, 6: 3})

#---------------------------------------------------------------
Point = namedtuple('Point', ['x', 'y'] )
p = Point(11, y=22)     # instantiate with positional or keyword arguments

E(  p[0]+p[1] == 33)
x, y = p #  x, y == (11, 22)

E(  (x,y)     == (11,22) )
E(  p.x+p.y   ==   33  )
E(  p         == Point(x=11, y=22) )
E(  p._fields == ('x', 'y')  )
E(  dict(p._asdict()) == {'x': 11, 'y': 22}  )
E(  p.x+p.y   == 33)
#-------------------------------------------------------

from itertools import count,product,zip_longest,accumulate

xs,ys,zs = range(10),range(10),range(10)

old = [(x,y,z) for x in xs for y in ys for z in zs ]
new = [(x,y,z) for x,y,z in product(xs,ys,zs)]
E( old == new )

out = list(zip_longest(["a","b","c"], ["A","B"]))
E( out == [('a','A'), ('b','B'), ('c',None)])

out = list(accumulate([1, 2, 3, 4, 5]))# can use another function as second input
E(  out == [1, 3, 6, 10, 15]  )

# count like a for loop but infinite, so like a while loop
for n in count():
    if n==90:
        break

#-------------------------------------------------------
from functools import partial , reduce, lru_cache
  
# A normal function 
def example_f(a, b, x, c=1): 
    return 1000*a + 100*b + 10*c + x 
example_l = [1,2,3,4,5,6]
  
g = partial(example_f, 1, 2, c=4) 
E( g(5) == 1245 ) 

out = reduce(lambda x,y:x*y, example_l)
E( out == 720 ) # sum(example_l)
#####################################

# if it has been calcualted previosly use that
@lru_cache(maxsize=None)
def fib(num):
    pass
#
    






#%%############################################################################
#                          Collections
###############################################################################
#namedtuple()#factory function for creating tuple subclasses with named fields
#deque#list-like container with fast appends and pops on either end
#Counter#dict subclass for counting hashable objects
#OrderedDict#dict subclass that remembers the order entries were added
#defaultdict#dict subclass that calls a factory function to supply missing values

from collections import Counter, namedtuple, deque

#---------------------------------------------------
cnt = Counter()
for word in ['red', 'blue', 'red', 'green', 'blue', 'blue']:
     cnt[word] += 1
     
out = Counter('abracadabra').most_common(3)
E(out, [('a', 5), ('b', 2), ('r', 2)])

out = dict(Counter([1,1,1,1,3,4,5,6,6,6]))
E(out, {1: 4, 3: 1, 4: 1, 5: 1, 6: 3})
#-----------------------------------------------------
d = deque(['g','h','i'])

d.append('j')                    # add a new entry to the right side
d.appendleft('f')                # add a new entry to the left side
E( d == deque(['f','g','h','i','j']) )
                          # return and remove the rightmost item
E(  d.pop()     == "j" )
E(  d.popleft() == "f" )                   # return and remove the leftmost item

#--------------------------------------------------------

Point = namedtuple('Point', ['x', 'y'], verbose=True)
p = Point(11, y=22)     # instantiate with positional or keyword arguments
p[0] + p[1]#>33
x, y = p #  x, y == (11, 22)
p.x + p.y   # 33
p#Point(x=11, y=22)
p._fields # ('x', 'y')
dict(p._asdict()) # {'x': 11, 'y': 22}








#%%############################################################################
#                       Itertools
###############################################################################



from itertools import count,cycle,permutations,combinations,product,zip_longest,accumulate

l1,l2 = [],[]

xs,ys,zs = range(10),range(10),range(10)
for x in xs:
    for y in ys:
        for z in zs:
            l1.append((x,y,z))

for x,y,z in product(xs,ys,zs):
     l2.append((x,y,z))

E( l1 == l2 )


#product      # order doesnt matter, can duplicate self
#permutations # order doesnt matter, can not duplicate self
#combinations # order does matter  , can not duplicate self

A1 = product('ABCD', repeat=2)
# A1 --> AA AB AC AD BA BB BC BD CA CB CC CD DA DB DC DD

A2 = permutations('ABCD', 2)
# A2 --> AB AC AD BA BC BD CA CB CD DA DB DC

A3 = combinations('ABCD', 2)
# A3 --> AB AC AD BC BD CD

#combinations_with_replacement # is like combinations but AA BB can appear


x = [1, 2, 3, 4]
y = ['a', 'b']
z1= list(zip(x, y))         # --> [(1, 'a'), (2, 'b')]
#fillvalue=None # default
z2= list(zip_longest(x, y)) # --> [(1, 'a'), (2, 'b'), (3, None), (4, None)]


list(accumulate([1, 2, 3, 4, 5])) # --> [1, 3, 6, 10, 15]


out = list(accumulate([1,2,3,4,5],lambda x,y:x+y))

# count like a for loop but infinite, so like a while loop
for n in count():
    if n==90:
        break

out = []
letters = cycle("ABC")
for i,a in enumerate(letters):
    out.append((i,a))
    if i==5:
        break
     
E( out ==  [(0,'A'),(1,'B'),(2,'C'),(3,'A'),(4,'B'),(5,'C')] )        
    
out = []
for a,b in zip(letters,range(6)):
    out.append((a,b))
 
E( out ==  [('A',0),('B',1),('C',2),('A',3),('B',4),('C',5)] )     



#%%############################################################################
#                       Functools
###############################################################################

from functools import partial , reduce, lru_cache
  
# A normal function 
def f(a, b, x, c=1): 
    return 1000*a + 100*b + 10*c + x 
  
g = partial(f, 7, 6, c=4) 
E( g(5) == 7645 )

#####################################
lis = [1,2,3,4,5,6]
out = reduce(lambda x,y:x*y, lis)
#####################################

# if it has been calcualted previosly use that
@lru_cache(maxsize=None)
def fib1(num):
    if num < 2:
        return num
    else:
        return fib1(num-1) + fib1(num-2)
    
def fib2(num):
    if num < 2:
        return num
    else:
        return fib2(num-1) + fib2(num-2)

# this one with 70 would crash
fib2(35)
fib1(70)

 
 
    
    
    
	