
#%%################################################################################################################################      
"""                 ***                                 NN FUNDERMENTAL                                  ***                   """       
################################################################################################################################### 
## remember a relu is max(x,0)  
                   
list()  
min()  max()    sum()
sorted()  reversed()  enumerate()   
range()    len()   
map()     filter()  
abs()      
divmod()   int()       round()   
all() any() and, or, xor not 
  
################################################
################################################

if elif else; while; for 

def             

lambda  del is  finally  continue  break 
################################################
################################################
#%%  list Sub-Operations
 in 
append pop insert  count index 
[:3] [3] [3:]  [2,3,4]  [-1] 

################################################

[&, |, ^, ]- [-,<=,=>] 

def add1(x):return(x+1)
x=0    
for n in range(7):
   x=add1(x)    
      
/   //   %
==
+  -  *   % 

################################################
(^-1) SIN COS TAN LOG   PI EXP GAUSSIAN ERROFUNCTION  INVERSE OF SOME OF THESE

CONVOLUTIONAL
multidemensioal arrays pytohn
count in a for loop
goto
complex()       zip()       bool()        
mean std standard error histograms binning gaussian distribtion
corrleation affine transfroms
pfor
reduce
################################################
pytohn problems
parellel 
goto
decleration
defaut type for multidmensional arrays same type
indexing
[1,2,3,4,5][1,3]# instead off
[ [1,2,3,4,5][n] for n in [1,3]]



resnet
relu
dropout
batchnormalization
#lstm
#maxpooling in cnn
#the hiercarl aracture in cnn
dynamic routing in capsule networks
################################################

print  (  list(map(lambda x,m:x*m,[5],[8]))  ,  list(map(lambda x:x[0]*x[1],[[5,8]]))  )

################################################

##  Two In  ##
a,b
relu(         a, b )
sorted(       a, b )
alargerb    ( a, b )
greaterthan0( a, b )

multi(        a, b )

relu2
aifb
bifa

divide(       a, b )
##################
exp() log()

##################
A = w1x1+w2x2+w3x3+b
B = w1x1+w2x2+w3x3+b


###############
func1  = lambda x:x**2
func1i = lambda x:x**(1/2)


#[  func2(  [x,i/(len(list__)+1)]  )   for x,i in enumerate(list__)  ]
#out = func1i(sum([func1(  ([x,i/(len(list__))+1])   for x,i in enumerate(list__)]))



def new_average(list_,func2=lambda x: x[0]*4*(0.5-abs(x[1]-0.5)) , div_len_correction=1 ):#an average that weights likE upsidedown V
    list_=sorted(list_)
    list_func=[ func2([    x/(len(list_)-div_len_correction),  i/(len(list_)-1)   ])  for i, x in enumerate(list_)]
    return(sum(list_func))
print(new_average([1,1,1,1,1,1,1,1,1,1,1,1,1]))
print(new_average([1,2,3,55,7,88,4,333,2,23]))
###############################



a [ X, X, X, X, X, X ]
b [ X, X, X, X, X, X ]
c [ X, X, X, X, X, X ]
d [ X, X, X, X, X, X ]
e [ X, X, X, X, X, X ]
f [ X, X, X, X, X, X ]
    A  B  C  D  E  F


[A * B]
[B if A>0]
[A if B>0]

[C if A>0]
[C if B>0]
[D if A>0]

[A if A>0]
[B if B>0]
[C if C>0]
[D if D>0]
[E if E>0]
[F if F>0]







def   # function repetition 
for   # iguess do on multiple ones
while # recurrent loop that breaks
if    # switch 
relu()
min()  max()    sum()
sorted()  
range()  #spatial encoding 
mean() meadian standard-deviation  
len()   
count()#weights
abs()   
   
divmod()   int()       round()   
all() any() and, or, xor not 

def add1(x):return(x+1)

[&, |, ^, ]- [-,<=,=>]       
/   //   %
==
+  -  *   % 

log exp






if genetic algroithums copying repertion:
collection of functions
#somehow in code look for repetetion and useful functions
Functions_dict={}
Functions_dict[1]#min

#%%################################################################################################################################## 
## Maybe a neural Netowork subblock

*things that may be useful location ?
-routing -sorting and moving data which is related to as well

for multiplication(a,b,c):return( (a/abs(a)) * min(abs(a*b),abs(c)),abs(a*b>c))
for division(a,b,c):return(min(a*b,c))

mean
count(n(x>0))
           
list()  
min()  max()    sum()
sorted()  reversed()  enumerate()   
range()    len()   
map()     filter()  
abs()      
divmod()   int()       round()   
all() any() and, or, xor not 

copy function
     x        y
I1  [1,4]  [1,4]
I2  [2,4]  [1,4]
I3  [3,4]  [1,4]
I4  [1,4]  [2,4]
I5  [2,4]  [2,4]
I6  [3,4]  [2,4]
I7  [1,4]  [3,4]
I8  [2,4]  [3,4]
I9  [3,4]  [3,4]

example I8
(relu(1-abs(2-xx))+relu(1-abs(4-xx))) * (relu(1-abs(3-yy))+relu(1-abs(4-yy)))

so for I458 (value,x,y)= 657.8,14,16
#xx,yy in
(relu(1-abs(x-xx))+relu(1-abs(4-xx))) * (relu(1-abs(x-yy))+relu(1-abs(4-yy)))
max(I001 : I999)=> [657.8,14,16] # MAX JUST USING TOP LAYER
# and when signal is sent back it can recreate it

# ALSO IF CORDINATED ARE ALTEABLE YOU CAN CREATE A DEFOMRABLE IMAGE
#advanced 
std
corr
#%%######################################################
"""   Basic Neural Network Equation(using small diffrences)          """
# x in; X out ; e in(error in); E out
# w1 normal wieght in matrix
# w2 used to calcualate the gradient dx
# w3 learning rate
# w4 wieght to zero
max(w1*(x+w3)+b,0)=X1
max(w1*(x-w3)+b,0)=X2
X=(X1+X2)/2
#diff=(X1-X2)/(2*w3)  ;  
E=((X1-X2)/(2*w3))*e
# I guess normally E=w2*e
w1 = w1+(w2*E) + (w4 if w1>0 else -w4)

maybe as well use mse(standard error)
## new activation function

# max(w1*(x+w3)+b,0)=> w1*(x+w3)+b if w1*(x+w3)+b>0 else 0 :::: but:: w1*(x2+w3)+b if w11*(x1+w3)+b>0 else 0 
# say here w2*(x1)+b1 => A1; w2*(x2)+b2 => A2
#also  ! star is 1-x   e.g. w2!= 1-w2
# new activation function


# (w1!+w1*A2)*A1 if (w2!*A1+w2*A2 ) else w3*(w1!+w1*A2)*A1+  w4term
# so in this example there is w1,w2,w3
# w1 multiply A1*A2
# w2 if A2 or A1
# w3 what else
# w4 term that when 0 means there isnt a step drop 



M1*A1+M2*A2+M3*A1*A2 if M4*A1+M5*A2+M6*A1*A2>2 else M7*A1+M8*A2+M9*A1*A2 
# so nine parameters
#%%######################################################
def layer(In,W):
  return(act_func(W*[In]+[1]))

out_3 = layer(layer(layer(In)))


#%%######################################################
""" Fractal Module Neurakl Network """

#%%

## this allows inter-lambda functions applies to two similar dataframes
"""   Elementwise combing two different dataframes   """


######################################
    
df1=HERD_df*2
df2=HERD_df
      
######################################
def func(x,y):
    print("p",x)
    return(3*x+y)

def create_func_conv(func):
    def func_conv(x):
        return(func(x[0],x[1]))
    return(func_conv)

func_conv = create_func_conv(func)  
#func_conv = create_func_conv(lambda x,y:x+y)  
 
#func_conv([2,5])
######################################

df12 = pd.concat([df1,df2], keys=[0, 1],axis=1)
df12_= df12.apply(func_conv,axis=1)  # lambda x:x[0]+x[1]
# this will have x[0] and x[1] which are series

######################################

pp=TOTAL.applymap(lambda x: str(type(x))[8:-2])
TOTAL.dtypes
######################################
def range2(*args): 
    if len(args)==1: args = (0,args[0],1)
    if len(args)==2: args = (args[0],args[1],1)      
    a,b,c = args        
    return([(c*n)+a for n in range(int((b-a)/c))])

######################################

** unpacks dictionaries.

This

func(a=1, b=2, c=3)
#is the same as
args = {'a': 1, 'b': 2, 'c':3}
func(**args)
######################################



















































