# -*- coding: utf-8 -*-
"""Created on Sat Aug  5 12:53:37 2017,@author: Alex"""

######################################################
##function template and example

m=54
def the_functions_name(var1,var2=2):# car2 has default variable
    global m#now function can change the m value
    hidden_variable=var1*var2
    return hidden_variable 

print( the_functions_name(1))
#######################################################
#######################################################
# a class is a blueprint to create a type of obeject#n instance
# instances are each seperate one
#indtancevariables areuniue class varibles are common

#inheritance

class classname(inputs1,inputs2,inputs3,inputs4=9):
    class_variable=1.5
    no_instances_created=0
    #class variables can be changed outside by classname.class_variable   
    #an single instances calss variable can be changed by examp2.class_variable=2...
    #hower in this instance self.class_variable does change
    
    def __init(self,*inputs2):#intialize - constrictor, this ran automaticaly
        self._class=inputs2  #self had the instance information
        self.input3=input3
        self.input4=input4
        classname.no_instances_created=classname.no_instances_created+1
        #this will have the same value amonst all instances at the same time
    def printinput3(self):#method
        return "{}".format(self.input3)
    def add_class_variable(self):
        self.input3=self.input3+self.class_variable
    #special method repr(object),used for debugging shows info can be used to recreate object
    def __repr__(self):
        return       "classname({},{})".format(self.input3, self.input4)
    #special method str(object), basic info to user
    def __str__(self):
        pass
    def __add__(self,other):#a+b objects how to act, dunner method
        return self.input2 + other.input2
    def __len__(self):# there is more may of these
        return len(self.input2)
###### Getters,Setteres and Deleters-methods that act like attributes
    @property
    def fullname(self):
        return "{},{}".format(self.first,self.last)
        
    @fullname.setter
    def fullname(self,name):
        first,last = name.split(" ")
        self.first = first
        self.last = last
        
    @fullname.deleter
    def fullname(self,name):
        first,last = name.split(" ")
        print("Delete")
        self.first = None
        self.last =  None
       
emp_1.fullname="Corey Schafer" 
#now with setters this attribute is changed as well as ones that follow on from it
  
examp1=classname(2,3,4)
examp2=classname(4,5,6)
examp2.add_class_variable()
print(classname.class_variable,examp2.class_variable)

print(examp2.printinput3())

__name__ #apperently get the name of the class of the object



#regular methods self
#class methods   cls   >can be used alterantively to create object(constructers)
#static methods   #nothing;they have some logical connection to the code
# if you dont use class or instance static method should be used

regular class method
@classmethod # decorator
def set_raise(cls,amount):#cls #cls class variable name
   cls.raise_amount=amount

@classmethod # decorator
def from_string(cls,strcls):
    a,b,c=strcls.split()
    return cls(a,b,c)

@staticmethod
def is_weekday(day):
   if day.weekday() == 5 or day.weekday() == 6
       

#### new class
class Developer(Emploee_class):
#inhertiance from empleeclass it cant find the __init__ so looks there
     raise_atm =1.10# changing this only effects the subclass but no affect in the parretn


#### new class_2
class Developer1(Emploee_class):
# allows you to reuse code easily
    def __init__(self,a,b,c,d,e=None)
       super().__init__(a,b,c)
       if e is None
          self.e=[]



class_name.set_raise(565)#all instances by the method(even if class instance name)

# inheritiance allows us two inhert attribures and methods
# useful to sub-class same functionality of parent and then add extra functionality
# help shows the info of this class


####
isinstance(mgr1,mgr)#tells you if it is an instance share inheritance to one another somehow
issubclass tells you if it inherites
##what does super
##as well as all built infuction relating to classes




### getattr delattr setattr
getattr(Person, 'name')

