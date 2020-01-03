# -*- coding: utf-8 -*-
"""Created on Fri Oct 26 13:44:28 2018@author: milroa1"""
#
#           Wrapper Example
#


#########################################################################################
"How to use __setattr__  __getattr__  __getattribute__"

#  __getattr__ : basically if not in dir it goes to this 

"  Using  __getattribute__ and __setattr__ is difficult and can cause infinte recurssion"
"  however here is the default behavoior"

class Default(object):

    def __getattribute__(self,attr):               #obj.attr
        return object.__getattribute__(self, attr)
    def __setattribute__(self,attr,val):           #obj.attr = val
        object.__setattr__(self,attr, val)
        
    def __getitem__(self,item):                    #obj[item]  
        return self.data[item]
    def __setitem__(self, item, value):            #obj[item] = value
        self.data[item] = value
        
    def __call__(self,*args,**kwargs):             #obj(args)
        self.obj(self,*args,**kwargs)
    def __dir__(self):                             #dir(obj)
         return list(set(list(self._self.__dict__.keys()) + dir(self._self.__class__))) #dir(self.obj )

#__iter__ is the first to be called and returns the object to iterate over(often itself)
#__next__ is what iterates over the object and when stops raise StopIteration
# both need to exist for this and can use iter(obj_in) , next(obj) instead of for loop
    def __iter__(self):
            return self
    def __next__(self):  
        if self.str > self.end: 
            raise StopIteration 
        else: 
            self.str += 1
            return self.str
########################################################################################
    @property
    def y(self):# so acts like n.x instead of n.x()
        return self.data[1]











def print_object_info(obj):
    """
    prints out an objects attributes and methods
    print_object_info({1:4,66:777})
    """
    dirobj=dir(obj)
    
    import inspect
    from copy import deepcopy
    
    def limit(objstr,no=40):
        return str(objstr).ljust(no)[:no]
    
    out=[]
    print("\n Attributes in Object \n")
    
    for attr_meth_name in dirobj:
        attr_meth = getattr(obj, attr_meth_name)
        if  inspect.ismethod(attr_meth):
            print("   ",limit(attr_meth_name)," : ",limit(attr_meth))
        else:
            out = out + [attr_meth_name]  
    print("\n Methods in Object \n")             
    for meth_name in out:
        objcopy=deepcopy(obj)
        value  = getattr(objcopy, meth_name)
        try :
            value2 = value()
            if value2 is None:
                try :
                    print("   ",limit(meth_name)," : ",limit(objcopy),"#Object Now")#,attr_meth) 
                except:
                    print("Something Wrong Here")
            else :
                    print("   ",limit(meth_name)," : ",limit(value2),"#Returned")
        except:
            print("   ",limit(meth_name)," : ",limit("Method Requires Variables"))#,attr_meth)  

print_object_info([232,12])








# Example of getting attribute and setting one
class Foo(object):
    def __getattribute__(self, attr):
        print(f"getting attribute: {attr}")
        return object.__getattribute__(self, attr)

    def __setattr__(self, attr, val):
        print(f"setting attribute: {attr} to {val}")
        return object.__setattr__(self, attr, val)

#>   should be printed out

foo=Foo()
a=67

foo.a=a #> setting attribute: a to 67
a=foo.a #> getting attribute: a

###################################################################
###################################################################
        
class wrapper(object):
    def __init__(self,obj):
        self.Y = 12
        self.Z = 9  
        self.F =lambda x:x
        self._wrapped_obj = obj
    def __dir__(self):
        return dir(getattr(self, "_wrapped_obj"))
    def __setattr__(self,attr,val):
        if "_wrapped_obj" not in vars(self):
             super().__setattr__(attr, val)
        else : 
            obj = getattr(self, "_wrapped_obj")
            if attr in dir(obj):#vars(obj):vars(self):
                setattr(obj, attr, val) 
            else:
                super().__setattr__(attr, val)
    def __getattr__(self,attr):  
        return getattr(self._wrapped_obj, attr)
#    def __getattribute__(self,attr):
#        out = object.__getattribute__(self, attr)
#        print(attr,out)
#        return out


class wrapped:
    def __init__(self):
        self.a=1
        self.b=2
        self.c=3  
    def printme(self):
        print(f"a:{self.a}, b:{self.b}, c:{self.c}")


examp = wrapper(wrapped())

print([a for a in dir(examp) if not a.startswith("_")])

Y=examp.Y
b=examp.b
examp.b=90
b=examp.b
examp.printme()
dir(examp)




#From Example
#    def __setattr__(self, name, value):
#            super().__setattr__(name, value)#






