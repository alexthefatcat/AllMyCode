# -*- coding: utf-8 -*-"""Created on Mon Jun 11 11:53:41 2018@author: milroa1"""
#%%
"""                      Expert Python                                                   """

#%%
# Decorators

def ntimes(n):
    def inner(f):
        def wrapper(*args,**kwargs):
            for _ in range(n):
                print("running {.__name__}".format(f))
                rv = f(*args,**kwargs)
            return rv
        return wrapper
    return inner

@ntimes(2)
def add(x,y=10):
    return(x+y)

print("add(10)",add(10))

#%% 
# Generators

def compute():
    for i in range(10):
        sleep(1)
        yield i 

#%%
meta classes
asyinco

#maybe class stuff
super 
insatance
inheritance
assert

context manager
inspect:-getsource, getfile

__defaults__
#hack python code

assert hasattr(derived , "bar")
#%%
#low expert
#dont run when not main script
if __name__ == "__main__":
    pass



#%% not expert    but advanced to intermidant      inheritiance saving memory
##########################################################################################################################################################    
class a:
    print("This runs only once on the first time the class is used")
    memory_global = 78
    import copy
    copy_obj = copy.copy
    def __init__(self, mem_unique, mem_shared):
        print(f"New Class Created where, mem_unique:{mem_unique},   mem_shared:{mem_shared}")
        self.memory_unique = mem_unique
        self.memory_shared = self.memory_shared_class(mem_shared)

    def __str__(self):
        return f"self.memory_shared(): {self.memory_shared()},\t self.memory_unique: {self.memory_unique},\t self.memory_global: {self.memory_global}"
    
    def __call__(self, call):
       if call =="memory":
           print("return memory_shared")
           return self.memory_shared
       else :
           print("CLONE")
           obj = a.copy_obj(self)
           obj.memory_unique = call
           return obj
        
    def change_global_memory(self,in_):
        a.memory_global = in_
        
    class memory_shared_class:
        def __init__(self,mem_shared):
            self.mem = mem_shared
        def __call__(self,call=None):
            if call is None :
                return self.mem
            else:
                self.mem = call
        def __str__(self):
            return f"memory_shared():\t{self.mem}"
    def print_update(examples,i,msg=""):# dont need self if it doest use it
        print("-"*8+str(i)+"-"*8+"  "+msg)
        for k,v in examples.items():
           print(f"\n\t examples[{k}] = ▾\n\t\t{v}")
        print("#"*120,"\n")  
##############################################################################################################################################  
print("#"*120,"\n")     
for i, msg in [(1, 'Create the first two'), (2, 'Clone'), (3, 'change shared mem'), (4, 'change global mem'), (5, 'mem seperate changed')] :
    if   i == 1:   examples={"1":a([5,4,9,9],[98,1,11]),"2":a([6,1,3,7,8],[78,12])}
    elif i == 2:   examples["2a"] = examples["2"](92)
    elif i == 3:   examples["2a"].memory_shared([9,9,9])
    elif i == 4:   examples["2"].change_global_memory(1)
    elif i == 5:
        mem = examples["2"]("memory")
        print(mem,"before mem changed")
        mem(1111)
        print(mem,"after mem changed")
    a.print_update(examples, i, msg)
##############################################################################################################################################





















