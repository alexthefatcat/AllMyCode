# -*- coding: utf-8 -*-
"""Created on Mon Nov 18 15:45:02 2019
@author: Alexm"""

# Dynamically add methods to a Object

if ("Dynamically add methods",True)[1]:
    
    class A:
        def __init__(self,name,age):
            self.name = name
            self.age = age
    a = A("monkeyman",(12,3))
    
    def printname(self):
        print(self.name)
    setattr(A, 'printname', printname)
    
    def calc_monthsold(self):
        self.months = (self.age[0]*12) +self.age[1]
    setattr(A, 'calc_monthsold', calc_monthsold)
    
    a.printname() 
    a.calc_monthsold() 
    a.months
    
    
    
class Foo():
    def __init__(self):
        self._bar = 0

    @property
    def bar(self):
        return self._bar + 5

fooy = Foo()
fooy.bar    




class Test:     
    def __getitem__(self, arg):
        return str(arg)*2
    def __call__(self, arg):
        return str(arg)*5
test = Test()

print(  test[0]          )
print(  test['kitten ']  )
# calling  __call__    obj()
print(  test(2)          )
print(  test('dog ')     )








