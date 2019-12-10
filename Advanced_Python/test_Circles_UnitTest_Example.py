# -*- coding: utf-8 -*-
"""Created on Fri Nov 15 09:08:23 2019@author: Alexm"""

import unittest
from Circles import circle_area
from math import pi

# Check some values should match, values are limited in range and type should be
# python -m unittest

class TestCircleArea(unittest.TestCase):
    def test_area(self):
      # Test areas when radius >= 0
      self.assertAlmostEqual(circle_area(1  ),pi)
      self.assertAlmostEqual(circle_area(0  ),0 )
      self.assertAlmostEqual(circle_area(2.1),pi*2.1*2)
      self.assertAlmostEqual(circle_area(2.1),pi*2.1*2.1)      
      
    def test_values(self):
        # Make sure value errors are raised when necessary
        self.assertRaises(ValueError, circle_area, -2)
    
    def test_types(self):
        # Make sure type errors are raised when necessary
        self.assertRaises(TypeError, circle_area, 3+5j)
        self.assertRaises(TypeError, circle_area, True)
        self.assertRaises(TypeError, circle_area, "Radius")

"""

import Circles 

# Normally

class Test_scriptname(unittest.TestCase):
    def test_functionname1():
       self.assertAlmostEqual( Circles.functionname1(2.1),pi*2.1*2.1)      
       self.assertRaises(ValueError, Circles.functionname1, -2  )
       self.assertRaises(TypeError, Circles.functionname1, 3+5j)         
     def test_functionname2():
  
       
########################################################################
# unit test testing a class   
#   where
#       obj = NewClass("Examp1")       

from Script import NewClass
                 
class TestNewClass(unittest.TestCase):

   @classmethod
   def setUpClass(cls): # This runs once but first
       print("setupClass")

   @classmethod
   def tearDownClass(cls): # This runs once but last
       print("setupClass")
   ##############################################
   def setUp(self): # ran before each of the tests
       self.obj1 = NewClass("Examp1")
       self.obj2 = NewClass("Examp2")
       
   def tearDown(self): # ran after eachof the tests
       pass
        
   # The tests dont neccasry run in order
   def test_attr1(self):
       self.assertEqual(self.obj1.attr1,"Examp1")
       self.assertEqual(self.obj2.attr1,"Examp2")

   def test_attr2(self):
       self.assertEqual(self.obj1.attr2,"1")
       self.assertEqual(self.obj2.attr2,"2")
         
         
         
         
         
         
         
         
         
"""
















if __name__ =="__main__":
    unittest.main()
    
    
    
    
    
    
    
    