# -*- coding: utf-8 -*-
"""Created on Fri Nov 22 12:56:59 2019@author: Alexm"""

Variable        : caseandseparated_with_underscores
function names  : caseandseparated_with_underscores
Named constants : ALL_CAPITAL_LETTERS
Classes         : CamelCase
Class Method    : caseandseparated_with_underscores 
Module          : my_module.py # caseandseparated_with_underscores # BUT KEEP NAMES SHORT
Package         : mypackage    # lowercasenounderscores            # BUT KEEP NAMES SHORT

Module	Use a short, lowercase word or words. Separate words with underscores to improve readability.	module.py,
Package	Use a short, lowercase word or words. Do not separate words with underscores.	package, 




Using a NAMED_CONSTANT defined in a single place makes changing the value easier and more consistent.
functions at the top of code

The variable name must describe the information represented by the variable. A variable name should tell you specifically in words what the variable stands for.
Your code will be read more times than it is written. Prioritize how easy your code is to understand rather than how quickly it is written.
Adopt standard conventions for naming so you can make one global decision instead of multiple local decisions.

X and y. If you’ve seen these several hundred times, you know they are features and targets, but that may not be obvious to other developers reading your code. Instead, use names that describe what these variables represent such as house_features and house_prices.
value. What does the value represent? It could be a velocity_mph, customers_served, efficiency, revenue_total. A name like value tells you nothing about the purpose of the variable and is easily confused.
temp . Even if you are only using a variable as a temporary value store, still give it a meaningful name. Perhaps it is a value where you need to convert the units, so in that case, make it explicit:

First, decide on common abbreviations: avg for average, max for maximum, std for standard deviation and so on. Make sure all team members agree and write these down.
Put the abbreviation at the end of the name. This puts the most relevant information, the entity described by the variable, at the beginning.  
    
velcoity_mean
velocity_std
velocity_max
velocity_mean

user_info


class MyClass(object):
    def __init__(self, param='some_value'):
        pass

    def public(self):
        'User, this public method is for you!'
        return 'public method'

    def _indicate_private(self):
        return 'private method'

    def __pseudo_private(self):
        return 'really private method'








styles of coding
  functional
  procudral
  class
  mix?
functions as the top ?
functions very descriptative
sectional programming
  

##############################################################################
def two_times(n):
    return 2*n
lis = [1,2,3,4,5,6]
lis_trans1 = [two_times(i) for i in lis]
lis_trans2 = list(map(two_times, lis))
def nmap(arg,func):
    """  Nested Version of list(map(nested_container,func))"""
    pass
##############################################################################    

functions at the top

when to use classes
   bundle of mutuable data
with assicated functions


data
config
foo
bar
temp
main
run
var


NamingConventionsForML
  train test
  x y YY real actual
  pred_probss
  pred_probs
  pred_real
  xContent
  yTopic
  ypTopic
  probability
  
filepaths filneame obj_fp

 
x__house_features 
y__house_prices
yp__houseprices





dataframe
url_name_df

Pep8


##########################################################################################################################################
#bad
for i in range(n):
    for j in range(m):
        for k in range(l): 
            temp_value = X[i][j][k] * 12.5
            new_array[i][j][k] = temp_value + 150

##########################################################################################################################################

PIXEL_NORMALIZATION_FACTOR = 12.5
PIXEL_OFFSET_FACTOR        = 150

for row_index in range(row_count):
    for column_index in range(column_count):
        for color_channel_index in range(color_channel_count):
            normalized_pixel_value = ( original_pixel_array[row_index][column_index][color_channel_index]* PIXEL_NORMALIZATION_FACTOR)
            transformed_pixel_array[row_index][column_index][color_channel_index] = ( normalized_pixel_value + PIXEL_OFFSET_FACTOR )
##########################################################################################################################################

PIXEL_NORMALIZATION_FACTOR = 12.5
PIXEL_OFFSET_FACTOR        = 150

def transform_pixel(org_pixel):
    norm_pixel = org_pixel * PIXEL_NORMALIZATION_FACTOR
    return norm_pixel + PIXEL_OFFSET_FACTOR   

for i__row_index in range(row_count):
    for j__column_index in range(column_count):
        for k__color_channel_index in range(color_channel_count):
            org_pixel   = original_pixel_array[i__row_index][j__column_index][k__color_channel_index]
            transformed_pixel_array[i__row_index][j__column_index][k__color_channel_index]   = transform_pixel( org_pixel  )

##########################################################################################################################################










instead of building_num
 use:
     building_count # total number of buildings / item_count
     building_index # specific building.        / item_index


for building_index in range(building_count):

for row_index in range(row_count):
    for column_index in range(column_count):

        
        
More Names to Avoid
    Avoid using numerals in variable names
    Avoid commonly misspelled words in English
    Avoid names with ambiguous characters
    Avoid names with similar meanings
    Avoid abbreviations in names
    Avoid names that sound similar






