# -*- coding: utf-8 -*-
"""Created on Fri Nov 15 11:43:25 2019@author: Alexm"""


#Logging

#levels Debug, Info, Warning, Error, Critical

import logging
# Create and configgure logger
LOGFILENAME = "QuadraticExquation_Logging_Examp.log"
LOG_FORMAT  = "%(levelname)s %(asctime)s - %(message)s"
logging.basicConfig(filename = LOGFILENAME,level = logging.DEBUG, format = LOG_FORMAT, filemode="w")
logger = logging.getLogger()

import math
# Test messages

#logger.debug("This is a harmless message")

def quaratic_formula(a,b,c):
    """  ax^2 +bx +c  """
    logger.info(f"quadrtaic_formula({a},{b},{c})")

    # Compute the discriminant
    logger.debug(" # Compute the discrimator")
    disc = b**2 - 4*a*c

    # Compute the two roots
    logger.debug(" # Compute the two roots")  
    root1 = (-b + math.sqrt(disc) ) / (2*a)
    root2 = (-b - math.sqrt(disc) ) / (2*a)

    # Return the roots
    logger.debug("# Return the roots")
    return root1,root2

roots = quaratic_formula(1,0,-4)
roots = quaratic_formula(1,0,1)







