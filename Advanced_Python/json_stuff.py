# -*- coding: utf-8 -*-
"""Created on Fri Nov 15 12:34:05 2019@author: Alexm"""

import json

"""
     Very Similar to Python Dicts but
        false is False,
        true is True,
        null is None

    json.load(f)   # Load JSON from a fileobj
    json.loads(s)  # Load JSON from a string
    json.dump(j,f) # Write JSON to a file
    json.dumps(j)  # output JSON obj as a string

"""




fp = "json file path"

# Read in JSON File
with open(fp,"r",encoding="utf-8") as json_file:
     data_dic = json.load(json_file)

# Write JSON file     
with open("new"+fp,"w",encoding="utf-8") as json_file:
     json.dump(json_file,data_dic)
     
     