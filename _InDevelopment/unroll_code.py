# -*- coding: utf-8 -*-
"""Created on Fri Sep 28 10:15:38 2018 @author: milroa1 """

___code ="""  fgdfsgd"""

FLAG_tripple_spech=False


def find_indexs_in_list(strin,strpat):
    from itertools import accumulate        
    lstrpat=len(strpat)
    v2 = [len(n) for n in strin.split(strpat)]
    v3 = list(accumulate(v2[:-1]))
    return [n+(i*lstrpat) for i,n in enumerate(v3)]


strin,strpat = "fdasdfsaadafwefasaadfpaaawefd" , "aa"
locs = find_indexs_in_list(strin,strpat)

for n in locs:
    try:
        print(strin[(n-1):(n+3)])
    except:
        pass

for ___i, ___line in enumerate(code.splitlines()):
    try:
        exec(___line)
    except:
        if "def " in ___line:
            pass


    
    
    
    