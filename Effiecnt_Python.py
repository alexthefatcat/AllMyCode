# -*- coding: utf-8 -*-
"""Created on Thu Dec 12 10:23:41 2019@author: Alexm"""

#Effeicnt porgramming Python

#	1. Searching in entity records

		records = [
		  {"id": 1, "name": "John", "surname": "Doe", "country": "ES"},
		  {"id": 2, "name": "Alice", "surname": "Bond", "country": "IT"},
		  {"id": 1000, "name": "Foo", "surname": "Bar", "country": "IN"}]
		ids = [1, 2, 50, 70, 87]

		# 0(n^2)
		for id in ids:
		  for r in records:
			if id == r['id']:
			  print(r)
			  break

		# 0(m)+0(n)
		records = {r['id']: r for r in records}
		for id in ids:
		  print(records.get(id))

#	2. Membership checks

		#0(n) * number of id's your looking for 
		70 in ids

		#0(n) + number of id's #  after you make it a set only 0(1) to search for more
		sids = set(ids)
		70 in sids

#	3. Improving a counter loop

		c = {}
		for r in records:
		  if r['country'] not in c: 
				c[r['country']] = 0
		  c[r['country']] += 1

		# if sparse this is a lot quicker
		c = {}
		for r in records:
		  try:
			c[r['country']] += 1
		  except KeyError:
			c[r['country']] = 1

		# what about mine
		c = {}
		for r in records:
		  c["country"] = c.get("country",0)+1
##################################################
          
          
          
          
          
          
          
          