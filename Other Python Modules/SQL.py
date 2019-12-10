# -*- coding: utf-8 -*-
"""Created on Mon Dec  2 12:07:32 2019@author: Alexm"""


"""
################################################
  SQL 5 core cocepts
-------------------------------------------
SELECT  : Returning data from a single table
WHERE   : Filtering Records
JOIN    : join two tables
ORDER BY: sorting data
DISTINCT: returning unique values
--------------------------------------------
effeinct SQL/ EFFIECNT DATABASE PORGRAMMING

use square barckets around column names with spaces in them
#####################################################
"""





*SELECT 
FROM [ArticleSitesXpaths]

SELECT homesite,homename
FROM [ArticleSitesXpaths]
ORDER BY homesite ASC

SELECT *
FROM [ArticleSitesXpaths]
WHERE homesite LIKE '%hr%'








#----------------------------------------------------------------------
table_name1, table_name2 = 'movie', 'rating'

# Pandas
table1, table2 = db[table_name1], db[table_name2]
INNER, ON = 'inner', 'mov_id'
FROM_CLAUSE_WITH_JOIN = JOIN(table1, table2, INNER, ON)
SELECT_CLAUSE = ['mov_title', 'rev_stars']
rs = FROM_CLAUSE_WITH_JOIN[SELECT_CLAUSE]

# SQL
rs = con.execute(f'''SELECT {table_name1}.mov_title, 
                            {table_name2}.rev_stars
                     FROM {table_name1}  AS {table_name1}
                     INNER JOIN {table_name2} AS {table_name2}
                     ON {table_name1}.mov_id = {table_name2}.mov_id       ''')
#----------------------------------------------------------------------



"""SELECT * FROM df WHERE  col1 > 56  """
df.query("col1 > 56")











