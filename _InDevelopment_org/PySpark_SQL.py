# -*- coding: utf-8 -*-
"""created on Mon Jan 29 11:31:11 2018@author: milroa1"""

#%%################################################################################################################################      
"""                            SQL in a Nushell                                         """      
###################################################################################################################################  
"""
A SQL query returns a table derived from one or more tables contained in a database.
Every SQL query is made up of commands that tell the database what you want to do with the data. 
The two commands that every query has to contain are SELECT and FROM.

** SELECT   ** FROM                                                                     

example 

SELECT * FROM my_table;

# * selects all columns, so this returns the entire table named my_table.

SELECT origin, dest, air_time / 60 FROM flights;

returns a table with the origin, destination, and duration in hours for each flight.

WHERE is also usefull/ in this case to only select students with As

SELECT * FROM students
WHERE grade = 'A';
#############################################################################################
 database task is aggregation.  GROUP BY

For example, if you wanted to count the number of flights from each of two origin destinations, you could use the query

SELECT origin, dest, COUNT(*) FROM flights

GROUP BY origin, dest;
GROUP BY origin;  # tells SQL that you want the output to have a row for each unique value of the origin column. 
The output will have a row for every combination of the values in origin and dest (i.e. a row listing each origin and destination that a flight flew to). There will also be a column with the COUNT() of all the rows in each group.

#AVG average   SELECT AVG(air_time) / 60 FROM flights


 """


########################################
"""   Basic Queries                 """
########################################
# Filter columns
" SELECT col1, col2 FROM table1"
" SELECT * FROM table1         "#all columns

# filter the rows
" WHERE col4=1 AND col5=2"

# aggregate the data
" GROUP by"

# limit aggregated data 
" HAVING count(*)>1"

# order of the results
" ORDER BY col2"
########################################
# Useful keywords for SELECTS
" DISTINCT       " # Return unique results 
" BETWEEN a AND b" # limit the range, the values can be numbers, text, or dates.
" LIKE           " # pattern search within the column text
" IN(a, b, c)    " # check if the value is contained among given
########################################

########################################
"""      Data Modification           """
########################################
"UPDATE table SET col1=1 WHERE col2=2"                                                    # update specific data with the WHERE clause
"INSERT INTO table1 (ID,FIRST_NAME,LAST_NAME) VALUES(1,'Rebel','Labs');"                  # insert values manually
"INSERT INTO table1 (ID,FIRST_NAME,LAST_NAME) SELECT id,last_name,first_name FROM table2" # or by using the result of a query

########################################
""" Views(this is a virtual table as a results of a query)  """
########################################
"""     CREATEVIEW view1 AS
        SELECT col1,col2
        FROM table1
        WHERE                       """
        
########################################
""" Reporting(Use aggregation functions)"""
########################################
"COUNT"   # Return the number of rows
"SUM"     # cumulate the values
"AVG"     # return the average for the group
"MIN/MAX" # smallest/ largest value

########################################
########################################













#%%################################################################################################################################      
"""                            Find the locations of the filepaths and input sheetnames                                         """      
###################################################################################################################################  













# Filter flights with a SQL string
long_flights1 = flights.filter("distance > 1000")

# Filter flights with a boolean column
long_flights2 = flights.filter(flights.distance > 1000)

# Examine the data to check they're equal
print(long_flights1.show())
print(long_flights2.show())

df.colName
.withColumn()
.select()

Spark variant of SQL's SELECT is the .select() method
The difference between .select() and .withColumn() methods is that .select() returns only the columns you specify, while .withColumn() returns all the columns of the DataFrame in addition to the one you defined.
df.select("col1", df.col2)


# Select the first set of columns
selected1 = flights.select("tailnum", "origin", "dest")

# Select the second set of columns
temp = flights.select(flights.origin, flights.dest, flights.carrier)

# Define first filter
filterA = flights.origin == "SEA"

# Define second filter
filterB = flights.dest == "PDX"

# Filter the data, first by filterA then by filterB
selected2 = temp.filter(filterA).filter(filterB)


# Define avg_speed
avg_speed = (flights.distance/(flights.air_time/60)).alias("avg_speed")

# Select the correct columns
speed1 = flights.select("origin", "dest", "tailnum", avg_speed)

# Create the same table using a SQL expression
speed2 = flights.selectExpr("origin", "dest", "tailnum", "distance/(air_time/60) as avg_speed")




[.min(), .max(),  .count() are GroupedData methods. 
df.groupBy().min("col").show()

# Find the shortest flight from PDX in terms of distance
flights.filter(flights.origin == "PDX").groupBy().min("distance").show()

# Find the longest flight from SEA in terms of duration
flights.filter(flights.origin == "SEA").groupBy().max("air_time").show()


# Average duration of Delta flights
flights.filter(flights.carrier == "DL").filter(flights.origin == "SEA").groupBy().avg("air_time").show()

# Total hours in the air
flights.withColumn("duration_hrs", flights.air_time/60).groupBy().sum("duration_hrs").show()

# Group by tailnum
by_plane = flights.groupBy("tailnum")

# Number of flights each plane made
by_plane.count().show()

# Group by origin
by_origin = flights.groupBy("origin")

# Average duration of flights from PDX and SEA
by_origin.avg("air_time").show()


# Import pyspark.sql.functions as F
import pyspark.sql.functions as F

# Group by month and dest
by_month_dest = flights.groupBy("month","dest")

# Average departure delay by month and destination
by_month_dest.avg("dep_delay").show()

# Standard deviation
by_month_dest.agg(F.stddev("dep_delay")).show()

############################################

# Examine the data
print(airports.show())

# Rename the faa column
airports = airports.withColumnRenamed("faa", "dest")

# Join the DataFrames
flights_with_airports = flights.join(airports, on="dest", how="leftouter")

# Examine the data again
print(flights_with_airports.show())

##############################################################
##############################################################
Machine Learning Pipelines

In the next two chapters you'll step through every stage of the machine learning pipeline, from data intake to model evaluation. Let's get to it!

At the core of the pyspark.ml module are the Transformer and Estimator classes. Almost every other class in the module behaves similarly to these two basic classes.

Transformer classes have a .transform() method that takes a DataFrame and returns a new DataFrame; usually the original one with a new column appended. For example, you might use the class Bucketizer to create discrete bins from a continuous feature or the class PCA to reduce the dimensionality of your dataset using principal component analysis.

Estimator classes all implement a .fit() method. These methods also take a DataFrame, but instead of returning another DataFrame they return a model object. This can be something like a StringIndexerModel for including categorical data saved as strings in your models, or a RandomForestModel that uses the random forest algorithm for classification or regression.

##############################################################

# Rename year column
planes = planes.withColumnRenamed("year","plane_year")

# Join the DataFrames
model_data = flights.join(planes, on="tailnum", how="leftouter")


.cast() works on columns, while .withColumn() works on DataFrames.
 you'll pass the argument "integer" and for decimal numbers you'll use "double"


model_data = model_data.withColumn("arr_delay", model_data.arr_delay.cast("integer"))




model_data = model_data.withColumn("plane_age", model_data.year - model_data.plane_year)



# Create is_late
model_data = model_data.withColumn("is_late", model_data.arr_delay > 0)

# Convert to an integer
model_data = model_data.withColumn("label", model_data.is_late.cast("integer")    )

# Remove missing values
model_data = model_data.filter("arr_delay is not NULL and dep_delay is not NULL and air_time is not NULL and plane_year is not NULL")















