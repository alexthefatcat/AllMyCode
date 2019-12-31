# -*- coding: utf-8 -*-
"""Created on Mon May 13 09:32:58 2019@author: milroa1"""


The anatomy of a SQL query
A SQL query consists of a few important keywords. Between those keywords, you add the specifics of what data, exactly, you want to see.
Here is a skeleton query without the specifics:

SELECT… FROM… WHERE…
GROUP BY… HAVING…
ORDER BY…
LIMIT… OFFSET…
###############################################################
import pandas as pd

airports = pd.read_csv('data/airports.csv')
airport_freq = pd.read_csv('data/airport-frequencies.csv')
runways = pd.read_csv('data/runways.csv')
###############################################################
SQL	Pandas

###############################################################
SELECT, WHERE, DISTINCT, LIMIT
###############################################################
airports
airports.head(3)
airports[airports.ident == 'KLAX'].id
airports.type.unique()

select * from airports	
select * from airports limit 3	
select id from airports where ident = 'KLAX'	
select distinct type from airport	

###############################################################
SELECT with multiple conditions
###############################################################
airports[(airports.iso_region == 'US-CA') & (airports.type == 'seaplane_base')]
airports[(airports.iso_region == 'US-CA') & (airports.type == 'large_airport')][['ident', 'name', 'municipality']]

select * from airports where iso_region = 'US-CA' and type = 'seaplane_base'	
select ident, name, municipality from airports where iso_region = 'US-CA' and type = 'large_airport'	

###############################################################
ORDER BY
###############################################################
airport_freq[airport_freq.airport_ident == 'KLAX'].sort_values('type')
airport_freq[airport_freq.airport_ident == 'KLAX'].sort_values('type', ascending=False)

select * from airport_freq where airport_ident = 'KLAX' order by type	
select * from airport_freq where airport_ident = 'KLAX' order by type desc	

###############################################################
IN… NOT IN
###############################################################
airports[airports.type.isin(['heliport', 'balloonport'])]
airports[~airports.type.isin(['heliport', 'balloonport'])]

select * from airports where type in ('heliport', 'balloonport')	
select * from airports where type not in ('heliport', 'balloonport')	

###############################################################
GROUP BY, COUNT, ORDER BY
###############################################################
airports.groupby(['iso_country', 'type']).size()
airports.groupby(['iso_country', 'type']).size().to_frame('size').reset_index().sort_values(['iso_country', 'size'], ascending=[True, False])

select iso_country, type, count(*) from airports group by iso_country, type order by iso_country, type	
select iso_country, type, count(*) from airports group by iso_country, type order by iso_country, count(*) desc	
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
airports.groupby(['iso_country', 'type']).size()
airports.groupby(['iso_country', 'type']).size().to_frame('size').reset_index().sort_values(['iso_country', 'size'], ascending=[True, False])

select iso_country, type, count(*) from airports group by iso_country, type order by iso_country, type	
select iso_country, type, count(*) from airports group by iso_country, type order by iso_country, count(*) desc	

###############################################################
HAVING
###############################################################
airports[airports.iso_country == 'US'].groupby('type').filter(lambda g: len(g) > 1000).groupby('type').size().sort_values(ascending=False)

select type, count(*) from airports where iso_country = 'US' group by type having count(*) > 1000 order by count(*) desc	

###############################################################
Top N records
###############################################################
by_country.nlargest(10, columns='airport_count')
by_country.nlargest(20, columns='airport_count').tail(10)

select iso_country from by_country order by size desc limit 10	
select iso_country from by_country order by size desc limit 10 offset 10	

###############################################################
Aggregate functions (MIN, MAX, MEAN)
###############################################################
runways.agg({'length_ft': ['min', 'max', 'mean', 'median']})

select max(length_ft), min(length_ft), mean(length_ft), median(length_ft) from runways	

###############################################################
JOIN
###############################################################
airport_freq.merge(airports[airports.ident == 'KLAX'][['id']], left_on='airport_ref', right_on='id', how='inner')[['airport_ident', 'type', 'description', 'frequency_mhz']]

select airport_ident, type, description, frequency_mhz from airport_freq join airports on airport_freq.airport_ref = airports.id where airports.ident = 'KLAX'	

###############################################################
UNION ALL and UNION
###############################################################
pd.concat([airports[airports.ident == 'KLAX'][['name', 'municipality']], airports[airports.ident == 'KLGB'][['name', 'municipality']]])

select name, municipality from airports where ident = 'KLAX' union all select name, municipality from airports where ident = 'KLGB'	

###############################################################
INSERT
###############################################################
df1 = pd.DataFrame({'id': [1, 2], 'name': ['Harry Potter', 'Ron Weasley']})
df2 = pd.DataFrame({'id': [3], 'name': ['Hermione Granger']})
**  There’s no such thing as an INSERT in Pandas. Instead, you would create a new dataframe containing new records, and then concat the two: **
pd.concat([df1, df2]).reset_index(drop=True)

create table heroes (id integer, name text);	
insert into heroes values (1, 'Harry Potter');	
insert into heroes values (2, 'Ron Weasley');	
insert into heroes values (3, 'Hermione Granger');	

###############################################################
UPDATE
###############################################################
airports.loc[airports['ident'] == 'KLAX', 'home_link'] = 'http://www.lawa.org/welcomelax.aspx'

update airports set home_link = 'http://www.lawa.org/welcomelax.aspx' where ident == 'KLAX'	

###############################################################
DELETE
###############################################################
lax_freq = lax_freq[lax_freq.type != 'MISC']
lax_freq.drop(lax_freq[lax_freq.type == 'MISC'].index)

delete from lax_freq where type = 'MISC'	




###############################################################
###############################################################
 Good SQL Programing
###############################################################
###############################################################
Adding an index wasn’t a bad idea; in fact, it’s one of the best ways to optimize query performance
Also filtering two large tables before joining them is faster


Table 1: [Citizen] citizen_id	last_name	first_name	birth_date	gender	address	height	ethnicity
Table 2: [Vehicle] plate_no	color	year	brand	model	owner_id
###############################################################
# Slow Query
SELECT first_name, last_name, address
FROM citizen JOIN vehicle ON citizen.citizen_id = vehicle.owner_id
WHERE year(birthdate) between 1978 and 1988 -- condition to check age in [30,40]
AND citizen.ethnicity = ‘caucasian’         -- condition to check ethnicity
AND vehicle.color = ‘green’                 -- condition to check car color
AND vehicle.brand = ‘Ford’                  -- condition to check car brand
AND vehicle.plate like ‘%853’               -- condition to check plate number

# Faster
1)
CREATE INDEX vehicle_i1 ON vehicle(owner_id);


2)
SELECT  * 
FROM        vehicles
WHERE   plate LIKE ‘%H%-853’    -- condition to check car color
AND         brand = ‘Ford’      -- condition to check car color
AND         color = ‘green’;        -- condition to check car color
## Filter before joining the two tables
3)
###############################################################    
WITH    (
SELECT  * 
FROM        vehicles
WHERE   plate LIKE ‘%H%-853’    -- condition to check car color
AND         brand = ‘Ford’      -- condition to check car color
AND         color = ‘green’;    -- condition to check car color
    ) as reduced_vehicle
SELECT  first_name, last_name, address
FROM        citizen, reduced_vehicle 
WHERE   citizen.citizen_id = reduced_vehicle.owner_id
AND         year(birthdate) between 1978 and 1988   -- condition to check age in [30,40]
AND         citizen.ethnicity = ‘caucasian’     -- condition to check plate number
###############################################################





SELECT LastName as SirName
FROM   Person.Person

SELECT UPPER(FirstName) + ' ' + UPPER(LastName) AS FullName
FROM   Person.Person

SELECT UnitPrice, OrderQty, 
       UnitPrice * OrderQty AS Total
FROM   Purchasing.PurchaseOrderDetail


SELECT   FirstName,
         LastName
FROM     Person.Person
ORDER BY LastName


# 10 to 15
SELECT   NationalIDNumber,
         JobTitle,
         HireDate
FROM     HumanResources.Employee
ORDER BY HireDate
         OFFSET 10 ROWS
         FETCH NEXT 5 ROWS ONLY




SELECT PurchaseOrderDetailID,
       ProductID,
       OrderQty,
       UnitPrice
FROM   Purchasing.PurchaseOrderDetail
WHERE  UnitPrice > 5.00
       AND OrderQty > 10
       AND UnitPrice * OrderQty <= 200





#first name starts with R but does not end in s
WHERE FirstName LIKE 'R%' AND FirstName NOT LIKE '%s'


7 Main data types in SQL : [INT,  VARCHAR, NVARCHAR,   DATETIME,   DECIMAL, FLOAT,   BIT]


SELECT JobTitle,
        CASE Gender
           WHEN 'M' THEN 'Male'
           WHEN '0' THEN 'Male'
           WHEN 'F' THEN 'Female'
           WHEN '1' THEN 'Female'
           ELSE 'Unknown Value'
        END
 FROM   HumanResources.Employee


VARCHAR popular is that values less than fifty characters take less space. The VARCHAR datatype can typically store a maximum of 8,000 characters.
NVARCHAR datatype is used to store Unicode text.  NVARCHAR columns can store a maximum of 4,000 characters.


between statement
    SELECT * from players
    where birthdate between "01/01/2018" and "12/31/2019"
in statment
    SELECT * from players
    where birthdtate in ("ID","WA","OR")
    #SELECT * FROM orders WHERE State="WA" or ...

# select only players from states which had players before 01/01/2010
# in ()brackets because of the distinct is a list
select * from players where birthstate in (select distinct birthstate from payers where
deathdate<"01/01/2010" and birthdtate is not null )

case statement
select namelast,namefirst,weight
    case when weight<150 then "light" # order of these is importnant
         when weight<200 then "Medium"
         else "Huge" end as Build


#####################################################################
ULTIMATE LIST OF 40 IMPORTANT SQL QUERIES
#####################################################################

1.   Query for Retrieving Tables
    SELECT * FROM My_Schema.Tables;
2.   Query for Selecting Columns from a Table
    SELECT Student_ID FROM STUDENT;
    SELECT * FROM STUDENT;
3.   Query for Outputting Data Using a Constraint
    SELECT EMP_ID, NAME FROM EMPLOYEE_TBL WHERE EMP_ID = '0000';
4. Query for Outputting Sorted Data Using ‘Order By’
    SELECT EMP_ID, LAST_NAME FROM EMPLOYEE
    WHERE CITY = 'Seattle' ORDER BY EMP_ID;
5.   Query for Outputting Sorted Data Using ‘Group By’
    SELECT Name, Age FROM Patients WHERE Age > 40 
    GROUP BY Age ORDER BY Name;
6. Data Manipulation Using COUNT
    SELECT COUNT(CustomerID), Country FROM Customers GROUP BY Country;
7. Data Manipulation Using SUM
    SELECT SUM(Salary)FROM Employee WHERE Emp_Age < 30;
8. Data Manipulation Using AVG
    SELECT AVG(Price)FROM Products;
9.   Query for Listing all Views
    SELECT * FROM My_Schema.views;
10. Query for Creating a View
    CREATE VIEW Failing_Students AS
    SELECT S_NAME, Student_ID
    FROM STUDENT
    WHERE GPA > 40;
11. Query for Retrieving a View
    SELECT * FROM Failing_Students;
12. Query for Updating a View
    CREATE OR REPLACE VIEW [ Product List] AS
    SELECT ProductID, ProductName, Category
    FROM Products
    WHERE Discontinued = No;
13. Query for Dropping a View
    DROP VIEW V1;
14. Query to Display User Tables
    SELECT * FROM Sys.objects WHERE Type='u'
15. Query to Display Primary Keys
    SELECT * from Sys.Objects WHERE Type='PK'
16. Query for Displaying Unique Keys
    SELECT * FROM Sys.Objects WHERE Type='uq'
17. Displaying Foreign Keys
    SELECT * FROM Sys.Objects WHERE Type='f'
18. Displaying Triggers
    SELECT * FROM Sys.Objects WHERE Type='tr'
19. Displaying Internal Tables
    SELECT * FROM Sys.Objects WHERE Type='it'
20. Displaying a List of Procedures
    SELECT * FROM Sys.Objects WHERE Type='p'
21. Swapping the Values of Two Columns in a table
    UPDATE Customers SET Zip=Phone, Phone=Zip
22. Returning a Column of Unique Values
    SELECT DISTINCT ID FROM Customers
23. Making a Top 25 with the SELECT TOP Clause
    SELECT TOP 25 FROM Customers WHERE Customer_ID<>NULL;
24. Searching for SQL Tables with Wildcards
    SELECT * From Customers WHERE Name LIKE 'Herb%'
25. Between Monday and Tuesday
    SELECT ID FROM Orders WHERE
    Date BETWEEN ‘01/12/2018’ AND ‘01/13/2018’
26. Finding the Intersection of Two Tables
    SELECT ID FROM Customers INNER
    JOIN Orders ON Customers.ID = Orders.ID
27. Doubling the Power with UNION
    SELECT phone FROM Customers 
    UNION SELECT item FROM Orders
28. Making Column Labels More Friendly
    SELECT Item AS item_description FROM Orders
29. Always and Everywhere!
    SELECT Item FROM Orders 
    WHERE id = ALL
    (SELECT ID FROM Orders
    WHERE quantity > 50)
30. Writing Developer Friendly SQL
    /* This query below is commented so it won't execute*/
    /*
    SELECT item FROM Orders 
    WHERE date ALL = (SELECT Order_ID FROM Orders
    WHERE quantity > 50)
    */
     
    /* the sql query below the will be executed 
    ignoring the text after "--"
    */
     
    SELECT item -- single comment  
    FROM Orders -- another single comment
    WHERE id 
    ALL = (SELECT ID FROM Orders
    WHERE quantity > 25)
31.  SQL queries for Database Management
    CREATE DATABASE AllSales
32. Adding Tables to Our New DB
    CREATE TABLE Customers (
    ID varchar(80),
    Name varchar(80),
    Phone varchar(20),
    ....
    );
33. Modifying and Deleting Tables with SQL
    ALTER TABLE Customers ADD Birthday varchar(80)
    DROP TABLE table_name
34. The Key to Successful Indexing
    CREATE TABLE Customers (
    ID int NOT NULL,
    Name varchar(80) NOT NULL,
    PRIMARY KEY (ID)
    );
36. Conditional Subquery Results
    SELECT Name FROM Customers WHERE EXISTS 
    (SELECT Item FROM Orders 
    WHERE Customers.ID = Orders.ID AND Price < 50)
37. Copying Selections from Table to Table
    INSERT INTO Yearly_Orders 
    SELECT * FROM Orders 
    WHERE Date<=1/1/2018
38. Catching NULL Results
    SELECT Item, Price * 
    (QtyInStock + IFNULL(QtyOnOrder, 0)) 
    FROM Orders
39. HAVING can be Relieving!
    SELECT COUNT(ID), Region
    FROM Customers
    GROUP BY Region
    HAVING COUNT(ID) > 0;
40. Tie things up with Strings!
    SELECT SUBSTRING_INDEX("www.bytescout.com", ".", 2);










35. Advanced Concepts For Improving Performance
Whenever practical, is always better to write the column name list into a SELECT statement
 rather than using the * delimiter as a wildcard to select all columns. SQL Server has to
 do a search and replace operation to find all the columns in your table and write them
 into the statement for you (every time the SELECT is executed). For example:
     
    SELECT * FROM Customers
Would actually execute much faster on our database as:
    SELECT Name, Birthday, Phone, 
    Address, Zip FROM Customers
    
Performance pitfalls can be avoided in many ways. For example, avoid the time sinkhole of
 forcing SQL Server to check the system/master database every time by using only a stored
 procedure name, and never prefix it with SP_. Also setting NOCOUNT ON reduces the time
 required for SQL Server to count rows affected by INSERT, DELETE, and other commands.
 Using INNER JOIN with a condition is much faster than using WHERE clauses with conditions.
 We advise developers to learn SQL server queries to an advanced level for this purpose. 
 For production purposes, these tips may be crucial to adequate performance. 
 Notice that our tutorial examples tend to favor the INNER JOIN.
 
 
 
 
 
###############################################################################################
##                     An Example of an Advanced Query                                       ## 
###############################################################################################
DECLARE @date DATETIME
SELECT @date = '10/31/09'

SELECT
      t1.EmpName,
      t1.Region,
      t1.TourStartDate,
      t1.TourEndDate,
      t1.FOrdDate,
      FOrdType  = MAX(CASE WHEN o.OrderDate = t1.FOrdDate THEN o.OrderType  ELSE NULL END),
      FOrdTotal = MAX(CASE WHEN o.OrderDate = t1.FOrdDate THEN o.OrderTotal ELSE NULL END),
      t1.LOrdDate,
      LOrdType  = MAX(CASE WHEN o.OrderDate = t1.LOrdDate THEN o.OrderType  ELSE NULL END),
      LOrdTotal = MAX(CASE WHEN o.OrderDate = t1.LOrdDate THEN o.OrderTotal ELSE NULL END)
  FROM 
      (--Derived table t1 returns the tourdates, and the order dates
      SELECT
            e.EmpId,
            e.EmpName,
            et.Region,
            et.TourStartDate,
            et.TourEndDate,
            FOrdDate = MIN(o.OrderDate),
            LOrdDate = MAX(o.OrderDate)
        FROM #Employees e INNER JOIN #EmpTours et
          ON e.EmpId = et.EmpId INNER JOIN #Orders o
          ON e.EmpId = o.EmpId
       WHERE et.TourStartDate <= @date
         AND (et.TourEndDate > = @date OR et.TourEndDate IS NULL)
         AND o.OrderDate BETWEEN et.TourStartDate AND @date
       GROUP BY e.EmpId,e.EmpName,et.Region,et.TourStartDate,et.TourEndDate
      ) t1 INNER JOIN #Orders o
    ON t1.EmpId = o.EmpId
   AND (t1.FOrdDate = o.OrderDate OR t1.LOrdDate = o.OrderDate)
 GROUP BY t1.EmpName,t1.Region,t1.TourStartDate,t1.TourEndDate,t1.FOrdDate,t1.LOrdDate 
############################################################################################### 
 
 





############################################################################################### 
SQL Quick Reference
############################################################################################### 
 




AND / OR	
    SELECT column_name(s)
    FROM table_name
    WHERE condition
    AND|OR condition
ALTER TABLE	
    ALTER TABLE table_name 
    ADD column_name datatype
    or
    ALTER TABLE table_name 
    DROP COLUMN column_name
AS (alias)	
    SELECT column_name AS column_alias
    FROM table_name
    or
    SELECT column_name
    FROM table_name  AS table_alias
BETWEEN	
    SELECT column_name(s)
    FROM table_name
    WHERE column_name
    BETWEEN value1 AND value2
CREATE DATABASE
    CREATE DATABASE database_name
CREATE TABLE	
    CREATE TABLE table_name
    (
    column_name1 data_type,
    column_name2 data_type,
    column_name3 data_type,
    ...
    )
CREATE INDEX	
    CREATE INDEX index_name
    ON table_name (column_name)
    or
    CREATE UNIQUE INDEX index_name
    ON table_name (column_name)
CREATE VIEW	
    CREATE VIEW view_name AS
    SELECT column_name(s)
    FROM table_name
    WHERE condition
DELETE	
    DELETE FROM table_name
    WHERE some_column=some_value
    or
    DELETE FROM table_name 
    (Note: Deletes the entire table!!)
    DELETE * FROM table_name 
    (Note: Deletes the entire table!!)

DROP DATABASE	
    DROP DATABASE database_name
DROP INDEX	
    DROP INDEX table_name.index_name (SQL Server)
    DROP INDEX index_name ON table_name (MS Access)
    DROP INDEX index_name (DB2/Oracle)
    ALTER TABLE table_name
    DROP INDEX index_name (MySQL)
DROP TABLE
	DROP TABLE table_name
EXISTS	
    IF EXISTS (SELECT * FROM table_name WHERE id = ?)
    BEGIN
    --do what needs to be done if exists
    END
    ELSE
    BEGIN
    --do what needs to be done if not
    END
GROUP BY	
    SELECT column_name, aggregate_function(column_name)
    FROM table_name
    WHERE column_name operator value
    GROUP BY column_name
HAVING	
    SELECT column_name, aggregate_function(column_name)
    FROM table_name
    WHERE column_name operator value
    GROUP BY column_name
    HAVING aggregate_function(column_name) operator value
IN	
    SELECT column_name(s)
    FROM table_name
    WHERE column_name
    IN (value1,value2,..)
INSERT INTO	
    INSERT INTO table_name
    VALUES (value1, value2, value3,....)
    or
    INSERT INTO table_name
    (column1, column2, column3,...)
    VALUES (value1, value2, value3,....)
INNER JOIN
    SELECT column_name(s)
    FROM table_name1
    INNER JOIN table_name2 
    ON table_name1.column_name=table_name2.column_name
LEFT JOIN	
    SELECT column_name(s)
    FROM table_name1
    LEFT JOIN table_name2 
    ON table_name1.column_name=table_name2.column_name
RIGHT JOIN	
    SELECT column_name(s)
    FROM table_name1
    RIGHT JOIN table_name2 
    ON table_name1.column_name=table_name2.column_name
FULL JOIN	
    SELECT column_name(s)
    FROM table_name1
    FULL JOIN table_name2 
    ON table_name1.column_name=table_name2.column_name
LIKE	
    SELECT column_name(s)
    FROM table_name
    WHERE column_name LIKE pattern
ORDER BY	
    SELECT column_name(s)
    FROM table_name
    ORDER BY column_name [ASC|DESC]
SELECT	
    SELECT column_name(s)
    FROM table_name
SELECT *	
    SELECT *
    FROM table_name
SELECT DISTINCT	
    SELECT DISTINCT column_name(s)
    FROM table_name
SELECT INTO	
    SELECT *
    INTO new_table_name [IN externaldatabase]
    FROM old_table_name
    or
    SELECT column_name(s)
    INTO new_table_name [IN externaldatabase]
    FROM old_table_name
SELECT TOP	
    SELECT TOP number|percent column_name(s)
    FROM table_name
TRUNCATE TABLE	
    TRUNCATE TABLE table_name
UNION	
    SELECT column_name(s) FROM table_name1
    UNION
    SELECT column_name(s) FROM table_name2
UNION ALL	
    SELECT column_name(s) FROM table_name1
    UNION ALL
    SELECT column_name(s) FROM table_name2
UPDATE	
    UPDATE table_name
    SET column1=value, column2=value,...
    WHERE some_column=some_value
WHERE	
    SELECT column_name(s)
    FROM table_name
    WHERE column_name operator value
































SQL Server String Functions
    Function	Description
    ASCII	Returns the ASCII value for the specific character
    CHAR	Returns the character based on the ASCII code
    CHARINDEX	Returns the position of a substring in a string
    CONCAT	Adds two or more strings together
    Concat with +	Adds two or more strings together
    CONCAT_WS	Adds two or more strings together with a separator
    DATALENGTH	Returns the number of bytes used to represent an expression
    DIFFERENCE	Compares two SOUNDEX values, and returns an integer value
    FORMAT	Formats a value with the specified format
    LEFT	Extracts a number of characters from a string (starting from left)
    LEN	Returns the length of a string
    LOWER	Converts a string to lower-case
    LTRIM	Removes leading spaces from a string
    NCHAR	Returns the Unicode character based on the number code
    PATINDEX	Returns the position of a pattern in a string
    QUOTENAME	Returns a Unicode string with delimiters added to make the string a valid SQL Server delimited identifier
    REPLACE	Replaces all occurrences of a substring within a string, with a new substring
    REPLICATE	Repeats a string a specified number of times
    REVERSE	Reverses a string and returns the result
    RIGHT	Extracts a number of characters from a string (starting from right)
    RTRIM	Removes trailing spaces from a string
    SOUNDEX	Returns a four-character code to evaluate the similarity of two strings
    SPACE	Returns a string of the specified number of space characters
    STR	Returns a number as string
    STUFF	Deletes a part of a string and then inserts another part into the string, starting at a specified position
    SUBSTRING	Extracts some characters from a string
    TRANSLATE	Returns the string from the first argument after the characters specified in the second argument are translated into the characters specified in the third argument.
    TRIM	Removes leading and trailing spaces (or other specified characters) from a string
    UNICODE	Returns the Unicode value for the first character of the input expression
    UPPER	Converts a string to upper-case

SQL Server Math/Numeric Functions
    Function	Description
    ABS	Returns the absolute value of a number
    ACOS	Returns the arc cosine of a number
    ASIN	Returns the arc sine of a number
    ATAN	Returns the arc tangent of a number
    ATN2	Returns the arc tangent of two numbers
    AVG	Returns the average value of an expression
    CEILING	Returns the smallest integer value that is >= a number
    COUNT	Returns the number of records returned by a select query
    COS	Returns the cosine of a number
    COT	Returns the cotangent of a number
    DEGREES	Converts a value in radians to degrees
    EXP	Returns e raised to the power of a specified number
    FLOOR	Returns the largest integer value that is <= to a number
    LOG	Returns the natural logarithm of a number, or the logarithm of a number to a specified base
    LOG10	Returns the natural logarithm of a number to base 10
    MAX	Returns the maximum value in a set of values
    MIN	Returns the minimum value in a set of values
    PI	Returns the value of PI
    POWER	Returns the value of a number raised to the power of another number
    RADIANS	Converts a degree value into radians
    RAND	Returns a random number
    ROUND	Rounds a number to a specified number of decimal places
    SIGN	Returns the sign of a number
    SIN	Returns the sine of a number
    SQRT	Returns the square root of a number
    SQUARE	Returns the square root of a number
    SUM	Calculates the sum of a set of values
    TAN	Returns the tangent of a number

SQL Server Date Functions
    Function	Description
    CURRENT_TIMESTAMP	Returns the current date and time
    DATEADD	Adds a time/date interval to a date and then returns the date
    DATEDIFF	Returns the difference between two dates
    DATEFROMPARTS	Returns a date from the specified parts (year, month, and day values)
    DATENAME	Returns a specified part of a date (as string)
    DATEPART	Returns a specified part of a date (as integer)
    DAY	Returns the day of the month for a specified date
    GETDATE	Returns the current database system date and time
    GETUTCDATE	Returns the current database system UTC date and time
    ISDATE	Checks an expression and returns 1 if it is a valid date, otherwise 0
    MONTH	Returns the month part for a specified date (a number from 1 to 12)
    SYSDATETIME	Returns the date and time of the SQL Server
    YEAR	Returns the year part for a specified date

SQL Server Advanced Functions
    Function	Description
    CAST	Converts a value (of any type) into a specified datatype
    COALESCE	Returns the first non-null value in a list
    CONVERT	Converts a value (of any type) into a specified datatype
    CURRENT_USER	Returns the name of the current user in the SQL Server database
    ISNULL	Return a specified value if the expression is NULL, otherwise return the expression
    ISNUMERIC	Tests whether an expression is numeric
    NULLIF	Returns NULL if two expressions are equal
    SESSION_USER	Returns the name of the current user in the SQL Server database
    SESSIONPROPERTY	Returns the session settings for a specified option
    SYSTEM_USER	Returns the login name for the current user
    USER_NAME	Returns the database user name based on the specified id




















