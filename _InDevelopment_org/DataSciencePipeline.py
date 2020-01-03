# -*- coding: utf-8 -*-
"""Created on Thu May  9 11:18:31 2019@author: milroa1"""


"""
Other One


# Question
# DATA
wrangle 
clean
explore>preprocess>model>>explore or
validate
tell the story
"""

# Where does feature enginerging fit it
# data mining
# desciptive stats are the gausian
# tring differnt models
# look at presnetation

Quick Overview
    obtain your data, clean your data,
    explore your data with visualizations,
    model your data with different machine learning algorithms,
    interpret your data by evaluation,
    and update your model.



Business Question
    So before we even begin the OSEMN pipeline, the most crucial and important step that we must take into consideration 
    is understanding what problem we’re trying to solve. Let’s say this again. Before we even begin doing anything with “Data Science”,
    we must first take into consideration what problem we’re trying to solve. If you have a small problem you want to solve, then at most
    you’ll get a small solution. If you have a BIG problem to solve, then you’ll have the possibility of a BIG solution.
    
    Ask yourself:    
        How can we translate data into dollars?
        What impact do I want to make with this data?
        What business value does our model bring to the table?
        What will save us lots of money?
        What can be done to make our business run more efficiently?
    
    Knowing this fundamental concept will bring you far and lead you to greater steps in being successful towards being a “Data Scientist”
     (from what I believe… sorry I’m not one!) But nonetheless, this is still a very important step you must do! No matter how well your model
     predicts, no matter how much data you acquire, and no matter how OSEMN your pipeline is… your solution or actionable insight will only be
     as good as the problem you set for yourself.

    “Good data science is more about the questions you pose of the data rather than data munging and analysis” — Riley Newman






O — Obtaining our data
S — Scrubbing / Cleaning our data
E — Exploring / Visualizing our data will allow us to find patterns and trends
M — Modeling our data will give us our predictive power as a wizard
N — Interpreting our data


Obtain Your Data
    You cannot do anything as a data scientist without even having any data. As a rule of thumb, there are some things you must take into
    consideration when obtaining your data. You must identify all of your available datasets (which can be from the internet or 
    external/internal databases). You must extract the data into a usable format (.csv, json, xml, etc..)
        Skills Required:
            Database Management: MySQL, PostgresSQL,MongoDB
            Querying Relational Databases
            Retrieving Unstructured Data: text, videos, audio files, documents
            Distributed Storage: Hadoops, Apache Spark/Flink 
 

Scrubbing / Cleaning Your Data
    Clean up on column 5! This phase of the pipeline should require the most time and effort. 
    Because the results and output of your machine learning model is only as good as what you put into it. 
    Basically, garbage in garbage out.
        Objective:
            Examine the data: understand every feature you’re working with, identify errors, missing values, and corrupt records
            Clean the data: throw away, replace, and/or fill missing values/errors
        Skills Required:   
            Scripting language: Python, R, SAS
            Data Wrangling Tools: Python Pandas, R
            Distributed Processing: Hadoop, Map Reduce / Spark
    
Exploring (Exploratory Data Analysis)
     Now during the exploration phase, we try to understand what patterns and values our data has.
     We’ll be using different types of visualizations and statistical testings to back up our findings.
     This is where we will be able to derive hidden meanings behind our data through various graphs and analysis. 
     Objective:
        Find patterns in your data through visualizations and charts
        Extract features by using statistics to identify and test significant variables
    Skills Required Python: Numpy, Matplotlib, Pandas, Scipy

Modeling (Machine Learning)
    Now comes the fun part. Models are general rules in a statistical sense.Think of a machine learning model as tools in your toolbox.
     You will have access to many algorithms and use them to accomplish different business goals. The better features you use the better
     your predictive power will be. After cleaning your data and finding what features are most important, using your model as a predictive
     tool will only enhance your business decision making.
     
     Predictive Power Example:
         One great example can be seen in Walmart’s supply chain. Walmart was able to predict that they would sell out all of their 
         Strawberry Pop-tarts during the hurricane season in one of their store location. Through data mining, their historical data
         showed that the most popular item sold before the event of a hurricane was Pop-tarts. As crazy it sounds, this is a true story
         and brings up the point on not to underestimate the power of predictive analytics.
    Objective:
        In-depth Analytics: create predictive models/algorithms
        Evaluate and refine the model
    Skills Required:
        Machine Learning: Supervised/Unsupervised algorithms
        Evaluation methods
        Machine Learning Libraries: Python (Sci-kit Learn) / R (CARET)
        Linear algebra & Multivariate Calculus

Interpreting (Data Storytelling)
     It’s story time! The most important step in the pipeline is to understand and learn how to explain your findings through communication.
     Telling the story is key, don’t underestimate it. It’s about connecting with people, persuading them, and helping them.
     The art of understanding your audience and connecting with them is one of the best part of data storytelling.

     Emotion plays a big role in data storytelling. People aren’t going to magically understand your findings. 
     The best way to make an impact is telling your story through emotion. We as humans are naturally influenced by emotions. 
     If you can tap into your audiences’ emotions, then you my friend, are in control. When you’re presenting your data, 
     keep in mind the power of psychology. The art of understanding your audience and connecting with them is one of the best part of data storytelling.
    
     Best Practice: A good practice that I would highly suggest to enhance your data storytelling is to rehearse it over and over.
     If you’re a parent then good news for you.Instead of reading the typical Dr. Seuss books to your kids before bed, try putting
     them to sleep with your data analysis findings! Because if a kid understands your explanation, then so can anybody, especially your Boss!

    Objective:
        Identify business insights: return back to business problem
        Visualize your findings accordingly: keep it simple and priority driven
        Tell a clear and actionable story: effectively communicate to non-technical audience
    Skills Required:
        Business Domain Knowledge
        Data Visualization Tools: Tablaeu, D3.JS, Matplotlib, GGplot, Seaborn
        Communication: Presenting/Speaking & Reporting/Writing

Update Model
    also when more data become avilabe its importnat to update model
    
    
    
    
    
