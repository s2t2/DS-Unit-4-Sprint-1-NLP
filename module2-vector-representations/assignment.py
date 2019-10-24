#!/usr/bin/env python
# coding: utf-8

# <img align="left" src="https://lever-client-logos.s3.amazonaws.com/864372b1-534c-480e-acd5-9711f850815c-1524247202159.png" width=200>
# <br></br>
# 
# # Vector Representations
# ## *Data Science Unit 4 Sprint 2 Assignment 2*

# In[1]:


import re
import string

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import spacy


# ## 1) *Optional:* Scrape 100 Job Listings that contain the title "Data Scientist" from indeed.com
# 
# At a minimum your final dataframe of job listings should contain
# - Job Title
# - Job Description
# 
# If you choose to not to scrape the data, there is a CSV with outdated data in the directory. Remeber, if you scrape Indeed, you're helping yourself find a job. ;)

# In[ ]:


##### Your Code Here #####
raise Exception("\nThis task is not complete. \nReplace this line with your code for the task."


# ## 2) Use Spacy to tokenize / clean the listings 

# In[ ]:


##### Your Code Here #####
raise Exception("\nThis task is not complete. \nReplace this line with your code for the task."


# ## 3) Use Scikit-Learn's CountVectorizer to get word counts for each listing.

# In[ ]:


##### Your Code Here #####
raise Exception("\nThis task is not complete. \nReplace this line with your code for the task."


# ## 4) Visualize the most common word counts

# In[ ]:


##### Your Code Here #####
raise Exception("\nThis task is not complete. \nReplace this line with your code for the task."


# ## 5) Use Scikit-Learn's tfidfVectorizer to get a TF-IDF feature matrix

# In[ ]:


##### Your Code Here #####
raise Exception("\nThis task is not complete. \nReplace this line with your code for the task."


# ## 6) Create a NearestNeighbor Model. Write the description of your ideal datascience job and query your job listings. 

# In[ ]:


##### Your Code Here #####
raise Exception("\nThis task is not complete. \nReplace this line with your code for the task."


# ## Stretch Goals
# 
#  - Try different visualizations for words and frequencies - what story do you want to tell with the data?
#  - Scrape Job Listings for the job title "Data Analyst". How do these differ from Data Scientist Job Listings
#  - Try and identify requirements for experience specific technologies that are asked for in the job listings. How are those distributed among the job listings?
#  - Use a clustering algorithm to cluster documents by their most important terms. Do the clusters reveal any common themes?
#   - **Hint:** K-means might not be the best algorithm for this. Do a little bit of research to see what might be good for this. Also, remember that algorithms that depend on Euclidean distance break down with high dimensional data.
#  - Create a labeled dataset - which jobs will you apply for? Train a model to select the jobs you are most likely to apply for. :) 
