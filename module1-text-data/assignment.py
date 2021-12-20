#!/usr/bin/env python
# coding: utf-8

# <img align="left" src="https://lever-client-logos.s3.amazonaws.com/864372b1-534c-480e-acd5-9711f850815c-1524247202159.png" width=200>
# <br></br>
# <br></br>
# 
# # Natural Language Processing (NLP)
# ## *Data Science Unit 4 Sprint 1 Assignment 1*
# 
# Your goal in assignment: find the attributes of the best & worst coffee shops in the dataset. The text is fairly raw: dates in the review, extra words in the `star_rating` column, etc. You'll probably want to clean that stuff up for a better analysis. 
# 
# Analyze the corpus of text using text visualizations of token frequency. Try cleaning the data as much as possible. Try the following techniques: 
# - Lemmatization
# - Custom stopword removal
# 
# Keep in mind the attributes of good tokens. Once you have a solid baseline, layer in the star rating in your visualization(s). Keep part in this assignment - produce a write-up of the attributes of the best and worst coffee shops. Based on your analysis, what makes the best the best and the worst the worst. Use graphs and numbesr from your analysis to support your conclusions. There should be plenty of markdown cells! :coffee:

# In[5]:


from IPython.display import YouTubeVideo

YouTubeVideo('Jml7NVYm8cs')


# In[2]:


get_ipython().run_line_magic('pwd', '')


# In[1]:


import pandas as pd

url = "https://raw.githubusercontent.com/LambdaSchool/DS-Unit-4-Sprint-1-NLP/master/module1-text-data/data/yelp_coffeeshop_review_data.csv"

shops = pd.read_csv(url)
shops.head()


# In[2]:


# Start here 


# ## How do we want to analyze these coffee shop tokens? 
# 
# - Overall Word / Token Count
# - View Counts by Rating 
# - *Hint:* a 'bad' coffee shops has a rating betweeen 1 & 3 based on the distribution of ratings. A 'good' coffee shop is a 4 or 5. 

# In[ ]:





# ## Can visualize the words with the greatest difference in counts between 'good' & 'bad'?
# 
# Couple Notes: 
# - Rel. freq. instead of absolute counts b/c of different numbers of reviews
# - Only look at the top 5-10 words with the greatest differences
# 

# In[46]:





# ## Stretch Goals
# 
# * Analyze another corpus of documents - such as Indeed.com job listings ;).
# * Play the the Spacy API to
#  - Extract Named Entities
#  - Extracting 'noun chunks'
#  - Attempt Document Classification with just Spacy
#  - *Note:* This [course](https://course.spacy.io/) will be of interesting in helping you with these stretch goals. 
# * Try to build a plotly dash app with your text data 
# 
# 
