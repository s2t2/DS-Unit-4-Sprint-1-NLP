#!/usr/bin/env python
# coding: utf-8

# Lambda School Data Science
#
# *Unit 4, Sprint 1, Module 1*
#
# ---
# <h1 id="moduleTitle"> Natural Language Processing Introduction (Prepare)</h1>
#
# "Natural" meaning - not computer languages but spoken/written human languages. The hard thing about NLP is that human languages are far less structured or consistent than computer languages. This is perhaps the largest source of difficulty when trying to get computers to "understand" human languages. How do you get a machine to understand sarcasm, and irony, and synonyms, connotation, denotation, nuance, and tone of voice --all without it having lived a lifetime of experience for context? If you think about it, our human brains have been exposed to quite a lot of training data to help us interpret languages, and even then we misunderstand each other pretty frequently.
#
#
# <h2 id='moduleObjectives'>Learning Objectives</h2>
#
# By the end of end of this module, a student should be able to:
# * <a href="#p1">Objective 1</a>: Tokenze text
# * <a href="#p1">Objective 2</a>: Remove stop words from text
# * <a href="#p3">Objective 3</a>: Perform stemming and lemmatization on tokens
#
# ## Conda Environments
#
# You will be completing each module this sprint on your machine. We will be using conda environments to manage the packages and their dependencies for this sprint's content. In a classroom setting, instructors typically abstract away environment for you. However, environment management is an important professional data science skill. We showed you how to manage environments using pipvirtual env during Unit 3, but in this sprint, we will introduce an environment management tool common in the data science community:
#
# > __conda__: Package, dependency and environment management for any language—Python, R, Ruby, Lua, Scala, Java, JavaScript, C/ C++, FORTRAN, and more.
#
# The easiest way to install conda on your machine is via the [Anaconda Distribution](https://www.anaconda.com/distribution/) of Python & R. Once you have conda installed, read ["A Guide to Conda Environments"](https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533). This article will provide an introduce into some of the conda basics. If you need some additional help getting started, the official ["Setting started with conda"](https://conda.io/projects/conda/en/latest/user-guide/getting-started.html) guide will point you in the right direction.
#
# To get the sprint environment setup:
#
# 1. Open your command line tool (Terminal for MacOS, Anaconda Prompt for Windows)
# 2. Navigate to the folder with this sprint's content. There should be both  `environment_mac.yml` and a `environment_windows.yml`.
# 3. Run `conda env create -n U4-S1-NLP -f /path/to/environment_mac.yml` => You should replace the operating system if you're a windows user. You can also rename the environment if you would like. Once the command completes, your conda environment should be ready.
# 4. We are going to also add an Ipython Kernel reference to your conda environment, so we can use it from JupyterLab. You will need to 'activate' the conda environment: `source activate U4-S1-NLP` on Terminal or `conda activate U4-S1-NLP` on Anaconda Prompt.
# 5. Next run `python -m ipykernel install --user --name U4-S1-NLP --display-name "U4-S1-NLP (Python3)"` => This will add a json object to an ipython file, so JupterLab will know that it can use this isolated instance of Python. :)
# 6. Deactivate your conda environment and launch JupyterLab. You should know see "U4-S1-NLP (Python3)" in the list of available kernels on launch screen.
#

# # Tokenze Text (Learn)
# <a id="p1"></a>

# ## Overview
#
# > **token**: an instance of a sequence of characters in some particular document that are grouped together as a useful semantic unit for processing
#
# > [_*Introduction to Information Retrival*_](https://nlp.stanford.edu/IR-book/)
#
#
# ### The attributes of good tokens
#
# * Should be stored in an iterable datastructure
#   - Allows analysis of the "semantic unit"
# * Should be all the same case
#   - Reduces the complexity of our data
# * Should be free of non-alphanumeric characters (ie punctuation, whitespace)
#   - Removes information that is probably not relevant to the analysis

# Let's pretend we are trying analyze the random sequence here. Question: what is the most common character in this sequence?

# In[6]:


random_seq = "AABAAFBBBBCGCDDEEEFCFFDFFAFFZFGGGGHEAFJAAZBBFCZ"


# A useful unit of analysis for us is going to be a letter or character

# In[7]:


tokens = list(random_seq)
print(tokens)


# Our tokens are already "good": in an iterable datastructure, all the same case, and free of noise characters (punctionation, whitespace), so we can jump straight into analysis.

# In[8]:


import seaborn as sns

sns.countplot(tokens);


# The most common character in our sequence is  "F". We can't just glance at the the sequence to know which character is the most common. We (humans) struggle to subitize complex data (like random text sequences).
#
# > __Subitize__ is the ability to tell the number of objects in a set, quickly, without counting.
#
# We need to chunk the data into countable pieces "tokens" for us to analyze them. This inability subitize text data is the motivation for our discussion today.

# ### Tokenizing with Pure Python

# In[8]:


sample = "Friends, Romans, countrymen, lend me your ears;"


# ##### Iterable Tokens
#
# A string object in Python is already iterable. However, the item you iterate over is a character not a token:
#
# ```
# from time import sleep
# for num, character in enumerate(sample):
#     sleep(.5)
#     print(f"Char {num} - {character}", end="\r")
# ```
#
# If we instead care about the words in our sample (our semantic unit), we can use the string method `.split()` to seperate the whitespace and create iterable units. :)

# In[11]:


sample.split(", ")


# ##### Case Normalization
# A common data cleaning data cleaning task with token is to standardize or normalize the case. Normalizing case reduces the chance that you have duplicate records for things which have practically the same semantic meaning. You can use either the `.lower()` or `.upper()` string methods to normalize case.
#
# Consider the following example:

# In[12]:


# Notice anything odd here?
df['brand'].value_counts()


# In[13]:


# Much cleaner
df['brand'] = df['brand'].apply(lambda x: x.lower())
df['brand'].value_counts()


# ##### Keep Only Alphanumeric Characters
# Yes, we only want letters and numbers. Everything else is probably noise: punctionation, whitespace, and other notation. This one is little bit more complicatd than our previous example. Here we will have to import the base package `re` (regular expressions).
#
# The only regex expression pattern you need for this is `'[^a-zA-Z ^0-9]'` which keeps lower case letters, upper case letters, spaces, and numbers.

# In[16]:


sample = sample+" 911"


# In[17]:


import re

re.sub(r'[^a-zA-Z ^0-9]', '', sample)


# #### Two Minute Challenge
# - Complete the function `tokenize` below
# - Combine the methods which we discussed above to clean text before we analyze it
# - You can put the methods in any order you want

# In[18]:


def tokenize(text):
    """Parses a string into a list of semantic units (words)

    Args:
        text (str): The string that the function will tokenize.

    Returns:
        list: tokens parsed out by the mechanics of your choice
    """

    tokens = re.sub(r'[^a-zA-Z ^0-9]', '', text)
    tokens = tokens.lower().split()

    return tokens


# ## Follow Along
#
# Our inability to analyze text data becomes quickly amphilfied in business context. Consider the following:
#
# A business which sells widgets also collects customer reviews of those widgets. When the business first started out, they had a human read the reviews to look for patterns. Now, the business sells thousands of widgets a month. The human readers can't keep up with the pace of reviews to synthesize an accurate analysis. They need some science to help them analyze their data.
#
# Now, let's pretend that business is Amazon, and the widgets are Amazon products such as the Alexa, Echo, or other AmazonBasics products. Let's analyze their reviews with some counts. This dataset is available on [Kaggle](https://www.kaggle.com/datafiniti/consumer-reviews-of-amazon-products/).

# In[11]:


"""
Import Statements
"""

# Base
from collections import Counter
import re

import pandas as pd

# Plotting
import squarify
import matplotlib.pyplot as plt
import seaborn as sns

# NLP Libraries
import spacy
from spacy.tokenizer import Tokenizer
from nltk.stem import PorterStemmer

nlp = spacy.load("en_core_web_lg")


# In[12]:


df = pd.read_csv('./data/Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv')


# In[13]:


df.head(2)


# In[7]:


# How can we count the raw text?
df['reviews.text'].value_counts(normalize=True)[:50]


# In[19]:


df['tokens'] = df['reviews.text'].apply(tokenize)


# In[20]:


df['tokens'].head()


# #### Analyzing Tokens

# In[22]:


# Object from Base Python
from collections import Counter

# The object `Counter` takes an iterable, but you can instaniate an empty one and update it.
word_counts = Counter()

# Update it based on a split of each of our documents
df['tokens'].apply(lambda x: word_counts.update(x))

# Print out the 10 most common words
word_counts.most_common(10)


# Let's create a fuction which takes a corpus of document and returns and dataframe of word counts for us to analyze.

# In[23]:


def count(docs):

        word_counts = Counter()
        appears_in = Counter()

        total_docs = len(docs)

        for doc in docs:
            word_counts.update(doc)
            appears_in.update(set(doc))

        temp = zip(word_counts.keys(), word_counts.values())

        wc = pd.DataFrame(temp, columns = ['word', 'count'])

        wc['rank'] = wc['count'].rank(method='first', ascending=False)
        total = wc['count'].sum()

        wc['pct_total'] = wc['count'].apply(lambda x: x / total)

        wc = wc.sort_values(by='rank')
        wc['cul_pct_total'] = wc['pct_total'].cumsum()

        t2 = zip(appears_in.keys(), appears_in.values())
        ac = pd.DataFrame(t2, columns=['word', 'appears_in'])
        wc = ac.merge(wc, on='word')

        wc['appears_in_pct'] = wc['appears_in'].apply(lambda x: x / total_docs)

        return wc.sort_values(by='rank')


# In[24]:


# Use the Function
wc = count(df['tokens'])


# In[25]:


wc.head()


# In[26]:


import seaborn as sns

# Cumulative Distribution Plot
sns.lineplot(x='rank', y='cul_pct_total', data=wc);


# In[28]:


wc[wc['rank'] <= 20]['cul_pct_total'].max()


# In[29]:


import squarify
import matplotlib.pyplot as plt

wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8 )
plt.axis('off')
plt.show()


# ### Processing Raw Text with Spacy
#
# Spacy's datamodel for documents is unique among NLP libraries. Instead of storing the documents components repeatively in various datastructures, Spacy indexes components and simply stores the lookup informaiton.
#
# This is often why Spacy is considered to be more production grade than library like NLTK.

# In[30]:


import spacy
from spacy.tokenizer import Tokenizer

nlp = spacy.load("en_core_web_md")

# Tokenizer
tokenizer = Tokenizer(nlp.vocab)


# In[31]:


# Print out list of tokens
[token.text for token in tokenizer(sample)]


# In[32]:


# Tokenizer Pipe

tokens = []

""" Make them tokens """
for doc in tokenizer.pipe(df['reviews.text'], batch_size=500):
    doc_tokens = [token.text for token in doc]
    tokens.append(doc_tokens)

df['tokens'] = tokens


# In[33]:


df['tokens'].head()


# In[34]:


wc = count(df['tokens'])


# In[35]:


wc.head()


# In[36]:


wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8 )
plt.axis('off')
plt.show()


# ## Challenge
#
# In the module project, you will apply tokenization to another set of review data and produce visualizations of those tokens.

# # Stop Words (Learn)
# <a id="p2"></a>

# ## Overview
# Section Agenda
# - What are they?
# - How do we get rid of them using Spacy?
# - Visualization
# - Libraries of Stop Words
# - Extending Stop Words
# - Statistical trimming
#
# If the visualizations above, you began to notice a pattern. Most of the words don't really add much to our undertanding of product reviews. Words such as "I", "and", "of", etc. have almost no semantic meaning to us. We call these useless words "stop words," because we should 'stop' ourselves from including them in the analysis.
#
# Most NLP libraries have built in lists of stop words that common english words: conjunctions, articles, adverbs, pronouns, and common verbs. The best practice, however, is to extend/customize these standard english stopwords for your problem's domain. If I am studying political science, I may want to exclude the word "politics" from my analysis; it's so common it does not add to my understanding.

# ## Follow Along
#
# ### Default Stop Words
# Let's take a look at the standard stop words that came with our spacy model:

# In[38]:


# Spacy's Default Stop Words
len(nlp.Defaults.stop_words)


# In[39]:


tokens = []

""" Update those tokens w/o stopwords"""
for doc in tokenizer.pipe(df['reviews.text'], batch_size=500):

    doc_tokens = []

    for token in doc:
        if (token.is_stop == False) & (token.is_punct == False):
            doc_tokens.append(token.text.lower())

    tokens.append(doc_tokens)

df['tokens'] = tokens


# In[40]:


df.tokens.head()


# In[41]:


wc = count(df['tokens'])

wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8 )
plt.axis('off')
plt.show()


# ### Extending Stop Words

# In[42]:


print(type(nlp.Defaults.stop_words))


# In[43]:


STOP_WORDS = nlp.Defaults.stop_words.union(['I', 'amazon', 'i', 'Amazon', 'it', "it's", 'it.', 'the', 'this',])


# In[44]:


STOP_WORDS


# In[45]:


tokens = []

for doc in tokenizer.pipe(df['reviews.text'], batch_size=500):

    doc_tokens = []

    for token in doc:
        if token.text not in STOP_WORDS:
            doc_tokens.append(token.text.lower())

    tokens.append(doc_tokens)

df['tokens'] = tokens


# In[46]:


wc = count(df['tokens'])
wc.head()


# In[47]:


wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8 )
plt.axis('off')
plt.show()


# ### Statistical Trimming
#
# So far, we have talked about stop word in relation to either broad english words or domain specific stop words. Another common approach to stop word removal is via statistical trimming. The basic idea: preserve the words that give the most about of variation in your data.
#
# Do you remember this graph?

# In[48]:


sns.lineplot(x='rank', y='cul_pct_total', data=wc);


# This graph tells us that only a *handful* of words represented 80% of words in the overall corpus. We can interpret this in two ways:
# 1. The words that appear most frequently may not provide any insight into the mean on the documens since they are so prevalent.
# 2. Words that appear infrequeny (at the end of the graph) also probably do not add much value, because the are mentioned so rarely.
#
# Let's take a look at the words at the bottom and the top and make a decision for ourselves:

# In[49]:


wc.head(50)


# In[50]:


wc['appears_in_pct'].describe()


# In[51]:


# Frequency of appears in documents
sns.distplot(wc['appears_in_pct']);


# In[ ]:


# Tree-Map w/ Words that appear in a least 98% of documents.


# ## Challenge
#
# In the module project, you will apply stop word removal to a new corpus. You will focus on applying dictionary based stop word removal, but as a stretch goal, you should consider applying statistical stopword trimming.

# # Stemming & Lemmatization (Learn)
# <a id="p3"></a>

# ## Overview
#
# You can see from our example above there is still some normalization to do to get a clean analysis. You notice that there many words (*i.e.* 'batteries', 'battery') which share the same root word. We can use either the process of stemming or lemmatization to trim our words down to the 'root' word.
#
# __Section Agenda__:
#
# - Which is which
# - why use one v. other
# - show side by side visualizations
# - how to do it in spacy & nltk
# - introduce PoS in here as well

# ## Follow Along

# ### Stemming
#
# > *a process for removing the commoner morphological and inflexional endings from words in English. Its main use is as part of a term normalisation process that is usually done when setting up Information Retrieval systems.* - [Martin Porter](https://tartarus.org/martin/PorterStemmer/)
#
# Some examples include:
# - 'ing'
# - 'ed'
# - 's'
#
# These rules are by no means comprehensive, but they are somewhere to start. Most stemming is done by well documented algorithms such as Porter, Snowball, and Dawson. Porter and its newer version Snowball are the most popular stemming algorithms today. For more information on various stemming algorithms check out [*"A Comparative Study of Stemming Algorithms"*](https://pdfs.semanticscholar.org/1c0c/0fa35d4ff8a2f925eb955e48d655494bd167.pdf)
#
#
# Spacy does not do stemming out of the box, but instead uses a different technique called *lemmatization* which we will discuss in the next section. Let's turn to an antique python package `nltk` for stemming.

# In[54]:


from nltk.stem import PorterStemmer

ps = PorterStemmer()

words = ["is","are","be","was"]

for word in words:
    print(ps.stem(word))


# ### Two Minute Challenge
#
# Apply the Porter stemming algorithm to the tokens in the `df` dataframe. Visualize the results in the tree graph we have been using for this session.

# In[ ]:


# Put in a new column `stems`


# In[ ]:


wc = count(df['stems'])

wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8 )
plt.axis('off')
plt.show()


# ### Lemmatization
#
# You notice immediately that results are kinda funky - words just oddly chopped off. The Porter algorithm did exactly what it knows to do: chop off endings. Stemming works well in applications where humans don't have to worry about reading the results. Search engines and more broadly information retrival algorithms use stemming. Why? Becuase it's fast.
#
# Lemmatization on the other hand is more methodical. The goal is to transform a word into's base form called a lemma. Plural nouns with funky spellings get transformed to singular tense. Verbs are all transformed to the transitive. Nice tidy data for a visualization. :) However, this tidy data can come at computational cost. Spacy does a pretty freaking good job of it though. Let's take a look:

# In[55]:


sent = "This is the start of our NLP adventure. We started here with Spacy."

doc = nlp(sent)

# Lemma Attributes
for token in doc:
    print(token.lemma_)


# In[56]:


# Wrap it all in a function
def get_lemmas(text):

    lemmas = []

    doc = nlp(text)

    # Something goes here :P
    for token in doc:
        if ((token.is_stop == False) and (token.is_punct == False)) and (token.pos_!= 'PRON'):
            lemmas.append(token.lemma_)

    return lemmas


# In[57]:


df['lemmas'] = df['reviews.text'].apply(get_lemmas)


# In[58]:


df['lemmas'].head()


# In[59]:


wc = count(df['lemmas'])
wc_top20 = wc[wc['rank'] <= 20]

squarify.plot(sizes=wc_top20['pct_total'], label=wc_top20['word'], alpha=.8 )
plt.axis('off')
plt.show()


# ## Challenge
#
# You should know how to apply lemmatization with Spacy to a corpus of text.

# # Review
#
# In this module project, you've seen us apply Natural Language Processing techniques (tokenization, stopword removal, and lemmatization) to a corpus of Amazon text reviews. We analyzed those reviews using these techniques and discovered that Amazon customers are generally statisfied with the battery life of Amazon products and generally appear statisfied.
#
# You will apply similiar techniques to today's [module project assignment](LS_DS_411_Text_Data_Assignment.ipynb) to analyze coffee shop reviews from yelp. Remeber that the techniques of processing the text are just the begining. There are many ways to slice and dice the data.

# # Sources
#
# * Spacy 101 - https://course.spacy.io
# * NLTK Book - https://www.nltk.org/book/
# * An Introduction to Information Retrieval - https://nlp.stanford.edu/IR-book/pdf/irbookonlinereading.pdf

# ## Advanced Resources & Techniques
# - Named Entity Recognition (NER)
# - Dependcy Trees
# - Generators
# - the major libraries (NLTK, Spacy, Gensim)
