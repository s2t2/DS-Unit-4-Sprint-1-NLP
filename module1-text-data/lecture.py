
#import os
#import pandas as pd
import re

#AMAZON_REVIEWS_CSV_FILEPATH = os.path.join(os.path.dirname(__file__), "data", "Datafiniti_Amazon_Consumer_Reviews_of_Amazon_Products_May19.csv")

# The only regex expression pattern you need for this is `'[^a-zA-Z ^0-9]'` which keeps lower case letters, upper case letters, spaces, and numbers
#ALPHANUMERIC_PATTERN = r'[^a-zA-Z ^0-9]' # r string literal treats slashes in a way that plays nice with regex
ALPHANUMERIC_PATTERN = "[^a-zA-Z ^0-9]" # ... but no slashes in here so this should be ok

def tokenize(text):
    """
    Parses a string into a list of semantic units (words)
    Args:
        text (str): The string that the function will tokenize.
    Returns:
        list: tokens parsed out by the mechanics of your choice
    """
    parsed_text = text
    parsed_text = parsed_text.lower()
    parsed_text = re.sub(ALPHANUMERIC_PATTERN, '', parsed_text)
    tokens = parsed_text.split() # do after removing special characters
    return tokens

#
# TOKENIZING
#
#  A "token" is a "useful semantic unit for processing" and is:
#    + iterable
#    + same case
#    + only alphanumeric (i.e. no punctuation or whitespace or other notation)
#    + reduced complexity (removes info that won't be helpful for analysis)
#

seq = "AABAAFBBBBCGCDDEEEFCFFDFFAFFZFGGGGHEAFJAAZBBFCZ"

print("--------------")
print("STRING CASE CONVERSIONS")
print(seq.upper())
print(seq.lower())

print("--------------")
print("STRING ITERATION")
print(list(seq))
for char in list(seq):
    print(char)

sentence = "Friends, Romans, countrymen, lend me your ears;"

print("--------------")
print("STRING SPLITTING")
print(sentence.split(", "))
print(sentence.split(" "))
print(sentence.split())

print("--------------")
print("STRING CLEANING (VIA REGULAR EXPRESSIONS)")

sentence += " 911"
parsed_sentence = re.sub(ALPHANUMERIC_PATTERN, '', sentence)
print(parsed_sentence) #> 'Friends Romans countrymen lend me your ears 911'

#
# TWO MINUTE CHALLENGE
#

print("--------------")
print("ALL TOGETHER NOW... (TWO MINUTE CHALLENGE)")
print(tokenize(sentence))
my_message = " Oh HeY there - so whatr'u up to later???? \n  Text me (123) 456-4444. k cool! "
print(my_message)
print(tokenize(my_message))
assert tokenize(my_message) == ['oh', 'hey', 'there', 'so', 'whatru', 'up', 'to', 'later', 'text', 'me', '123', '4564444', 'k', 'cool']

exit()


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
