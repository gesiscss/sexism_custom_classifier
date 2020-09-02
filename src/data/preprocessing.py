#!/usr/bin/env python
# coding: utf-8

# In[3]:


import preprocessor as p
from pycontractions import Contractions
#from nltk.tokenize import RegexpTokenizer
from nltk.stem import PorterStemmer
import string
import re

from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()
cont = Contractions(api_key="glove-twitter-100")
#regex_tokenizer = RegexpTokenizer(r'\w+')
porter_stemmer = PorterStemmer()

def remove_new_lines(text):
    return " ".join(text.splitlines())

def remove_RT(text):
    return text.replace('RT', ' ')

def remove_repeating_chars(text):
    ''' happyyyyyy ==  happyy''' 
    return re.sub(r'(.)\1+', r'\1\1', text)

def lower(text):
    return text.lower()

def clean_tweet(text):
    '''Removes URLs, Mentions, Hastags, Reserved, Emojis, Smilies, Numbers from text
    
    Args:
    text: Message text
    
    Returns:
    text: Cleaned text.
    '''
    return p.clean(text)


def expand_contractions(text):
    '''Expands english contractions in text by using pycontractions package.
    
    Args:
    text: Message text
    
    Returns:
    text: Text with expanded contractions.
    '''
    return list(cont.expand_texts([text], precise=True))[0]

def remove_numbers(text):
    return re.sub(r'\d+', '', text)

def remove_punctuations(text):
    return "".join([char if char not in string.punctuation else " " for char in text]) 
    
def tokenize(text):
    return tweet_tokenizer.tokenize(text)

def stem(text):
    return porter_stemmer.stem(text)