import os
import pandas as pd
import pickle    
import time
    
def read_csv(path, delimiter='\t'):
    return pd.read_csv(path, delimiter=delimiter)

def save_to_csv(df, path, delimiter='\t'):
    df.to_csv(path, index=False, sep=delimiter)

def save_to_pickle(data, path):
    timestr = time.strftime("%Y%m%d-%H%M%S")
    file_name=''.join((path, timestr, '.pkl'))
        
    with open(file_name, 'wb+') as f:
        pickle.dump(data, f)
    return file_name

def read_pickle(file_name):
    df=[]
    with open(file_name, 'rb') as f:
        df = pickle.load(f)
    return df

def get_object(objects, name: object = None) -> object:
    '''Factory'''
    return objects[name]()

def get_attr(object_, method_name):
    return getattr(object_, method_name)()
    
#######################################################
import spacy

import preprocessor as pre_tweet
pre_tweet.set_options(pre_tweet.OPT.SMILEY, pre_tweet.OPT.EMOJI)

from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()

from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()

from pycontractions import Contractions
cont = Contractions(api_key="glove-twitter-100")

import re
import string
    
class Preprocessing():
    def tokenize_spacy(self, text):
        nlp = spacy.load("en_core_web_sm")
        doc=nlp(text)
        return [token.text for token in doc]

    def tokenize_tweettokenizer(self, text):
        return tweet_tokenizer.tokenize(text)

    def stem_porter(self, text):
        return porter_stemmer.stem(text)

    def expand_contractions(self, text):
        return list(cont.expand_texts([text], precise=True))[0]

    def remove_usernames(self, text):
        return re.sub(r'@\w+ ?', '', text)

    def remove_hashtags(self, text):
        return re.sub(r'#\w+ ?', '', text)

    def remove_URLs(self, text):
        return re.sub(r"http\S+", '', text)

    def remove_new_lines(self, text):
        return ' '.join(text.splitlines())

    def replace_whitespace_with_single_space(self, text):
        return ' '.join(text.split())

    def remove_numbers(self, text):
        return re.sub(r'\d+', '', text)

    def remove_punctuations(self, text):
        if text.isalnum() == True:
            return text
        return ''.join([char if char not in string.punctuation else ' ' for char in text]) 

    def compress_words(self, text):
        return re.sub(r'(.)\1+', r'\1\1', text)

    def lower_text(self, text):
        return text.lower()

    def clean_tweet(self, text):
        return pre_tweet.clean(text)
    
    def remove_RT(self, text):
        if text.startswith('RT:'):
            text=text[3:]
        elif text.startswith('RT :'):
            text=text[4:]
        elif text.startswith('RT '):
            text=text[3:]
   
        text=re.sub(r" RT ", " ", text)
        text=re.sub(r"RT:", " ", text)
        text=re.sub(r"RT :", " ", text)
    
        return text