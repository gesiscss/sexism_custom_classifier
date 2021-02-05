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
    if name in objects.keys():
        return objects[name]()
    return None

def get_attr(object_, method_name):
    return getattr(object_, method_name)()
    
#######################################################
from nltk.tokenize import TweetTokenizer
tweet_tokenizer = TweetTokenizer()

from nltk.stem import PorterStemmer
porter_stemmer = PorterStemmer()

import re
import string
import emoji

class Preprocessing():
    def tokenize_tweettokenizer(self, text):
        return tweet_tokenizer.tokenize(text)

    def stem_porter(self, text):
        return porter_stemmer.stem(text)

    def remove_emojis(self, text, replace=' '):
        emoji_pattern = re.compile("["
        u"\U0001F600-\U0001F64F"  # emoticons
        u"\U0001F300-\U0001F5FF"  # symbols & pictographs
        u"\U0001F680-\U0001F6FF"  # transport & map symbols
        u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                           "]+", flags=re.UNICODE)
        return re.sub(emoji_pattern, replace, text)
        
    def replace_emojis(self, text):
        return re.sub('::', ': :', emoji.demojize(text))
    
    def remove_mention(self, text):
        return re.sub(r'@\w+ ?', '', text)

    def remove_hashtag(self, text):
        return re.sub(r'#\w+ ?', '', text)
    
    def remove_rt(self, text):
        rt_pattern = re.compile('([^a-zA-Z0-9]|^)(RT|rt)([^a-zA-Z0-9]|$)', flags=re.UNICODE)
        return re.sub(rt_pattern, '', text)
    
    def remove_urls(self, text):
        return re.sub(r'http\S+', '', text)
    
    def remove_non_alnum(self, text):
        return re.sub(r'[^a-zA-Z0-9\']', ' ', text)
    
    def remove_space(self, text):
        return re.sub(r'\s+', ' ', text)
    
    def lower_text(self, text):
        return text.lower()
    
    def strip_text(self, text):
        return text.strip()

    def compress_words(self, text):
        return re.sub(r'(.)\1+', r'\1\1', text)
        #return re.sub(r'(.)\1+', r'\1', text)

#######################################################
import time as t
from functools import wraps

def execution_time_calculator(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start_time = t.time()
        result = function(*args, **kwargs)
        end_time = t.time() - start_time
        formatted_end_time = "{:.2f}".format((end_time / 60))
        print('FUNCTION: {} TIME ELAPSED : {}'.format(function.__name__, formatted_end_time))
        return result
    return wrapper

def start_time_calculator(function):
    @wraps(function)
    def wrapper(*args, **kwargs):
        start = t.localtime()
        start_time = t.strftime("%H:%M:%S", start)
        print('Start time is %s' % start_time)
        result = function(*args, **kwargs)
        return result
    return wrapper

#######################################################
import json
from src.enums import Model, Domain, Feature

def object_hook_method(obj):
        if '__tuple__' in obj:
            return tuple(obj['items'])
        elif '__tuple_list__' in obj:
            return [tuple(i) for i in obj['items']]
        elif '__model__' in obj:
            return [getattr(Model, model) for model in obj['items']]
        elif '__domain__' in obj:
            return [{'name':domain , 'value':getattr(Domain, domain)} for domain in obj['items']]
        elif '__feature__' in obj:
            return getattr(Feature, obj['item'])
        else:
            return obj

class Params():
    def __init__(self, json_path):
        self.update(json_path)
    
    def update(self, json_path):
        """Loads parameters from json file"""
        with open(json_path) as f:
            params = json.load(f, object_hook=object_hook_method)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['data_file']`"""
        return self.__dict__