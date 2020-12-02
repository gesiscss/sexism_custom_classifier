#src module
from src.utilities import Preprocessing

#sklearn
from sklearn.base import BaseEstimator

import re
import nltk
from nltk.corpus import stopwords

class PreprocessTextVec(BaseEstimator):
    '''Preporcesses data for BERT document embedding features.'''
    def clean_str(self, string):
        stops = set(stopwords.words('english'))
        stops.discard('not')
    
        string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        string = re.sub(r",", " , ", string)
        string = re.sub(r"!", " ! ", string)
        string = re.sub(r"\(", " \( ", string)
        string = re.sub(r"\)", " \) ", string)
        string = re.sub(r"\?", " \? ", string)
        string = re.sub(r"\s{2,}", " ", string)
    
        string_words = string.split(" ")
        string_words = list(word for word in string_words if word not in stops)
        string = " ".join(string_words)
        return string.strip().lower()
    
    def preprocess(self, text):
        try:
            upre=Preprocessing()
            
            text=upre.remove_new_lines(text)
            text=upre.replace_whitespace_with_single_space(text)
            text=upre.remove_URLs(text)
            #text=upre.remove_usernames(text)
            #text=upre.remove_hashtags(text)
            #text=upre.clean_tweet(text)
            
            tokens=upre.tokenize_tweettokenizer(text)
    
            tokens=[upre.lower_text(item) for item in tokens]
            tokens=[upre.compress_words(item) for item in tokens]
            text=' '.join(tokens)
            #text=text.replace('..', '.')
            
            #tokens=text.split('.')
            #text=list(filter(None, tokens))
            
            #text=self.clean_str(text)
            
            return text
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)
    
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        return raw_docs.apply(lambda x: self.preprocess(x))