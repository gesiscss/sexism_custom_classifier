#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src.features.build_features import BuildFeature

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

class BuildNgramFeature(BuildFeature):
    '''Extracts Ngram Features'''
    
    def __init__(self, ngram_range=(1,3)):
        '''
        Args:
        ngram_range (tuple (min_n, max_n)) = Ngram range for  features (e.g. (1, 2) means that extracts unigrams and bigrams)
        '''
        self.ngram_range=ngram_range
        self.tfidf_vectorizer = None
   
    def get_feature_names(self):
        '''Returns feature names'''
        return self.tfidf_vectorizer.get_feature_names()
    
    def fit(self, x, y=None):
        self.tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=self.ngram_range)
        self.tfidf_vectorizer.fit(x)
        return self

    def transform(self, texts):
        return self.tfidf_vectorizer.transform(texts)