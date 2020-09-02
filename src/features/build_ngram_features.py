#!/usr/bin/env python
# coding: utf-8

# In[3]:

from sklearn.feature_extraction.text import TfidfVectorizer
from src import utilities as u


class BuildNgramFeature:
    '''Extracts Ngram Features'''
    
    def __init__(self, ngram_range=(1,1)):
        '''
        Args:
        ngram_range (tuple (min_n, max_n)) = Ngram range for  features (e.g. (1, 2) means unigrams and bigrams)
        '''
        self.tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=ngram_range)
    
    def build_features(self, X_train, X_test):
        self.tfidf_vectorizer.fit(X_train)
        
        X_train_features = self.tfidf_vectorizer.transform(X_train)
        X_test_features = self.tfidf_vectorizer.transform(X_test)
        
        print('shape of train features : ' + str(X_train_features.shape))
        print('shape of test features : ' + str(X_test_features.shape))
        
        return X_train_features, X_test_features