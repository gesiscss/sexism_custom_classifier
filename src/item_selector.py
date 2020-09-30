
# coding: utf-8

# In[2]:

#sklearn
from sklearn.base import BaseEstimator, TransformerMixin

#other
import inspect
from collections import defaultdict

class ItemSelector(BaseEstimator):
    
    def __init__(self, key=''):
        self.key = key

    def fit(self, x, y=None):
        return self #check for ngrams

    def transform(self, data_dict):
        return data_dict[self.key]