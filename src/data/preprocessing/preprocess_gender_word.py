#src module
import re
import unidecode

#sklearn
from sklearn.base import BaseEstimator

class PreprocessGenderWord(BaseEstimator):
    '''Preporcesses data for gender word model.'''
    
    def preprocess(self, text):
        return unidecode.unidecode(text).lower()
        
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        return raw_docs.apply(lambda x: self.preprocess(x))