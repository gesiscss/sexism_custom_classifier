#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

class BuildNgramFeature(BaseEstimator):
    '''Extracts Ngram Features'''
    
    def __init__(self, ngram_range=(1,1)):
        '''
        Args:
        ngram_range (tuple (min_n, max_n)) = Ngram range for  features (e.g. (1, 2) means that extracts unigrams and bigrams)
        '''
        self.ngram_range=ngram_range
        self.tfidf_vectorizer = None
    
    def fit(self, x, y=None):
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, preprocessor=' '.join).fit(x)
        return self

    def transform(self, texts):
        return self.tfidf_vectorizer.transform(texts)
    
    def get_feature_names(self):
        return self.tfidf_vectorizer.get_feature_names()