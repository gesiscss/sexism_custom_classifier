#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

from nltk.corpus import stopwords

class BuildNgramFeature(BaseEstimator):
    '''Extracts Ngram Features'''
    
    def __init__(self, ngram_range=(1,1)):
        '''
        Args:
        ngram_range (tuple (min_n, max_n)) = Ngram range for  features (e.g. (1, 2) means that extracts unigrams and bigrams)
        '''
        self.ngram_range=ngram_range
        self.tfidf_vectorizer = None
        self.feature_dimension=0
    
    def fit(self, x, y=None):
        stops = set(stopwords.words('english'))
        stops.discard('not')
        stops.discard('but')
        
        
        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, preprocessor=' '.join, stop_words=stops).fit(x)
        #self.tfidf_vectorizer = TfidfVectorizer(ngram_range=self.ngram_range, stop_words=stops).fit(x)
        self.feature_dimension=len(self.get_feature_names())
        return self

    def transform(self, texts):
        return self.tfidf_vectorizer.transform(texts)
    
    def get_feature_names(self):
        return self.tfidf_vectorizer.get_feature_names()