#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

#other
import inspect
from collections import defaultdict

class BuildNgramFeature(BaseEstimator):
    '''Extracts Ngram Features'''
    
    def __init__(self, ngram_range=(1,3)):
        '''
        Args:
        ngram_range (tuple (min_n, max_n)) = Ngram range for  features (e.g. (1, 2) means that extracts unigrams and bigrams)
        '''
        self.ngram_range=ngram_range
        self.tfidf_vectorizer = None
    
    def fit(self, x, y=None):
        #print('fit BuildNgramFeature start x.shape {}'.format(x.shape))
        #print('fit BuildNgramFeature {} {} {}'.format(len(x), x, y))
        self.tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=self.ngram_range)
        self.tfidf_vectorizer.fit(x)
        #print('fit BuildNgramFeature end tfidf feature count: {}'.format(len(self.tfidf_vectorizer.get_feature_names())))
        return self

    def transform(self, texts):
        #print('transform BuildNgramFeature start texts.shape {}'.format(texts.shape))
        #print('transform BuildNgramFeature len {} {}'.format(len(texts), texts))
        features = self.tfidf_vectorizer.transform(texts)
        #print('transform BuildNgramFeature end features {}'.format(features.shape))
        return features