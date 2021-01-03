#src module
from src.utilities import Preprocessing

#sklearn
from sklearn.base import BaseEstimator

class PreprocessSentiment(BaseEstimator):
    '''Preporcesses data for sentiment features.'''
    def preprocess(self, text):
        try:
            upre=Preprocessing()
            text=upre.remove_mention(text)
            text=upre.remove_rt(text)
            text=upre.remove_urls(text)
            text=upre.remove_space(text)
            return text
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)
    
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        return [self.preprocess(raw_doc) for raw_doc in raw_docs]