#src module
from src.utilities import Preprocessing

#sklearn
from sklearn.base import BaseEstimator

class PreprocessBertDocEmb(BaseEstimator):
    '''Preporcesses data for BERT document embedding features.'''
    def preprocess(self, text):
        try:
            upre=Preprocessing()
            # TODO
            return text
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)
    
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        return [self.preprocess(raw_doc) for raw_doc in raw_docs]