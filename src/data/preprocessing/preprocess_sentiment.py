#src module
from src.utilities import Preprocessing

#sklearn
from sklearn.base import BaseEstimator

class PreprocessSentiment(BaseEstimator):
    '''Preporcesses data for sentiment features.'''
    def preprocess(self, text):
        try:
            upre=Preprocessing()
            
            text = upre.remove_new_lines(text)
            text = upre.remove_URLs(text)
            
            text = upre.remove_usernames(text)
            text = upre.replace_whitespace_with_single_space(text)
        
            text = upre.remove_RT(text)
            text = upre.replace_whitespace_with_single_space(text)
        
            return text
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)
    
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        #return raw_docs.apply(lambda x: self.preprocess(x))
        return [self.preprocess(raw_doc) for raw_doc in raw_docs]