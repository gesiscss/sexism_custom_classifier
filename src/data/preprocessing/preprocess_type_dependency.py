#src module
from src.utilities import Preprocessing
from src.data.preprocessing.jha2017_preprocessing import preprocess as jha_preprocess

#sklearn
from sklearn.base import BaseEstimator

class PreprocessTypeDependency(BaseEstimator):
    '''Preporcesses data for type dependency features.'''
    
    def preprocess(self, text):
        try:
            upre=Preprocessing()
            
            text=upre.remove_new_lines(text)
            text=upre.replace_whitespace_with_single_space(text)
            text=upre.remove_URLs(text)
            text=upre.remove_usernames(text)
            text=upre.remove_hashtags(text)
            text=upre.clean_tweet(text)
            return text
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)
    
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        #return raw_docs.apply(lambda x: self.preprocess(x))
        return raw_docs.apply(lambda x: jha_preprocess(x, False))