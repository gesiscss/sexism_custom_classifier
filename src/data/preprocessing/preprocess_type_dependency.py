#src module
from src.utilities import Preprocessing

#sklearn
from sklearn.base import BaseEstimator

class PreprocessTypeDependency(BaseEstimator):
    '''Preporcesses data for type dependency features.'''
    
    def preprocess(self, text):
        try:
            upre=Preprocessing()
            
            text=upre.remove_emojis(text)
            text=upre.remove_hashtag(text)
            text=upre.remove_mention(text)
            text=upre.remove_urls(text)
            text=upre.remove_rt(text)
            
            text=upre.remove_non_alnum(text)
            text=upre.remove_space(text)
            text=upre.lower_text(text)
            text=upre.strip_text(text)
            text=upre.compress_words(text)
            
            return text
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)
    
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        return raw_docs.apply(lambda x: self.preprocess(x))