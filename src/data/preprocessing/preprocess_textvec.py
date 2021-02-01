#src module
from src.utilities import Preprocessing

#sklearn
from sklearn.base import BaseEstimator

class PreprocessTextVec(BaseEstimator):
    '''Preporcesses data for TextVectorization.'''
    
    def preprocess(self, text):
        upre=Preprocessing()
        
        text=upre.replace_emojis(text)
        text=upre.remove_mention(text)
        text=upre.remove_rt(text)
        text=upre.remove_urls(text)
        
        text=upre.remove_non_alnum(text)
        text=upre.remove_space(text)
        text=upre.lower_text(text)
        text=upre.strip_text(text)
        text=upre.compress_words(text)
        
        return text
    
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        return raw_docs.apply(lambda x: self.preprocess(x))