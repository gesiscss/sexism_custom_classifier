#src module
from src.utilities import Preprocessing

#sklearn
from sklearn.base import BaseEstimator

class PreprocessBertDocEmb(BaseEstimator):
    '''Preporcesses data for BERT document embedding features.'''
    def preprocess(self, text):
        try:
            upre=Preprocessing()
            
            text=upre.remove_new_lines(text)
            text=upre.replace_whitespace_with_single_space(text)
            text=upre.remove_URLs(text)
            text=upre.remove_usernames(text)
            text=upre.remove_hashtags(text)
            text=upre.clean_tweet(text)
            
            tokens=upre.tokenize_tweettokenizer(text)
    
            tokens=[upre.lower_text(item) for item in tokens]
            tokens=[upre.compress_words(item) for item in tokens]
            text=' '.join(tokens)
            #text=text.replace('..', '.')
            
            #tokens=text.split('.')
            #text=list(filter(None, tokens))
            
            return text
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)
    
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        return [self.preprocess(raw_doc) for raw_doc in raw_docs]