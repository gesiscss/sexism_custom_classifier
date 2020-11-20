#src module
from src.utilities import Preprocessing

#sklearn
from sklearn.base import BaseEstimator

class PreprocessNgram(BaseEstimator):
    '''Preporcesses data for ngram features.'''
    
    def preprocess(self, text):
        try:
            upre=Preprocessing()
            
            text=upre.remove_new_lines(text)
            text=upre.replace_whitespace_with_single_space(text)
            text=upre.remove_URLs(text)
            text=upre.remove_usernames(text)
            text=upre.remove_hashtags(text)
            text=upre.clean_tweet(text)
            
            text=text.replace('-', ' ')
            text=text.replace('.', ' ')
            text=upre.expand_contractions(text)
        
            tokens=upre.tokenize_tweettokenizer(text)
    
            tokens=[upre.lower_text(item) for item in tokens]
            tokens=[upre.compress_words(item) for item in tokens]
            tokens=[upre.remove_punctuations(item) for item in tokens]
            tokens=[upre.remove_numbers(item) for item in tokens]
            
            tokens=[item.replace(' ', '') for item in tokens]
            tokens=[item for item in tokens if item != '']
            tokens=[item for item in tokens if item != 'rt']
            
            stems = [ upre.stem_porter(str(item)) for item in tokens]
            return list(stems)
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)
    
    def fit(self, raw_docs, y=None):
        return self

    def transform(self, raw_docs):
        return raw_docs.apply(lambda x: self.preprocess(x))