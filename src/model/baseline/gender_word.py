#sklearn
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

#other
import re

class GenderWord(BaseEstimator):
    '''Gender Word baseline model.'''
    def __init__(self, vocab_path=''):
        self.vocab_path=vocab_path   
        self.gender_words=None
        
    def fit(self, X, y):
        with open(self.vocab_path) as f:
            self.gender_words = set([line.strip() for line in f.readlines()])

        self.gender_words = set(map(lambda x: x.replace("_", " "), self.gender_words))
        self.gender_words = re.compile(r"\b(" +"|".join(map(re.escape, self.gender_words))+r")\b")
        return self
    
    def predict(self, X):
        return list(map(lambda x: self.has_gender_words(x, self.gender_words), X))
    
    def has_gender_words(self, text, gender_words):
        return int(len(re.findall(gender_words, text))>0)