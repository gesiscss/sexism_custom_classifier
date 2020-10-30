#sklearn
from sklearn.base import BaseEstimator

# TODO

class CNN(BaseEstimator):
    '''Builds model.'''
    def __init__(self):
        self.estimator=None

    def fit(self, X, y):
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y)