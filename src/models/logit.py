#sklearn
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

class Logit(BaseEstimator):
    '''Logistic Regression model.'''
    def __init__(self, penalty='l2', C=1):
        self.penalty=penalty
        self.C=C
        self.estimator=None
        
    @property
    def coef_(self):
        return self.estimator.coef_ 

    def fit(self, X, y):
        self.estimator=LogisticRegression(penalty=self.penalty, C=self.C).fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y)