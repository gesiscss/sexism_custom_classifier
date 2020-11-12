#sklearn
from sklearn.base import BaseEstimator
from sklearn.linear_model import LogisticRegression

class Logit(BaseEstimator):
    '''Logistic Regression model.'''
    def __init__(self, n_jobs=-1, penalty='l2', C=1, class_weight=None, max_iter=100):
        self.n_jobs=n_jobs
        self.penalty=penalty
        self.C=C
        self.class_weight=class_weight
        self.max_iter=max_iter
        self.estimator=None
        
    @property
    def coef_(self):
        return self.estimator.coef_ 

    def fit(self, X, y):
        self.estimator=LogisticRegression(n_jobs=self.n_jobs, penalty=self.penalty, C=self.C, class_weight=self.class_weight, max_iter=self.max_iter).fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y)