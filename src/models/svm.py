#src module
from src.enums import Model

#sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator

class SVM(BaseEstimator):
    '''SVM model.'''
    def __init__(self, kernel='rbf', C=1):
        self.kernel=kernel
        self.C=C
        self.estimator=None
    
    @property
    def coef_(self):
        return self.estimator.coef_ 
        
    def fit(self, X, y):
        self.estimator=SVC(kernel=self.kernel, C=self.C).fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y)