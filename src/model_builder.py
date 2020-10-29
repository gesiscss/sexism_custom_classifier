#src module
from src.enums import Model

#sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.base import BaseEstimator

class BuildLogisticRegression(BaseEstimator):
    '''Builds model.'''
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
    
class BuildSVM(BaseEstimator):
    '''Builds model.'''
    def __init__(self, penalty='l2', C=1):
        self.penalty=penalty
        self.C=C

    def fit(self, X, y):
        self.estimator=SVC().fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y)
    
class BuildCNN(BaseEstimator):
    '''Builds model.'''
    def __init__(self, penalty='l2', C=1):
        self.penalty=penalty
        self.C=C

    def fit(self, X, y):
        # TODO
        self.estimator=SVC().fit(X, y)
        return self
    
    def predict(self, X):
        return self.estimator.predict(X)
    
    def score(self, X, y, sample_weight=None):
        return self.estimator.score(X, y)

class ModelBuilder(BaseEstimator):
    '''Builds model.'''
    def get_logistic_regression(self):
        return BuildLogisticRegression()
    
    def get_cnn(self):
        return BuildCNN()
    
    def get_svm(self):
        return BuildSVM()

    def get_model(self, model_name):
        method_name = 'get_' + model_name
        method = getattr(self, method_name)
        return method()