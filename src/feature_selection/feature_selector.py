#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.model_selection import StratifiedKFold

#src module
from src.utilities import *

class FeatureSelector(BaseEstimator):

    def __init__(self, name='SelectKBest', cv=5, random_state=0, k=0):
        self.name=name
        self.cv=cv
        self.random_state=random_state
        
        self.k=k
        self.feature_dimension=0
        self.selector=None
        self.estimator=SVR(kernel="linear")
        
    def fit(self, x, y=None):
        if self.name == 'RFECV':
            return self.fit_rfecv(x, y)
        elif self.name == 'SelectKBest':
            return self.fit_selectkbest(x, y)
    
    def fit_selectkbest(self, x, y=None):
        self.feature_dimension=x.shape[1]
        self.selector=SelectKBest(k=self.k).fit(x, y)
        return self
    
    @execution_time_calculator
    def fit_rfecv(self, x, y=None):
        self.feature_dimension=x.shape[1]
        sf=StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        self.selector=RFECV(estimator=self.estimator, cv=sf).fit(x, y)
        return self
    
    def transform(self, texts):
        return self.selector.transform(texts)