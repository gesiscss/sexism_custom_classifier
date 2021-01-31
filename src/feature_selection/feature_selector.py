#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.linear_model import RidgeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.utilities import *

from sklearn.feature_selection import SelectKBest, chi2
from sklearn.feature_selection import SelectFromModel

class FeatureSelector(BaseEstimator):

    def __init__(self, name='SelectKBest', cv=5, random_state=0, k=0):
        self.name=name
        self.cv=cv
        self.random_state=random_state
        
        self.selector=None
        self.estimator=SVR(kernel="linear")
        self.k=k
        self.feature_dimension=0
    
    def fit(self, x, y=None):
        if self.name == 'RFECV':
            return self.fit_rfecv(x, y)
        elif self.name == 'SelectKBest':
            return self.fit_selectkbest(x, y)
    
    #@execution_time_calculator
    def fit_selectkbest(self, x, y=None):
        self.feature_dimension=x.shape[1]
        #if self.k == 0:
        #    self.k=x.shape[1]-int(x.shape[1]/3)
        #print('SelectKBest fit start. ', x.shape, self.k)
        self.selector=SelectKBest(k=self.k).fit(x, y)
        return self
    
    @execution_time_calculator
    def fit_rfecv(self, x, y=None):
        #print('SelectorRFECV fit start. ', x.shape)
        sf=StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        self.selector=RFECV(estimator=self.estimator, cv=sf).fit(x, y)
        return self
    
    def transform(self, texts):
        ret= self.selector.transform(texts)
        #print('Selector transform end. ', ret.shape)
        return ret