#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from src.utilities import *

class SelectorRFECV(BaseEstimator):
    
    def __init__(self, model=None):
        self.selector=None
    
    #@execution_time_calculator
    def fit(self, x, y=None):
        #self.selector=RFECV(estimator=SVR(kernel="linear"), cv=2, scoring='accuracy').fit(x, y)
        self.selector=RFECV(estimator=LogisticRegression(), cv=2, scoring='accuracy').fit(x, y)
        return self
    
    def transform(self, texts):
        return self.selector.transform(texts)