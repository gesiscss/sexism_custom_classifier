#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.svm import SVR

class SelectorRFECV(BaseEstimator):
    
    def __init__(self, model=None):
        self.selector=None
    
    def fit(self, x, y=None):
        self.selector=RFECV(estimator=SVR(kernel="linear"), step=1, cv=2).fit(x, y)
        return self
    
    def transform(self, texts):
        return self.selector.transform(texts)