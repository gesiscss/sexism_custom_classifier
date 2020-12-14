#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
#from sklearn.svm import SVR
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from src.utilities import *

class SelectorRFECV(BaseEstimator):
    
    def __init__(self, cv=5, random_state=0):
        self.cv=cv
        self.random_state=random_state
        self.selector=None
        
    @execution_time_calculator
    def fit(self, x, y=None):
        print('SelectorRFECV fit ', x.shape, self.cv)
        #self.selector=RFECV(estimator=SVR(kernel="linear"), cv=2, scoring='accuracy').fit(x, y)
        sf=StratifiedKFold(n_splits=self.cv, random_state=self.random_state, shuffle=True)
        self.selector=RFECV(estimator=LogisticRegression(max_iter=5000), cv=sf, scoring='accuracy').fit(x, y)
        print('£££ n_features_ selector.n_features_ ', self.selector.n_features_)
        return self
    
    def transform(self, texts):
        return self.selector.transform(texts)