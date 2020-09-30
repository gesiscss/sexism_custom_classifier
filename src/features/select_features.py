#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression

class SelectFeatures(BaseEstimator):
    
    def __init__(self, estimator=None):
        self.estimator=estimator
        self.selector=None
    
    def fit(self, x, y=None):
        print('fit SelectFeatures start x.shape {} y.shape {}'.format(x.shape, y.shape))
        #print('fit SelectFeatures {} {}'.format(x, y))
        #self.estimator=SVC(kernel="linear")
        self.estimator=LogisticRegression()
        self.selector=RFECV(self.estimator, step=1, cv=5)
        self.selector.fit(x, y)
        s=self.selector.support_
        print('fit SelectFeatures end == selector.support_.shape {} selected features count {}'.format(s.shape, len(s[s == True]) ) )
        return self

    def transform(self, texts):
        #print('transform SelectFeatures start texts.shape {}'.format(texts.shape))
        selected_fetures= self.selector.transform(texts)
        #print('transform SelectFeatures end selected_fetures.shape {}'.format(selected_fetures.shape))
        return selected_fetures