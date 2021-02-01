#sklearn
from sklearn.base import BaseEstimator

class ModelBuilder(BaseEstimator):
    def __init__(self, estimator = None):
        self.estimator=estimator
        self.feature_dimension=0
        
    def fit(self, X, y=None, **kwargs):
        self.feature_dimension=X.shape[len(X.shape)-1]
        self.estimator.fit(X, y)
        return self

    def predict(self, X, y=None):
        #print('estimator predict ', X.shape)
        return self.estimator.predict(X)

    def predict_proba(self, X):
        return self.estimator.predict_proba(X)

    def score(self, X, y):
        return self.estimator.score(X, y)