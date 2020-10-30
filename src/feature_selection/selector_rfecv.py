#src module
from src.builder.model_builder import ModelBuilder

#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_selection import RFECV

class SelectorRFECV(BaseEstimator):
    
    def __init__(self, model=None):
        self.model=model
        self.selector=None
    
    def fit(self, x, y=None):
        estimator=ModelBuilder().get_model(self.model)
        self.selector=RFECV(estimator, step=1, cv=5).fit(x, y)
        return self
    
    def transform(self, texts):
        return self.selector.transform(texts)