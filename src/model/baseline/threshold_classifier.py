#sklearn
from sklearn.base import BaseEstimator
from sklearn.metrics import roc_curve

#other
import numpy as np
import re

class ThresholdClassifier(BaseEstimator):
    '''Toxicity baseline model.'''
    def __init__(self, threshold=0):
        self.threshold=0
        
    def fit(self, X, y):
        self.threshold = self.compute_best_thold(X, y)
        return self
    
    def predict(self, X):
        return list(map(lambda x: int(x>=self.threshold), X))
    
    def compute_best_thold(self, X, y):
        tp, fp, tholds = roc_curve(y, X)

        min_distance_01 = np.infty
        min_distance_01_idx = np.nan
        for idx, point in enumerate(zip(tp, fp)):
            #take the point that is closest to (0, 1), maximize acc
            distance_from_01 = np.sqrt((point[0])**2 + (1- point[1])**2)
            if distance_from_01 < min_distance_01:
                min_distance_01 = distance_from_01
                min_distance_01_idx = idx

        best_thold = tholds[min_distance_01_idx]
        return best_thold