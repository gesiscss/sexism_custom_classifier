#src module
from src.enums import Model
from src.models.logit import Logit
from src.models.svm import SVM
from src.models.cnn import CNN

#sklearn
from sklearn.base import BaseEstimator

class ModelBuilder(BaseEstimator):
    '''Builds model.'''
    def get_logistic_regression(self):
        return Logit()
    
    def get_svm(self):
        return SVM()
    
    def get_cnn(self):
        return CNN()

    def get_model(self, model_name):
        method_name = 'get_' + model_name
        method = getattr(self, method_name)
        return method()