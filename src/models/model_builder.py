#src module
from src.utilities import get_object
from src.enums import Model
from src.models.logit import Logit
from src.models.svm import SVM
from src.models.cnn import CNN

#sklearn
from sklearn.base import BaseEstimator

class ModelBuilder(BaseEstimator):
    '''Builds model.'''
    def __init__(self):
        self.build_model_objects={
                Model.LR: Logit,
                Model.SVM: SVM,
                Model.CNN: CNN,
        }
        
    def get_model(self, model_name):
        return get_object(self.build_model_objects, model_name)