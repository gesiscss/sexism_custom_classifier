#sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#src module
from src.model.cnn import CNN
from src.model.baseline.gender_word import GenderWord
from src.model.baseline.threshold_classifier import ThresholdClassifier
from src.builder.model_builder import ModelBuilder
from src.builder.item_selector import ItemSelector
from src.data.preprocessing.preprocess_gender_word import PreprocessGenderWord

from src.utilities import get_object, get_attr
from src.enums import Model

build_model_objects={
                Model.LR: LogisticRegression,
                Model.SVM: SVC,
                Model.CNN: CNN,
                Model.GENDERWORD: GenderWord,
                Model.THRESHOLDCLASSIFIER: ThresholdClassifier,
        }

class PipelineBuilderModels():
    def __init__(self):
        self.model_obj=None
        
    def build_pipeline_logistic_regression(self):
        return Pipeline([('model', ModelBuilder(self.model_obj))])
    
    def build_pipeline_svm(self):
        return Pipeline([('model', ModelBuilder(self.model_obj))])
    
    def build_pipeline_cnn(self):
        return Pipeline([('model', ModelBuilder(self.model_obj))])
        
    def build_pipeline_gender_word(self):
        return Pipeline([('selector', ItemSelector(key='text')),
                         ('preprocessing', PreprocessGenderWord()),
                         ('model', ModelBuilder(self.model_obj))
        ])
    
    def build_pipeline_threshold_classifier(self):
        return Pipeline([('selector', ItemSelector(key='toxicity')),
                         ('model', ModelBuilder(self.model_obj))
        ])
                        
    def build_pipeline(self, model):
        self.model_obj=get_object(build_model_objects, model)
        return get_attr(self, ''.join(('build_pipeline_', model)))