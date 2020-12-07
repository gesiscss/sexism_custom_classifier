#sklearn
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

#src module
from src.model.cnn import CNN
from src.model.baseline.gender_word import GenderWord
from src.model.baseline.threshold_classifier import ThresholdClassifier

from src.builder.feature_union_builder import FeatureUnionBuilder
from src.builder.model_builder import ModelBuilder
from src.builder.item_selector import ItemSelector

from src.utilities import get_attr, get_object
from src.enums import Model
from src.data.preprocessing.preprocess_gender_word import PreprocessGenderWord

build_model_objects={
                Model.LR: LogisticRegression,
                Model.SVM: SVC,
                Model.CNN: CNN,
                Model.GENDERWORD: GenderWord,
                Model.THRESHOLDCLASSIFIER: ThresholdClassifier,
        }

class PipelineBuilder():
    def __init__(self, features, model_name):
        self.features=features
        self.model=model_name
        self.model_obj=get_object(build_model_objects, model_name)
        
    def get_pipeline_gender_word(self):
        return Pipeline([('selector', ItemSelector(key='text')),
                         ('preprocessing', PreprocessGenderWord()),
                         ('model', ModelBuilder(self.model_obj))
        ])
    
    def get_pipeline_threshold_classifier(self):
        return Pipeline([('selector', ItemSelector(key='toxicity')),
                         ('model', ModelBuilder(self.model_obj))
        ])
    
    def get_pipeline_svm(self):
        return Pipeline([("features", FeatureUnionBuilder().get_feature_union(self.features)),
                         ('model', ModelBuilder(self.model_obj))
        ])

    def get_pipeline_cnn(self):
        return Pipeline([("features", FeatureUnionBuilder().get_feature_union(self.features)),
                         ('model', ModelBuilder(self.model_obj))
        ])
    
    def get_pipeline_logistic_regression(self):
        return Pipeline([("features", FeatureUnionBuilder().get_feature_union(self.features)),
                         ('model', ModelBuilder(self.model_obj))
        ])
                        
    def build_pipeline(self):
        return get_attr(self, ''.join(('get_pipeline_', self.model)))