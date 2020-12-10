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
    
    def build_pipeline(self, model):
        return Pipeline([('model', ModelBuilder(get_object(build_model_objects, model)))])