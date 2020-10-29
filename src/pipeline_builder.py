#src module
from src.transformers import Transformers
from src.models.model_builder import ModelBuilder

#sklearn
from sklearn.pipeline import Pipeline

class PipelineBuilder():
    def __init__(self, model, features):
        self.model=model
        self.features=features
        
    def build_pipeline(self):
        return Pipeline([
            ("features", Transformers().get_combined_features(self.features)),
            ('model', ModelBuilder().get_model(self.model)),
        ])