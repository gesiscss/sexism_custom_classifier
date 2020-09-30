#src module
from src.transformers import Transformers
from src.model_builder import ModelBuilder

#sklearn
from sklearn.pipeline import Pipeline

class PipelineBuilder():
    def __init__(self, model, features):
        self.model=model
        self.features=features
        self.transformers=Transformers()
        self.model_builder=ModelBuilder()
    
    def build_pipeline_for_features(self):
        return Pipeline([
            ("features", self.transformers.get_combined_features(self.features)),
        ]) 
        return self
    
    def get_estimator(self):
        return Pipeline([
            ('model', self.model_builder.get_model(self.model)),
        ])