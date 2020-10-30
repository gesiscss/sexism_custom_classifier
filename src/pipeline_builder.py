#src module
from src.feature_union_builder import FeatureUnionBuilder
from src.models.model_builder import ModelBuilder

#sklearn
from sklearn.pipeline import Pipeline

class PipelineBuilder():
    def __init__(self, model, features):
        self.model=model
        self.features=features
        
    def build_pipeline(self):
        return Pipeline([
            ("features", FeatureUnionBuilder().get_feature_union(self.features)),
            ('model', ModelBuilder().get_model(self.model)),
        ])