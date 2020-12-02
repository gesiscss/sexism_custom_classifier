#src module
from src.builder.feature_union_builder import FeatureUnionBuilder
from src.builder.model_builder import ModelBuilder

#sklearn
from sklearn.pipeline import Pipeline

class PipelineBuilder():
    def __init__(self, features):
        self.features=features
        
    def build_pipeline(self):
        return Pipeline([
            ("features", FeatureUnionBuilder().get_feature_union(self.features)),
            ('model', ModelBuilder()),
        ])