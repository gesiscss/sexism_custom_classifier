#src module
from src.builder.pipeline_builder_models import PipelineBuilderModels
from src.builder.pipeline_builder_features import PipelineBuilderFeatures

class PipelineBuilder():
    def __init__(self):
        self.pb_models=PipelineBuilderModels()
        self.pf_features=PipelineBuilderFeatures()
        
    def build_model_pipeline(self, model):
        return self.pb_models.build_pipeline(model)
        
    def build_feature_pipeline(self, name, feature_selection=False):
        return self.pf_features.build_pipeline(name, feature_selection)
       
    def build_feature_union(self, combination, extracted_features):
        return self.pf_features.build_feature_union(combination, extracted_features)