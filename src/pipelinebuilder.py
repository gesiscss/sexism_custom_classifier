
# coding: utf-8

# In[ ]:

#src module
from src.transformers import Transformers
from src.model_builder import ModelBuilder

#sklearn
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

class PipelineBuilder():
    def __init__(self):
        self.transformers = Transformers()
        self.modelbuilder = ModelBuilder()

    def get_pipeline(self, model='', **features):
        return Pipeline([
            ("features", self.transformers.get_combined_features(**features)),
            ('model', self.modelbuilder.get_model(model))
        ]) 
    
    def fit_and_predict(self, split_dict, pipeline):
        for value in split_dict.values():
            X_train, y_train = value['X_train'], value['y_train']
            X_test, y_test = value['X_test'], value['y_test']

            pipeline.fit(X_train, y_train)
            y = pipeline.predict(X_test)
            print(classification_report(y, y_test))

