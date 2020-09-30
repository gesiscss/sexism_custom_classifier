#src module
from src.enums import Feature
from src import factory
from src.item_selector import ItemSelector
from src.features.build_sentiment_features import BuildSentimentFeature
from src.features.build_ngram_features import BuildNgramFeature
from src.features.build_type_dependency_features import BuildTypeDependencyFeature
from src.features.build_bert_doc_emb_features import BuildBERTDocumentEmbeddingsFeature
from src.features.select_features import SelectFeatures

#sklearn
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
 
class Transformers():
    
    def __init__(self):
        self.build_feature_objects={
                Feature.SENTIMENT: BuildSentimentFeature,
                Feature.NGRAM: BuildNgramFeature,
                Feature.TYPEDEPENDENCY: BuildTypeDependencyFeature,
                Feature.BERTDOCEMB: BuildBERTDocumentEmbeddingsFeature
        }
        
    def get_feature_builder(self, feature, params):
        feature_builder=factory.get_object(self.build_feature_objects, feature)
        feature_builder.set_params(**params)
        return feature_builder
    
    def get_transformer_sentiment(self, params):
        return (Feature.SENTIMENT, Pipeline
                ([
                    ('selector', ItemSelector(key=Feature.SENTIMENT)),
                    ('features', self.get_feature_builder(Feature.SENTIMENT, params)),
                ])
               )

    def get_transformer_ngram(self, params):
        return (Feature.NGRAM, Pipeline
                ([
                    ('selector', ItemSelector(key=Feature.NGRAM)),
                    ('features', self.get_feature_builder(Feature.NGRAM, params)),
                    ('feature_selection', SelectFeatures()),
                ])
               )
    def get_transformer_type_dependency(self, params):
        return (Feature.TYPEDEPENDENCY, Pipeline
                ([
                    ('selector', ItemSelector(key=Feature.TYPEDEPENDENCY)),
                    ('features', self.get_feature_builder(Feature.TYPEDEPENDENCY, params)),
                    ('feature_selection', SelectFeatures()),
                ])
               )
    
    def get_transformer_bert_doc_emb(self, params):
        return (Feature.BERTDOCEMB, Pipeline
                ([
                    ('selector', ItemSelector(key=Feature.BERTDOCEMB)),
                    ('features', self.get_feature_builder(Feature.BERTDOCEMB, params)),
                ])
               )

    def get_transformer(self, feature, params):
        method_name = 'get_transformer_' + feature
        method = getattr(self, method_name)
        return method(params)
    
    def get_transformer_list(self, features):
        '''Gets transformer list. 
        
        Args:
        features (list): Features that will be added to the pipeline
        
        Example:
        >>>features = { Feature.NGRAM: {'ngram_range':(1,1)}, }
        >>>get_transformer_list(features)
        '''
        transformer_list=[]
        for key, value in features.items():
            transformer_list.append(self.get_transformer(key, value))
        return transformer_list

    def get_combined_features(self, features):
        return FeatureUnion(transformer_list=self.get_transformer_list(features))