#src module
from src.enums import Feature

from src.feature_extraction.build_sentiment_features import BuildSentimentFeature
from src.feature_extraction.build_ngram_features import BuildNgramFeature
from src.feature_extraction.build_type_dependency_features import BuildTypeDependencyFeature
from src.feature_extraction.build_bert_doc_emb_features import BuildBERTDocumentEmbeddingsFeature

from src.data.preprocessing.preprocess_ngram import  PreprocessNgram
from src.data.preprocessing.preprocess_sentiment import PreprocessSentiment
from src.data.preprocessing.preprocess_type_dependency import PreprocessTypeDependency
from src.data.preprocessing.preprocess_bert_doc_emb import PreprocessBertDocEmb

from src.feature_selection.select_features import SelectFeatures

#sklearn
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
 
class Transformers():
    
    def get_transformer_sentiment(self):
        return (Feature.SENTIMENT, Pipeline
                ([
                    ('preprocessing', PreprocessSentiment()),
                    ('feature_extraction', BuildSentimentFeature()),
                ])
               )

    def get_transformer_ngram(self):
        return (Feature.NGRAM, Pipeline
                ([
                    ('preprocessing', PreprocessNgram()),
                    ('feature_extraction', BuildNgramFeature()),
                    ('feature_selection', SelectFeatures()),
                ])
               )
    
    def get_transformer_type_dependency(self):
        return (Feature.TYPEDEPENDENCY, Pipeline
                ([
                    ('preprocessing', PreprocessTypeDependency()),
                    ('feature_extraction', BuildTypeDependencyFeature()),
                    ('feature_selection', SelectFeatures()),
                ])
               )
    
    def get_transformer_bert_doc_emb(self):
        return (Feature.BERTDOCEMB, Pipeline
                ([
                    ('preprocessing', PreprocessBertDocEmb()),
                    ('feature_extraction', BuildBERTDocumentEmbeddingsFeature()),
                ])
               )

    def get_transformer(self, feature):
        method_name = 'get_transformer_' + feature
        method = getattr(self, method_name)
        return method()
    
    def get_transformer_list(self, features):
        '''Gets transformer list. 
        
        Args:
        features (list): Features that will be added to the pipeline
        
        Example:
        >>>features = [Feature.SENTIMENT, Feature.NGRAM, ]
        >>>get_transformer_list(features)
        '''
        transformer_list=[]
        for feature in features:
            transformer_list.append(self.get_transformer(feature))
        return transformer_list

    def get_combined_features(self, features):
        return FeatureUnion(transformer_list=self.get_transformer_list(features))