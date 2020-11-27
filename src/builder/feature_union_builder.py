#src module
from src.enums import Feature
from src.utilities import get_attr

from src.feature_extraction.build_sentiment_features import BuildSentimentFeature
from src.feature_extraction.build_ngram_features import BuildNgramFeature
from src.feature_extraction.build_type_dependency_features import BuildTypeDependencyFeature
from src.feature_extraction.build_bert_features import BuildBERTFeature
from src.feature_extraction.build_text_vect_features import BuildTextVecFeature

from src.data.preprocessing.preprocess_ngram import  PreprocessNgram
from src.data.preprocessing.preprocess_sentiment import PreprocessSentiment
from src.data.preprocessing.preprocess_type_dependency import PreprocessTypeDependency
from src.data.preprocessing.preprocess_bert import PreprocessBert
from src.data.preprocessing.preprocess_textvec import PreprocessTextVec

from src.feature_selection.selector_rfecv import SelectorRFECV

#sklearn
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion
 
class FeatureUnionBuilder():
    
    def get_transformer_textvec(self):
        return (Feature.TEXTVEC, Pipeline
                ([
                    ('preprocessing', PreprocessTextVec()),
                    ('feature_extraction', BuildTextVecFeature()),
                ])
               )
    
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
                    ('feature_selection', SelectorRFECV()),
                ])
               )
    
    def get_transformer_type_dependency(self):
        return (Feature.TYPEDEPENDENCY, Pipeline
                ([
                    ('preprocessing', PreprocessTypeDependency()),
                    ('feature_extraction', BuildTypeDependencyFeature()),
                    ('feature_selection', SelectorRFECV()),
                ])
               )
    
    def get_transformer_bert(self):
        return (Feature.BERT, Pipeline
                ([
                    ('preprocessing', PreprocessBert()),
                    ('feature_extraction', BuildBERTFeature()),
                ])
               )

    def get_transformer(self, feature):
        method_name = 'get_transformer_' + feature
        return get_attr(self, method_name)
    
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

    def get_feature_union(self, features):
        return FeatureUnion(transformer_list=self.get_transformer_list(features))