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
    
    def get_pipeline_textvec(self):
        return [('preprocessing', PreprocessTextVec()), ('feature_extraction', BuildTextVecFeature())]
    
    def get_pipeline_sentiment(self):
        return [('preprocessing', PreprocessSentiment()), ('feature_extraction', BuildSentimentFeature())]

    def get_pipeline_ngram(self):
        return [('preprocessing', PreprocessNgram()), ('feature_extraction', BuildNgramFeature())]
    
    def get_pipeline_type_dependency(self):
        return [('preprocessing', PreprocessTypeDependency()), ('feature_extraction', BuildTypeDependencyFeature())]
    
    def get_pipeline_bert_doc(self):
        return [('preprocessing', PreprocessBert()), ('feature_extraction', BuildBERTFeature())]
    
    def get_pipeline_bert_word(self):
        return [('preprocessing', PreprocessBert()), ('feature_extraction', BuildBERTFeature())]
    
    def get_pipeline(self, name, feature_selection):
        method_name = 'get_pipeline_' + name
        
        pipeline=get_attr(self, method_name)
        
        if feature_selection:
            pipeline.append(('feature_selection', SelectorRFECV()))
        
        return (name, Pipeline(pipeline))
    
    def get_transformer_list(self, features):
        '''Gets transformer list. 
        
        Args:
        features (list): Features that will be added to the pipeline
        
        Example:
        >>>features = [{'name': 'sentiment', 'feature_selection': False}, {'name': 'ngram', 'feature_selection': True}]
        >>>get_transformer_list(features)
        '''
        print('feature union ', features)
        transformer_list=[]
        for feature in features:
            transformer_list.append(self.get_pipeline(**feature))
        return transformer_list

    def get_feature_union(self, features):
        return FeatureUnion(transformer_list=self.get_transformer_list(features))