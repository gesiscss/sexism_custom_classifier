#src module
from src.data.preprocessing.preprocess_sentiment import PreprocessSentiment
from src.data.preprocessing.preprocess_ngram import  PreprocessNgram
from src.data.preprocessing.preprocess_type_dependency import PreprocessTypeDependency
from src.data.preprocessing.preprocess_bert import PreprocessBert
from src.data.preprocessing.preprocess_textvec import PreprocessTextVec

from src.feature_extraction.build_sentiment_features import BuildSentimentFeature
from src.feature_extraction.build_ngram_features import BuildNgramFeature
from src.feature_extraction.build_type_dependency_features import BuildTypeDependencyFeature
from src.feature_extraction.build_bert_features import BuildBERTFeature
from src.feature_extraction.build_text_vect_features import BuildTextVecFeature

from src.utilities import get_attr
from src.builder.item_selector import ItemSelector
from src.feature_selection.selector_rfecv import SelectorRFECV

#sklearn
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator
from sklearn.pipeline import FeatureUnion


class PipelineBuilderFeatures():
    
    def build_pipeline_sentiment(self):
        return [('selector', ItemSelector(key='text')),
                ('preprocessing', PreprocessSentiment()), 
                ('feature_extraction', BuildSentimentFeature())]

    def build_pipeline_ngram(self):
        return [('selector', ItemSelector(key='text')),
                ('preprocessing', PreprocessNgram()), 
                ('feature_extraction', BuildNgramFeature())]
    
    def build_pipeline_type_dependency(self):
        return [('selector', ItemSelector(key='text')),
                ('preprocessing', PreprocessTypeDependency()), 
                ('feature_extraction', BuildTypeDependencyFeature())]
    
    def build_pipeline_bert_doc(self):
        return [('selector', ItemSelector(key='text')),
                ('preprocessing', PreprocessBert()), 
                ('feature_extraction', BuildBERTFeature())]
    
    def build_pipeline_bert_word(self):
        return [('selector', ItemSelector(key='text')),
                ('preprocessing', PreprocessBert()), 
                ('feature_extraction', BuildBERTFeature())]
    
    def build_pipeline_textvec(self):
        return [('selector', ItemSelector(key='text')),
                ('preprocessing', PreprocessTextVec()), 
                ('feature_extraction', BuildTextVecFeature())]
    
    def build_pipeline(self, name, feature_selection=False):
        pipeline=get_attr(self, ''.join(('build_pipeline_', name)))
        if feature_selection:
            pipeline.append(('feature_selection', SelectorRFECV()))
        return Pipeline(pipeline)
    
    def build_feature_union(self, combination, extracted_features):
        '''
        #e.g., params : combination = (0, 2)     extracted_features=[ {'name': 'sentiment',       'value':[...]},
        #                                                             {'name': 'ngram',           'value':[...]},
        #                                                             {'name': 'type_dependency', 'value':[...]}  ]
        #transformer_list=[
        #                        ('sentiment',       FeatureClass([...]))
        #                        ('type_dependency', FeatureClass([...]))
        #                      ]
        '''
        transformer_list=[]
        for c in combination:
            #Without FeatureClass, it raises error.
            transformer_list.append( (extracted_features[c]['name'], FeatureClass(extracted_features[c]['value'])) )
        return FeatureUnion(transformer_list=transformer_list)
    
class FeatureClass(BaseEstimator):
    def __init__(self, features=None):
        self.features=features
    
    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return self.features