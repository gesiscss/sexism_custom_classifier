#src module
from src.data.preprocessing.preprocess_sentiment import PreprocessSentiment
from src.data.preprocessing.preprocess_ngram import  PreprocessNgram
from src.data.preprocessing.preprocess_type_dependency import PreprocessTypeDependency
from src.data.preprocessing.preprocess_bert import PreprocessBert
from src.data.preprocessing.preprocess_textvec import PreprocessTextVec
from src.data.preprocessing.preprocess_gender_word import PreprocessGenderWord

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
    
    def build_pipeline_gender_word(self):
        return [('selector', ItemSelector(key='text')),
                ('preprocessing', PreprocessGenderWord())]
    
    def build_pipeline_toxicity(self):
        return [('selector', ItemSelector(key='toxicity'))]
    
    def build_pipeline(self, name, feature_selection=False):
        pipeline=get_attr(self, ''.join(('build_pipeline_', name)))
        if feature_selection:
            pipeline.append(('feature_selection', SelectorRFECV()))
        return Pipeline(pipeline)
    
    def build_feature_union(self, combination, extracted_features):
        '''Retrieves combineed polarity scores of text (neutral and compoound scores). 
    
        Args:
        combination (list): Feature names.
        extracted_features (dict): Extracted features.
    
        Returns:
        object: FeatureUnion object that includes list of transformer objects to be applied to the data.
    
        Example:
            >>> combination = ['sentiment', 'ngram']
            >>> extracted_features={ 'sentiment':[0.4, 0.5], 'ngram':[0.003,...,0.2] }
            >>> PipelineBuilderFeatures().build_feature_union(combination, extracted_features)
            FeatureUnion([
                            ('sentiment', FeatureClass([0.4, 0.5])), 
                            ('type_dependency', FeatureClass([0.003,...,0.2]))
                         ]
        '''
        transformer_list=[]
        for name in combination:
            transformer_list.append( (name, FeatureClass(extracted_features[name])) )
        return FeatureUnion(transformer_list=transformer_list)
    
class FeatureClass(BaseEstimator):
    def __init__(self, features=None):
        self.features=features
    
    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return self.features