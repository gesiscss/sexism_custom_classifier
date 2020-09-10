
# coding: utf-8

# In[1]:

#src modue
from src.item_selector import ItemSelector
from src.features.build_sentiment_features import BuildSentimentFeature
from src.features.build_ngram_features import BuildNgramFeature

#sklearn
from sklearn.pipeline import Pipeline
from sklearn.pipeline import FeatureUnion

class Transformers():
    #TODO: Feature selection options
    def get_transformer_sentiment(self):
        return ('sentiment', Pipeline
                ([
                    ('selector', ItemSelector(key='text_sentiment')),
                    ('features', BuildSentimentFeature())
                ])
               )

    def get_transformer_ngram(self):
        return ('ngram', Pipeline
                ([
                    ('selector', ItemSelector(key='text_ngram')),
                    ('features', BuildNgramFeature())
                ])
               )
    def get_transformer_type_dependency(self):
        #TODO
        return ()
    
    def get_transformer_bert_doc_emb(self):
        #TODO
        return ()

    def get_transformer_list(self, **features):
        transformer_list=[]
    
        for key, value in features.items():
            if value == True:
                method_name = 'get_transformer_' + key
                method = getattr(self, method_name)
                transformer_list.append(method())
            
        return transformer_list

    def get_combined_features(self, **features):
        return FeatureUnion(transformer_list=self.get_transformer_list(**features))