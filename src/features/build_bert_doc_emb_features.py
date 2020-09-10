#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src.features.build_features import BuildFeature

class BuildBERTDocumentEmbeddingsFeature(BuildFeature):
    '''Extracts BERT document embeddings Features'''
    
    def fit(self, x, y=None):
        #TODO
        return self

    def transform(self, texts):
        #TODO
        return None