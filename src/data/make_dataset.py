#!/usr/bin/env python
# coding: utf-8

# In[3]:

from src.data.preprocessing import preprocess_sentiment as p_s
from src.data.preprocessing import preprocess_ngram as p_ng
from src.data.preprocessing import preprocess_type_dependency as p_td
from src.data.preprocessing import preprocess_bert_doc_emb as p_bde
import pandas as pd

class MakeDataset:
    '''Preporcesses data for sentiment, ngram, type dependency and Bert document embedding features.'''
    def __init__(self):
        self.data_full_path = '../../data/raw/all_data.csv'
            
    def read_data(self):
        return pd.read_csv(self.data_full_path, delimiter='\t')
    
    def preprocess_sentiment(self, data):
        ps = p_s.PreprocessSentiment()
        ps.data = data
        ps.preprocess()
    
    def preprocess_ngram(self, data):
        png = p_ng.PreprocessNgram()
        png.data = data
        png.preprocess()
        
    def preprocess_type_dependency(self, data):
        ptd = p_td.PreprocessTypeDependency()
        ptd.data = data
        ptd.preprocess()

    def preprocess_bert_doc_emb(self, data):
        pbde = p_bde.PreprocessBertDocEmb()
        pbde.data = data
        pbde.preprocess()