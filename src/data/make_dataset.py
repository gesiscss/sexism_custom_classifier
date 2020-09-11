#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src import utilities as u
from src.data.preprocessing import preprocess_sentiment as p_s
from src.data.preprocessing import preprocess_ngram as p_ng
from src.data.preprocessing import preprocess_type_dependency as p_td
from src.data.preprocessing import preprocess_bert_doc_emb as p_bde

#other
import pandas as pd


class MakeDataset:
    '''Preporcesses data for sentiment, ngram, type dependency and Bert document embedding features.'''
    def __init__(self):
        self.data_full_path = '../../data/raw/all_data.csv'
        self.preprocessed_data_path = '../../data/processed/'
            
    def read_data(self):
        return pd.read_csv(self.data_full_path, delimiter='\t')
    
    def save(self, data, path, file_name):
        u.save(data, path + file_name)
        print('Saved preprocessed data: ' + str(path + file_name))
    
    def preprocess_sentiment(self, data):
        ps = p_s.PreprocessSentiment()
        ps.data = data
        pre_data, file_name = ps.preprocess()
        self.save(pre_data, self.preprocessed_data_path, file_name)
    
    def preprocess_ngram(self, data):
        png = p_ng.PreprocessNgram()
        png.data = data
        pre_data, file_name = png.preprocess()
        self.save(pre_data, self.preprocessed_data_path, file_name)
        
    def preprocess_type_dependency(self, data):
        ptd = p_td.PreprocessTypeDependency()
        ptd.data = data
        pre_data, file_name = ptd.preprocess()
        self.save(pre_data, self.preprocessed_data_path, file_name)

    def preprocess_bert_doc_emb(self, data):
        pbde = p_bde.PreprocessBertDocEmb()
        pbde.data = data
        pre_data, file_name = pbde.preprocess()
        self.save(pre_data, self.preprocessed_data_path, file_name)