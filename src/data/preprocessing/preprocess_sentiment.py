#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src.data.preprocessing.preprocess import Preprocess
from src import utilities as u
from src import utilities_preprocessing as pre
from src.decorators import * 

class PreprocessSentiment(Preprocess):
    '''Preporcesses data for sentiment features.'''
    def __init__(self):
        super().__init__()
        self.file_name = 'preprocessed_data_sentiment.csv'
        self.column_name_sentiment = 'text_sentiment'

    @property
    def file_name(self):
        return self._file_name
 
    @file_name.setter
    def file_name(self,value):
        self._file_name = value
        
    @property
    def data(self):
        return self._data 
    
    @data.setter
    def data(self,value):
        self._data = value
                
    def preprocess(self):
        print('\nSTARTED: Preprocessing sentiment started.\n')
        
        print('Dropping NaN values...')
        self.data = pre.drop_nan_values(self.data, self.text_column)
        
        print('Removing URLs...')
        self.data[self.column_name_sentiment] = self.data[self.text_column].apply(lambda x: pre.remove_URLs(str(x)))
        
        #Save
        full_path=self.processed_data_path + self.file_name
        u.save(self.data, full_path)
        print('Saved preprocessed data: ' + full_path)
        
        print('\nFINISHED: Preprocessing sentiment finished.\n')