#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src.enums import Feature
from src.data.preprocessing.preprocess import Preprocess
from src import utilities as u
from src import utilities_preprocessing as pre
from src.decorators import * 

class PreprocessSentiment(Preprocess):
    '''Preporcesses data for sentiment features.'''
    def __init__(self):
        super().__init__()

    @property
    def data(self):
        return self._data 
    
    @data.setter
    def data(self,value):
        self._data = value
        
    def clean(self, text):
        text = pre.remove_URLs(text)
        text = pre.remove_new_lines(text)
        return text
                
    def preprocess(self):
        '''1.Preprocesses text column from self.data. 
           2.Inserts preprocessed text as a new 'sentiment' column
           
           Returns:
           data: Data including preprocessed text.
        '''
        print('\nSTARTED: Preprocessing sentiment started.\n')
        
        print('Dropping NaN values...')
        self.data = pre.drop_nan_values(self.data, self.text_column)
        
        print('Cleaning URLs, Mentions, Hastags, RT, new lines, contractions')
        self.data[Feature.SENTIMENT] = self.data.copy()[self.text_column].apply(lambda x: self.clean(str(x)))
        
        print('Dropping NaN values...')
        self.data = pre.drop_nan_values(self.data, Feature.SENTIMENT)
        
        self.data['sexist'] = self.data.copy()['sexist'].astype(int)
        self.data = self.data.drop(['toxicity', 'tweet_id'], axis=1)
        
        return self.data