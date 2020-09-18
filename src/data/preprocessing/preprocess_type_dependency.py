#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src.enums import Feature
from src.data.preprocessing.preprocess import Preprocess
from src import utilities_preprocessing as pre
from src.decorators import * 

class PreprocessTypeDependency(Preprocess):
    '''Preporcesses data for type dependency features.'''
    def __init__(self):
        super().__init__()

    @property
    def data(self):
        return self._data 
    
    @data.setter
    def data(self,value):
        self._data = value
        
    def clean(self, text):
        text = pre.clean_tweet(text)
        text = pre.remove_RT(text)
        text = pre.remove_new_lines(text)
        text = pre.expand_contractions(text)
        text = pre.remove_numbers(text)
        text = pre.lower(text)
        text = pre.remove_repeating_chars(text)
        return text
        
    def preprocess(self):
        '''1.Preprocesses text column from self.data. 
           2.Inserts preprocessed text as a new 'type_dependency' column
           
           Returns:
           data: Data including preprocessed text.
        '''
        print('\nSTARTED: Preprocessing type dependency started.\n')
        
        print('Dropping NaN values...')
        self.data = pre.drop_nan_values(self.data, self.text_column)
        
        print('Cleaning URLs, Mentions, Hastags, Reserved, Emojis, Smilies, RT, new lines, contractions, repeating chars...')
        self.data[Feature.TYPEDEPENDENCY] = self.data.copy()[self.text_column].apply(lambda x: self.clean(str(x)))
        
        print('Dropping NaN values...')
        self.data = pre.drop_nan_values(self.data, Feature.TYPEDEPENDENCY)
        self.data = self.data.copy()[self.data[Feature.TYPEDEPENDENCY] != '']
        
        self.data['sexist'] = self.data.copy()['sexist'].astype(int)
        self.data = self.data.drop(['toxicity', 'tweet_id'], axis=1)
        
        return self.data