#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src.data.preprocessing.preprocess import Preprocess
from src import utilities as u
from src import utilities_preprocessing as pre
from src.decorators import * 

class PreprocessNgram(Preprocess):
    '''Preporcesses data for ngram features.'''
    def __init__(self):
        super().__init__()
        self.file_name = 'preprocessed_data_ngram.csv'
        self.column_name_ngram = 'text_ngram'

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
        
    def clean_for_ngrams(self, text):
        text = pre.clean_tweet(text)
        text = pre.remove_RT(text)
        text = pre.remove_new_lines(text)
        text = pre.expand_contractions(text)
        text = pre.remove_numbers(text)
        text = pre.remove_punctuations(text)
        text = pre.lower(text)
        text = pre.remove_repeating_chars(text)
        text = pre.tokenize(text)
        text = [pre.stem(str(word)) for word in text]
        return text
        
    def preprocess(self):
        print('\nSTARTED: Preprocessing ngram started.\n')
        
        print('Dropping NaN values...')
        self.data = pre.drop_nan_values(self.data, self.text_column)
        
        print('Cleaning URLs, Mentions, Hastags, Reserved, Emojis, Smilies, Numbers, RT, new lines, contractions, punctuations, repeating chars, tokenizing, stemming...')
        self.data[self.column_name_ngram] = self.data.copy()[self.text_column].apply(lambda x: self.clean_for_ngrams(str(x)))
        
        print('Dropping NaN values...')
        self.data = pre.drop_nan_values(self.data, self.column_name_ngram)
        
        #Save
        full_path=self.processed_data_path + self.file_name
        u.save(self.data, full_path)
        print('Saved preprocessed data: ' + full_path)
        
        print('\nFINISHED: Preprocessing ngram finished.\n')