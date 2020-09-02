#!/usr/bin/env python
# coding: utf-8

# In[3]:

from src import utilities as u
from src.decorators import * 
from src.data import preprocessing as pre

import pandas as pd


class MakeDataset:
    '''Preprocesses and saves dataset.'''
    
    def __init__(self, df, project_path):
        '''Constructs a MakeDataset.
        Args:
        df: pandas dataframe
        project_path : Project path (e.g. C:/Users/username/Desktop/)
        '''
        self.df = df
        self.project_path = project_path
        self.processed_data_path = 'cc/data/processed/'
    
    def save(self, file_name):
        u.save(self.df, self.project_path + self.processed_data_path + file_name)
        
    def clean_for_ngrams(self, text):
        text = pre.clean_tweet(text)
        text = pre.remove_RT(text)
        text = pre.remove_new_lines(text)
        text = pre.expand_contractions(text)
        text = pre.remove_numbers(text)
        text = pre.remove_punctuations(text)
        text = pre.lower(text)
        text = pre.remove_repeating_chars(text)
        return text
    
    @execution_time_calculator
    def make_dataset_for_sentiment(self):
        '''Preprocesses and saves self.df'''
        
        #Drop NaN values
        print('Dropping NaN values...')
        self.df = self.df.copy()[self.df['text'].notna()]
        
        #Save
        print('Saving preprocessed file...')
        self.save('all_data_preprocessed_for_sentiment.csv')
        
        print('Finished.')
    
    @execution_time_calculator
    def make_dataset_for_ngrams(self):
        '''Preprocesses and saves self.df'''
        
        #Drop NaN values
        print('Dropping NaN values...')
        self.df = self.df.copy()[self.df['text'].notna()]
        
        #Clean for ngrams
        print('Cleaning URLs, Mentions, Hastags, Reserved, Emojis, Smilies, Numbers, RT, new lines, contractions, punctuations, repeating chars...')
        self.df['text'] = self.df.copy()['text'].apply(lambda x: self.clean_for_ngrams(str(x)))
        
        #Tokenize
        print('Tokenizing...')
        self.df.loc[:, 'text'] = self.df.copy()['text'].apply(lambda x: pre.tokenize(str(x)))
        
        #Stemming
        print('Stemming...')
        self.df.loc[:, 'text'] = self.df.copy()['text'].apply(lambda x: [pre.stem(str(word)) for word in x])
        
        #Save
        print('Saving preprocessed file...')
        self.save('all_data_preprocessed_for_ngrams.csv')
        
        print('Finished.')