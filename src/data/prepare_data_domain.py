#!/usr/bin/env python
# coding: utf-8

# In[9]:

#src module
from src import utilities as u
from src.enums import Domain

#sklearn
from sklearn.model_selection import StratifiedShuffleSplit

#other
import pandas as pd
 

class PrepareDataDomain:
    '''Prepares data domains.'''
    def __init__(self, n_splits=5, test_size=0.3):
        '''
        Args:
        n_split = number of re-shuffling & splitting iterations
        test_size = the proportion of the dataset to include in the test split
        '''
        
        self.sss = StratifiedShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=0)

    def get_n_splits(self, X, y):
        '''Split data into training and test set.
        Returns:
        data_dict (dictionary) = dictionary that includes training and  test set
        '''
        
        data_dict = {}
        i=1
        for train_index, test_index in self.sss.split(X, y):
            data_dict[i] = {
                'X_train': X.iloc[train_index], 
                'X_test': X.iloc[test_index], 
                'y_train': y.iloc[train_index], 
                'y_test': y.iloc[test_index]
            }
            i=i+1
        return data_dict
    
    #TODO: Add other feature paths
    def read_preprocessed_data(self):
        '''Gets preprocessed data for each feature set.'''
        
        pre_data_sentiment_full_path = '../../data/processed/preprocessed_data_sentiment.csv'
        pre_data_ngrams_full_path = '../../data/processed/preprocessed_data_ngram.csv'
        
        df_sentiment = u.read(pre_data_sentiment_full_path)
        df_ngrams = u.read(pre_data_ngrams_full_path)
        return df_sentiment, df_ngrams 
    
    def get_preprocessed_data_bhocs(self):
        '''Gets preprocessed original data domain - bhocs
        
        Returns:
        X (dataframe) = A dataframe that includes preprocessed text columns for each feature.
        y (dataframe) = A dataframe that includes labels.
        '''
        
        #Read preprocessed data
        df_sentiment, df_ngrams = self.read_preprocessed_data()
        
        #Get bhocs domain
        df_sentiment = df_sentiment[df_sentiment['of_id'].isnull()]
        df_ngrams = df_ngrams[df_ngrams['of_id'].isnull()]
        
        X = pd.concat([df_sentiment['text_sentiment'], df_ngrams['text_ngram']], axis=1, 
                         keys=['text_sentiment', 'text_ngram'])
        y = df_sentiment['sexist']
        
        return X, y
    
    #TODO: Add other domains
    def get_preprocessed_data(self, domain=''):
        '''Gets preprocessed data domain.
        
        Args:
        domain (enum) : Domain enum. {Domain.BHOCS}
        
        X (dataframe) = A dataframe that includes preprocessed text columns for each feature.
        y (dataframe) = A dataframe that includes labels.
        '''
        
        if domain == Domain.BHOCS:
            return self.get_preprocessed_data_bhocs()
        else:
            return None