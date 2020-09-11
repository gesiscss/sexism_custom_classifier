#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src.data.preprocessing.preprocess import Preprocess
from src import utilities as u
from src import utilities_preprocessing as pre
from src.decorators import * 

class PreprocessTypeDependency(Preprocess):
    '''Preporcesses data for type dependency features.'''
    def __init__(self):
        super().__init__()
        self.file_name = 'preprocessed_data_type_dependency.csv'
        self.data = None

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
        #TODO
        print('\nSTARTED: Preprocessing type dependency started.\n')
        
        return self.data, self.file_name