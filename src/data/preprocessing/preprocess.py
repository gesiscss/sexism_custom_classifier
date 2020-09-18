#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from abc import ABCMeta, abstractmethod

class Preprocess(metaclass = ABCMeta):
    
    def __init__(self):
        self.text_column = 'text'
        self.data = None
        
    @property
    @abstractmethod
    def data(self):
        pass
 
    @data.setter
    @abstractmethod
    def data(self,value):
        pass

    @abstractmethod
    def preprocess(self):
        pass