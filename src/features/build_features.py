#!/usr/bin/env python
# coding: utf-8

# In[ ]:


from abc import ABCMeta, abstractmethod

class BuildFeature(metaclass = ABCMeta):
        
    @abstractmethod
    def fit(self, x, y=None):
        pass

    @abstractmethod
    def transform(self, texts):
        pass