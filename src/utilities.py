#!/usr/bin/env python
# coding: utf-8

# In[7]:


import os
import pandas as pd
    
def read_csv(path, delimiter='\t'):
    return pd.read_csv(path, delimiter=delimiter)

def save_to_csv(df, path, delimiter='\t'):
    df.to_csv(path, index=False, sep=delimiter)