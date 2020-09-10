
# coding: utf-8

# In[1]:


from enum import Enum

class Domain(Enum):
    BHO=1
    C=2
    BHOCS=3
    S=4
    BHOM=5
    CM=6
    BHOOCSM=7

class Model(Enum):
    LR=1
    CNN=2
    SVM=3

