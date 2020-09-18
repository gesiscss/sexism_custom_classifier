
# coding: utf-8

# In[1]:

def get_object(objects, name: object = None) -> object:
    '''Factory'''
    return objects[name]()

