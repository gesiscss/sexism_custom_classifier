
# coding: utf-8

# In[1]:

from src.enums import Model
from sklearn.svm import SVC

class ModelBuilder():
    '''Builds model.'''
    def get_model(self, model):
        #TODO
        if model == Model.LR:
            return SVC(kernel='linear')
        elif model == Model.CNN:
            return SVC(kernel='linear')
        elif model == Model.SVM:
            return SVC(kernel='linear')
        else:
            return None

