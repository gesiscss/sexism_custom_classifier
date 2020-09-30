
# coding: utf-8

# In[1]:

#src module
from src.enums import Model

#sklearn
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

class ModelBuilder():
    '''Builds model.'''
    
    def get_logistic_regression(self):
        return LogisticRegression()
    
    def get_cnn(self):
        #TODO
        return SVC()
    
    def get_svm(self):
        return SVC()

    def get_model(self, model):
        method_name = 'get_' + model
        method = getattr(self, method_name)
        return method()

