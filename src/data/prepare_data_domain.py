#!/usr/bin/env python
# coding: utf-8

# In[9]:


from sklearn.model_selection import StratifiedShuffleSplit

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
            data_dict[i] = {'X_train': X[train_index], 'X_test': X[test_index], 'y_train': y[train_index], 'y_test': y[test_index]}
            i=i+1
        return data_dict