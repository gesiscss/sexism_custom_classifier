#!/usr/bin/env python
# coding: utf-8

# In[9]:

#src module
from src.data.make_dataset import MakeDataset
from src.enums import *

#sklearn
from sklearn.utils import shuffle
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
        
    def get_original_data(self, data):
        '''Gets original data.'''
        return data[data['of_id'].isnull()]
    
    def get_adversarial_examples(self, data):
        '''Gets adversarial examples.'''
        return data[data['of_id'].notna()]
    
    def get_original_data_by_domain_names(self, data, domain_names):
        '''Gets original data by domain names.'''
        original_data = self.get_original_data(data)
        return original_data[original_data['dataset'].isin(domain_names)]
    
    def get_modified_data_domain(self, data, domain):
        '''Modifies given data domain by injecting adversarial examples (while maintaining equal size)
        Step 1.Sample half of the sexist examples from original data domain
        Step 2.Retrieve the modified version of sexist examples on step 1 from adversarial examples
        Step 3.Discard a corresponding number of non-sexist examples from the original data set. (to maintain equal size)
        Step 4.Inject retrieved adversarial examples on step 2
        '''
        
        original_data = self.get_data_domain(data, domain)
        adversarial_examples = self.get_adversarial_examples(data)
        #print('begin original_data {}'.format(len(original_data)))

        #Step 1.Sample half of the sexist examples
        sexist = original_data[original_data['sexist'] == Label.SEXIST]
        sexist = shuffle(sexist, random_state=0)
        sample_count = int(len(sexist) / 2)
        sexist = sexist.head(sample_count)
        #print('sample_count {}'.format(sample_count))
              
        #Step 2.Retrieve the modified version of sexist examples on step 1
        adversarial_examples = adversarial_examples[adversarial_examples['of_id'].isin(sexist['_id'])]  
        #There might be more than one modified example for a sexist tweet. Select the first one.
        adversarial_examples = adversarial_examples.drop_duplicates(subset ="of_id", keep = 'first')
        #print('adversarial_examples {}'.format(len(adversarial_examples)))
        
        #Step 3.Discard a corresponding number of non-sexist examples from the original data set. (to maintain equal size)
        non_sexist = original_data[original_data['sexist'] == Label.NONSEXIST]
        non_sexist = shuffle(non_sexist, random_state=0)
        non_sexist = non_sexist.head(len(adversarial_examples))
        original_data = original_data[~original_data['_id'].isin(non_sexist['_id'])]
        
        #Step 4.Inject retrieved adversarial examples on step 2
        original_data = pd.concat([original_data, adversarial_examples], axis=0)
        #print('end original_data {}'.format(len(original_data)))
        
        return original_data
        
    def get_data_domain_bho(self, data):
        '''Get original data domain. (bhocs: benevoolent, hostile, other)'''
        return self.get_original_data_by_domain_names(data, [Dataset.BENEVOLENT, 
                                                             Dataset.HOSTILE, 
                                                             Dataset.OTHER])

    def get_data_domain_c(self, data):
        '''Get original data domain. (c: callme)'''
        return self.get_original_data_by_domain_names(data, [Dataset.CALLME])

    def get_data_domain_bhocs(self, data):
        '''Get original data domain. (bhocs: benevoolent, hostile, other, callme, scales)'''
        return self.get_original_data_by_domain_names(data, [Dataset.BENEVOLENT, 
                                                             Dataset.HOSTILE, 
                                                             Dataset.OTHER, 
                                                             Dataset.CALLME, 
                                                             Dataset.SCALES])
    
    def get_data_domain_s(self, data):
        '''Get original data domain. (s: scales)'''
        return self.get_original_data_by_domain_names(data, [Dataset.SCALES])

    def get_data_domain_bhom(self, data):
        '''Get modified data domain. (bho-M)'''
        return self.get_modified_data_domain(data, Domain.BHO)

    def get_data_domain_cm(self, data):
        '''Get modified data domain. (c-M)'''
        return self.get_modified_data_domain(data, Domain.C)

    def get_data_domain_bhocsm(self, data):
        '''Get modified data domain. (bhocs-M)'''
        return self.get_modified_data_domain(data, Domain.BHOCS)
    
    def get_data_domain(self, data, domain):
        method_name = 'get_data_domain_' + domain
        method = getattr(self, method_name)
        return method(data)
            
    def get_preprocessed_data(self, features, domain):
        '''Gets preprocessed data domain.
        
        Returns:
        features (list(src.Enum.Feature)): Feature list.
        domain (src.Enums.Domain) = Domain name.
        '''
        #Read preprocessed data
        md = MakeDataset()
        data = md.read_preprocessed_data(features)
        
        #Get training data by domain name
        X = self.get_data_domain(data, domain)
        y = X['sexist']
        return X, y