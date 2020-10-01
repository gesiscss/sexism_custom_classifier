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
        self.n_splits=n_splits
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
        return data[data[DataColumn.ADVERSARIAL].isnull()]
    
    def get_adversarial_examples(self, data):
        '''Gets adversarial examples.'''
        return data[data[DataColumn.ADVERSARIAL].notna()]
    
    def get_data_by_domain_names(self, data, domain_names):
        '''Gets data and labels by domain names.'''
        X, y=pd.DataFrame(), pd.DataFrame()
        
        if len(domain_names) > 0:
            X=pd.DataFrame(data[data[DataColumn.DATASET].isin(domain_names)])
            y=pd.DataFrame(X[DataColumn.LABEL])
            
        return X, y
    
    def get_modified_data(self, original_data, adversarial_examples, sample_proportion=0.5):
        '''Modifies given data domain by injecting adversarial examples (while maintaining equal size)
        Step 1.Sample 'sample_proportion' of the sexist examples from original data domain.
        Step 2.Retrieve the modified version of sexist examples on step 1 from adversarial examples
        Step 3.Discard a corresponding number of non-sexist examples from the original data set. (to maintain equal size)
        Step 4.Inject retrieved adversarial examples on step 2
        Step 5.Shuffle 
        
        Example:
        >>> training_sample_proportion=0.5, test_sample_proportion=1
        >>> get_modified_data(original_data, adversarial_examples, training_sample_proportion)
        '''
        original_data = shuffle(original_data, random_state=0)
        
        ### To check the size end of the method
        begin_len_sexist = len(original_data[original_data['sexist'] == Label.SEXIST])
        begin_len_nonsexist = len(original_data[original_data['sexist'] == Label.NONSEXIST])
        ###
    
        #Step 1.Sample 'sample_proportion' of the sexist examples
        sexist = original_data[original_data['sexist'] == Label.SEXIST]
        sample_count = int(len(sexist) * sample_proportion)
        sexist = sexist.head(sample_count)
    
        #Step 2.Retrieve the modified version of sexist examples on step 1
        adversarial_examples = adversarial_examples[adversarial_examples[DataColumn.ADVERSARIAL].isin(sexist[DataColumn.ID])]  
        #There might be more than one modified example for a sexist tweet. Select the first one.
        adversarial_examples = adversarial_examples.drop_duplicates(subset =DataColumn.ADVERSARIAL, keep = 'first')
        
        #Step 3.Discard a corresponding number of non-sexist examples from the original data set. (to maintain equal size)
        non_sexist = original_data[original_data[DataColumn.LABEL] == Label.NONSEXIST].head(len(adversarial_examples))
        original_data = original_data[~original_data[DataColumn.ID].isin(non_sexist[DataColumn.ID])]
        
        #Step 4.Inject retrieved adversarial examples on step 2 
        #NOTE: When sexist examples > nonsexist examples, adversarial_examples might be more than non_sexist example count.
        #To maintain equal size in that case, concat original_data with: adversarial_examples.head(len(non_sexist)
        original_data = pd.concat([original_data, adversarial_examples.head(len(non_sexist))], axis=0)
        
        ######### To compare the data size before and after injection ##############
        end_len_sexist = len(original_data[original_data['sexist'] == Label.SEXIST])
        end_len_nonsexist = len(original_data[original_data['sexist'] == Label.NONSEXIST])
        
        if begin_len_sexist != end_len_sexist or begin_len_nonsexist != end_len_nonsexist:
            print('equal size did not maintain: sexist begin {} sexist end {} nonsexist begin {} nonsexist end {}'.format(begin_len_sexist, end_len_sexist, begin_len_nonsexist, end_len_nonsexist))
        ##########################################################################
        
        #Step 5.Shuffle
        original_data=shuffle(original_data, random_state=0)
        return original_data, pd.DataFrame(original_data['sexist'])

    def get_symmetric_difference_data(self, original_data, domain_names, modified=False, adversarial_examples=None):
        ''' '''
        X, y=pd.DataFrame(), pd.DataFrame()
    
        for domain in domain_names:
            X_domain, y_domain=self.get_data_by_domain_names(original_data, [domain])
        
            if modified: # and len(X_domain) > 0:
                X_domain, y_domain=self.get_modified_data(X_domain, adversarial_examples, sample_proportion=0.5)
        
            X=X.append(X_domain, ignore_index=True)
            y=y.append(y_domain, ignore_index=True)
        
        return X, y
    
    def get_intersection_data(self, original_data, domain_names, train_modified=False, test_modified=False, adversarial_examples=None):
        ''' '''
        #Initialize split dictionary that will be returned
        dic_item={'X_train': pd.DataFrame(), 'y_train': pd.DataFrame(), 'X_test': pd.DataFrame(), 'y_test': pd.DataFrame()}
        data_dict = [dic_item.copy() for i in range(self.n_splits)]
    
        for domain in domain_names:
            i=0
            X_both_have, y_both_have = self.get_data_by_domain_names(original_data, [domain])
            split_dict=self.get_n_splits(X_both_have, y_both_have)
        
            for split in split_dict.values():
                X_train_s, y_train_s=split['X_train'], split['y_train'] 
                X_test_s, y_test_s=split['X_test'], split['y_test']
            
                if train_modified:
                    X_train_s, y_train_s=self.get_modified_data(X_train_s, adversarial_examples, sample_proportion=0.5)
                
                if test_modified:
                    X_test_s, y_test_s=self.get_modified_data(X_test_s, adversarial_examples, sample_proportion=1)
        
                data_dict[i]['X_train'] = data_dict[i]['X_train'].append(X_train_s, ignore_index=True)
                data_dict[i]['y_train'] = data_dict[i]['y_train'].append(y_train_s, ignore_index=True)
            
                data_dict[i]['X_test'] = data_dict[i]['X_test'].append(X_test_s, ignore_index=True)
                data_dict[i]['y_test'] = data_dict[i]['y_test'].append(y_test_s, ignore_index=True)
                
                i=i+1
    
        return data_dict
    
    def get_split_data(self, data, train_domain_names, test_domain_names, train_modified=False, test_modified=False):
        '''Gets splits for training and test data domains. 
        #Step 1. Find the symmetric difference and the intersection of training and test domain names
        #Step 2. Get the symmetric difference data that only TRAIN has
        #Step 3. Get the symmetric difference data that only TEST has
        #Step 4. Get data that the intersection of training and test domains  
        #Step 5. Append the symmetric difference data to the intersection data
        '''
        
        original_data = self.get_original_data(data)
        adversarial_examples = self.get_adversarial_examples(data)
        
        #Step 1. Find the symmetric difference and the intersection of training and test domain names
        domain_names_only_train=list(set(train_domain_names) - set(test_domain_names))
        domain_names_only_test= list(set(test_domain_names) - set(train_domain_names))
        domain_names_intersection=list(set(train_domain_names).intersection(test_domain_names))
    
        #Step 2. Get the symmetric difference data that only TRAIN has
        X_only_train, y_only_train=self.get_symmetric_difference_data(original_data, domain_names_only_train, modified=train_modified, adversarial_examples=adversarial_examples)
    
        #Step 3. Get the symmetric difference data that only TEST has
        X_only_test, y_only_test=self.get_symmetric_difference_data(original_data, domain_names_only_test, modified=test_modified, adversarial_examples=adversarial_examples)
        
        #Step 4. Get data that the intersection of training and test domains   
        data_dict=self.get_intersection_data(original_data, domain_names_intersection, train_modified=train_modified, test_modified=test_modified, adversarial_examples=adversarial_examples)
    
        #Step 5. Append the symmetric difference data to the intersection data
        for i in range(len(data_dict)):
            data_dict[i]['X_train'] = data_dict[i]['X_train'].append(X_only_train, ignore_index=True)
            data_dict[i]['y_train'] = data_dict[i]['y_train'].append(y_only_train, ignore_index=True)
            
            data_dict[i]['X_test'] = data_dict[i]['X_test'].append(X_only_test, ignore_index=True)
            data_dict[i]['y_test'] = data_dict[i]['y_test'].append(y_only_test, ignore_index=True)
        
        return data_dict