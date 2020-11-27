#src module
from src import utilities as u
from src.enums import *
from src.utilities import Preprocessing

#sklearn
from sklearn.model_selection import ShuffleSplit
from sklearn.utils import shuffle
from sklearn.utils import resample

#other
import pandas as pd

class MakeDataset:
    '''Prepares data domains.'''
    def __init__(self, n_splits=1, test_size=0.3, random_state=None):
        '''
        Args:
        n_split = number of re-shuffling & splitting iterations
        test_size = the proportion of the dataset to include in the test split
        '''
        self.n_splits=n_splits
        self.test_size=test_size
        self.random_state=random_state
    
    def read_csv(self, full_path, delimiter='\t'):
        '''Reads raw data from the full path.'''
        return u.read_csv(full_path, delimiter=delimiter)

    def preprocess(self, text):
        try:
            upre=Preprocessing()
            
            text=upre.remove_new_lines(text)
            text=upre.replace_whitespace_with_single_space(text)
            text=upre.remove_URLs(text)
            text=upre.remove_usernames(text)
            text=upre.remove_hashtags(text)
            text=upre.clean_tweet(text)
            return text
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)

    def preprocess_data(self, data):
        data=data[~data.sexist.isnull()]
        data=data[data.text.notna()]
        data=data[data.text != '']
        data['preprocessed']=[self.preprocess(raw_doc) for raw_doc in data['text']]
        data=data[data.preprocessed != '']
        data['sexist'] = data.copy()['sexist'].astype(int)
        data=pd.DataFrame(data[['_id', 'sexist', 'text', 'of_id','dataset']])
        return data
    
    def read_data(self, full_path):
        data = self.read_csv(full_path)
        return self.preprocess_data(data)

    def get_n_splits(self, X, n_splits, test_size, random_state):
        '''Split data into training and test set.
            
        Returns:
        split_dict (dictionary) = dictionary that includes split training and test set
        '''
        split_dict = []
        ss = ShuffleSplit(n_splits=n_splits, test_size=test_size, random_state=random_state)
        
        for train_index, test_index in ss.split(X):
            split = {'X_train': X.iloc[train_index], 'X_test': X.iloc[test_index] }
            split_dict.append(split)
        
        return split_dict 
    
    def get_splits(self, X):
        split_dict=self.get_n_splits(X, self.n_splits, self.test_size, self.random_state)
        
        return split_dict[0]['X_train'], split_dict[0]['X_test']
    
    def get_original_data(self, data):
        '''Gets original data.'''
        return data[data['of_id'].isnull()]
    
    def get_adversarial_examples(self, data):
        '''Gets adversarial examples.'''
        return data[data['of_id'].notna()]
       
    def get_data_by_domain_name(self, data, domain_name):
        '''Gets data by domain name.'''
        return pd.DataFrame(data[data['dataset'] == domain_name])
    
    def get_modified_data(self, original_data, adversarial_examples, sample_proportion):
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
        begin_len_sexist = len(original_data[original_data['sexist'] == 1])
        begin_len_nonsexist = len(original_data[original_data['sexist'] == 0])
        ###
    
        #Step 1.Sample 'sample_proportion' of the sexist examples
        sexist = original_data[original_data['sexist'] == 1]
        sample_count = int(len(sexist) * sample_proportion)
        sexist = sexist.head(sample_count)
    
        #Step 2.Retrieve the modified version of sexist examples on step 1
        adversarial_examples = adversarial_examples[adversarial_examples['of_id'].isin(sexist['_id'])]  
        #There might be more than one modified example for a sexist tweet. Select the first one.
        adversarial_examples = adversarial_examples.drop_duplicates(subset ='of_id', keep = 'first')
        
        #Step 3.Discard a corresponding number of non-sexist examples from the original data set. (to maintain equal size)
        non_sexist = original_data[original_data['sexist'] == 0].head(len(adversarial_examples))
        original_data = original_data[~original_data['_id'].isin(non_sexist['_id'])]
        
        #Step 4.Inject retrieved adversarial examples on step 2 
        #NOTE: When sexist examples > nonsexist examples, adversarial_examples might be more than non_sexist example count.
        #To maintain equal size in that case, concat original_data with: adversarial_examples.head(len(non_sexist)
        original_data = pd.concat([original_data, adversarial_examples.head(len(non_sexist))], axis=0)
        
        ######### To compare the data size before and after injection ##############
        end_len_sexist = len(original_data[original_data['sexist'] == 1])
        end_len_nonsexist = len(original_data[original_data['sexist'] == 0])
        
        if begin_len_sexist != end_len_sexist or begin_len_nonsexist != end_len_nonsexist:
            print('equal size did not maintain: sexist begin {} sexist end {} nonsexist begin {} nonsexist end {}'.format(begin_len_sexist, end_len_sexist, begin_len_nonsexist, end_len_nonsexist))
        ##########################################################################
        
        #Step 5.Shuffle
        original_data=shuffle(original_data, random_state=0)
        return original_data
      
    def get_symmetric_difference_data(self, original_data, domain_names, modified=False, adversarial_examples=None, sample_proportion=0):
        ''' '''
        X=pd.DataFrame()
    
        for domain in domain_names:
            X_domain=self.get_data_by_domain_name(original_data, domain)
        
            if modified and len(X_domain) > 0:
                X_domain=self.get_modified_data(X_domain, adversarial_examples, sample_proportion)
        
            X=X.append(X_domain, ignore_index=True)
        
        return X
    
    def get_intersection_data_domains(self, original_data, domain_names, train_modified=False, test_modified=False, adversarial_examples=None):
        ''' '''
        X_train, X_test=pd.DataFrame(), pd.DataFrame()
        
        for domain in domain_names:
            X_domain = self.get_data_by_domain_name(original_data, domain)
            
            X_train_s, X_test_s=self.get_splits(X_domain)
            
            if train_modified:
                X_train_s=self.get_modified_data(X_train_s, adversarial_examples, sample_proportion=0.5)
                
            if test_modified:
                X_test_s=self.get_modified_data(X_test_s, adversarial_examples, sample_proportion=1)
           
            X_train=X_train.append(X_train_s, ignore_index=True)
            X_test=X_test.append(X_test_s, ignore_index=True)
        
        return X_train, X_test
        
    def get_train_and_test_sets(self, data, train_domain_names, test_domain_names, train_modified=False, test_modified=False):
        '''Gets splits for training and test data domains. 
        #Step 1. Find the symmetric difference and the intersection of training and test domain names
        #Step 2. Get the symmetric difference data that only TRAIN has
        #Step 3. Get the symmetric difference data that only TEST has
        #Step 4. Get data that the intersection of training and test domains  
        #Step 5. Append the symmetric difference data to the intersection data
        '''
        X_train, X_test=pd.DataFrame(), pd.DataFrame()
        
        original_data = self.get_original_data(data)
        adversarial_examples = self.get_adversarial_examples(data)
        
        #Step 1. Find the symmetric difference and the intersection of training and test domain names
        domain_names_only_train=list(set(train_domain_names) - set(test_domain_names))
        domain_names_only_test= list(set(test_domain_names) - set(train_domain_names))
        domain_names_intersection=list(set(train_domain_names).intersection(test_domain_names))
    
        #Step 2. Get the symmetric difference data that only TRAIN has
        X_train_sym_diff=self.get_symmetric_difference_data(original_data, domain_names_only_train, modified=train_modified, adversarial_examples=adversarial_examples, sample_proportion=0.5)
        
        #Step 3. Get the symmetric difference data that only TEST has
        X_test_sym_diff=self.get_symmetric_difference_data(original_data, domain_names_only_test, modified=test_modified, adversarial_examples=adversarial_examples, sample_proportion=1)
        
        #Step 4. Get data that the intersection of training and test domains   
        X_train_intersection, X_test_intersection=self.get_intersection_data_domains(original_data, domain_names_intersection, train_modified=train_modified, test_modified=test_modified, adversarial_examples=adversarial_examples)
        
        #Step 5. Append the symmetric difference data to the intersection data
        X_train=X_train.append(X_train_sym_diff, ignore_index=True)
        X_train=X_train.append(X_train_intersection, ignore_index=True)
        
        X_test=X_test.append(X_test_sym_diff, ignore_index=True)
        X_test=X_test.append(X_test_intersection, ignore_index=True)
        
        return X_train, X_test
        
    def downsample(self, df):
        '''Balances dataset by downsampling the majority class. 
           Uses sklearn resample method to downsample.
        '''
        nonsexist_count=len(df[df.sexist==0])
        sexist_count=len(df[df.sexist==1])
    
        # Separate majority and minority classes
        df_minority, df_majority=None, None
        n_samples= 0
        
        if sexist_count < nonsexist_count:
            df_minority = df[df.sexist==1]
            df_majority = df[df.sexist==0]
            n_samples=sexist_count
        else:
            df_minority = df[df.sexist==0]
            df_majority = df[df.sexist==1]
            n_samples=nonsexist_count
    
        # Downsample majority class
        df_majority_downsampled = resample(df_majority, 
                                 replace=False,           # sample without replacement
                                 n_samples=n_samples,     # to match minority class
                                 random_state=123)        # reproducible results
 
        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
        return df_downsampled

    def downsample_domains(self, df):
        df_downsampled=pd.DataFrame()
        
        domains=df.groupby(['dataset']).groups        
        for domain in domains:
            df_domain=df[df['dataset'].isin([domain])]
            df_domain=self.downsample(df_domain)
            df_downsampled=pd.concat([df_downsampled, df_domain])
            
        return df_downsampled

    def downsample_test(self, df, verson=''):
        df_downsampled=df
        
        print('===== BEFORE DOWNSAMPLE')
        print()
        print(pd.DataFrame(df.groupby(['sexist']).size()))
        print()
        print(pd.DataFrame(df.groupby(['dataset', 'sexist']).size()))
        print()
        if verson=='v1':
            df_downsampled=self.downsample(df)
        elif verson=='v2':
            df_downsampled=self.downsample_domains(df)
        
        print('===== AFTER DOWNSAMPLE')
        print()
        print(pd.DataFrame(df_downsampled.groupby(['sexist']).size()))
        print()
        print(pd.DataFrame(df_downsampled.groupby(['dataset', 'sexist']).size()))
        print()
        return df_downsampled
        
    def get_balanced_data_split(self, data, train_domain, test_domain):
        ''' '''
        train_modified, train_domain_names=train_domain['modified'], train_domain['dataset']
        test_modified,  test_domain_names=test_domain['modified'], test_domain['dataset']
        
        # Step 1. Gets splits for training and test data domains. 
        X_train, X_test=self.get_train_and_test_sets(data, train_domain_names, test_domain_names, train_modified, test_modified)
        
        # Step 2. Balance
        X_train=self.downsample(X_train)
        X_test=self.downsample(X_test)
        
        # Step 3. Shuffle
        X_train=shuffle(X_train, random_state=0)
        X_test=shuffle(X_test, random_state=0)
        
        X_train.set_index('_id', inplace=True)
        X_test.set_index('_id', inplace=True)
        
        X_train, y_train=X_train['text'], X_train['sexist'].ravel()
        X_test, y_test=X_test['text'], X_test['sexist'].ravel()
        
        return X_train, y_train, X_test, y_test