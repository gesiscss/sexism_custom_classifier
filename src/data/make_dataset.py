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
    '''Prepares data.'''
    
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
    
    def get_splits(self, X, n_splits, test_size, random_state):
        split_dict=self.get_n_splits(X, n_splits, test_size, random_state)
        
        return split_dict[0]['X_train'], split_dict[0]['X_test']
    
    def get_original_data(self, data):
        '''Gets original data.'''
        return data[data['of_id'].isnull()]
    
    def get_adversarial_examples(self, data):
        '''Gets adversarial examples.'''
        return data[data['of_id'].notna()]
    
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
    
    def prepare_data_splits(self, data, random_state):
        original_data = self.get_original_data(data)
        adversarial_examples = self.get_adversarial_examples(data)
        
        dataset_list=[Dataset.BENEVOLENT, Dataset.HOSTILE, Dataset.OTHER, Dataset.CALLME, Dataset.SCALES]
        
        splits_original, splits_modified={}, {}
        for dataset in dataset_list:
            X=original_data[original_data.dataset == dataset]
            X_train, X_test=self.get_splits(X, n_splits=1, test_size=0.3, random_state=random_state)
            
            splits_original[dataset]={'X_train':X_train, 'X_test':X_test}
            splits_modified[dataset]={
                'X_train':self.get_modified_data(X_train, adversarial_examples, 0.5), 
                'X_test':self.get_modified_data(X_test, adversarial_examples, 1)
            }
            
        return splits_original, splits_modified
    
    def get_data_split(self, domain, splits_original, splits_modified, train=False, test=True):
        splits=splits_original if domain['modified'] == False else splits_modified
        return self.get_balanced_data(domain, splits, train, test)
        
    def get_balanced_data(self, data_domain, splits, train, test):
        col='X_train' if train else 'X_test' if test else ''
        
        X=pd.DataFrame()
        for domain in data_domain['dataset']:
            X=pd.concat([X, self.downsample(splits[domain][col])])
    
        X=shuffle(X, random_state=0)
        X.set_index('_id', inplace=True)
        X, y=X['text'], X['sexist'].ravel()
        
        return X, y