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
            
            text=upre.remove_emojis(text)
            text=upre.remove_hashtag(text)
            text=upre.remove_mention(text)
            text=upre.remove_rt(text)
            text=upre.remove_urls(text)
        
            text=upre.remove_non_alnum(text)
            text=upre.remove_space(text)
            text=upre.lower_text(text)
            text=upre.strip_text(text)
            text=upre.compress_words(text)
            return text
        except Exception as e:
            print('text> {}'.format(text))
            raise Exception(e)
    
    def preprocess_data(self, data):
        data=data[~data.sexist.isnull()]
        data=data[data.text.notna()]
        data=data[data.text != '']
        
        data['preprocessed']=data.text.apply(lambda x: self.preprocess(x))
        data=data[data.preprocessed != '']
        data['sexist'] = data.copy()['sexist'].astype(int)
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
        split_dict_s=self.get_n_splits(X[X.sexist == 1], n_splits, test_size, random_state)
        split_dict_ns=self.get_n_splits(X[X.sexist == 0], n_splits, test_size, random_state)
        
        X_train=pd.concat([split_dict_s[0]['X_train'], split_dict_ns[0]['X_train']])
        X_test=pd.concat([split_dict_s[0]['X_test'], split_dict_ns[0]['X_test']])
        return X_train, X_test
    
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
        #print(len(adversarial_examples))
        if len(adversarial_examples) > sample_count:
            #There might be more than one modified example for a sexist tweet. Select the first one.
            adversarial_examples = adversarial_examples.drop_duplicates(subset ='of_id', keep = 'first')
            #print(len(adversarial_examples))
            
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
        
    def downsample(self, df, random_state):
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
                                 random_state=random_state)        # reproducible results
 
        # Combine minority class with downsampled majority class
        df_downsampled = pd.concat([df_majority_downsampled, df_minority])
 
        return df_downsampled
    
    def prepare_data_splits1(self, data, random_state):
        original_data = self.get_original_data(data)
        adversarial_examples = self.get_adversarial_examples(data)
        
        domain_list=[{'name': 'BHO', 'value':Domain.BHO},
                     {'name': 'BHOCS', 'value':Domain.BHOCS},
                     {'name': 'C', 'value':Domain.C},
                     {'name': 'S', 'value':Domain.S}]
        
        splits={}
        for domain in domain_list:
            X = original_data[original_data.dataset.isin(domain['value']['dataset'])]
            
            #1.Balance
            X=self.downsample(X, random_state)
            
            #2.Split
            X_train, X_test=self.get_splits(X, n_splits=1, test_size=0.3, random_state=random_state)
            
            splits[domain['name']]={'X_train':X_train, 'X_test':X_test}
            
            #3.Adversarial Injection
            splits[''.join((domain['name'], 'M'))]={
                'X_train':self.get_modified_data(X_train, adversarial_examples, 0.5), 
                'X_test':self.get_modified_data(X_test, adversarial_examples, 1)
            }
            
        return splits
    
    def prepare_data_splits(self, data, random_state):
        original_data = self.get_original_data(data)
        adversarial_examples = self.get_adversarial_examples(data)
        
        domain_list=[{'name': 'BHO', 'value':Domain.BHO},
                     {'name': 'BHOCS', 'value':Domain.BHOCS},
                     {'name': 'C', 'value':Domain.C},
                     {'name': 'S', 'value':Domain.S}]
        
        bho=original_data[original_data.dataset.isin(Domain.BHO['dataset'])]
        bho=self.downsample(bho, random_state)
        
        callme=original_data[original_data.dataset.isin(Domain.C['dataset'])]
        callme=self.downsample(callme, random_state)
        
        scales=original_data[original_data.dataset.isin(Domain.S['dataset'])]
        scales=self.downsample(scales, random_state)
        
        # Original
        #BHO
        X_train_bho, X_test_bho=self.get_splits(bho, n_splits=1, test_size=0.3, random_state=random_state)
        
        #C
        X_train_callme, X_test_callme=self.get_splits(callme, n_splits=1, test_size=0.3, random_state=random_state)
        
        #S
        X_train_scales, X_test_scales=self.get_splits(scales, n_splits=1, test_size=0.3, random_state=random_state)
        
        #BHOCS
        X_train_bhosc=pd.concat([X_train_bho, X_train_callme, X_train_scales])
        X_test_bhosc=pd.concat([X_test_bho, X_test_callme, X_test_scales])
        
        # Modified
        #BHO
        X_train_bho_modified=self.get_modified_data(X_train_bho, adversarial_examples, 0.5)
        X_test_bho_modified=self.get_modified_data(X_test_bho, adversarial_examples, 1)
        
        #C
        X_train_callme_modified=self.get_modified_data(X_train_callme, adversarial_examples, 0.5)
        X_test_callme_modified=self.get_modified_data(X_test_callme, adversarial_examples, 1)
        
        #S
        X_train_scales_modified=self.get_modified_data(X_train_scales, adversarial_examples, 0.5) 
        X_test_scales_modified=self.get_modified_data(X_test_scales, adversarial_examples, 1)
        
        #BHOCS
        X_train_bhosc_modified=pd.concat([X_train_bho_modified, X_train_callme_modified, X_train_scales_modified])
        X_test_bhosc_modified=pd.concat([X_test_bho_modified, X_test_callme_modified, X_test_scales_modified])
        
        splits={}
        splits['BHO']={'X_train':X_train_bho, 'X_test':X_test_bho}
        splits['C']={'X_train':X_train_callme, 'X_test':X_test_callme}
        splits['S']={'X_train':X_train_scales, 'X_test':X_test_scales}
        splits['BHOCS']={'X_train':X_train_bhosc, 'X_test':X_test_bhosc}
        
        splits['BHOM']={'X_train':X_train_bho_modified, 'X_test':X_test_bho_modified}
        splits['CM']={'X_train':X_train_callme_modified, 'X_test':X_test_callme_modified}
        splits['BHOCSM']={'X_train':X_train_bhosc_modified, 'X_test':X_test_bhosc_modified}
           
        return splits
    
    def get_data_split(self, domain_name, splits, train=False, test=True, random_state=0):
        col='X_train' if train else 'X_test' if test else ''
        
        X=splits[domain_name][col]
        
        X=shuffle(X, random_state=random_state)
        X.set_index('_id', inplace=True)
        #X, y=X[['text', 'toxicity']], X['sexist'].ravel()
        X, y=X, X['sexist'].ravel()
        
        return X, y