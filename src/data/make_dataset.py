#src module
from src import utilities as u
from src.enums import *
from src import factory

from src.data.preprocessing.preprocess_sentiment import PreprocessSentiment
from src.data.preprocessing.preprocess_ngram import PreprocessNgram
from src.data.preprocessing.preprocess_type_dependency import PreprocessTypeDependency
from src.data.preprocessing.preprocess_bert_doc_emb import PreprocessBertDocEmb

#other
import pandas as pd

class MakeDataset:
    '''Preporcesses data for sentiment, ngram, type dependency and Bert document embedding features.'''
    
    def __init__(self):
        self.raw_data_full_path='../data/raw/all_data.csv'
        self.preprocessed_data_path='../data/processed/'
        self.preprocessing_objects={
            Feature.SENTIMENT: PreprocessSentiment,
            Feature.NGRAM: PreprocessNgram,
            Feature.TYPEDEPENDENCY: PreprocessTypeDependency,
            Feature.BERTDOCEMB: PreprocessBertDocEmb
        }
        
    def read_csv(self, delimiter='\t'):
        '''Read data from raw data full path.'''
        return u.read_csv(self.raw_data_full_path, delimiter=delimiter)
    
    def save_to_csv(self, df, file_name):
        '''Saves dataframe to the preprocessed data path.'''
        path=self.preprocessed_data_path + file_name
        u.save_to_csv(df, path) 
        print('Saved preprocessed data: ' + path)
            
    def preprocess(self, data, features=[]):
        '''For each item in features, preprocesses data and saves to the preprocessed data path.'''
        for feature in features:
            preprocessor = factory.get_object(self.preprocessing_objects, feature)
            preprocessor.data = data
            df = preprocessor.preprocess()
            file_name = 'preprocessed_data_' + feature + '.csv'
            self.save_to_csv(df, file_name)
    
    def read_preprocessed_data_by_feature(self, feature):
        '''Gets preprocessed data for each feature set.
        
        Args:
        feature (src.Enum.Feature): Feature name.
        
        Returns:
        df (pandas.DataFrame): Preprocessed data.
    
        Example:
            >>> read_preprocessed_data(Feature.SENTIMENT)
            pandas.DataFrame Columns ['_id', 'dataset', 'of_id', 'sexist', 'text', 'sentiment'
        '''
        path='/Users/ezgidaldal/sexism_custom_classifier/data/processed/preprocessed_data_' + feature + '.csv'
        return u.read_csv(path)
        
    def read_preprocessed_data(self, features):
        '''Gets preprocessed data for each feature set.
        
        Args:
        features (list(src.Enum.Feature)): Feature list.
        
        Returns:
        df (pandas.DataFrame): Preprocessed data.
    
        Example:
            >>> read_preprocessed_data(features=[Feature.SENTIMENT, Feature.NGRAM])
            pandas.DataFrame Columns ['_id', 'dataset', 'of_id', 'sexist', 'text', 'sentiment', 'ngram']
        '''
        df=pd.DataFrame()
        for feature in features:
            feature_df = self.read_preprocessed_data_by_feature(feature)
            if len(df) > 0:
                df = pd.merge(df, feature_df, how='inner', on=[DataColumn.ID, DataColumn.ADVERSARIAL,
                                                               DataColumn.DATASET, DataColumn.TEXT,
                                                               DataColumn.LABEL])
            else:
                df = pd.concat([df, feature_df], axis=1)
        
        return df