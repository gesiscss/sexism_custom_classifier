#src module
from src.utilities import *

#sklearn
from sklearn.base import BaseEstimator
from sklearn.feature_extraction.text import TfidfVectorizer

#other
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer
import os
import time
import urllib

class BuildTypeDependencyFeature(BaseEstimator):
    '''Extracts Type Dependency Features'''
    
    def __init__(self, ngram_range=(1,1), add_relation=True, model_path=None, 
                 extract=False, save_path='', type_dep_file_name=''):
        '''
        Args:
        ngram_range (tuple (min_n, max_n)) = Ngram range for  features (e.g. (1, 2) means that extracts unigrams and bigrams)
        '''
        self.ngram_range=ngram_range
        self.model_path=model_path
        self.vectorizer=None
        self.add_relation=add_relation
        self.extract=extract
        self.save_path=save_path
        self.type_dep_file_name=type_dep_file_name
    
    def start_CoreNLPServer(self):
        url='http://localhost:9000'
        status_code=0
        try:
            status_code = urllib.request.urlopen(url).getcode()
        except:
            pass
        
        if status_code != 200:
            print('CoreNLPServer is starting {}'.format(url))
            try:
                os.environ['CLASSPATH'] = self.model_path 
                server=CoreNLPServer(port=9000)
                server.start()
                
                status_code = urllib.request.urlopen(url).getcode()
                print('server started {}'.format(status_code))
                
            except Exception as e:
                print(url, e)
                raise Exception(e)
    
    def process(self, text):
        upre=Preprocessing()
        text=upre.lower_text(text)
        return upre.stem_porter(text)
        
    def get_combined_feature(self, parser, text):
        ''' 
        Gets combines features.
        
        Args:
        parser (nltk.parse.corenlp.CoreNLPDependencyParser): Type dependency parser.
        text (string): Sentence. 
        
        Returns:
        features (string): Type dependency relationshhips.
    
        Example:
            >>> get_combined_feature(parser, 'i am not sexist.')
            'sexist_nsubj_i sexist_cop_am sexist_advmod_not'
        '''
        try:
            parse, = parser.raw_parse(text)
            triples = parse.triples()
            features=''
            for governor, dep, dependent in triples:
                if str(dep) not in ['punct']:
                    new_dep=dep.replace(':', '_')
                    processed_governor=self.process(governor[0])
                    processed_dependent=self.process(dependent[0])
                    feature=''
                    if self.add_relation:
                        feature = processed_governor + '_' +  new_dep + '_' + processed_dependent + ' '
                    else:
                        feature = processed_governor + '_' + processed_dependent + ' '
                    features = features + feature
            return features
        except Exception as e:
            print('text >> ', text, e)
            raise Exception(e)

    #@execution_time_calculator
    def get_type_dependency_relationships(self, data):
        '''Gets type dependency relationships.'''
        try:
            self.start_CoreNLPServer()
            parser=CoreNLPDependencyParser(url='http://localhost:9000')
            return [self.get_combined_feature(parser, text) for text in data]
        except Exception as e:
            raise Exception(e)
            
    def get_features_from_file(self, texts):
        df=read_pickle(self.type_dep_file_name)
        df.set_index('_id', inplace=True)
        filtered_df=df[df.index.isin(texts.index)]
        
        feature_list=[]
        for i in range(len(texts)):
            feature_list.append(filtered_df.loc[texts.index[i]].type_dependencies)
        return feature_list
    
    def save_features(self, features, data, file_name):
        data_dic=[]
        for i in  range(len(data)):
            row={'_id': data.index[i], 'type_dependencies':features[i] }
            data_dic.append(row)
        
        df=pd.DataFrame.from_dict(data_dic)
        return save_to_pickle(df, '/'.join((self.save_path, file_name)))
    
    def fit(self, x, y=None):
        if self.extract:
            self.add_relation=True
            x_with_relation = self.get_type_dependency_relationships(x)
            self.save_features(x_with_relation, x, 'type_dep_with_relation')
            
            self.add_relation=False
            x_without_relation = self.get_type_dependency_relationships(x)
            self.save_features(x_without_relation, x, 'type_dep_without_relation')
        else:
            x=self.get_features_from_file(x)
            self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range).fit(x)
            self.feature_dimension=len(self.get_feature_names())
        
        return self

    def transform(self, texts):
        texts=self.get_features_from_file(texts)
        return self.vectorizer.transform(texts)
    
    def get_feature_names(self):
        return self.vectorizer.get_feature_names()