#src module
from src.decorators import *
from src.utilities import Preprocessing

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
    
    def __init__(self, ngram_range=(1,3), model_path=None):
        '''
        Args:
        ngram_range (tuple (min_n, max_n)) = Ngram range for  features (e.g. (1, 2) means that extracts unigrams and bigrams)
        '''
        self.ngram_range=ngram_range
        self.model_path=model_path
        self.vectorizer=None
    
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
                    processed_governor=self.process(governor[0])
                    processed_dependent=self.process(dependent[0])
                    feature = processed_governor + '_' +  dep + '_' + processed_dependent + ' '
                    features = features + feature
            return features
        except Exception as e:
            print(text, e)
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
            
    def fit(self, x, y=None):
        x = self.get_type_dependency_relationships(x)
        self.vectorizer = TfidfVectorizer(ngram_range=self.ngram_range).fit(x)
        return self

    def transform(self, texts):
        texts = self.get_type_dependency_relationships(texts)
        return self.vectorizer.transform(texts)