#!/usr/bin/env python
# coding: utf-8

# In[3]:

#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src.features.build_features import BuildFeature

#sklearn
from sklearn.feature_extraction.text import TfidfVectorizer

#other
from nltk.parse.corenlp import CoreNLPDependencyParser, CoreNLPServer
import os

class BuildTypeDependencyFeature(BuildFeature):
    '''Extracts Type Dependency Features'''
    
    def __init__(self, ngram_range=(1,3)):
        '''
        Args:
        ngram_range (tuple (min_n, max_n)) = Ngram range for  features (e.g. (1, 2) means that extracts unigrams and bigrams)
        '''
        self.ngram_range=ngram_range
        self.tfidf_vectorizer=None
        self.model_path=None
   
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
                #TODO: Decide which dependency relationships to include. 
                #Available relationships : (http://universaldependencies.org/docsv1/u/dep/index.html)
                if str(dep) not in ['punct', 'dep', 'root']:
                    feature = governor[0] + '_' +  dep + '_' + dependent[0] + ' '
                    features = features + feature
            return features
        except Exception as e:
            print(text)
            print(e)
        
    def get_type_dependency_relationships(self,data):
        os.environ['CLASSPATH'] = self.model_path
        '''Gets type dependency relationships.'''
        #NOTE: If CoreNLPServer does not used, CoreNLPDependencyParser throws ConnectionError exeption.
        with CoreNLPServer(port=9000) as server:
            parser = CoreNLPDependencyParser(url='http://localhost:9000')
            return [self.get_combined_feature(parser, text) for text in data]
        
    def fit(self, x, y=None):
        x = self.get_type_dependency_relationships(x)
        self.tfidf_vectorizer = TfidfVectorizer(smooth_idf=True, use_idf=True, ngram_range=self.ngram_range)
        self.tfidf_vectorizer.fit(x)
        return self

    def transform(self, texts):
        texts = self.get_type_dependency_relationships(texts)
        return self.tfidf_vectorizer.transform(texts)