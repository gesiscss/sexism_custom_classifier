
# coding: utf-8

# In[1]:


#src module
from src.data.prepare_data_domain import PrepareDataDomain
from src.pipeline_builder import PipelineBuilder
from src.enums import *

class RunPipeline():
    '''Runs pipeline for a given data domain, model and features.'''
    def __init__(self, training_domain, test_domains, model, features):
        '''
        Args:
        training_domain (src.Enum.Domain): Training domain that is used to train model.
        test_domains (src.Enum.Domain):    Training domain that is used to train model.
        model (src.Enum.Model):            Model that is traine as a part of the sklearn pipeline.
        features (list(src.Enum.Feature)): Feature(s) that is used to train model on top of.
        Example:
            >>> features={ 
                    Feature.SENTIMENT:      { Parameter.Sentiment.SCORE_NAMES: ['neu', 'compound'] },
                    Feature.NGRAM:          { Parameter.Ngram.NGRAM_RANGE: (2,2) },
                    Feature.TYPEDEPENDENCY: { Parameter.Ngram.NGRAM_RANGE: (1,1) }
                }
            >>> RunPipeline(Domain.BHOCS, Domain.BHOCSM, Model.LR, features)
        '''
        self.training_domain=training_domain
        self.test_domains=test_domains
        self.model=model
        self.features=features
        
    def run(self):
        '''Runs pipeline for given features and model and preprocessed data domain.'''
        p = PrepareDataDomain()
        X, y = p.get_preprocessed_data(list(self.features.keys()), self.training_domain)

        #Get split data dictionary that includes training and  test set. 
        #Default params : 
        #n_splits=5 : number of re-shuffling & splitting iterations 
        #test_size=0.3 : the proportion of the dataset to include in the test split 
        split_dict = p.get_n_splits(X, y)
            
        pb = PipelineBuilder()
        pipeline = pb.get_pipeline(self.model, self.features)
        pb.fit_and_predict(split_dict, pipeline)