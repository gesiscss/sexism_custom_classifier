#sklearn
from sklearn.base import BaseEstimator

#huggingface
from transformers import AutoConfig, AutoTokenizer, AutoModel 

#other
import torch

#TODOS:
#1.After retrieving embeddings, normalize them
#2.Prevent overfitting

class BuildBERTDocumentEmbeddingsFeature(BaseEstimator):
    '''Extracts BERT document embeddings Features'''
    def __init__(self, aggregated=True, model_name='bert-base-cased', return_tensors='pt', padding=True, output_hidden_states=True):
        '''
        Args:
        aggregated (boolean) = True  : an aggregated representation for the whole sequence
                               False : the generated representation for every token in the sequence
        return_tensors (string) : { 'tf', 'pt' }
        '''
        torch.set_grad_enabled(False)
        
        self.aggregated=aggregated
        self.model_name=model_name
        self.return_tensors=return_tensors
        self.padding=padding
        self.output_hidden_states=output_hidden_states
    
    def extract_features(self, data):
        '''Extracts features. 
        Steps:
        1.Initialize a configuration (e.g., a BERT bert-base-uncased style configuration)
        2.Create the tokenizer and model (pretrained_weights)
        3.Tokenize
        4.Retrieve features of shape:
            1. Token-based representation  : (BATCH_SIZE, NB_TOKENS, REPRESENTATION_SIZE)
            2. Aggregated representaton    : (BATCH_SIZE, REPRESENTATION_SIZE)
            3. Token-based representation with hidden layers : (LAYER_SIZE, BATCH_SIZE, NB_TOKENS, REPRESENTATION_SIZE)
            4. Aggregated representaton with hidden layers : Model does not output(?) (TODO)
        '''
        # 1.Initialize a configuration (e.g., a BERT bert-base-uncased style configuration)
        config=AutoConfig.from_pretrained(self.model_name)
        config.output_hidden_states=self.output_hidden_states
    
        # 2.Create the tokenizer and model (pretrained_weights)
        tokenizer=AutoTokenizer.from_pretrained(self.model_name)
        model=AutoModel.from_pretrained(self.model_name, config=config)
        
        # 3.Tokenize
        tokens=tokenizer(data, padding=self.padding , return_tensors=self.return_tensors)
        
        # 4. Retrieve features
        # Model outputs Tuple of torch.FloatTensor (e.g.,  MODEL_NAME = 'bert-base-cased' and return_tensors='pt')
        # If output_hidden_states=False: 
        #    Model outputs Tuple of 2 tensors: 
        #        ( (BATCH_SIZE, NB_TOKENS, REPRESENTATION_SIZE), (BATCH_SIZE, REPRESENTATION_SIZE) )
        # If output_hidden_states=True: 
        #    Model outputs Tuple of 3 tensors: 
        #        ( (BATCH_SIZE, NB_TOKENS, REPRESENTATION_SIZE), (BATCH_SIZE, REPRESENTATION_SIZE), (LAYER_SIZE, BATCH_SIZE, NB_TOKENS, REPRESENTATION_SIZE) ) where LAYER_SIZE = initial embeddings + 12 BERT layers
        outputs = model(**tokens)
        return outputs
    
    def fit(self, x, y=None):
        #TODO
        return self

    def transform(self, texts):
        #TODO
        return None