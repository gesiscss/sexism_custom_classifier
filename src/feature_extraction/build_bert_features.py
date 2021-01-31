#sklearn
from sklearn.base import BaseEstimator

#huggingface
from transformers import AutoConfig, AutoTokenizer, AutoModel 

#other
import tensorflow as tf
import pandas as pd
from src import utilities as u
import numpy as np

class BuildBERTFeature(BaseEstimator):
    '''Extracts BERT word and document embeddings features.
   
    Pretrained models : https://huggingface.co/transformers/pretrained_models.html
    
    bert-large-uncased      : 24-layer, 1024-hidden, 16-heads, 336M parameters. Trained on lower-cased English text.
    bert-base-uncased       : 12-layer, 768-hidden, 12-heads, 110M parameters. Trained on lower-cased English text.
    distilbert-base-uncased : 6-layer, 768-hidden, 12-heads, 66M parameters. The DistilBERT model distilled from the BERT model bert-base-uncased checkpoint
    '''
    def __init__(self, aggregated=True, model_name='bert-base-uncased', return_tensors='pt', padding=True, output_hidden_states=True, extract=False, save_path='', embedding_file_name=''):
        '''
        Args:
        aggregated (boolean) = True  : an aggregated representation for the whole sequence
                               False : the generated representation for every token in the sequence
        return_tensors (string) : { 'tf', 'pt' }
        '''
        
        self.aggregated=aggregated
        self.model_name=model_name
        self.return_tensors=return_tensors
        self.padding=padding
        self.output_hidden_states=output_hidden_states
        self.extract=extract
        self.save_path=save_path
        self.embedding_file_name=embedding_file_name
    
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
        tokens=tokenizer(list(data), padding=self.padding , return_tensors=self.return_tensors)
        
        # 4. Retrieve features
        # Model outputs Tuple of torch.FloatTensor (e.g.,  MODEL_NAME = 'bert-base-cased' and return_tensors='pt')
        # If output_hidden_states=False: 
        #    Model outputs Tuple of 2 tensors: 
        #        ( (BATCH_SIZE, NB_TOKENS, REPRESENTATION_SIZE), (BATCH_SIZE, REPRESENTATION_SIZE) )
        # If output_hidden_states=True: 
        #    Model outputs Tuple of 3 tensors: 
        #        ( (BATCH_SIZE, NB_TOKENS, REPRESENTATION_SIZE), (BATCH_SIZE, REPRESENTATION_SIZE), (LAYER_SIZE, BATCH_SIZE, NB_TOKENS, REPRESENTATION_SIZE) ) where LAYER_SIZE = initial embeddings + 12 BERT layers
        outputs = model(**tokens)
        
        #Save word embeddings
        file_name_word_emb=self.save_features(outputs[0], data, 'word_embeddings')
        
        #Save document embeddings
        file_name_doc_emb=self.save_features(outputs[1], data, 'doc_embeddings')
        
        return file_name_word_emb, file_name_doc_emb
    
    def save_features(self, features, data, file_name):
        data_dic=[]
        for i in  range(len(data)):
            row={'_id': data.index[i], 'embedding':features[i].numpy() }
            data_dic.append(row)
        
        df=pd.DataFrame.from_dict(data_dic)
        return u.save_to_pickle(df, '/'.join((self.save_path, file_name)))
    
    def fit(self, x, y=None):
        if self.extract:
            return self.extract_features(x)
        return self

    def transform(self, texts):
        df=u.read_pickle(self.embedding_file_name)
        df.set_index('_id', inplace=True)
        filtered_df=df[df.index.isin(texts.index)]
        
        embedding_list=[]

        for i in range(len(texts)):
            embedding_list.append(filtered_df.loc[texts.index[i]].embedding)
        
        self.feature_dimension=embedding_list[0].shape[0]
        return tf.convert_to_tensor(embedding_list, dtype=tf.float32)