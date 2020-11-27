#sklearn
from sklearn.base import BaseEstimator

from tensorflow.keras.layers.experimental.preprocessing import TextVectorization
import tensorflow as tf

class BuildTextVecFeature(BaseEstimator):
    '''Extracts TextVectorization Features'''
    
    def __init__(self, max_tokens = 2400, output_sequence_length = 60, output_mode='int'):
        self.max_tokens = max_tokens  # Maximum vocab size.
        self.output_sequence_length = output_sequence_length  # Sequence length to pad the outputs to.
        self.output_mode=output_mode
        
        self.model=None
        self.vocab_processor=None
    
    def fit(self, x, y=None):
        vectorize_layer=TextVectorization(max_tokens=self.max_tokens, output_mode=self.output_mode, 
                                        output_sequence_length=self.output_sequence_length)

        vectorize_layer.adapt(list(x))
        model = tf.keras.models.Sequential()
        model.add(tf.keras.Input(shape=(1,), dtype=tf.string))
        model.add(vectorize_layer)
        
        self.model=model 
        self.vocab_processor=vectorize_layer
        
        return self

    def transform(self, texts):
        return self.model.predict(list(texts))
    
    def get_feature_names(self):
        return self.vocab_processor.get_vocabulary()