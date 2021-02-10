#sklearn
from sklearn.base import BaseEstimator

#tensorflow
from tensorflow import keras
import tensorflow as tf

#other
import numpy as np

class CNN(BaseEstimator):
    '''Builds CNN model.'''
    
    def __init__(self, 
                 num_filters=100,           # 3.1 Conv2D Layer
                 filter_sizes=(3,4,5),      # 3.1 Conv2D Layer
                 l2=1e-3,                   # 3.1 Conv2D Layer
                 global_max_pool=False,     # 3.2 Pooling Layer: (True : GlobalMaxPool2D, False : MaxPool2D )
                 dropout_prob=0.5,          # 6.  Dropout Layer
                 optimizer="Adam",          # 7.  Create Model   (Adadelta, Adam)
                 print_model=False,         # build_model method 
                 num_epochs=100,             # Fit method : estimator.fit
                 batch_size=50,             # Fit method : estimator.fit
                 verbose=0                  # Fit method : estimator.fit 
                ):
        
        self.num_filters=num_filters
        self.filter_sizes=filter_sizes
        self.l2=l2
        self.global_max_pool=global_max_pool
        self.dropout_prob=dropout_prob
        self.optimizer=optimizer
        self.print_model=print_model
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.verbose=verbose
        
        self.estimator=None
    
    def build_model(self):
        '''
        Steps:
        0. Model Input Layer 
        1. Embedding layer ( 1.1 Pre-trained Embedding Layer  or  1.2 Random Embedding Layer )
        2. Expand dimension
        3. Convolution Layer
            3.1 Conv2D Layer
            3.2 Pooling Layer
        4. Combine all the pooled features
        5. Add dropout
        6. Model Output Layer (Dense layer with softmax)
        7. Create and compile Model with 'Model Input' and 'Model Output'
        '''
        model=None
        
        # 0. Input Layer and 1. Embedding layer
        model_input, x=None, None
        if self.input_shape == 3:  # 1.1 Pre-trained Embedding Layer
            model_input = keras.layers.Input(shape=(self.sequence_length, self.embedding_dim))
            x=model_input
        elif self.input_shape == 2: # 1.2 Random Embedding Layer
            model_input = keras.layers.Input(shape=(self.sequence_length,), dtype='int32')
            x = keras.layers.Embedding(self.max_words, self.embedding_dim)(model_input)
            
        # 2. Expand dimension
        x=tf.expand_dims(x, -1)
        
        # 3. Convolution Layer
        conv_kernel_reg = keras.regularizers.L2(self.l2)
        conv_bias_reg = keras.regularizers.L2(self.l2) 
        
        conv_blocks = []
        for filter_size in self.filter_sizes:
            # 3.1 Conv2D Layer
            conv = keras.layers.Conv2D(filters=self.num_filters, 
                      kernel_size=filter_size,
                      strides=(1, 1), 
                      padding='valid',
                      activation='relu',
                      kernel_regularizer=conv_kernel_reg, 
                      bias_regularizer=conv_bias_reg)(x)
            
            # 3.2 Pooling Layer
            if self.global_max_pool:
                conv = keras.layers.GlobalMaxPool2D()(conv)
            else:
                conv = keras.layers.MaxPool2D()(conv)
                conv = keras.layers.Flatten()(conv)
            
            conv_blocks.append(conv)
        
        # 4. Combine all the pooled features
        x = keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    
        # 5.Add dropout
        x = keras.layers.Dropout(self.dropout_prob)(x)

        # 6. Model Output
        model_output = keras.layers.Dense(2, activation='softmax')(x)
        
        # 7. Create Model
        model_optimizer=keras.optimizers.Adam()
        if self.optimizer=='Adadelta':
            model_optimizer=keras.optimizers.Adadelta()
        model = keras.Model(model_input, model_output)
        model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True), optimizer=model_optimizer, metrics=["accuracy"])
        
        if self.print_model:
            print(model.summary())
        
        return model

    def prepare_parameters(self, X):
        # 0.  Input Layer: shape[0]  
        self.sequence_length=0 
        
        #1. Embedding Layer
        self.input_shape=0 # 2, 3
        self.embedding_dim=0
        self.max_words=0
        
        #Prepare         
        self.input_shape=len(X.shape)
        
        if self.input_shape == 2: # Random Embedding Layer        (TEXTVEC shape  : (646, 60))
            self.embedding_dim=128
            self.max_words=max(map(max, X))
            self.max_words=self.max_words+2
        elif self.input_shape == 3: # Pre-trained Embedding Layer  (BERTWORD shape : (646, 46, 768))
            self.embedding_dim=X.shape[2] # 768
            
        self.sequence_length=X.shape[1]
        
    
    def fit(self, X, y):
        self.prepare_parameters(X)
        
        #Build model
        self.estimator=self.build_model()
        
        #Fit model
        self.estimator.fit(X, tf.one_hot(y, 2), batch_size=self.batch_size, epochs=self.num_epochs,
                    verbose=self.verbose)
        return self
    
    def predict(self, X):
        y_pred=self.estimator.predict(X)
        y_pred = (y_pred > 0.5).astype(np.int)
        return tf.argmax(y_pred, axis=1)
        
    def score(self, X, y, sample_weight=None):
        loss, accuracy = self.estimator.evaluate(X, y, verbose=False)
        return accuracy