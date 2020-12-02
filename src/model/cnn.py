#sklearn
from sklearn.base import BaseEstimator

#tensorflow
from tensorflow import keras
import tensorflow as tf

import numpy as np

class CNN(BaseEstimator):
    '''Builds CNN model.'''
    
    def __init__(self, 
                 global_max_pool=False,
                 num_epochs=15, 
                 batch_size=64, 
                 num_filters=100, 
                 hidden_dims=50, 
                 filter_sizes=(2,3,4,5,6), 
                 embedding_dim=150, 
                 l2=0.01, 
                 output_activation='softmax',
                 embedding_layer=True,
                 max_words=3400,
                 sequence_length=60,
                 dropout_prob=(0.2,0.5),
                 verbose=0,
                 print_model=False
                ):
        
        self.global_max_pool=global_max_pool
        self.num_epochs=num_epochs
        self.batch_size=batch_size
        self.num_filters=num_filters
        self.hidden_dims=hidden_dims
        self.filter_sizes=filter_sizes
        self.embedding_dim=embedding_dim
        self.l2=l2
        self.output_activation=output_activation
        self.embedding_layer=embedding_layer
        self.max_words=max_words
        self.sequence_length=sequence_length
        self.dropout_prob=dropout_prob
        self.verbose=verbose
        self.print_model=print_model
        
        self.estimator=None
        
    def build_cnn(self):
        #TODO
        return None
    
    def build_cnn_with_emb(self):
        model_input = keras.layers.Input(shape=(self.sequence_length,), dtype='int32')
    
        # 1.Embedding layer
        x = keras.layers.Embedding(self.max_words, self.embedding_dim)(model_input)
        
        # 2. Expand dimension
        x=tf.expand_dims(x, -1)
        
        # 3.Add dropout
        x = keras.layers.SpatialDropout2D(self.dropout_prob[0])(x)
       
        # 4.Create a convolution + pooling layer for each filter size
        conv_kernel_reg = keras.regularizers.L2(self.l2)
        conv_bias_reg = keras.regularizers.L2(self.l2)  
        
        conv_blocks = []
        for filter_size in self.filter_sizes:
            # 4.1 Convolution layer
            conv = keras.layers.Conv2D(filters=self.num_filters, 
                      kernel_size=filter_size,
                      strides=(1, 1), 
                      padding='valid',
                      activation='relu',
                      kernel_regularizer=conv_kernel_reg, 
                      bias_regularizer=conv_bias_reg)(x)
            
            # 4.2 Pooling layer
            if self.global_max_pool:
                conv = keras.layers.GlobalMaxPool2D()(conv)
            else:
                conv = keras.layers.MaxPool2D()(conv)
                conv = keras.layers.Flatten()(conv)
            
            conv_blocks.append(conv)
        
        # 5.Combine all the pooled features
        x = keras.layers.Concatenate()(conv_blocks) if len(conv_blocks) > 1 else conv_blocks[0]
    
        # 6.Add dropout
        x = keras.layers.Dropout(self.dropout_prob[1])(x)

        # 7. Dense layer
        x = keras.layers.Dense(self.hidden_dims, activation="relu")(x)
        
        # 8. Model Output
        output_num=2 #softmax
        if self.output_activation== 'sigmoid':
            output_num=1
        model_output = keras.layers.Dense(output_num, activation=self.output_activation)(x)
        
        # 9. Create Model
        model = keras.Model(model_input, model_output)
        model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=True),
                      optimizer=keras.optimizers.Adam(),
                      metrics=["accuracy"])
        
        if self.print_model:
            print(model.summary())
        
        return model
    
    def build_model(self):
        if self.embedding_layer:  #Creates model with embedding layer(randomly initialized weights)
            return self.build_cnn_with_emb()
        else:
            return self.build_cnn()
        return None

    def fit(self, X, y):
        #Build model
        self.estimator=self.build_model()
        
        #Fit model
        if self.output_activation=='softmax':
            y=tf.one_hot(y, 2) 
        
        self.estimator.fit(X, y, batch_size=self.batch_size, epochs=self.num_epochs,
                    verbose=self.verbose)
        return self
    
    def predict(self, X):
        y_pred=self.estimator.predict(X)
        
        if self.output_activation== 'softmax':
            y_pred = (y_pred > 0.5).astype(np.int)
            return tf.argmax(y_pred, axis=1)
        else: #sigmoid
            return [int(i > 0.5) for i in y_pred]  
    
    def score(self, X, y, sample_weight=None):
        loss, accuracy = self.estimator.evaluate(X, y, verbose=False)
        return accuracy