
# coding: utf-8

# In[1]:


from enum import Enum

class Model():
    LR='logistic_regression'
    CNN='cnn'
    SVM='svm'

class Dataset():
    BENEVOLENT='benevolent'
    HOSTILE='hostile'
    OTHER='other'
    CALLME='callme'
    SCALES='scales'
    
class Domain():
    BHO={'modified': False, 'dataset': [Dataset.BENEVOLENT, Dataset.HOSTILE, Dataset.OTHER]}
    C={'modified': False, 'dataset': [Dataset.CALLME]}
    BHOCS={'modified': False, 'dataset': [Dataset.BENEVOLENT, Dataset.HOSTILE, Dataset.OTHER, Dataset.CALLME, Dataset.SCALES]}
    S={'modified': False, 'dataset': [Dataset.SCALES]}
    BHOM={'modified': True, 'dataset': [Dataset.BENEVOLENT, Dataset.HOSTILE, Dataset.OTHER]}
    CM={'modified': True, 'dataset': [Dataset.CALLME]}
    BHOCSM={'modified': True, 'dataset': [Dataset.BENEVOLENT, Dataset.HOSTILE, Dataset.OTHER, Dataset.CALLME, Dataset.SCALES]}
    
class Label():
    NONSEXIST=int('0')
    SEXIST=int('1')

class Feature():
    SENTIMENT='sentiment'
    NGRAM='ngram'
    TYPEDEPENDENCY='type_dependency'
    BERTDOCEMB='bert_doc_emb'

class Parameter():
    class Sentiment():
            SCORE_NAMES='score_names' # ['neg', 'neu', 'pos', 'compound']
    class Ngram():
            NGRAM_RANGE='ngram_range' # (2,2)
    class TypeDependency():
            MODEL_PATH='model_path'
            NGRAM_RANGE='ngram_range' # (2,2)
   
            
class DataColumn():
    ID='_id'
    ADVERSARIAL='of_id'
    DATASET='dataset'
    TEXT='text'
    LABEL='sexist'