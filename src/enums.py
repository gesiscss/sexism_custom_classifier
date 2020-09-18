
# coding: utf-8

# In[1]:


from enum import Enum

class Model(Enum):
    LR=1
    CNN=2
    SVM=3

class Dataset():
    BENEVOLENT='benevolent'
    HOSTILE='hostile'
    OTHER='other'
    CALLME='callme'
    SCALES='scales'
    
class Domain():
    BHO='bho'
    C='c'
    BHOCS='bhocs'
    S='s'
    BHOM='bhom'
    CM='cm'
    BHOCSM='bhocsm'
    
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