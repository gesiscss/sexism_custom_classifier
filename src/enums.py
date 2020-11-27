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
    BERT='bert'
    TEXTVEC='textvec'

class DataColumn():
    ID='_id'
    ADVERSARIAL='of_id'
    DATASET='dataset'
    TEXT='text'
    LABEL='sexist'