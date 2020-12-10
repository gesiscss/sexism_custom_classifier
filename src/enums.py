class Model():
    LR='logistic_regression'
    CNN='cnn'
    SVM='svm'
    GENDERWORD='gender_word'
    THRESHOLDCLASSIFIER='threshold_classifier'    

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

class Feature():
    SENTIMENT='sentiment'
    NGRAM='ngram'
    TYPEDEPENDENCY='type_dependency'
    BERTDOC='bert_doc'
    BERTWORD='bert_word'
    TEXTVEC='textvec'
    GENDERWORD='gender_word'
    TOXICITY='toxicity'