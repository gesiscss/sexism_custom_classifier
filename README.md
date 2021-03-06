[![Binder](https://notebooks.gesis.org/binder/badge.svg)](https://notebooks.gesis.org/binder/v2/gh/gesiscss/sexism_custom_classifier/master?filepath=notebooks)

# Sexism Custom Classifier

## Introduction

Sexism custom classifier is implemented to detect sexism automatically in the field of natural language processing. In this work, the dataset from [Samory et al. (2020)](https://arxiv.org/abs/2004.12764) was used to train three models on top of four different feature sets (both individual and combinations). The aim of the experiments is to answer two research questions:  

* What is the informativeness of different feature sets on the sexism detection task?
* What are the improvements on the sexism classifiers’ performance when different feature sets are introduced?

## Support Features

* Sentiment : The sentiment intensity provided by [VADER](http://eegilbert.org/papers/icwsm14.vader.hutto.pdf) sentiment analysis method.
* Word N-grams : Term Frequency/Inverse Frequency (TF-IDF) weights of word n-grams proviided by [scikit-learn](https://jmlr.csail.mit.edu/papers/v12/pedregosa11a.html).
* Type Dependency : The type dependency relationships provided by [Stanford Parser](https://nlp.stanford.edu/~wcmac/papers/td-lrec06.pdf). The parser can be dowloaded from [here](https://nlp.stanford.edu/software/lex-parser.shtml#Download).  
* Document Embeddings : The document embeddings provided by [BERT](https://arxiv.org/abs/1810.04805) language representation model.

## Support Models

* Logistic Regression
* Convolutional Neural Network
* Support Vector Machine

## Experiments

This example code trains Logistic Regression on top of sentiment and uni-gram features. To replicate the results, run the scripts in the 'experiments/scripts.txt'

```shell
export PARAMS_FILE='experiments/params.json'
export HYPERPARAMS_FILE='experiments/hyperparams.json'

python run.py \
  --params_file=$PARAMS_FILE \
  --hyperparams_file=$HYPERPARAMS_FILE \
```

This example code obtains the pre-trained BERT embeddings.

```shell
export DATA_FILE=/path/data.csv

python run_bert_feature_extraction.py \
	--data_file=$DATA_FILE \
```
