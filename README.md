[![Binder](img/launch_binder.png)](https://notebooks.gesis.org/binder/v2/gh/gesiscss/sexism_custom_classifier/master?filepath=notebooks)

# Sexism Custom Classifier

## Introduction

Sexism custom classifier is implemented to detect sexism automatically in the field of natural language processing. In this work, we used a dataset from [Samory et al. (2020)](https://arxiv.org/abs/2004.12764). We will train three models on top of the different feature sets. The aim of the experiments is to answer two research questions:  

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