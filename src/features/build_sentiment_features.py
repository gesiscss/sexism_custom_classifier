#!/usr/bin/env python
# coding: utf-8

# In[3]:

#src module
from src.features.build_features import BuildFeature

#nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class BuildSentimentFeature(BuildFeature):
    '''Extracts sentiment features by using VADER sentiment analysis model.'''
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.score_names=['neg', 'neu', 'pos', 'compound']
        
    def get_polarity_scores(self, text):
        '''Retrieves combineed polarity scores of text (neutral and compoound scores). 
    
        Args:
        text: Message text
    
        Returns:
        dictionary: Combined polarity scores (neutral and compoound).
    
        Example:
            >>> bsf = BuildSentimentFeature(score_names=['neu', 'compound'])
            >>> bsf.get_polarity_scores("The file is super cool.")
            [0.326, 0.7351]
        '''
        
        #Example scores = {'neg': 0.0, 'neu': 0.326, 'pos': 0.674, 'compound': 0.7351}
        scores = self.sid.polarity_scores(text)
        return [scores[name] for name in self.score_names]
    
    def fit(self, x, y=None):
        return self

    def transform(self, texts):
        return [self.get_polarity_scores(str(text)) for text in texts]