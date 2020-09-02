#!/usr/bin/env python
# coding: utf-8

# In[3]:

from nltk.sentiment.vader import SentimentIntensityAnalyzer

class BuildSentimentFeature:
    '''Extracts Sentiment Features'''
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
    
    def get_sentiment_intensity_scores(self, text):
        '''Retrieves polarity scores of text. 
    
        Args:
        text: Message text
    
        Returns:
        score (dictionary): Polarity scores.
    
        Example:
            >>> get_sentiment_intensity_scores("The file is super cool.")
            {'neg': 0.0, 'neu': 0.326, 'pos': 0.674, 'compound': 0.7351}
        '''
        return self.sid.polarity_scores(text)
    
    def combine_polarity_scores(self, scores):
        '''Combines polarity scores into a single feature vector. 
    
        Args:
        score (dictionary): Polarity scores.
    
        Returns:
        dictionary: Combined polarity scores.
    
        Example:
            >>> combine_polarity_scores({'neg': 0.0, 'neu': 0.326, 'pos': 0.674, 'compound': 0.7351})
            [0.0, 0.326, 0.674, 0.7351]
        '''
        return [scores['neg'], scores['neu'], scores['pos'], scores['compound']]
    
    def get_polarity_scores(self, text):
        scores = self.get_sentiment_intensity_scores(text)
        return self.combine_polarity_scores(scores)
    
    def build_features(self, X_train, X_test):
        X_train_features = [self.get_polarity_scores(str(text)) for text in X_train]
        X_test_features = [self.get_polarity_scores(str(text)) for text in X_test]
        
        print('shape of train features : ' + str(len(X_train_features)))
        print('shape of test features : ' + str(len(X_test_features)))
        
        return X_train_features, X_test_features