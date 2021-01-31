#sklearn
from sklearn.base import BaseEstimator

#other
from nltk.sentiment.vader import SentimentIntensityAnalyzer

class BuildSentimentFeature(BaseEstimator):
    '''Extracts sentiment features by using VADER sentiment analysis model.'''
    
    def __init__(self, score_names=['neg', 'neu', 'pos', 'compound']):
        self.score_names=score_names
        self.sid = SentimentIntensityAnalyzer()
        self.feature_dimension=len(self.score_names)
        
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