
# coding: utf-8

# In[3]:


#src module
from src.data.make_dataset import MakeDataset
from src.enums import Feature

class RunPreprocessing():
    def run(self, features = [Feature.SENTIMENT, Feature.NGRAM]):
        m = MakeDataset()
        data = m.read_csv()
        m.preprocess(data, features)

