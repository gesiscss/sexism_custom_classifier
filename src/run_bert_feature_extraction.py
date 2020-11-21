#Extracts and saves BERT embeddings by using all data.

from src.data.make_dataset import MakeDataset
from src.data.preprocessing.preprocess_bert import PreprocessBert
from src.feature_extraction.build_bert_features import BuildBERTFeature
from src.enums import * 
import os

DATA_PATH='../data/raw/all_data_augmented.csv'
SAVE_DIR='bert_embeddings'

save_path = os.path.abspath(SAVE_DIR) 
if not os.path.exists(save_path):
    os.makedirs(save_path)

make_dataset = MakeDataset()
data = make_dataset.read_data(DATA_PATH)
data.set_index('_id', inplace=True)

pre=PreprocessBert()
preprocessed=pre.transform(data.text)

bf=BuildBERTFeature(output_hidden_states=False, extract=True, save_path=save_path)
file_name=bf.fit(preprocessed)

print('file name ==> {}'.format(file_name))