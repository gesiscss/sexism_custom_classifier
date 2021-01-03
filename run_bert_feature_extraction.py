#Extracts and saves BERT embeddings by using all data.

from src.data.make_dataset import MakeDataset
from src.data.preprocessing.preprocess_bert import PreprocessBert
from src.feature_extraction.build_bert_features import BuildBERTFeature
from src.enums import * 
import os
from absl import app, flags

FLAGS = flags.FLAGS

# Required parameters

flags.DEFINE_string(
    "data_file", None, 
    "The .csv file that contains data.")

def extract_features():
    save_path = os.path.abspath('experiments') 
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. Read Data
    data = MakeDataset().read_data(FLAGS.data_file)
    data.set_index('_id', inplace=True)

    # 2. Preprocess
    preprocessed=PreprocessBert().transform(data.text)
    
    # 3. Extract features and then save
    file_name=BuildBERTFeature(output_hidden_states=False, extract=True, save_path=save_path).fit(preprocessed)
    
    return file_name

def main(argv):
    file_name=extract_features()
    print(file_name)

if __name__ == '__main__':
    # Required flag.
    flags.mark_flag_as_required("data_file")
    
    app.run(main)