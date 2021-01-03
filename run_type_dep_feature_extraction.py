#Extracts and saves type dependency features by using all data.

from src.data.make_dataset import MakeDataset
from src.data.preprocessing.preprocess_type_dependency import PreprocessTypeDependency
from src.feature_extraction.build_type_dependency_features import BuildTypeDependencyFeature
from src.enums import * 
import os
from absl import app, flags

FLAGS = flags.FLAGS

# Required parameters

flags.DEFINE_string(
    "data_file", None, 
    "The .csv file that contains data.")

flags.DEFINE_string(
    "model_path", None, 
    "")


def extract_features():
    save_path = os.path.abspath('experiments') 
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # 1. Read Data
    data = MakeDataset().read_data(FLAGS.data_file)
    data.set_index('_id', inplace=True)

    # 2. Preprocess
    preprocessed=PreprocessTypeDependency().transform(data.text)
    
    # 3. Extract features and then save
    BuildTypeDependencyFeature(model_path=FLAGS.model_path, save_path=save_path, extract=True).fit(preprocessed)
    
def main(argv):
    extract_features()
    print('FINISHED')

if __name__ == '__main__':
    # Required flag.
    flags.mark_flag_as_required("data_file")
    flags.mark_flag_as_required("model_path")
    
    
    app.run(main)