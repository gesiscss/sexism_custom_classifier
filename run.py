#src module
from src.data.make_dataset import MakeDataset
from src.builder.pipeline_builder import PipelineBuilder
from src.utilities import *

from src.enums import *

#sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import classification_report

#other
import pickle
import time
import pandas as pd
import itertools
from absl import app, flags

FLAGS = flags.FLAGS

# Required parameters

flags.DEFINE_string(
    "data_file", None, 
    "The .csv file that contains data.")

# Other parameters
flags.DEFINE_integer(
    "iteration_count", 5, 
    "")

flags.DEFINE_list(
    "features", 'sentiment,ngram', 
    "")

flags.DEFINE_list(
    "models", 'logistic_regression,svm', 
    "")

flags.DEFINE_string(
    "stanford_corenlp_dir", None,
    "The type dependency parser model directory. It should contain the .jar files (or other data files) for the task.")

flags.DEFINE_string(
    "bert_doc_emb_file", None, 
    "")

flags.DEFINE_string(
    "bert_word_emb_file", None, 
    "")

class RunPipeline():
    '''Runs pipeline for a given data domain, model and features.'''
        
    def get_classification_report(self, y_true, y_pred):
        target_names = ['nonsexist', 'sexist']
        classification_dict =  classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

        filter_list=['nonsexist', 'sexist', 'macro avg']
        filtered_dict={k:v for (k,v) in classification_dict.items() if k in filter_list}
        filtered_dict={ ' '.join((key, metric)):value for key, value_dict in filtered_dict.items() for metric, value in value_dict.items()}
    
        return pd.Series(filtered_dict)
        
    @execution_time_calculator
    def run(self):
        '''Runs the steps below.
        Step 1. Read data
        Step 2. Split data
        Step 3. Build Pipeline
        Step 4. Fit and predict with sklearn GridSearchCV
        Step 5. Get classification report
        '''
        features_list=self.get_features()
        models=self.get_models()
        
        train_domain=Domain.BHO
        test_domain=Domain.BHO
        
        # Step 1. Read data
        make_dataset = MakeDataset()
        data = make_dataset.read_data(FLAGS.data_file)
        
        results_df = pd.DataFrame(columns=['train_domain', 'train_domain_modified','test_domain', 'test_domain_modified',
                                            'model_name', 'feature_names', 'param_grid', 'best_params', 
                                            'y_test', 'y_pred', 'y_test_ids'])
        
        for i in range(FLAGS.iteration_count):
        
            # Step 2. Split data
            X_train, y_train, X_test, y_test=make_dataset.get_balanced_data_split(data, train_domain, test_domain)
            
            for model, param_grid_model in models.items():
                
                for features in features_list:
                    feature_names=features['features']
                    param_grid={**features['param_grid'], **param_grid_model} 
                    print('Running the pipeline for {} and {}'.format(model, feature_names))
                
                    # Step 3. Build Pipeline
                    pb=PipelineBuilder(model, feature_names)
                    pipeline=pb.build_pipeline()
                   
                    # Step 4. Fit and predict
                    grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='f1_macro')
                    grid_search.fit(X_train, y_train)
                    y_pred=grid_search.predict(X_test)
            
                    results_df=results_df.append({'train_domain':train_domain['dataset'], 
                                                  'train_domain_modified':train_domain['modified'], 
                                                  'test_domain':test_domain['dataset'], 
                                                  'test_domain_modified':test_domain['modified'], 
                                                  'model_name':model, 'feature_names':feature_names, 
                                                  'param_grid':param_grid, 'best_params': grid_search.best_params_, 
                                                  'y_test':y_test, 'y_pred':y_pred, 'y_test_ids':X_test.index}, 
                                                 ignore_index=True)
            
        # Step 5. Get classification report  
        metrics_df=results_df.apply(lambda r: self.get_classification_report(r['y_test'], r['y_pred']), axis=1)
        
        results_df=pd.concat((results_df, metrics_df), axis=1)
        
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name=''.join(('results/results_', timestr, '.pkl'))
        
        with open(file_name, 'wb+') as f:
            pickle.dump(results_df, f)
            
        return file_name
    
    def get_feature_combinations(self, features):
        comb_list=[]
        feature_count=len(features)
        for i in range(feature_count):
            comb_list.extend(list(itertools.combinations(range(feature_count), (i+1))))

        feature_combinations=[]
        for combination in comb_list:
            feature_names=[]
            param_grids={}
    
            for f in combination:
                feature_names.append(features[f]['name'])
                param_grids.update(features[f]['param_grid'])
        
            feature_combinations.append({'features': feature_names, 'param_grid': param_grids})
    
        return feature_combinations
    
    def get_features(self):
        #param grids for features
        param_grid_sentiment={
            'features__sentiment__feature_extraction__score_names': [['neg', 'compound'], ['neg', 'compound', 'neu', 'pos'], ],
        }   

        param_grid_ngram={
            'features__ngram__feature_extraction__ngram_range': [(1, 2),],
            #'features__ngram__feature_selection__model': [model, ],
        }   

        param_grid_type_dependency={
            'features__type_dependency__feature_extraction__ngram_range': [(1, 1),],
            'features__type_dependency__feature_extraction__model_path': [FLAGS.stanford_corenlp_dir, ],
            #'features__type_dependency__feature_selection__model': [model, ],
        }   

        param_grid_bert={
            'features__bert__feature_extraction__embedding_file_name': [FLAGS.bert_doc_emb_file, ],
            'features__bert__feature_extraction__extract': [False, ],
            'features__bert__feature_extraction__model_name': ['bert-base-uncased', ],
        }   

        param_grid_text_vec={
            'features__textvec__feature_extraction__max_tokens': [2400, ],
            'features__textvec__feature_extraction__output_sequence_length': [60, ],
        }
        
        sentiment={'name': Feature.SENTIMENT, 'param_grid': param_grid_sentiment}
        ngram={'name': Feature.NGRAM, 'param_grid': param_grid_ngram}
        type_dependency={'name': Feature.TYPEDEPENDENCY, 'param_grid': param_grid_type_dependency}
        bert={'name': Feature.BERT, 'param_grid': param_grid_bert}
        text_vec={'name': Feature.TEXTVEC, 'param_grid': param_grid_text_vec}
        
        #Define implemented features and parameter grids
        valid_features=[sentiment, ngram, type_dependency, bert, text_vec]
        
        #Filter feature list with FLAGS.features
        features = [f for f in valid_features if f['name'] in FLAGS.features]
        
        #Return feature combinations
        return self.get_feature_combinations(features)
        
    def get_models(self):
        #param grids for models
        param_grid_logit={
            'model__C': [10,], #[1, 10, 100], #[0.001,0.01,0.1,1,10,100],
            #'model__penalty': ['l2'],  #['l1', 'l2']
            #'model__class_weight':[None, 'balanced', ],  
            'model__max_iter':[2500] # [100, 600, ], 
        }

        param_grid_svm={
            'model__C': [1, 10, 100],  #[0.001,0.01,0.1,1,10,100],
        }

        param_grid_cnn={
            'model__num_filters': [8], #[120, 100, 80],  #conv output
            #'model__hidden_dims': [64], #[64, 50, 10], 
            'model__num_epochs': [30], #[15, 30, 50],
            'model__filter_sizes':[(2,3,4,5,6), ],
            #'model__filter_sizes':[(2,3,4,5,6), (2,3,4), (3,4,5)], 
            #'model__l2':[0.00001, 0.01, 0.1],
            'model__embedding_dim':[50 ,]
        }
        
        valid_models={Model.LR:param_grid_logit, Model.SVM:param_grid_svm, Model.CNN:param_grid_cnn, }
        return {k:v for (k,v) in valid_models.items() if k in FLAGS.models}
        

def main(argv):
    if 'type_dependency' in FLAGS.features and FLAGS.stanford_corenlp_dir is None:
        raise ValueError("To use type dependency feature, stanford_corenlp_dir should be given.")
    if 'bert' in FLAGS.features and (FLAGS.bert_doc_emb_file is None or FLAGS.bert_word_emb_file is None):
        raise ValueError("To use bert embedding feature, bert_doc_emb_file and bert_word_emb_file should be given.")
    
    file_name=RunPipeline().run()
    print(file_name)

if __name__ == '__main__':
    # Required flag.
    flags.mark_flag_as_required("data_file")
    
    app.run(main)