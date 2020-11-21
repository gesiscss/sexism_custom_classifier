#src module
from src.data.make_dataset import MakeDataset
from src.builder.pipeline_builder import PipelineBuilder
from src.decorators import *
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

#param grids for models
param_grid_logit = {
    'model__C': [1, 10, 100],  #[0.001,0.01,0.1,1,10,100],
    #'model__penalty': ['l2'],  #['l1', 'l2']
    #'model__class_weight':[None, 'balanced', ],  
    #'model__max_iter':[100, 600, ], 
}

param_grid_svm = {
    'model__C': [1, 10, 100],  #[0.001,0.01,0.1,1,10,100],
}


#param grids for features
param_grid_sentiment = {
    'features__sentiment__feature_extraction__score_names': [['neg', 'compound'], ['neg', 'compound', 'neu', 'pos'], ],
}   

param_grid_ngram = {
    'features__ngram__feature_extraction__ngram_range': [(1, 2),],
    #'features__ngram__feature_selection__model': [model, ],
}   

param_grid_type_dependency= {
    'features__type_dependency__feature_extraction__ngram_range': [(1, 1),],
    'features__type_dependency__feature_extraction__model_path': ['/Users/ezgidaldal/sexism_custom_classifier/stanford-corenlp-4.1.0', ],
    #'features__type_dependency__feature_selection__model': [model, ],
}   

param_grid_bert= {
    'features__bert__feature_extraction__embedding_file_name': ['../src/bert_embeddings/doc_embeddings20201121-161343.pkl', ],
    'features__bert__feature_extraction__extract': [False, ],
    'features__bert__feature_extraction__model_name': ['bert-base-uncased', ],
}   

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
    def run(self, train_domain, test_domain, models, features_list, iteration_count=1, file_name='results_'):
        '''Runs the steps below.
        Step 1. Read data
        Step 2. Split data
        Step 3. Build Pipeline
        Step 4. Fit and predict with sklearn GridSearchCV
        Step 5. Get classification report
        '''
        
        # Step 1. Read data
        make_dataset = MakeDataset()
        data = make_dataset.read_data('../data/raw/all_data_augmented.csv')
        
        results_df = pd.DataFrame(columns=['train_domain', 'train_domain_modified','test_domain', 'test_domain_modified',
                                            'model_name', 'feature_names', 'param_grid', 'best_params', 
                                            'y_test', 'y_pred', 'y_test_ids'])
        
        for i in range(iteration_count):
        
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
        file_name=''.join(('../results/', file_name, timestr, '.pkl'))
        
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
    
    def run_experiments_research_question_1(self):
        train_domain=Domain.BHO
        test_domain=Domain.BHO
        
        models={Model.LR:param_grid_logit, Model.SVM:param_grid_svm}
        
        sentiment={'name': Feature.SENTIMENT, 'param_grid': param_grid_sentiment}
        ngram={'name': Feature.NGRAM, 'param_grid': param_grid_ngram}
        type_dependency={'name': Feature.TYPEDEPENDENCY, 'param_grid': param_grid_type_dependency}
        bert={'name': Feature.BERT, 'param_grid': param_grid_bert}
        
        features=[sentiment, ngram, type_dependency, bert]
        
        features_list=self.get_feature_combinations(features)
                
        file_name=self.run(train_domain, test_domain, models, features_list, iteration_count=1, file_name='results_rq1_')
        
        return file_name