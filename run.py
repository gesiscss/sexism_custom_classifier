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
import nltk
nltk.download('vader_lexicon')

FLAGS = flags.FLAGS

# Required parameters

flags.DEFINE_string(
    "params_file", None, 
    "")

flags.DEFINE_string(
    "hyperparams_file", None, 
    "")

flags.DEFINE_string(
    "file_name", None, 
    "")


class RunPipeline():
    '''Runs pipeline for a given data domain, model and features.'''
    def __init__(self):
        self.params=Params(FLAGS.params_file)
        self.hyperparams=Params(FLAGS.hyperparams_file)
    
    def prepare_parameters(self):
        self.make_dataset = MakeDataset()
        self.data_file=self.params.dict['data_file']
        self.iteration=self.params.dict['iteration']
        self.train_domains=self.params.dict['train_domains'][0]
        self.test_domains=self.params.dict['test_domains'][0]
        self.models=self.get_models()
        self.results_df = pd.DataFrame(columns=['iteration','train_domain', 'test_domain', 'model_name', 'features',
                                            'param_grid', 'best_params', 
                                            'y_test', 'y_pred', 'y_test_ids'])
        
        
    @execution_time_calculator
    def run(self):
        '''Runs the steps below.
        Step 1. Read data
        Step 2. Split data
        Step 3. Build Pipeline
        Step 4. Fit with sklearn GridSearchCV
        Step 5. Predict
        Step 6. Get and save classification report
        '''
        self.prepare_parameters()
        
        # Step 1. Read data
        data = self.make_dataset.read_data(self.data_file)
        
        for i in self.iteration:
            train_num=0
            
            splits_original, splits_modified=self.make_dataset.prepare_data_splits(data, random_state=i)
            
            for train_domain in self.train_domains:
                X_train, y_train=self.make_dataset.get_data_split(train_domain, splits_original, splits_modified, train=True)
    
                for model_name, features_set in self.models.items():
                    for f in features_set:
                        train_num=train_num+1
                        
                        features, param_grid=f['features'], f['param_grid']
                        print('{}.{} Running the pipeline for: {}, {}, {}'.format(i, train_num, model_name, 
                                                                            [k['name'] for k in features], train_domain))
                        # Step 3. Build Pipeline
                        pipeline=PipelineBuilder(features, model_name).build_pipeline()
                        
                        # Step 4. Fit
                        grid_search = GridSearchCV(pipeline, param_grid, cv=2, scoring='f1_macro')
                        grid_search.fit(X_train, y_train)
                        
                        # Step 5. Predict
                        for test_domain in self.test_domains:
                            X_test, y_test=self.make_dataset.get_data_split(test_domain, splits_original, splits_modified, test=True)
                            
                            y_pred=grid_search.predict(X_test)
            
                            self.results_df=self.results_df.append({
                                                  'iteration':'.'.join((str(i), str(train_num))),
                                                  'train_domain':train_domain, 'test_domain':test_domain, 
                                                  'model_name':model_name, 'features':features, 
                                                  'param_grid':param_grid, 
                                                  'best_params': grid_search.best_params_, 
                                                  'y_test':y_test, 'y_pred':y_pred, 
                                                  'y_test_ids':X_test.copy().reset_index()['_id']}, 
                                                 ignore_index=True)
                            
        # Step 6. Get and save classification report  
        metrics_df=self.results_df.apply(lambda r: self.get_classification_report(r['y_test'], r['y_pred']), axis=1)
        
        results_df=pd.concat((self.results_df, metrics_df), axis=1)
        
        file_name=''.join(('experiments/results_', FLAGS.file_name, '_')) if FLAGS.file_name != None else 'experiments/results_'
        timestr = time.strftime("%Y%m%d-%H%M%S")
        file_name=''.join((file_name, timestr, '.pkl'))
        
        with open(file_name, 'wb+') as f:
            pickle.dump(results_df, f)
            
        return file_name
    
    def get_models(self):
        valid_models=[Model.LR, Model.SVM, Model.CNN, Model.GENDERWORD, Model.THRESHOLDCLASSIFIER]
        params_models=list(set(self.params.dict['models'][0]).intersection(valid_models))
        
        models={}
        for name in params_models:
            features=[]
            
            if name == Model.CNN:
                features=self.get_cnn_features()
            elif name == Model.SVM or name == Model.LR:
                features=self.get_svm_and_logit_features(name)
            else:
                features=[{'features':[{'name':'baseline', 'feature_selection':False}], 
                           'param_grid':self.hyperparams.dict[name]}]
                
            models[name]=features
        return models
    
    def get_cnn_features(self):
        params_features=self.params.dict['features']
        
        valid_features=[Feature.TEXTVEC, Feature.BERTWORD]
        
        features=[]
        for k, v in params_features.items():
            name=v['name']
            if name in valid_features:
                param_grid_model=self.hyperparams.dict[''.join((Model.CNN, '_', name))]
                features.append({
                    'features':{'name':name, 'feature_selection':v['feature_selection']}, 
                    'param_grid':{**v['param_grid'], **param_grid_model}
                })
        return [{'features':[i['features']], 'param_grid':i['param_grid']} for i in features] 
    
    def get_svm_and_logit_features(self, model):
        valid_features=[Feature.SENTIMENT, Feature.NGRAM, Feature.TYPEDEPENDENCY, Feature.BERTDOC]
        features=self.get_features(valid_features)
        
        #Return feature combinations
        return self.get_feature_combinations(features, model)
    
    def get_features(self, valid_features):
        params_features=self.params.dict['features']
        
        features=[]
        for k, v in params_features.items():
            name=v['name']
            if name in valid_features:
                features.append({
                    'features':{'name':name, 'feature_selection':v['feature_selection']}, 
                    'param_grid':v['param_grid']
                })
         
        return features
    
    def get_feature_combinations(self, features, model):
        param_grid_model=self.hyperparams.dict[model]
        
        #1.Create index combinations
        comb_list=[]
        feature_count=len(features)
        for i in range(feature_count):
            comb_list.extend(list(itertools.combinations(range(feature_count), (i+1))))

        #2.Create feature combination list by using index combinations
        feature_combinations=[]
        for combination in comb_list:
            comb_features, feature_names=[], []
            param_grid_features={}
    
            for f in combination:
                #Prevent adding same features in a combination (e.g. ngram (1,1) and ngram(1,2))
                if features[f]['features']['name'] in feature_names:
                    feature_names=[]
                    break
                
                feature_names.append(features[f]['features']['name'])
                
                comb_features.append(features[f]['features'])
                param_grid_features.update(features[f]['param_grid'])
            if len(feature_names) > 0:
                feature_combinations.append(
                    {'features': comb_features, 'param_grid': {**param_grid_features, **param_grid_model}})
    
        return feature_combinations
    
    def get_classification_report(self, y_true, y_pred):
        target_names = ['nonsexist', 'sexist']
        classification_dict =  classification_report(y_true, y_pred, target_names=target_names, output_dict=True)

        filter_list=['nonsexist', 'sexist', 'macro avg']
        filtered_dict={k:v for (k,v) in classification_dict.items() if k in filter_list}
        filtered_dict={ ' '.join((key, metric)):value for key, value_dict in filtered_dict.items() for metric, value in value_dict.items()}
    
        return pd.Series(filtered_dict)

def main(argv):
    file_name=RunPipeline().run()
    print()
    print(file_name)
    print()

if __name__ == '__main__':
    # Required flag.
    flags.mark_flag_as_required("params_file")
    flags.mark_flag_as_required("hyperparams_file")
    
    app.run(main)