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
from sklearn.model_selection import StratifiedKFold

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
        
        
    
    def extract_features(self, splits_original, splits_modified, train_domain, test_domains):
        train_features, test_features=[],[]
        
        #1. Prepare train split for train_domain
        X_train, y_train=self.make_dataset.get_data_split(train_domain, splits_original, splits_modified, train=True)
        train_features={'train_domain':train_domain, 'X_train':X_train, 'y_train':y_train, 'extracted_features': []}
        
        #2. Prepare test split for each test_domains
        test_features=[]
        for test_domain in self.test_domains:
            X_test, y_test=self.make_dataset.get_data_split(test_domain, splits_original, splits_modified, test=True)
            test_features.append({'test_domain':test_domain, 'X_test':X_test, 'y_test':y_test, 'extracted_features': []})
        
        #3. Extract train and test features
        params_features=self.params.dict['features']
        
        for k, v in params_features.items():
            feature_selection, name=v['feature_selection'], v['name']
                        
            #3.1 Extract train features
            fit_params=v['feature_pipeline_params']
            
            pipe=PipelineBuilder().build_feature_pipeline(name, feature_selection)
            pipe.set_params(**fit_params)
            
            train_feature_list=train_features['extracted_features']
            train_feature_list.append({'name':name, 'value':pipe.fit_transform(X_train, y_train)})
            train_features['extracted_features']=train_feature_list
                    
            
            #3.2 Extract test features for each test domain
            for t in test_features:
                feature_list=t['extracted_features']
                feature_list.append({'name':name, 'value':pipe.transform(t['X_test'])})
                t['extracted_features']=feature_list
        
        return train_features, test_features
    
    @execution_time_calculator
    def run(self):
        '''Runs the steps below.
        Step 1. Read data
        Step 2. Extract Features
        Step 3. Build Feature Union For Train
        Step 4. Build Model Pipeline
        Step 5. Fit with sklearn GridSearchCV
        Step 6. Predict
                6.1 Build Feature Union For Test
                6.2 Predict
        Step 7. Get and save classification report
        '''
        self.prepare_parameters()
        
        # Step 1. Read data
        data = self.make_dataset.read_data(self.data_file)
        
        for i in self.iteration:
            train_num=0
            
            splits_original, splits_modified=self.make_dataset.prepare_data_splits(data, random_state=i)
            
            for train_domain in self.train_domains:
                #Step 2. Extract Features
                train_features, test_features=self.extract_features(splits_original, splits_modified, train_domain, self.test_domains)
                        
                for model_name, features_set in self.models.items():
                    
                    for fs in features_set:
                        train_num=train_num+1
                        param_grid=self.hyperparams.dict[model_name]
                        combination, features=fs['combination'], fs['features']
                        
                        
                        print('{}.{}/{} Running the pipeline for: {}, {}, {}'.format(i, train_num, len(features_set), 
                                                                                     model_name, combination, train_domain))
                        
                        # 4. Build Feature Union For Train
                        pb=PipelineBuilder()
                        feature_union_train=pb.build_feature_union(combination, train_features['extracted_features'])
                        X_train, y_train=train_features['X_train'], train_features['y_train']
                        X_train=feature_union_train.fit_transform(X_train, y_train)
                        #print('X_train.shape', X_train.shape)
                        
                        # 5. Fit
                        sf=StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
                        pipeline_g=pb.build_model_pipeline(model_name)
                        grid_search = GridSearchCV(pipeline_g, param_grid=param_grid, cv=sf, scoring='f1_macro', n_jobs=-1)
                        grid_search.fit(X_train, y_train)
                        
                        # 6. Predict
                        for t in test_features:
                            # 6.1 Build Feature Union For Test
                            feature_union_test=pb.build_feature_union(combination, t['extracted_features'])
                            
                            X_test, y_test, test_domain=t['X_test'], t['y_test'], t['test_domain']
                            y_test_ids=X_test.copy().reset_index()['_id']
                            X_test=feature_union_test.fit_transform(X_test)
                            #print('X_test.shape', X_test.shape)
                            
                            # 6.2 Predict
                            y_pred=grid_search.predict(X_test)
                            
                            self.results_df=self.results_df.append({
                                                  'iteration':'.'.join((str(i), str(train_num))),
                                                  'train_domain':train_domain, 'test_domain':test_domain, 
                                                  'model_name':model_name, 'features':features, 
                                                  'param_grid':param_grid, 
                                                  'best_params': grid_search.best_params_, 
                                                  'y_test':y_test, 'y_pred':y_pred, 
                                                  'y_test_ids':y_test_ids }, 
                                                  ignore_index=True)
                            
                            
        # Step 7. Get and save classification report  
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
        models=list(set(self.params.dict['models'][0]).intersection(valid_models))
        return {name:self.get_model_features(name) if name in [Model.CNN,Model.SVM,Model.LR] else [] for name in models}
        
    def get_model_features(self, name):
        params_features=self.params.dict['features']
        
        if name == Model.CNN:
            valid_features=[Feature.TEXTVEC, Feature.BERTWORD]
            features=self.filter_features(valid_features, params_features)
            return self.get_feature_combinations(features, feature_count=1)
        else:
            valid_features=[Feature.SENTIMENT, Feature.NGRAM, Feature.TYPEDEPENDENCY, Feature.BERTDOC]
            features=self.filter_features(valid_features, params_features)
            return self.get_feature_combinations(features, len(features))
    
    def filter_features(self, valid_features, params_features):
        features=[]
        for k, v in params_features.items():
            if v['name'] in valid_features:
                features.append(v)
        return features
    
    def get_feature_combinations(self, features, feature_count):
        #1.Create index combinations
        comb_list=[]
        for i in range(feature_count):
            comb_list.extend(list(itertools.combinations(range(feature_count), (i+1))))
        
        #2.Create feature combination list by using index combinations
        feature_combinations=[]
        for combination in comb_list:
            comb_features, feature_names=[], []
            for f in combination:
                #To prevent adding same features in a combination (e.g. ngram (1,1) and ngram(1,4))
                if features[f]['name'] in feature_names:
                    feature_names=[]
                    break
                
                feature_names.append(features[f]['name'])
                
                comb_features.append(features[f])
            
            if len(feature_names) > 0:
                feature_combinations.append({'features': comb_features, 'combination': combination})
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