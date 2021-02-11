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
nltk.download('stopwords')

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
        self.all_domains=self.params.dict['all_domains']
        self.feature_combination=self.params.dict['feature_combination']
        self.grid_search_cnn=True
        self.models=self.get_models()
        self.use_grid_search=self.params.dict['use_grid_search']
        self.results_df = pd.DataFrame(columns=['iteration','train_domain', 'test_domain', 'model_name', 'features', 
                                                'feature_dimensions',
                                                'param_grid', 'best_params',
                                                'y_test', 'y_pred', 'y_test_ids'])
        
    
    @execution_time_calculator
    def run(self):
        '''Runs the steps below.
        Step 1. Prepare parameters using "params.json" and "hyperparams.json" files.
        Step 2. Read data
        Step 3. Prepare train and test splits for seven domains.
        Step 4. Build Pipeline
        Step 5. Fit
        Step 6. Predict
        Step 7. Get and save classification report
        '''
        
        # Step 1. Prepare parameters using "params.json" and "hyperparams.json" files.
        self.prepare_parameters()
        
        # Step 2. Read data
        data = self.make_dataset.read_data(self.data_file)
        
        for i in self.iteration:
            
            # Step 3. Prepare train and test splits for seven domains.
            splits=self.make_dataset.prepare_data_splits(data, random_state=i)
            
            for td in self.train_domains:
                train_domain_name=td['name']
                train_domain=td['value']
                
                X_train, y_train= self.make_dataset.get_data_split(train_domain_name, splits, train=True, random_state=i)
                        
                for model_name, features_set in self.models.items():
                    train_num=0
                    
                    for f in features_set:
                        train_num=train_num+1
                        
                        features, param_grid=f['features'], f['param_grid']
                        
                        print()
                        print('{}.{}/{} Running the pipeline for: {}, {}, {}'.format(
                            i, train_num, len(features_set), model_name, [k['comb_name'] for k in features], train_domain_name))
                        print()
                        
                        # Step 4. Build Pipeline
                        pipeline=PipelineBuilder(features, model_name).build_pipeline()
                       
                        gs=pipeline
                        
                        if model_name == Model.CNN and self.grid_search_cnn == False:
                            #grid_search_cnn is set to False while training across datasets
                            gs.set_params(**param_grid)
                        else:
                            sf=StratifiedKFold(n_splits=5, random_state=0, shuffle=True)
                            gs=GridSearchCV(pipeline, param_grid=param_grid, cv=sf, scoring='f1_macro', n_jobs=-1)
                        
                        # Step 5. Fit
                        gs.fit(X_train, y_train)
                        
                        for test_domain in self.test_domains:
                            test_domain_name=test_domain['name']
                            
                            X_test, y_test=self.make_dataset.get_data_split(test_domain_name, splits, test=True, random_state=i)
                            y_test_ids=X_test.copy().reset_index()
                            
                            # Step 6. Predict
                            y_pred=gs.predict(X_test)
                            
                            feature_dimensions = self.get_feature_dimensions(gs.best_estimator_, model_name)
                            
                            self.results_df=self.results_df.append({
                                                  'iteration':'.'.join((str(i), str(train_num))),
                                                  'train_domain':train_domain_name, 'test_domain':test_domain_name, 
                                                  'model_name':model_name, 'features':features,
                                                  'feature_dimensions':feature_dimensions,
                                                  'param_grid':param_grid, 
                                                  'best_params': gs.best_params_ if self.use_grid_search else param_grid, 
                                                  'y_test':y_test, 'y_pred':y_pred, 
                                                  'y_test_ids':y_test_ids }, 
                                                  ignore_index=True)
                            
                            
            # Step 7. Get and save classification report  
            metrics_df=self.results_df.apply(lambda r: self.get_classification_report(r['y_test'], r['y_pred']), axis=1)
        
            results_df=pd.concat((self.results_df, metrics_df), axis=1)
        
            file_name=''.join(('experiments/results_', FLAGS.file_name, '_')) if FLAGS.file_name != None else 'experiments/results_'
            timestr = time.strftime("%Y%m%d-%H%M%S")
            file_name=''.join((file_name, timestr, '_', str(i), '.pkl'))
        
            with open(file_name, 'wb+') as f:
                pickle.dump(results_df, f)
            
        return 'FINISHED'
    
    def get_models(self):
        if self.all_domains:
            #Research Question 2
            return self.get_models_rq2()
        else:
            #Research Question 1
            return self.get_models_rq1()
    
    def get_models_rq2(self):
        retVal={}
        
        params_features=self.params.dict['features']
        models=self.params.dict['models']
        for k, v in models.items():
            model_name=v['name'][0]
            if model_name in [Model.GENDERWORD, Model.THRESHOLDCLASSIFIER]:
                retVal[model_name]=[
                    {'features': [{'name':model_name, 'comb_name':model_name, 'feature_selection':False}],
                     'param_grid':self.hyperparams.dict[model_name]}]
            elif model_name in [Model.LR, Model.SVM]:
                best_features=v['best_features']
                bf={}
                for f in best_features:
                    bf[f]={
                    'features':{
                        'name':params_features[f]['name'], 
                        'feature_selection':params_features[f]['feature_selection']}, 
                    'param_grid':params_features[f]['param_grid']
                    }
                
                retVal[model_name]=self.get_feature_combinations(bf, model_name, comb_max=len(bf), comb_min=len(bf)-1)
            else:
                self.grid_search_cnn = False
                
                best_features=v['best_features']
                bf={}
                for f in best_features:
                    bf[f]={
                    'features':{
                        'name':params_features[f]['name'], 
                        'feature_selection':params_features[f]['feature_selection']}, 
                    'param_grid':params_features[f]['param_grid']
                    }
                
                retVal[model_name]=self.get_cnn_features()
        
        return retVal
    
    def get_models_rq1(self):
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
                features=[
                    {'features':
                     [{'name':name, 'comb_name':name,
                       'feature_selection':False}],
                     'param_grid':self.hyperparams.dict[name]}]
                
            models[name]=features
            
        return models
    
    def get_cnn_features(self):
        params_features=self.params.dict['features']
        valid_features=[Feature.BERTWORD]
        
        features=[]
        for k, v in params_features.items():
            name=v['name']
            if name in valid_features:
                comb_name='_'.join((k, name))
                                   
                param_grid_feature=self.format_param_grid(comb_name, v['param_grid'])
                param_grid_model=self.hyperparams.dict[Model.CNN]
                
                features.append({
                    'features':[{
                        'name':name, 'comb_name':comb_name,
                        'feature_selection':v['feature_selection'] }],
                    'param_grid':{**param_grid_feature, **param_grid_model}
                })
        
        return features
    
    def get_svm_and_logit_features(self, model):
        valid_features=[Feature.SENTIMENT, Feature.NGRAM, Feature.TYPEDEPENDENCY, Feature.BERTDOC]
        features=self.get_features(valid_features)
        
        if self.feature_combination:
            #Gets combination features
            return self.get_feature_combinations(features, model, comb_max=len(features), comb_min=0)
        else:
            #Gets individual features for gridsearch over k
            return self.get_feature_combinations(features, model, comb_max=1, comb_min=0)
    
    def get_features(self, valid_features):
        params_features=self.params.dict['features']
        
        features={}
        for k, v in params_features.items():
            name=v['name']
            if name in valid_features:
                features[k]={
                    'features':{'name':name, 'feature_selection':v['feature_selection']}, 
                    'param_grid':v['param_grid']
                }
         
        return features
    
    def get_feature_combinations(self, features, model, comb_max, comb_min=0):
        param_grid_model=self.hyperparams.dict[model]
        
        #1.Create index combinations
        comb_list=[]
        for i in range(comb_min, comb_max):
            comb_list.extend(list(itertools.combinations(list(features.keys()), (i+1))))
        
        #2.Create feature combination list by using index combinations
        feature_combinations=[]
        for combination in comb_list:
            comb_features, feature_names=[], []
            param_grid_features={}
    
            for f in combination:
                #Prevent adding same features in a combination (e.g. ngram (1,1) and ngram (1,4))
                if features[f]['features']['name'] in feature_names:
                    feature_names=[]
                    comb_feature_names=[]
                    break
                
                feature_names.append(features[f]['features']['name'])
                features[f]['features']['comb_name']='_'.join((str(f), features[f]['features']['name']))
                
                comb_features.append(features[f]['features'])
                param_grid_feature=self.format_param_grid(features[f]['features']['comb_name'], features[f]['param_grid'])
                param_grid_features.update(param_grid_feature)
            if len(feature_names) > 0:
                feature_combinations.append(
                    {'features': comb_features, 'param_grid': {**param_grid_features, **param_grid_model}})
        
        return feature_combinations
    
    def format_param_grid(self, comb_name, param_grid):
        param_grid_feature={}
        for key, val in param_grid.items():
            name='__'.join(('features', comb_name, key))
            param_grid_feature[name]=val
        return param_grid_feature
    
    def get_feature_dimensions(self, best_estimator, model_name):
        feature_dimensions={}
        
        total=best_estimator.steps[-1][1].feature_dimension
        feature_dimensions['total']=total
        
        if model_name in [Model.SVM, Model.LR]:
            tl=best_estimator.steps[0][1].transformer_list
            for p in tl:
                steps=p[1].steps
                #Includes feature selection
                if len(steps) == 4:
                    feature_dimensions[p[0][2:]]={
                    'dimension':steps[3][1].feature_dimension,
                    'reduced':steps[3][1].k
                }
                elif len(steps) == 3:
                    feature_dimensions[p[0][2:]]={
                    'dimension':steps[2][1].feature_dimension
                }
                
        return feature_dimensions
        
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