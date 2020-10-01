#src module
from src.data.make_dataset import MakeDataset
from src.data.prepare_data_domain import PrepareDataDomain
from src.pipeline_builder import PipelineBuilder
from src.decorators import *

#sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class RunPipeline():
    '''Runs pipeline for a given data domain, model and features.'''
    
    def __init__(self, train_domain, test_domain, model, features, param_grid):
        '''
        Args:
        train_domain (src.Enum.Domain): Training domain that is used to train model.
        test_domain (src.Enum.Domain):    Test domain that is used to test model.
        model (src.Enum.Model):            Model that is traine as a part of the sklearn pipeline.
        features (list(src.Enum.Feature)): Feature(s) that is used to train model on top of.
        param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings
        
        Example:
            >>> features={ 
                    Feature.SENTIMENT:      { Parameter.Sentiment.SCORE_NAMES: ['neu', 'compound'] },
                    Feature.NGRAM:          { Parameter.Ngram.NGRAM_RANGE: (2,2) },
                    Feature.TYPEDEPENDENCY: { Parameter.Ngram.NGRAM_RANGE: (1,1) }
                }
            >>> param_grid = {
                    'model__C':[10, 100], 
                    'model__penalty':['l1', 'l2']
                }
            >>> RunPipeline(Domain.BHOCS, Domain.BHOCSM, Model.LR, features)
        '''
        self.train_domain=train_domain
        self.test_domain=test_domain
        self.model=model
        self.features=features
        self.param_grid=param_grid
        self.train_modified, self.train_domain_names=self.train_domain['modified'], self.train_domain['dataset']
        self.test_modified,  self.test_domain_names=self.test_domain['modified'], self.test_domain['dataset']
        
    @execution_time_calculator
    def run(self):
        '''Runs the pipeline.
        #Step 1. Read preprocessed data for the given features
        #Step 2. Prepare data splits for training and test domains
        #Step 3. For each split, extract features of training and test set
        #Step 4. For each split, fit and predict with sklearn GridSearchCV
        '''
        #Step 1. Read preprocessed data
        md=MakeDataset()
        data=md.read_preprocessed_data(self.features)
        
        #Step 2. Prepare data splits for training and test domains
        p=PrepareDataDomain()
        data_dict=p.get_split_data(data, self.train_domain_names, self.test_domain_names, self.train_modified, self.test_modified) 
        
        grid_searches=[]
        
        for value in data_dict:
            X_train, y_train=value['X_train'], value['y_train']['sexist'].ravel()
            X_test, y_test=value['X_test'], value['y_test']['sexist'].ravel()
            #print('split train test {} {} {} {} '.format(len(X_train), len(X_test), len(y_train), len(y_test)))  
            
            
            #Step 3. Extract features for training and test set
            pb=PipelineBuilder(self.model, self.features)
            
            pipeline_features=pb.build_pipeline_for_features()
            pipeline_features.fit(X_train, y_train)
            
            X_train=pipeline_features.transform(X_train)
            X_test=pipeline_features.transform(X_test)
            
            
            #Step 4. Fit and predict
            estimator=pb.get_estimator()
            grid_search = GridSearchCV(estimator, self.param_grid, cv=2)  #, verbose=2, n_jobs=-1)
            grid_search.fit(X_train, y_train)
            y=grid_search.predict(X_test)

            #print("accuracy:   %0.3f" % accuracy_score(y_test, y))
            print('f1_score {}'.format(f1_score(y_test, y, average='macro')))
            
            print("grid_search Best score: %0.3f" % grid_search.best_score_)
            print('===============================')
            
            grid_searches.append(grid_search)
            #break
             
        return data_dict, grid_searches