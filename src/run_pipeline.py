#src module
from src.data.make_dataset import MakeDataset
from src.pipeline_builder import PipelineBuilder
from src.decorators import *

#sklearn
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score

class RunPipeline():
    '''Runs pipeline for a given data domain, model and features.'''
    
    def __init__(self, train_domain, test_domain, model, features, param_grid, iteration_count=1):
        '''
        Args:
        train_domain (src.Enum.Domain): Training domain that is used to train model.
        test_domain (src.Enum.Domain):    Test domain that is used to test model.
        model (src.Enum.Model):            Model that is traine as a part of the sklearn pipeline.
        features (list(src.Enum.Feature)): Feature(s) that is used to train model on top of.
        param_grid (dict): Dictionary with parameters names (str) as keys and lists of parameter settings
        
        Example:
            >>> features=[ Feature.NGRAM, Feature.SENTIMENT, Feature.TYPEDEPENDENCY]
            >>> param_grid = {
                    'model__C':[10, 100], 
                    'model__penalty':['l1', 'l2']
                }
            >>> RunPipeline(Domain.BHOCS, Domain.BHOCSM, Model.LR, features, param_grid, 5)
        '''
        self.train_domain=train_domain
        self.test_domain=test_domain
        self.model=model
        self.features=features
        self.param_grid=param_grid
        self.iteration_count=iteration_count
        
    @execution_time_calculator
    def run(self):
        '''Runs the steps below.
        Step 1. Read data
        Step 2. Split data
        Step 3. Build Pipeline
        Step 4. Fit and predict with sklearn GridSearchCV
        Step 5. Save labels(true and predicted) and model parameters  (TODO)
        '''
        # Step 1. Read data
        make_dataset = MakeDataset()
        data = make_dataset.read_data('../../data/raw/all_data_augmented.csv')
        
        for i in range(self.iteration_count):
        
            # Step 2. Split data
            X_train, y_train, X_test, y_test=make_dataset.get_balanced_data_split(data, self.train_domain, self.test_domain)
            
            # Step 3. Build Pipeline
            pb=PipelineBuilder(self.model, self.features)
            pipeline=pb.build_pipeline()
           
            # Step 4. Fit and predict
            grid_search = GridSearchCV(pipeline, self.param_grid, cv=2, scoring='f1_macro')
            grid_search.fit(X_train, y_train)
            y=grid_search.predict(X_test)
            
            # TODO : Macro average f1 calculation
            print('f1_score {}'.format(f1_score(y_test, y, average='macro')))
            
            # Step 5. Save labels(true and predicted) and model parameters
            # TODO