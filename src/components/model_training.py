import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression 
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV , StratifiedKFold
from src.utils import save_object
from sklearn.metrics import accuracy_score
import warnings
warnings.filterwarnings("ignore")



@dataclass
class ModelConfig:
    
    model_path = os.path.join('artifacts' , 'model.pkl')

class ModelTraining:

    def __init__(self):
        self.model_config = ModelConfig()


    def initiate_model_training(self , train_arr , test_arr):

        try:
            
            logging.info('Initializing model training')
            
            X_train , y_train , X_test , y_test = (
                train_arr[:,:-1],
                train_arr[:,-1],
                test_arr[:,:-1],
                test_arr[:,-1]

            )

            param = {
            'penalty' : [ 'l1', 'l2' , 'elasticnet' , None],
            'C' : [1.0 , 0.5 , 0.9 , 0.8 , 0.3],
            'class_weight' : ['balanced'],
            'solver' : ['lbfgs', 'liblinear', 'newton-cg' , 'newton-cholesky' , 'sag' , 'saga' ] ,
            'max_iter' : [100 , 500 , 50 , 300  ,1000],
            'l1_ratio' : [0 , 0.3 ,  0.5 , 0.9]             
            }


            logistic_model = LogisticRegression()


            logging.info("GridSearchCV evaluate model started")
            gs = GridSearchCV(logistic_model, param_grid= param , cv = StratifiedKFold(n_splits=5 , shuffle=True , random_state=42))    

            gs.fit(X_train , y_train)

            logistic_model.set_params(**gs.best_params_)
            logistic_model.fit(X_train , y_train)

            y_test_pred = logistic_model.predict(X_test)

            #accuracy = accuracy_score(y_test_pred , y_train)

            #logging.info(f"accuracy_score {accuracy}")

            save_object(
                
                self.model_config.model_path ,
                logistic_model
            )

        except Exception as e:
            raise CustomException(e, sys)