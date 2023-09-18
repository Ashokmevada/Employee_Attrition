import os
import sys
from dataclasses import dataclass
from sklearn.linear_model import LogisticRegression 
from src.exception import CustomException
from src.logger import logging
from sklearn.model_selection import GridSearchCV , StratifiedKFold
from src.utils import save_object , evaluteModel
from sklearn.metrics import accuracy_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier , AdaBoostClassifier , GradientBoostingClassifier
# from xgboost import XGBClassifier
from sklearn.neighbors import KNeighborsClassifier
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


            models = {
                "Logistic Regression" : LogisticRegression(),
                "Decision Tree" : DecisionTreeClassifier(),
                "SVC" : SVC(),
                "Guassian Naive Bayes" : GaussianNB(),
                "RandomForestClassifier" : RandomForestClassifier(),
                "Adaboost Classifier" : AdaBoostClassifier(),
                "GradientClassifier" : GradientBoostingClassifier(),
                # "xgboost" : XGBClassifier(),
                "KNN" : KNeighborsClassifier()
            }


            hyperparameters = {
                "Logistic Regression": {
                    "penalty": ["l1", "l2"],
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "class_weight" : ['balanced'],
                    "solver" : ['lbfgs', 'liblinear', 'newton-cg' , 'newton-cholesky' , 'sag' , 'saga' ]
                }
                ,
                "Decision Tree": {
                    "criterion": ["gini", "entropy"],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10]
                },
                "SVC": {
                    "C": [0.01, 0.1, 1.0, 10.0],
                    "kernel": ["linear", "poly", "rbf", "sigmoid"],
                    "gamma" : ["scaled" , "auto"],
                    
                    "class_weight" : ['balanced']
                },
                "Guassian Naive Bayes": {},
                "RandomForestClassifier": {
                    "n_estimators": [10, 50, 100],
                    "max_depth": [None, 10, 20, 30],
                    "min_samples_split": [2, 5, 10],
                    "criterion": ["gini", "entropy"]
                },
                "Adaboost Classifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 1.0]
                },
                "GradientClassifier": {
                    "n_estimators": [50, 100, 200],
                    "learning_rate": [0.01, 0.1, 0.2],
                    "max_depth": [3, 4, 5]
                },
                # "xgboost": {
                #     "n_estimators": [50, 100, 200],
                #     "learning_rate": [0.01, 0.1, 0.2],
                #     "max_depth": [3, 4, 5]
                # },
                "KNN": {
                    "n_neighbors": [3, 5, 7, 9],
                    "weights": ["uniform", "distance"],
                    "algorithm": ["auto", "ball_tree", "kd_tree", "brute"]
                }
            }           

            logging.info("evaluate model started")

            model_performance:dict = evaluteModel(X_train , X_test , y_train , y_test , models , hyperparameters)
          
            best_model_score = max(sorted(model_performance.values()))

            best_model_name = list(model_performance.keys())[
                list(model_performance.values()).index(best_model_score)
            ]

            best_model = models[best_model_name]

            if best_model_score < 0.6:
                raise CustomException("No best model found")            

            print( best_model_name , best_model_score)

            save_object(
                
                self.model_config.model_path ,
                best_model
            )


            return accuracy_score(best_model.predict(X_test) , y_test)

        except Exception as e:
            raise CustomException(e, sys)