import os
import pickle
import sys
from src.exception import CustomException
from src.logger import logging
import numpy as np
import pandas as pd
from sklearn.model_selection import GridSearchCV , StratifiedKFold
from sklearn.metrics import accuracy_score


def save_object(file_path , object):

    try:

        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path , exist_ok=True)

        with open(file_path , 'wb') as f:
            pickle.dump(object , f)

    except Exception as e:
        raise CustomException(e , sys)
    

def load_object(file_path):

    try:
        
        with open(file_path , 'rb') as f:
            return pickle.load(f)


    except Exception as e:
        raise CustomException(e , sys)

def evaluteModel(X_train , X_test , y_train, y_test , models , params):

    try:
        
        report = {}

        for i in range(len(list(models))):
            model = list(models.values())[i]
            parameteres = params[list(models.keys())[i]]

            gs = GridSearchCV(model , param_grid= parameteres , cv=3)
            gs.fit(X_train , y_train)

            model.set_params(**gs.best_params_)
            model.fit(X_train , y_train)

            y_train_pred = model.predict(X_train)

            y_test_pred = model.predict(X_test)

            train_model_score = accuracy_score(y_train_pred , y_train)

            test_model_score = accuracy_score(y_test_pred , y_test)

            report[list(models.keys())[i]] = test_model_score


        return report
    
    except Exception as e:

        raise CustomException(e , sys)
