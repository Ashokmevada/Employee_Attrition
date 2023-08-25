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
