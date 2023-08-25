import sys
from dataclasses import dataclass
from typing import Any
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler , LabelEncoder
from src.exception import CustomException
from src.logger import logging
import os
from src.utils import save_object


@dataclass
class DataTransformerConfig:
    preprocessor_obj_file_path = os.path.join('artifacts' , "preprocessor.pkl")

class DataTransformation:
    
    def __init__(self):
        self.data_transformation_config = DataTransformerConfig()

    
        
    def initiate_data_transformation(self , train_path , test_path):

        try:

            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logging.info("Read train and test data set Successfully")

            columns_to_be_removed = ['Education' , 'EmployeeCount' , 'EmployeeNumber' , 'JobInvolvement' , 'MonthlyRate' , 'Over18' , 'PerformanceRating' , 'RelationshipSatisfaction' , 'WorkLifeBalance' , 'TrainingTimesLastYear' , 'StockOptionLevel' , 'StandardHours']

            train_df.drop(columns= columns_to_be_removed , inplace=True , axis = 1)
            test_df.drop(columns= columns_to_be_removed , inplace=True , axis = 1) 

            outlier_column = ['Age' , 'JobLevel' , 'MonthlyIncome' , 'NumCompaniesWorked' , 'PercentSalaryHike' , 'TotalWorkingYears', 'YearsAtCompany' , 'YearsInCurrentRole' , 'YearsSinceLastPromotion' , 'YearsWithCurrManager']

            logging.info("Removing Outliers for train set")
            for column in outlier_column:

                q1 = train_df[column].quantile(0.25)
                q3 = train_df[column].quantile(0.75)
                IQR = q3 - q1

                lower_bound = q1  - 1.5 * IQR
                upper_bound = q3 + 1.5 * IQR

                train_df = train_df[(train_df[column]>=lower_bound) & (train_df[column]<=upper_bound)]

            logging.info("Removing Outliers for test set")
            for column in outlier_column:

                q1 = test_df[column].quantile(0.25)
                q3 = test_df[column].quantile(0.75)
                IQR = q3 - q1

                lower_bound = q1  - 1.5 * IQR
                upper_bound = q3 + 1.5 * IQR

                test_df = test_df[(test_df[column]>=lower_bound) & (test_df[column]<=upper_bound)]


            logging.info("Outliers Removed Successfully ")

            #feature Selection
            
            not_feature = ['JobRole' , 'JobLevel' , 'YearsInCurrentRole' , 'YearsWithCurrManager']

            train_df.drop(not_feature , axis=1 , inplace=True)
            test_df.drop(not_feature , axis=1 , inplace=True)

            logging.info("Feature Selection Added Successfully")

            scaler = StandardScaler()
            

            category_columns= list(train_df.select_dtypes(include = 'object').columns)        

            label_dict = {}
            for column in category_columns:
                labelencoder = LabelEncoder() 
                train_df[column] = labelencoder.fit_transform(train_df[column])
                test_df[column] = labelencoder.transform(test_df[column])
                if column != "Attrition":
                    label_dict[column] = labelencoder

            logging.info(f"{label_dict}")
                         
            save_object(
                
                os.path.join('artifacts' , "label_encoders.pkl"),
                label_dict
            )
            
            target_column = "Attrition"
            X_train = train_df.drop(target_column, axis=1)
            y_train = train_df[target_column]


            X_test = test_df.drop(target_column, axis=1)
            y_test = test_df[target_column]    

           

            preprocessor = ColumnTransformer(
                transformers=[
                    ("scaler" , scaler , X_train.columns)
                ]
            )

            

            train_arr = preprocessor.fit_transform(X_train)
            test_arr = preprocessor.transform(X_test)

            train_arr = np.c_[

                train_arr , np.array(y_train)
            ]

            test_arr = np.c_[
                test_arr , np.array(y_test)
            ]


            logging.info("Preprocessing completed")

            save_object(

                file_path = self.data_transformation_config.preprocessor_obj_file_path, 
                object = preprocessor

            )


            return (
                train_arr, test_arr , self.data_transformation_config.preprocessor_obj_file_path
            )
        
        except Exception as e:
            raise CustomException(e,sys)



