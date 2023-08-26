import sys
import os
import pandas as pd
from src.exception import CustomException
from src.logger import logging
from src.utils import load_object

class PredictPipeline:
    def __init__(self):
        pass

    def predict(self , features):

        try:

            model_path = os.path.join('artifacts' , "model.pkl")
            preprocessor_path = os.path.join('artifacts' , 'preprocessor.pkl')
            label_encoder_dict_path = os.path.join('artifacts' , 'label_encoders.pkl')
            model = load_object(model_path)
            preprocessor = load_object(preprocessor_path)
            label_encoder_dict = load_object(label_encoder_dict_path)
            cat_columns = features.select_dtypes(include = 'object')
            logging.info(f"dict {label_encoder_dict}")            
            for column in cat_columns:

                labelEncoder = label_encoder_dict[column]
                features[column] = labelEncoder.transform(features[column])

            data_transform = preprocessor.transform(features)
            pred = model.predict_proba(data_transform)[0][1]
            percentage = pred.round(2)*100

            return percentage
            



        except Exception as e:
            raise CustomException(e , sys)

class CustomData:

    def __init__(self , Age,
                BusinessTravel : str,
                DailyRate,
                Department :str,
                DistanceFromHome,
                EducationField :str,
                EnvironmentSatisfaction,
                Gender :str,
                HourlyRate,
                JobSatisfaction,
                MaritalStatus : str,
                MonthlyIncome,
                NumCompaniesWorked,
                OverTime : str,
                PercentSalaryHike,
                TotalWorkingYears,
                YearsAtCompany,
                YearsSinceLastPromotion):
        
        self.Age = Age
        self.BusinessTravel = BusinessTravel
        self.DailyRate = DailyRate
        self.Department = Department
        self.DistanceFromHome = DistanceFromHome
        self.EducationField = EducationField
        self.EnvironmentSatisfaction = EnvironmentSatisfaction
        self.Gender = Gender
        self.HourlyRate = HourlyRate
        self.JobSatisfaction = JobSatisfaction
        self.MaritalStatus = MaritalStatus
        self.MonthlyIncome = MonthlyIncome
        self.NumCompaniesWorked = NumCompaniesWorked
        self.OverTime = OverTime
        self.PercentSalaryHike = PercentSalaryHike
        self.TotalWorkingYears = TotalWorkingYears
        self.YearsAtCompany = YearsAtCompany
        self.YearsSinceLastPromotion = YearsSinceLastPromotion

    def get_data_as_dataframe(self):

        try:

            custom_data_input_dict = {
                "Age": [self.Age],
                "BusinessTravel": [self.BusinessTravel],
                "DailyRate": [self.DailyRate],
                "Department": [self.Department],
                "DistanceFromHome": [self.DistanceFromHome],
                "EducationField": [self.EducationField],
                "EnvironmentSatisfaction": [self.EnvironmentSatisfaction],
                "Gender": [self.Gender],
                "HourlyRate": [self.HourlyRate],
                "JobSatisfaction": [self.JobSatisfaction],
                "MaritalStatus": [self.MaritalStatus],
                "MonthlyIncome": [self.MonthlyIncome],
                "NumCompaniesWorked": [self.NumCompaniesWorked],
                "OverTime": [self.OverTime],
                "PercentSalaryHike": [self.PercentSalaryHike],
                "TotalWorkingYears": [self.TotalWorkingYears],
                "YearsAtCompany": [self.YearsAtCompany],
                "YearsSinceLastPromotion": [self.YearsSinceLastPromotion]
                }

            return pd.DataFrame(custom_data_input_dict)

        except Exception as e:

            raise CustomException(e,sys)
        







