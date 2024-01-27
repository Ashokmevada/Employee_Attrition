from flask import Flask , request , render_template
import pandas as pd
import numpy as np

from src.pipeline.predict_pipeline import PredictPipeline , CustomData


application = Flask(__name__)
app = application

@app.route('/')
def index():

    return render_template('index.html')

@app.route('/predict' , methods=['GET' , 'POST'])
def predict_attrition():

    if request.method == 'GET':
        return render_template('home.html')
    
    else :

        data = CustomData(
            Age=int(request.form.get('Age')),
            BusinessTravel=request.form.get('BusinessTravel'),
            DailyRate=int(request.form.get('DailyRate')),
            Department=request.form.get('Department'),
            DistanceFromHome=int(request.form.get('DistanceFromHome')),
            EducationField=request.form.get('EducationField'),
            EnvironmentSatisfaction=int(request.form.get('EnvironmentSatisfaction')),
            Gender=request.form.get('Gender'),
            HourlyRate=int(request.form.get('HourlyRate')),
            JobSatisfaction=int(request.form.get('JobSatisfaction')),
            MaritalStatus=request.form.get('MaritalStatus'),
            MonthlyIncome=int(request.form.get('MonthlyIncome')),
            NumCompaniesWorked=int(request.form.get('NumCompaniesWorked')),
            OverTime=request.form.get('OverTime'),
            PercentSalaryHike=int(request.form.get('PercentSalaryHike')),
            TotalWorkingYears=int(request.form.get('TotalWorkingYears')),
            YearsAtCompany=int(request.form.get('YearsAtCompany')),
            YearsSinceLastPromotion=int(request.form.get('YearsSinceLastPromotion'))
            )
        
        pred_df = data.get_data_as_dataframe()

        predict_pipeline = PredictPipeline()

        result = predict_pipeline.predict(pred_df)

        return render_template('home.html' , result=result)
    
        
    

if __name__ == '__main__':

    app.run(host= '0.0.0.0')




