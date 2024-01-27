FROM python:3.11-slim

WORKDIR /employee_attrition

RUN python3 -m pip install --upgrade pip

COPY requirements.txt requirements.txt
RUN pip install -r requirements.txt

COPY . .

CMD ["python" , "application.py"]