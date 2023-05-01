#Import necessary libraries
import pickle
import os
import joblib
from django.apps import AppConfig
from django.conf import settings
from rest_framework.decorators import api_view
from rest_framework.response import Response
from django.shortcuts import render
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
labelencoder = LabelEncoder()
# Create your views here.
@api_view(['GET'])
def index_page(request):
    return_data = {
        "error" : "0",
        "message" : "Successful",
    }
    return Response(return_data)

@api_view(["POST"])
def predict_type(request):
    try:
        HEALTHCARE_COVERAGE= request.data.get('HEALTHCARE_COVERAGE',None)
        HEALTHCARE_EXPENSES = request.data.get('HEALTHCARE_EXPENSES',None)
        age = request.data.get('age',None)
        CITY = request.data.get('CITY',None)
        Life_Span = request.data.get('Life_Span',None)
        COUNTY = request.data.get('COUNTY',None)
        GENDER = request.data.get('GENDER',None)
        RACE = request.data.get('RACE',None)
        jok=1
        fields = np.array([[HEALTHCARE_COVERAGE, HEALTHCARE_EXPENSES, age, CITY, Life_Span, COUNTY, GENDER, RACE]])
        if not None in fields:
            #Datapreprocessing Convert the values to float
            jok=0
            df0 = pd.DataFrame(fields, columns = ['HEALTHCARE_COVERAGE','HEALTHCARE_EXPENSES','age','CITY','Life Span','COUNTY','GENDER','RACE'])
            #print(df0)
            df0 = df0.apply(LabelEncoder().fit_transform)
            #Passing data to model & loading the model from disks
            model_path = 'ml_model/nb.joblib'
            classifier = joblib.load(open(model_path, 'rb'))
            #prediction=1
            prediction = classifier.predict(df0)
            #conf_score =  np.max(classifier.predict_proba([df0]))*100
            predictions = {
                'error' : '0',
                'message' : 'Successfull1',
                'prediction' : prediction,
                #'confidence_score' : conf_score
            }
        else:
            predictions = {
                'error' : '1',
                'message': 'Invalid Parametes',   
                'jok' : fields             
            }
    except Exception as e:
        predictions = {
            'error' : '2',
            "message": str(e)
        }
    
    return Response(predictions)