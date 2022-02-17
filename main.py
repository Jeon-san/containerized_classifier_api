# -*- coding: utf-8 -*-
"""
Created on Mon Feb 14 20:03:16 2022

@author: San
"""

import uvicorn
from fastapi import FastAPI
import pickle
from clean_data import clean_data # Import clean_data function from clean_data.py
import pandas as pd
from enum import Enum

app = FastAPI()

# load the pickle model 
model_rf = pickle.load(open("classifier_model.sav", 'rb'))

# Create classes to validate input data

class AvailableContract(str, Enum):
    MonthToMonth = "Month-to-month"
    TwoYear = "Two year"
    OneYear = "One year"
    
class AvailableDependents(str, Enum):
    Yes = "Yes"
    No = "No"
    
class AvailableDeviceProtection(str, Enum):
    Yes = "Yes"
    No = "No"
    NoInternet = "No internet service"
    
class Availablegender(str, Enum):
    Male = "Male"
    Female = "Female"
    
class AvailableInternetService(str, Enum):
    FiberOptic = "Fiber optic"
    DSL = "DSL"
    No = "No"
    
class AvailableMultipleLines(str, Enum):
    Yes = "Yes"
    No = "No"
    NoInternet = "No phone service"

class AvailableOnlineBackup(str, Enum):
    Yes = "Yes"
    No = "No"
    NoInternet = "No internet service"

class AvailableOnlineSecurity(str, Enum):
    Yes = "Yes"
    No = "No"
    NoInternet = "No internet service"

class AvailablePaperlessBilling(str, Enum):
    Yes = "Yes"
    No = "No"

class AvailablePartner(str, Enum):
    Yes = "Yes"
    No = "No"

class AvailablePaymentMethod(str, Enum):
    ElectronicCheck = "Electronic check"
    MailedCheck = "Mailed check"
    BankTransfer = "Bank transfer (automatic)"
    CreditCard = "Credit card (automatic)"

class AvailablePhoneService(str, Enum):
    Yes = "Yes"
    No = "No"

class AvailableSeniorCitizen(str, Enum):
    Yes = "Yes"
    No = "No"
    
class AvailableStreamingMovies(str, Enum):
    Yes = "Yes"
    No = "No"
    NoInternet = "No internet service"
    
class AvailableStreamingTV(str, Enum):
    Yes = "Yes"
    No = "No"
    NoInternet = "No internet service"
    
class AvailableTechSupport(str, Enum):
    Yes = "Yes"
    No = "No"
    NoInternet = "No internet service"

   
    
@app.get("/predict/")
async def read_item(contract: AvailableContract, 
                    dependents: AvailableDependents,
                    deviceprotection: AvailableDeviceProtection,
                    gender: Availablegender,
                    internetservice: AvailableInternetService,
                    multiplelines: AvailableMultipleLines,
                    onlinebackup: AvailableOnlineBackup,
                    onlinesecurity: AvailableOnlineSecurity,
                    paperlessbilling: AvailablePaperlessBilling,
                    partner: AvailablePartner,
                    paymentmethod: AvailablePaymentMethod,
                    phoneservice: AvailablePhoneService,
                    seniorcitizen: AvailableSeniorCitizen,
                    streamingmovies: AvailableStreamingMovies,
                    streamingtv: AvailableStreamingTV,
                    techsupport: AvailableTechSupport,
                    tenure: int,
                    monthlycharges: float,
                    totalcharges: float):
    
        input_data = {'customerID': ["placeholder"],
                      'gender': [gender],
                      'SeniorCitizen': [seniorcitizen],
                      'Partner': [partner],
                      'Dependents': [dependents],
                      'tenure': [tenure],
                      'PhoneService': [phoneservice],
                      'MultipleLines': [multiplelines],
                      'InternetService': [internetservice],
                      'OnlineSecurity': [onlinesecurity],
                      'OnlineBackup': [onlinebackup],
                      'DeviceProtection': [deviceprotection],
                      'TechSupport': [techsupport],
                      'StreamingTV': [streamingtv],
                      'StreamingMovies':[streamingmovies],
                      'Contract': [contract],
                      'PaperlessBilling': [paperlessbilling],
                      'PaymentMethod': [paymentmethod],
                      'MonthlyCharges': [monthlycharges],
                      'TotalCharges': [totalcharges],
                      }
        
        df = pd.DataFrame(data=input_data) # Put variables into a dataframe
        cleaned_df = clean_data(df) # Clean/process the data
        cleaned_df = cleaned_df.drop(columns=['customerID']) # Drop the target variable and customerID
        ### Predict the output
        
        result = model_rf.predict_proba(cleaned_df)
        #no_def_proba = result[0,0]
        def_proba = result[0,1]        
        
        return {"default probability":def_proba}


if __name__ == "__main__":
    uvicorn.run("hello_world_fastapi:app")
