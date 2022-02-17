# -*- coding: utf-8 -*-
"""
Created on Tue Feb 15 09:48:27 2022

@author: San
"""
import pandas as pd

def clean_data(df):
    
    # Remove all rows with null
    df = df[~pd.isnull(df).any(axis=1)]

    # Process binary categorical data
    df["gender"] = df["gender"].map({'Female': 0, 'Male': 1})
    df["SeniorCitizen"] = df["SeniorCitizen"].map({"No": 0, "Yes": 1})
    df["Partner"] = df["Partner"].map({"No": 0, "Yes": 1})
    df["Dependents"] = df["Dependents"].map({'No': 0, 'Yes': 1})
    df["PhoneService"] = df["PhoneService"].map({'No': 0, 'Yes': 1})
    df["PaperlessBilling"] = df["PaperlessBilling"].map({'No': 0, 'Yes': 1})
    # df["Default"] = df["Default"].map({'No': 0, 'Yes': 1}) # No need for default since default is not in query

    # Process ordinal categorical data (Some variables are repeated e.g. 'No internet Service')
    # df.loc[:,"Contract"] = df["Contract"].map({'Month-to-month': 0,  'One year': 1,'Two year': 2})
    df.loc[:,"DeviceProtection"] = df["DeviceProtection"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
    # df.loc[:,"InternetService"] = df["InternetService"].map({'Fiber optic': 2, 'DSL': 1, 'No': 0})
    df.loc[:,"MultipleLines"] = df["MultipleLines"].map({'No': 0, 'Yes': 1, 'No phone service': 0})
    df.loc[:,"OnlineBackup"] = df["OnlineBackup"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
    df.loc[:,"OnlineSecurity"] = df["OnlineSecurity"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
    df.loc[:,"PaymentMethod"] = df["PaymentMethod"].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1})
    df.loc[:,"StreamingMovies"] = df["StreamingMovies"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
    df.loc[:,"StreamingTV"] = df["StreamingTV"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
    df.loc[:,"TechSupport"] = df["TechSupport"].map({'No': 0, 'Yes': 1, 'No internet service': 0})

    # Create new onehot encoding columns
    df["Contract-_One year"] = df['Contract'].apply(lambda x: 1 if x == "One year" else 0)
    df["Contract-_Two year"] = df['Contract'].apply(lambda x: 1 if x == "Two year" else 0)
    df["Contract-_Month-to-month"] = df['Contract'].apply(lambda x: 1 if x == "Month-to-month" else 0)
    df["InternetService-_Fiber optic"] = df['InternetService'].apply(lambda x: 1 if x == "Fiber optic" else 0)
    df["InternetService-_No"] = df['InternetService'].apply(lambda x: 1 if x == "No" else 0)
    df["InternetService-_DSL"] = df['InternetService'].apply(lambda x: 1 if x == "DSL" else 0)
    
    # Drop original columns
    df = df.drop(columns=['Contract','InternetService']) # Drop the target variable and customerID

    '''
    # Create function to do one-hot encoding for non-ordinal variables
    
    def one_hot(df,col_name):
        new_cols = pd.get_dummies(df[col_name],prefix=col_name+"-") # Encoded columns
        new_cols = new_cols.astype(int)
        df = df.drop(columns=[col_name])
        df_encoded = pd.concat([df, new_cols], axis=1, join="inner")
        return df_encoded    

    one_hot_list = ["Contract","InternetService"]
    for var in one_hot_list:
        df = one_hot(df, var)
    '''

    # Process "Charges" column
    def convert_num_only(number):
        try:
            output = float(number)
        except ValueError: # If not convertible, output NaN instead
            output = None
        return output

    df.loc[:,"TotalCharges"] = df.loc[:,"TotalCharges"].apply(convert_num_only)
    df.loc[:,"TotalCharges"] = df.loc[:,"TotalCharges"].fillna(df["TotalCharges"].mean()) # Fill null with mean
    
    return df