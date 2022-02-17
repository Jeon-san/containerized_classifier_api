# -*- coding: utf-8 -*-
"""
Created on Sun Feb 13 14:25:13 2022

@author: San
"""

# Import Libraries
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import f1_score
import matplotlib.pyplot as plt
from sklearn.metrics import plot_confusion_matrix,classification_report
import pickle

# Get CSV
url = 'https://drive.google.com/file/d/1TXwkqMin6UphS-qjnxXbHCbs9vU6hQvH/view?usp=sharing'
url='https://drive.google.com/uc?id=' + url.split('/')[-2] # Change to download link

df = pd.read_csv(url,index_col= False ) # Read csv into a pandas dataframe

### Data Cleaning ###

# Count null values
null_count = df.isnull().sum(axis = 0)
null_rows = df[pd.isnull(df).any(axis=1)] # Return all rows with null

# Seems like 4 rows are completely empty and one has only the gender field
# Therefore, drop all these rows

df = df[~pd.isnull(df).any(axis=1)]
null_count = df.isnull().sum(axis = 0) # Check there is no null values now

# Explore unique values of columns

# Look at target variable distribution
target_dist = df.groupby("Default").size() # About 5.1k no default, 1.9k default

# Function to get unique values of columns with categorical data
cat_columns = list(df.columns)
cat_columns = [i for i in cat_columns if i not in ["customerID","MonthlyCharges","tenure","TotalCharges"]]
count_dic = {} # Initiatlize dictionary
for col in cat_columns:
    count_dic[col] = df[col].value_counts().to_dict() # Record down unique values & count

# Process categorical data
'''
def encode_col(df): # Encodes a pandas df column *without order*
    for col in df.columns:
        column = df[col]   
        uni_array = column.unique()
        mapping_dict = {}
        for i,var in enumerate(uni_array):
            mapping_dict[var] = i
        df[col] = column.map(mapping_dict)
    return df
'''

cat_columns_multi = ["Contract","DeviceProtection","InternetService","MultipleLines","OnlineBackup","OnlineSecurity","PaymentMethod","StreamingMovies","StreamingTV","TechSupport"]
cat_columns_bina = [i for i in cat_columns if i not in cat_columns_multi]

'''
df[cat_columns_bina] = encode_col(df[cat_columns_bina]) # Process binary categorical data
'''
# Process binary categorical data
df["gender"] = df["gender"].map({'Female': 0, 'Male': 1})
df["SeniorCitizen"] = df["SeniorCitizen"].map({0.0: 0, 1.0: 1})
df["Partner"] = df["Partner"].map({'No': 0, 'Yes': 1})
df["Dependents"] = df["Dependents"].map({'No': 0, 'Yes': 1})
df["PhoneService"] = df["PhoneService"].map({'No': 0, 'Yes': 1})
df["PaperlessBilling"] = df["PaperlessBilling"].map({'No': 0, 'Yes': 1})
df["Default"] = df["Default"].map({'No': 0, 'Yes': 1})

# Process ordinal categorical data (Some variables are repeated e.g. 'No internet Service')
#df["Contract"] = df["Contract"].map({'Month-to-month': 0,  'One year': 1,'Two year': 2})
df["DeviceProtection"] = df["DeviceProtection"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
#df["InternetService"] = df["InternetService"].map({'Fiber optic': 2, 'DSL': 1, 'No': 0})
df["MultipleLines"] = df["MultipleLines"].map({'No': 0, 'Yes': 1, 'No phone service': 0})
df["OnlineBackup"] = df["OnlineBackup"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
df["OnlineSecurity"] = df["OnlineSecurity"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
df["PaymentMethod"] = df["PaymentMethod"].map({'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 1, 'Credit card (automatic)': 1})
df["StreamingMovies"] = df["StreamingMovies"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
df["StreamingTV"] = df["StreamingTV"].map({'No': 0, 'Yes': 1, 'No internet service': 0})
df["TechSupport"] = df["TechSupport"].map({'No': 0, 'Yes': 1, 'No internet service': 0})


# Create new onehot encoding columns
df["Contract-_One year"] = df['Contract'].apply(lambda x: 1 if x == "One year" else 0)
df["Contract-_Two year"] = df['Contract'].apply(lambda x: 1 if x == "Two year" else 0)
df["Contract-_Month-to-month"] = df['Contract'].apply(lambda x: 1 if x == "Month-to-month" else 0)
df["InternetService-_Fiber optic"] = df['InternetService'].apply(lambda x: 1 if x == "Fiber optic" else 0)
df["InternetService-_No"] = df['InternetService'].apply(lambda x: 1 if x == "No" else 0)
df["InternetService-_DSL"] = df['InternetService'].apply(lambda x: 1 if x == "DSL" else 0)

# Drop original columns
df = df.drop(columns=['Contract','InternetService']) # Drop the target variable and customerID

# Process "Charges" column
def convert_num_only(number):
    try:
        output = float(number)
    except ValueError: # If not convertible, output NaN instead
        output = None
    return output

df_copy = df.copy(deep=True)
df["TotalCharges"] = df["TotalCharges"].apply(convert_num_only)
unconverted_rows = df_copy[df["TotalCharges"].isnull()] # Seems like these rows have " " for Total Charges
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].mean()) # Fill null with mean

# Seems like most of them have 2/3 unique values, perhaps it is better to use a decision tree
# based model - e.g. Random Forest

# Split into train & test
X = df.drop(columns=['Default','customerID'])
y = df['Default']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=1)

# Create the model with 100 trees
model_rf = RandomForestClassifier(n_estimators=100, 
                               class_weight="balanced") # Class weight set to balanced as output is not ~50:50
# Fit on training data
model_rf.fit(X_train, y_train)

# Function to evaluate model performance
def evaluate(model, X, y):
    y_pred = model.predict(X)
    accuracy = accuracy_score(y, y_pred)*100
    f1 = f1_score(y, y_pred)*100
    print('Accuracy = {:0.2f}%.'.format(accuracy))
    print('F1 score = {:0.2f}%.'.format(f1))
    
    return accuracy

print("----------------Pre Tuning-----------------\n")
print('Train Model Performance')
train_accuracy = evaluate(model_rf,X_train,y_train)
print('Test Model Performance')
test_accuracy = evaluate(model_rf,X_test,y_test)
print("\n----------------Pre Tuning-----------------")

## Tune Hyperparameters using CVgridsearch
'''
# Number of trees in random forest
n_estimators = [i for i in range(100,300,20)]
# Number of features to consider at every split
max_features = ['auto','sqrt']
# Maximum number of levels in tree
max_depth = [i for i in range(4,20,1)]
# Minimum number of samples required to split a node
min_samples_split = [i for i in range(100,1000,20)]
# Minimum number of samples required at each leaf node
min_samples_leaf = [i for i in range(100,600,20)]
# Method of selecting samples for training each tree
bootstrap = [True,False]
'''

# Tuned parameters
# Number of trees in random forest
n_estimators = [200]
# Number of features to consider at every split
max_features = ['sqrt']
# Maximum number of levels in tree
max_depth = [10]
# Minimum number of samples required to split a node
min_samples_split = [180]
# Minimum number of samples required at each leaf node
min_samples_leaf = [100]
# Method of selecting samples for training each tree
bootstrap = [True]

param_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}
print(param_grid)

# Create the grid model
grid_rf = RandomizedSearchCV(estimator = model_rf, param_distributions = param_grid, n_iter = 100, cv = 7, verbose=0, random_state=42, n_jobs = -1)
grid_rf.fit(X_train, y_train)
best_grid = grid_rf.best_estimator_

print("----------------Post Tuning-----------------\n")
print('Train Model Performance')
grid_accuracy = evaluate(best_grid,X_train,y_train)
print('Test Model Performance')
grid_accuracy = evaluate(best_grid,X_test,y_test)
print("\n")
#print(grid_rf.best_params_) #Print parameters of best model
print("\n----------------Post Tuning-----------------")




y_pred = best_grid.predict(X_test)
plot_confusion_matrix(best_grid, X_test, y_test)  
plt.show()

print("\n\n**************Model performance Summary**************\n\n")
print(classification_report(y_test, y_pred))
print("\n\n**************Model performance Summary**************")

if __name__ == "__main__":
    ## Export the model using pickle
    filename = 'classifier_model.sav'
    pickle.dump(best_grid, open(filename, 'wb'))
    
















