#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
import random
from datetime import datetime
import urllib.parse
import os
from wwo_hist import retrieve_hist_data
import os


# In[2]:


from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
import seaborn as sns
from sklearn.metrics import classification_report, ConfusionMatrixDisplay


# In[3]:





# In[4]:


import warnings

# Turn off warnings
warnings.filterwarnings("ignore")


# In[5]:


# date_time, maxtempC, mintempC, totalSnow_cm, sunHour, uvIndex, uvIndex, moon_illumination, moonrise, moonset, sunrise, sunset, DewPointC, FeelsLikeC, HeatIndexC, WindChillC, WindGustKmph, cloudcover, humidity, precipMM, pressure, tempC, visibility, winddirDegree, windspeedKmph


# Combined Daily Weather Report

# In[6]:


#Raw content URL for the dataset
url = "https://raw.githubusercontent.com/hastighsh/Ellehacks_2024/main/datasets/FinalDataSet2.csv"

#Specify the delimiter
delimiter = ','

#Read the data into a DataFrame
daily_weather_df = pd.read_csv(url, delimiter=delimiter)

#Let's create a backup copy of the dataset
daily_weather_df_backup = daily_weather_df.copy()


# In[7]:


daily_weather_df


# In[8]:


header_list = daily_weather_df.columns.tolist()
#print(header_list)


# In[9]:


daily_weather_df.drop(columns=['uvIndex', 'moon_illumination', 'moonrise', 'moonset', 'sunrise','sunset', 'DewPointC'], inplace=True)


# In[10]:


header_list = daily_weather_df.columns.tolist()
#print(header_list)
#daily_weather_df


# Power Outage

# In[ ]:





# In[ ]:





# In[11]:


# power_outages_data.drop(["Unnamed: 0", "NERC Region", "Demand Loss (MW)", "Number of Customers Affected"], axis=1)


# In[12]:


daily_weather_df = daily_weather_df.rename(columns={'new_column': 'target'})
#daily_weather_df


# In[13]:


# tempd = daily_weather_df.drop(columns=["date_time","location"])
# labels = tempd['target']
# features = tempd.drop('target', axis=1)

# # Create a heatmap
# plt.figure(figsize=(24, 18))
# sns.heatmap(features, annot=True, cmap='coolwarm', fmt='.2f')
# plt.title('Heatmap of Features')
# plt.show()


# In[14]:


tempd = daily_weather_df.drop(columns=["date_time","location"])


# In[15]:


tempd


# In[16]:


df_normalized = (tempd - tempd.min()) / (tempd.max() - tempd.min())


# In[17]:


df_normalized = df_normalized.sample(frac=1).reset_index(drop=True)


# In[18]:


df_normalized


# In[19]:


features = df_normalized.columns[:-1]  # Assuming last column is the label
average_labels = []

for feature in features:
    if pd.api.types.is_numeric_dtype(df_normalized[feature]):
        average_label = df_normalized.groupby('target')[feature].mean()
        average_labels.append(average_label)
    else:
        average_labels.append(None)

plt.figure(figsize=(12, 8))
for i, feature in enumerate(features):
    if average_labels[i] is not None:
        plt.bar(feature, average_labels[i])
    else:
        print(f"Skipping non-numeric feature: {feature}")

plt.xlabel('Features')
plt.ylabel('Average Label')
plt.title('Average Label Value for Each Feature')
plt.xticks(rotation=45)
plt.show()


# In[20]:


# importing the libraries
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.impute import SimpleImputer


# In[21]:


# Define the numeric columns
X = df_normalized.drop('target',axis=1)
y = df_normalized['target']
num_cols = X.select_dtypes(include='number').columns.to_list()


# In[22]:


# Create pipelines for numeric columns
num_pipeline = make_pipeline(SimpleImputer(strategy='mean'), StandardScaler())


# In[23]:


#Preprocessing for numerical data
numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('scaler', StandardScaler())
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, num_cols)],
#         ('cat', categorical_transformer, cat_cols)],
         remainder='passthrough'
)


# In[24]:


# Create and apply the preprocessing pipeline
data_prepared = preprocessor.fit_transform(X)
feature_names = preprocessor.get_feature_names_out()
data_prepared = pd.DataFrame(data=data_prepared, columns=feature_names)


# In[25]:


from sklearn.feature_selection import RFE
from sklearn.ensemble import RandomForestClassifier

#inital the estimator as RandomForestClassifier
y = y.ravel()
estimator = RandomForestClassifier(n_estimators=100, random_state=42)
#use RFE to choose the attribute
selector = RFE(estimator, n_features_to_select=16, step=1)
selector = selector.fit(data_prepared, y)
#get attributes' name
selected_features = data_prepared.columns[selector.support_]


# In[26]:


data_prepared


# In[27]:


# Assuming 'data_prepared' is your feature matrix and 'y' is your target variable
# Replace 'selected_features' with the actual features you want to use for prediction
X = data_prepared[selected_features]
y = y.ravel()

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

# Print the shapes of the datasets
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")


# In[28]:


from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
# Import necessary libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report


# In[29]:


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Logistic Regression
log_reg_model = LogisticRegression()
log_reg_model.fit(X_train_scaled, y_train)

# Predictions
log_reg_predictions = log_reg_model.predict(X_test_scaled)

# Model evaluation
log_reg_accuracy = accuracy_score(y_test, log_reg_predictions)
print("Logistic Regression Accuracy:", log_reg_accuracy)
print("Classification Report:")
print(classification_report(y_test, log_reg_predictions))

# Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Predictions
rf_predictions = rf_model.predict(X_test)

# Model evaluation
rf_accuracy = accuracy_score(y_test, rf_predictions)
print("\nRandom Forest Accuracy:", rf_accuracy)
print("Classification Report:")
print(classification_report(y_test, rf_predictions))

# Gradient Boosting (XGBoost)
xgb_model = xgb.XGBClassifier(objective='binary:logistic', random_state=42)
xgb_model.fit(X_train, y_train)

# Predictions
xgb_predictions = xgb_model.predict(X_test)

# Model evaluation
xgb_accuracy = accuracy_score(y_test, xgb_predictions)
print("\nXGBoost Accuracy:", xgb_accuracy)
print("Classification Report:")
print(classification_report(y_test, xgb_predictions))


# In[30]:


import random

# Define the range for each feature
pressure_range = (0, 1)
tempC_range = (0, 1)
visibility_range = (0, 10)
winddirDegree_range = (0, 360)
windspeedKmph_range = (0, 100)


# Random Test
num_iterations = 100

for i in range(num_iterations):
    # Generate random values for each feature within the specified range
    sample_instance = {
        'maxtempC': random.uniform(0, 1),
        'mintempC': random.uniform(0, 1),
        'totalSnow_cm': random.uniform(0, 1),
        'sunHour': random.uniform(0, 1),
        'FeelsLikeC': random.uniform(0, 1),
        'HeatIndexC': random.uniform(0, 1),
        'WindChillC': random.uniform(0, 1),
        'WindGustKmph': random.uniform(0, 1),
        'cloudcover': random.uniform(0, 1),
        'humidity': random.uniform(0, 1),
        'precipMM': random.uniform(0, 1),
        'pressure': random.uniform(0, 1),
        'tempC': random.uniform(0, 1),
        'visibility': random.uniform(0, 1),
        'winddirDegree': random.uniform(0, 1),
        'windspeedKmph': random.uniform(0, 1)
    }
    
    # Convert dictionary to numpy array
    instance = np.array([[v for k, v in sample_instance.items()]])
    
    # Make a prediction
    prediction = log_reg_model.predict(instance)
    
    # Print the prediction
    print("Prediction for iteration", i+1, ":", prediction)


# In[31]:


def mainRun(city,date):
    api_key = '54278ea2bf104965bef84824241802'
    location = city
    encoded_location = urllib.parse.quote(location)
    start_date = date
    end_date = date # Use a different date for the end date
    frequency = 24

    response = retrieve_hist_data(api_key=api_key,
                                   location_list=[encoded_location],
                                   start_date=start_date,
                                   end_date=end_date,
                                   frequency=frequency)
    
    df = pd.read_csv(city + '.csv')

    # Convert the DataFrame into an array of dictionaries

    for index, row in df.iterrows():
        sample_instance = {
            'maxtempC': row['maxtempC'],
            'mintempC': row['mintempC'],
            'totalSnow_cm': row['totalSnow_cm'],
            'sunHour': row['sunHour'],
            'FeelsLikeC': row['FeelsLikeC'],
            'HeatIndexC': row['HeatIndexC'],
            'WindChillC': row['WindChillC'],
            'WindGustKmph': row['WindGustKmph'],
            'cloudcover': row['cloudcover'],
            'humidity': row['humidity'],
            'precipMM': row['precipMM'],
            'pressure': row['pressure'],
            'tempC': row['tempC'],
            'visibility': row['visibility'],
            'winddirDegree': row['winddirDegree'],
            'windspeedKmph': row['windspeedKmph']
        }
        
    
    # Convert dictionary to numpy array
    instance = np.array([[v for k, v in sample_instance.items()]])
    instance_normalized = (instance - instance.min()) / (instance.max() - instance.min())
    print(instance_normalized)
    
    # Make a prediction
    prediction = log_reg_model.predict(instance_normalized)
    
    return prediction
    


# In[32]:


# mainRun('Illinois','2/7/2014')


# In[33]:


##to use a period of time and get fractions

# from datetime import datetime, timedelta

# def mainRun1(state, sdate, edate):
#     # Convert string dates to datetime objects
#     sdate = datetime.strptime(sdate, "%m/%d/%Y")
#     edate = datetime.strptime(edate, "%m/%d/%Y")
    
#     # Initialize variables to keep track of sum and count
#     sum_values = 0
#     count_dates = 0
    
#     # Iterate over the range of dates
#     current_date = sdate
#     while current_date <= edate:
#         # Call mainRun function for each date
#         result = mainRun(state, current_date)
        
#         # Update sum and count
#         sum_values += result
#         count_dates += 1
        
#         # Move to the next date
#         current_date += timedelta(days=1)
    
#     # Return the fraction as a tuple (numerator, denominator)
#     return sum_values, count_dates


# In[ ]:





# In[ ]:




