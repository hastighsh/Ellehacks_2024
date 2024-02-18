#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn


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


# date_time, maxtempC, mintempC, totalSnow_cm, sunHour, uvIndex, uvIndex, moon_illumination, moonrise, moonset, sunrise, sunset, DewPointC, FeelsLikeC, HeatIndexC, WindChillC, WindGustKmph, cloudcover, humidity, precipMM, pressure, tempC, visibility, winddirDegree, windspeedKmph


# Combined Daily Weather Report

# In[4]:


#Raw content URL for the dataset
url = "https://raw.githubusercontent.com/hastighsh/Ellehacks_2024/main/datasets/FinalDataSet2.csv"

#Specify the delimiter
delimiter = ','

#Read the data into a DataFrame
daily_weather_df = pd.read_csv(url, delimiter=delimiter)

#Let's create a backup copy of the dataset
daily_weather_df_backup = daily_weather_df.copy()


# In[5]:


daily_weather_df


# In[6]:


header_list = daily_weather_df.columns.tolist()
print(header_list)


# In[7]:


daily_weather_df.drop(columns=['uvIndex', 'moon_illumination', 'moonrise', 'moonset', 'sunrise','sunset', 'DewPointC'], inplace=True)


# In[8]:


header_list = daily_weather_df.columns.tolist()
print(header_list)
daily_weather_df


# Power Outage

# In[9]:


#Raw content URL for adult-all.txt
url = "https://raw.githubusercontent.com/hastighsh/Ellehacks_2024/main/PowerOutageFinal.csv"

#Specify the delimiter (assuming it's a tab-separated file)
delimiter = ','

#Read the data into a DataFrame
power_outages_data = pd.read_csv(url, delimiter=delimiter)

#Let's create a backup copy of the dataset
outage_backup = power_outages_data.copy()


# In[10]:


power_outages_data


# In[11]:


# power_outages_data.drop(["Unnamed: 0", "NERC Region", "Demand Loss (MW)", "Number of Customers Affected"], axis=1)


# In[12]:


daily_weather_df = daily_weather_df.rename(columns={'new_column': 'target'})
daily_weather_df


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


# In[70]:


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


# In[71]:


data_prepared


# In[72]:


# Assuming 'data_prepared' is your feature matrix and 'y' is your target variable
# Replace 'selected_features' with the actual features you want to use for prediction
X = data_prepared[selected_features]
y = y.ravel()

# Split the data into training, validation, and testing sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.6, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.4, random_state=42)

# Print the shapes of the datasets
print(f"Training set: {X_train.shape}, Validation set: {X_val.shape}, Test set: {X_test.shape}")


# In[73]:


from sklearn.ensemble import IsolationForest
from sklearn.svm import OneClassSVM
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split


# In[ ]:





# In[74]:


# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import xgboost as xgb
from sklearn.metrics import accuracy_score, classification_report

# Assuming you have your data loaded into a DataFrame named 'data'
# Split data into features (X) and target variable (y)
# X = data_prepared.drop('target', axis=1)  # Features
# y = data_prepared['target']               # Target variable

# # Split the data into training and testing sets
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=42)

# Feature scaling (optional, but can be beneficial for some models like logistic regression)
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


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[76]:


sample_instance = {
    'maxtempC': 0.684,
    'mintempC': 0.723,
    'totalSnow_cm': 0.215,
    'sunHour': 0.845,
    'FeelsLikeC': 0.537,
    'HeatIndexC': 0.629,
    'WindChillC': 0.732,
    'WindGustKmph': 0.456,
    'cloudcover': 0.321,
    'humidity': 0.478,
    'precipMM': 0.123,
    'pressure': 0.589,
    'tempC': 0.701,
    'visibility': 0.874,
    'winddirDegree': 0.259,
    'windspeedKmph': 0.567
}

# sample_instance = {
#     'pressure': 0.889,
#     'tempC': 0.801,
#     'visibility': 0.674,
#     'winddirDegree': 0.959,
#     'windspeedKmph': 0.167
# }

# sample_instance2 = {
#     'pressure': 0.649123,
#     'tempC': 0.586207,
#     'visibility': 1,
#     'winddirDegree': 0.615616,
#     'windspeedKmph': 0.282609
# }


# In[77]:


# Convert dictionary to numpy array
instance = np.array([[v for k, v in sample_instance.items()]])

# Make a prediction
prediction = log_reg_model.predict(instance)

# Print the prediction
print("Prediction:", prediction)


# In[65]:


# Convert dictionary to numpy array
instance = np.array([[v for k, v in sample_instance2.items()]])

# Make a prediction
prediction = log_reg_model.predict(instance)

# Print the prediction
print("Prediction:", prediction)


# In[47]:


import random

# Define the range for each feature
pressure_range = (0, 1)
tempC_range = (0, 1)
visibility_range = (0, 10)
winddirDegree_range = (0, 360)
windspeedKmph_range = (0, 100)


# In[66]:


import numpy as np

# Number of iterations
num_iterations = 100

for i in range(num_iterations):
    # Generate random values for each feature within the specified range
    sample_instance2 = {
        'pressure': random.uniform(*pressure_range),
        'tempC': random.uniform(*tempC_range),
        'visibility': random.uniform(*visibility_range),
        'winddirDegree': random.uniform(*winddirDegree_range),
        'windspeedKmph': random.uniform(*windspeedKmph_range)
    }
    print(sample_instance2)
    
    # Convert dictionary to numpy array
    instance = np.array([[v for k, v in sample_instance2.items()]])
    
    # Make a prediction
    prediction = log_reg_model.predict(instance)
    
    # Print the prediction
    print("Prediction for iteration", i+1, ":", prediction)


# In[122]:


# Number of iterations
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


# In[ ]:


import pandas as pd

# Sample data (replace with your own data)
data = {'date': ['2024-01-01', '2024-01-02', '2024-01-03'],
        'prediction': [1, 0, 1]}  # 1 indicates outage, 0 indicates no outage
df = pd.DataFrame(data)
df['date'] = pd.to_datetime(df['date'])

# Calculate daily probabilities
prob_outage_per_day = df['prediction'].mean()

# Calculate probability over the entire time range
start_date = df['date'].min()
end_date = df['date'].max()
num_days = (end_date - start_date).days + 1  # Including both start and end dates
overall_probability = prob_outage_per_day * num_days / len(df)

print(f"Overall probability of outage: {overall_probability}")


# In[82]:


import urllib.parse
import os
import pandas as pd


# In[85]:


get_ipython().system('pip install wwo-hist')


# In[86]:


from wwo_hist import retrieve_hist_data

import os


# In[130]:


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
#     instance_normalized = ['maxtempC':0.586207,'mintempC':0.584906,'totalSnow_cm':0.0,'totalSnow_cm':0.600000,'sunHour':0.647059,'FeelsLikeC':0.596491,
#                            'HeatIndexC':0.698413,'WindChillC':0.253731,'WindGustKmph':0.19,'cloudcover':0.604651,'humidity':0.000962,
#                            'precipMM':0.649123,0.586207,1.000000,0.615616,0.282609]
    
    # Make a prediction
    prediction = log_reg_model.predict(instance_normalized)
    
    return prediction
    


# In[131]:


mainRun('Illinois','2/7/2014')


# In[109]:


import random
from datetime import datetime


# In[111]:


def generate_random_date(start_date, end_date):
    # Generating a random date between start_date and end_date
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    end_date = datetime.strptime(end_date, '%m/%d/%Y')
    random_date = start_date + (end_date - start_date) * random.random()
    return random_date.strftime('%m/%d/%Y')


# In[129]:


state = 'Illinois'
date = '30/06/2014'
mainRun(state, date)


# In[121]:


import random
from datetime import datetime

    
def generate_random_date(start_date, end_date):
    # Generating a random date between start_date and end_date
    start_date = datetime.strptime(start_date, '%m/%d/%Y')
    end_date = datetime.strptime(end_date, '%m/%d/%Y')
    random_date = start_date + (end_date - start_date) * random.random()
    return random_date.strftime('%m/%d/%Y')

# Example usage
state = 'Illinois'
for _ in range(50): # Adjust the range as needed
    date = generate_random_date('1/1/2014', '12/31/2014')
    mainRun(state, date)


# In[ ]:




