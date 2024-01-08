#!/usr/bin/env python
# coding: utf-8

# In[21]:


import streamlit as st
import pickle
import numpy as np
import pandas as pd
import joblib


# In[22]:


# Function to load the preprocessor
def load_preprocessor(preprocessor_file_path):
    return joblib.load(preprocessor_file_path)

# Load your preprocessor
preprocessor = load_preprocessor(r'C:\Users\angxi\OneDrive\Documents\Condo Data Project\Machine Learning\preprocessor.joblib')


# In[23]:


# Function to transform the input for prediction
def transform_input(project_name, postal_code, area, year, month):
    # Create a DataFrame from the user inputs
    input_df = pd.DataFrame({
        'Project Name': [project_name],
        'Area (SQFT)': [area],
        'Postal Code': [postal_code],
        'Year': [year],
        'Month': [month]
    })
    # Apply the preprocessor to the input data
    input_transformed = preprocessor.transform(input_df)
    return input_transformed


# In[24]:


pickle_file_path = r'C:\Users\angxi\linear_regression_model 0124(ml).pkl'

def load_model(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        return pickle.load(file)


# In[25]:


# Load your preprocessor and model
preprocessor = load_preprocessor(r'C:\Users\angxi\OneDrive\Documents\Condo Data Project\Machine Learning\preprocessor.joblib')
lr_model = load_model(r'C:\Users\angxi\linear_regression_model 0124(ml).pkl')


# In[26]:


import pandas as pd

# Load the mapping file
file_path = r'C:\Users\angxi\OneDrive\Documents\Condo Data Project\Machine Learning\Streamlit App\project_to_postal_mapping.csv'
project_to_postal_df = pd.read_csv(file_path)

# Display the first few rows to understand its structure
project_to_postal_df.head()


# In[29]:


import streamlit as st
import pickle

# Function to load the model
def load_model(pickle_file_path):
    with open(pickle_file_path, 'rb') as file:
        return pickle.load(file)

# Load your Linear Regression model
lr_model = load_model(r'C:\Users\angxi\linear_regression_model 0124(ml).pkl')

# Streamlit UI components
st.title("Singapore Condo Price Predictor")

# Load the property-postal code mapping DataFrame
property_postal_mapping = pd.read_csv(r'C:\Users\angxi\OneDrive\Documents\Condo Data Project\Machine Learning\Streamlit App\project_to_postal_mapping.csv')

# Convert the 'Postal Code' column back to lists of strings
property_postal_mapping['Postal Code'] = property_postal_mapping['Postal Code'].apply(lambda x: x.strip("[]").replace("'", "").split(", "))

# Property name selector
selected_property = st.selectbox("Select Property", property_postal_mapping['Project Name'].unique())

# Update postal code selector based on selected property
relevant_postal_codes = property_postal_mapping[property_postal_mapping['Project Name'] == selected_property]['Postal Code'].values[0]
selected_postal_code = st.selectbox("Select Postal Code", relevant_postal_codes)

# Input for Area
area = st.number_input("Enter Area (SQFT)", min_value=0)

# Slider for Year
year = st.slider("Select Year", 2024, 2027)

# Dropdown for Month
month = st.selectbox("Select Month", list(range(1, 13)))

# Predict Button and prediction logic
if st.button("Predict"):
    # Get transformed input data
    input_features = transform_input(selected_property, selected_postal_code, area, year, month)
    # Make prediction
    prediction = lr_model.predict(input_features)
    st.write(f"Predicted Price: ${prediction[0]:,.2f}")


# In[ ]:




