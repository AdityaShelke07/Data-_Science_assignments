# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:30:25 2023

@author: excel
"""

import streamlit as st
import pandas as pd
import pickle

# Load the trained model and encoder
model = pickle.load(open('model.pkl', 'rb'))
encoder = pickle.load(open('encoder.pkl', 'rb'))  # Load the encoder

# Define the app title
st.title('Titanic Survival Prediction')

# Create input fields for user input
age = st.number_input('Age', min_value=0, max_value=100, value=30)
sibsp = st.number_input('Number of Siblings/Spouses', min_value=0, max_value=10, value=0)
parch = st.number_input('Number of Parents/Children', min_value=0, max_value=10, value=0)
fare = st.number_input('Fare', min_value=0.0, max_value=500.0, value=50.0)
sex = st.selectbox('Sex', ['male', 'female'])
embarked = st.selectbox('Embarked', ['S', 'C', 'Q'])

# Create a function to preprocess the input data

def preprocess_data(age, sibsp, parch, fare, sex, embarked):
    input_data = {'Age': [age], 'SibSp': [sibsp], 'Parch': [parch], 'Fare': [fare], 'Sex': [sex], 'Embarked': [embarked], 'Pclass': [3]} 
    input_df = pd.DataFrame(input_data)

    encoded_features = encoder.transform(input_df[['Sex', 'Embarked']])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(['Sex', 'Embarked']))

    preprocessed_data = pd.concat([input_df[['Age', 'SibSp', 'Parch', 'Fare', 'Pclass']], encoded_df], axis=1)
    
    # Reorder columns to match the original order
    original_feature_order = [ 'Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'Sex_female',
       'Sex_male', 'Embarked_C', 'Embarked_Q', 'Embarked_S']  # Replace with your actual feature order
    preprocessed_data = preprocessed_data[original_feature_order] 
    
    return preprocessed_data

# Create a button to trigger predictions
if st.button('Predict'):
    # Preprocess the input data
    preprocessed_data = preprocess_data(age, sibsp, parch, fare, sex, embarked)
    
    # Make prediction
    prediction = model.predict(preprocessed_data)[0]
    
    # Display the prediction
    if prediction == 1:
        st.success('The passenger is predicted to have survived.')
    else:
        st.error('The passenger is predicted to have not survived.')