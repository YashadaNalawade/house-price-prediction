# app.py
import streamlit as st
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from load_data import get_data

# Load the dataset
X, y = get_data()

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Linear Regression model
model = LinearRegression()

# Train the model on the training data
model.fit(X_train, y_train)

# Define a function to predict house price given specific features
def predict_house_price(features):
    features = np.array(features).reshape(1, -1)
    price = model.predict(features)[0]
    return price

# Streamlit UI
st.title("California House Price Prediction")

st.write("Enter the features of the house:")

# Input fields for the features
MedInc = st.number_input("Median Income (MedInc)", min_value=0.0, max_value=20.0, value=3.0)
HouseAge = st.number_input("House Age (HouseAge)", min_value=0, max_value=100, value=25)
AveRooms = st.number_input("Average Number of Rooms (AveRooms)", min_value=1.0, max_value=20.0, value=5.0)
AveBedrms = st.number_input("Average Number of Bedrooms (AveBedrms)", min_value=1.0, max_value=10.0, value=2.0)
Population = st.number_input("Population (Population)", min_value=1, max_value=10000, value=1000)
AveOccup = st.number_input("Average House Occupancy (AveOccup)", min_value=1.0, max_value=10.0, value=3.0)
Latitude = st.number_input("Latitude (Latitude)", min_value=32.0, max_value=42.0, value=34.0)
Longitude = st.number_input("Longitude (Longitude)", min_value=-125.0, max_value=-114.0, value=-120.0)

# Create a button for prediction
if st.button("Predict House Price"):
    features = [MedInc, HouseAge, AveRooms, AveBedrms, Population, AveOccup, Latitude, Longitude]
    predicted_price = predict_house_price(features)
    st.write(f"The predicted house price is: ${predicted_price * 100000:.2f}")

# Show evaluation metrics
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

