#Import needed libraries
import numpy as np
import pandas as pd
import pickle
import streamlit as st
import joblib
import prophet
import matplotlib.pyplot as plt
from PIL import Image
#import plotly.graph_objects as go  # For interactive plots

import base64

def add_bg_from_local(image_file):
    with open(image_file, "rb") as img_file:
        encoded_string = base64.b64encode(img_file.read()).decode()

    # Apply CSS for background
    bg_css = f"""
    <style>
    .stApp {{
        background-image: url("data:image/jpg;base64,{encoded_string}");
        background-size: cover;
        background-position: center;
        background-attachment: fixed;
        background: linear-gradient(rgba(0, 0, 0, 0.5), rgba(0, 0, 0, 0.5)));
        color: white;
    }}
    </style>
    """
    st.markdown(bg_css, unsafe_allow_html=True)

# Call function to set background
add_bg_from_local("back.jpg")

# Load the trained Prophet model
model = joblib.load("prophet_model.pkl")


def predict_for_date(year, month):
    # Create a DataFrame for the user-input year and month
    future_date = pd.DataFrame({'ds': [pd.Timestamp(f"{year}-{month:02d}-01")]})

    # Make prediction
    forecast = model.predict(future_date)

    # Extract values
    predicted_value = round(forecast['yhat'].values[0], 2)
    lower_bound = round(forecast['yhat_lower'].values[0], 2)
    upper_bound = round(forecast['yhat_upper'].values[0], 2)

    return predicted_value, lower_bound, upper_bound, forecast


# Streamlit UI
st.title("Monthly Temperature Anomaly Prediction 🌡️")

# User input: Year & Month
future_year = st.number_input("Enter a future year (e.g., 2030)", min_value=2025, max_value=2100, step=1)
future_month = st.selectbox("Select a month",
                            ["January", "February", "March", "April", "May", "June",
                             "July", "August", "September", "October", "November", "December"])

# Convert month name to number
month_dict = {month: i + 1 for i, month in enumerate([
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December"])}
future_month_num = month_dict[future_month]

if st.button("Predict"):
    predicted_value, lower_bound, upper_bound, forecast = predict_for_date(future_year, future_month_num)

    # Display user-friendly message
    st.success(f"🌍 The estimated temperature anomaly for **{future_month} {future_year}** is "
               f"**{predicted_value}°C** (Range: {lower_bound}°C to {upper_bound}°C).")

    # Generate full forecast from 1880 to the selected year
    future_dates = pd.DataFrame({'ds': pd.date_range(start="1880-01-01",
                                                     end=f"{future_year}-{future_month_num:02d}-01",
                                                     freq="MS")})  # MS = Month Start

    full_forecast = model.predict(future_dates)

    # Plot trend
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(full_forecast['ds'], full_forecast['yhat'], label="Predicted Anomaly", color="blue")
    ax.fill_between(full_forecast['ds'], full_forecast['yhat_lower'], full_forecast['yhat_upper'],
                    color="blue", alpha=0.2, label="Confidence Interval")

    # Highlight the user-selected prediction
    user_prediction_date = pd.Timestamp(f"{future_year}-{future_month_num:02d}-01")
    ax.scatter(user_prediction_date, predicted_value, color="red", label="User's Prediction", s=100)

    ax.set_xlabel("Year")
    ax.set_ylabel("Temperature Anomaly (°C)")
    ax.set_title("Temperature Anomaly Trend from 1880 to Prediction Date")
    ax.legend()

    # Show plot in Streamlit
    st.pyplot(fig)
