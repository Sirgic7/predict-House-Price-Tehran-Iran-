import streamlit as st
import numpy as np
import pandas as pd
import joblib
import os

# -----------------------------
# Load model & columns
# -----------------------------
BASE_DIR = os.path.dirname(os.path.dirname(__file__))  
model_path = os.path.join(BASE_DIR, "model", "final_xgboost_model.pkl")
scaler_path = os.path.join(BASE_DIR, "model", "scaler.pkl")
model_columns_path = os.path.join(BASE_DIR, "model", "model_columns.pkl")

model = joblib.load(model_path)
scaler = joblib.load(scaler_path)
model_columns = joblib.load(model_columns_path)

# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

# -----------------------------
# Title
# -----------------------------
st.title("🏡 **predict House Price Tehran Iran**")
st.write("Predict the price of a property based on the XGBoost model by entering its specifications.")

# -----------------------------
# Extract address columns
# -----------------------------
base_cols = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', "Price"]
address_cols = [col for col in model_columns if col not in base_cols]
address_map = {name: name for name in address_cols}

# -----------------------------
# User Input Section
# -----------------------------
st.header("📋Property information")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("📏 Area (Square meter)", min_value=30, max_value=200, step=1)
    parking = st.selectbox("🚗 parking", ["no", "Yes"])
    warehouse = st.selectbox("📦 Warehouse",  ["no", "Yes"])

with col2:
    rooms = st.number_input("🛏 Number of rooms", min_value=0, max_value=5, step=1)
    elevator = st.selectbox("⬆️ Elevator",["no", "Yes"])
    address = st.selectbox("📍 Area", address_cols)

# تبدیل امکانات به عدد
parking_val = 1 if parking == "Yes" else 0
elevator_val = 1 if elevator == "Yes" else 0
warehouse_val = 1 if warehouse == "Yes" else 0

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("📊Prediction result")

if st.button("🔍Price Prediction"):
    # 1) Remove Price column
    model_columns = [c for c in model_columns if c != "Price"]
    # 2) Prepare input DF
    input_df = pd.DataFrame(0, index=[0], columns=model_columns)

    # 3) Fill values
    input_df.at[0, 'Area'] = area
    input_df.at[0, 'Room'] = rooms
    input_df.at[0, 'Parking'] = parking_val
    input_df.at[0, 'Warehouse'] = warehouse_val
    input_df.at[0, 'Elevator'] = elevator_val

    # 4) Address one-hot
    input_df.at[0, address] = 1

    # 5) Reorder columns
    input_df = input_df[model_columns]

    # 6) Predict
    input_for_model = scaler.transform(input_df) 
    prediction = model.predict(input_for_model)[0]
    usd_pred = prediction / 30000
    st.success(f"💰 **Predicted Price: {prediction:,.0f} Toman**")
    st.success(f"💰 **With an exchange rate of 30,000 Tomans per dollar, the predicted price is: {usd_pred:,.0f} $**")





