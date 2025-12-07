import streamlit as st
import numpy as np
import pandas as pd
import joblib

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

st.markdown("""
<style>
/* کل صفحه RTL و راست‌چین می‌شود */
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)
# -----------------------------
# Page Config
# -----------------------------
st.set_page_config(page_title="House Price Predictor", page_icon="🏡", layout="centered")

# -----------------------------
# Title
# -----------------------------
st.title("🏡 **پیش‌بینی قیمت خانه**")
st.write("با وارد کردن مشخصات ملک، قیمت آن را بر اساس مدل XGBoost پیش‌بینی کنید.")

# -----------------------------
# Extract address columns
# -----------------------------
base_cols = ['Area', 'Room', 'Parking', 'Warehouse', 'Elevator', "Price"]
address_cols = [col for col in model_columns if col not in base_cols]
address_map = {name: name for name in address_cols}

# -----------------------------
# User Input Section
# -----------------------------
st.header("📋 اطلاعات ملک")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input("📏 متراژ (متر مربع)", min_value=30, max_value=200, step=1)
    parking = st.selectbox("🚗 پارکینگ", ["ندارد", "دارد"])
    warehouse = st.selectbox("📦 انباری", ["ندارد", "دارد"])

with col2:
    rooms = st.number_input("🛏 تعداد اتاق", min_value=0, max_value=5, step=1)
    elevator = st.selectbox("⬆️ آسانسور", ["ندارد", "دارد"])
    address = st.selectbox("📍 منطقه", address_cols)

# تبدیل امکانات به عدد
parking_val = 1 if parking == "دارد" else 0
elevator_val = 1 if elevator == "دارد" else 0
warehouse_val = 1 if warehouse == "دارد" else 0

# -----------------------------
# Prediction Section
# -----------------------------
st.subheader("📊 نتیجه پیش‌بینی")

if st.button("🔍 پیش‌بینی قیمت"):
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
    input_for_model = scaler.transform(input_df)  # خروجی ndarray
    prediction = model.predict(input_for_model)[0]
    usd_pred = prediction / 30000
    st.success(f"💰 **قیمت پیش‌ بینی‌ شده: {prediction:,.0f} تومان**")
    st.success(f"💰 **با دلار 30,000 تومان قیمت پیش‌ بینی‌ شده: {usd_pred:,.0f} دلار**")



