import streamlit as st
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
# ---- TITLE SECTION ----
st.markdown("""
<div style="text-align:center; padding:15px 0;">
    <h1 style="color:#4A90E2;">🏡 predict House Price Tehran Iran Project</h1>
    <h4 style="color:gray; margin-top:-10px;">(Dataset Overview)</h4>
</div>
""", unsafe_allow_html=True)

# ---- CARD STYLE ----
st.markdown("""
<style>
.big-card {
    background-color: #f9f9f9;
    padding: 20px 25px;
    border-radius: 15px;
    border: 1px solid #e6e6e6;
    box-shadow: 0 2px 6px rgba(0,0,0,0.05);
}
.small-card {
    background-color: white;
    padding: 15px;
    border-radius: 12px;
    border: 1px solid #eeeeee;
    text-align:center;
    box-shadow: 0 1px 4px rgba(0,0,0,0.05);
}
</style>
""", unsafe_allow_html=True)


# ---- MAIN DATASET CARD ----
st.markdown('<div class="big-card">', unsafe_allow_html=True)

st.header( "📊 Dataset Introduction" , divider=True)


st.markdown("""
This project consists of approximately 3500 housing advertisement samples from the year 1399 (Persian calendar).

<strong>Each record includes the following information:</strong>

- Area in *square meters** -> Area
- Number of **bedrooms** -> Room
- status **Parking** -> Parking
- status **Elevator** -> Elevator
- status **Warehouse** -> Warehouse
- **region** Property -> Address
- **Price** -> Price

🔹 **Currency exchange rate:**
**1 dollar = 30,000 Tomans**

""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# ---- FEATURES IN CARDS ----
st.subheader("🔎 Features in the dataset")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="small-card"><h4>📐 Area</h4><p>in square meters</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-card"><h4>🚗 Parking</h4><p>true / false </p></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="small-card"><h4>🛏 Room</h4><p>Number</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-card"><h4>📦 Warehouse</h4><p>true / false</p></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="small-card"><h4>🛗 Elevator</h4><p>true / false</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-card"><h4>📍 Area</h4><p>region / Zone</p></div>', unsafe_allow_html=True)


st.write("")
st.write("")

# ---- MODEL INFORMATION ----
st.markdown("""
<div class="big-card">
    <h3>🤖 Model Used</h3>
    <p>
    This project employs the powerful XGBoost Regressor model for house price prediction.
    <strong>XGBoost Regressor</strong>
    <br>
    The model has been trained based on the available features to estimate property prices.
    <br>
    </p>
    <strong>The best parameters found for the XGBoost model are:</strong>
    <ul>
        <li><strong>learning_rate:</strong> 0.5</li>
        <li><strong>max_depth:</strong> 3</li>
        <li><strong>n_estimators:</strong> 200</li>
    </ul>
    <strong>The best results for the XGBRegressor model are:</strong>
    <ul>   
        <li>Coefficient of Determination (R²) on the train set: <strong>94.20٪</strong></li>
        <li>Coefficient of Determination (R²) on the test set: <strong>86.36٪</strong></li>
        <li>Root Mean Squared Error (RMSE): <strong>1,074,807,672.45</strong></li>
    </ul>   
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class="big-card">
    <strong>🤖 The parameters tested for the xgboost model are:</strong>
    <ul>
        <li><strong>learning_rate:</strong> [0.01, 0.1, 0.5,0.2]</li>
        <li><strong>max_depth:</strong> [3, 5, 7,9]</li>
        <li><strong>n_estimators:</strong> [50, 100, 200,250,300]</li>
    </ul>
</div>
""", unsafe_allow_html=True)
            
st.write("")
st.write("")
st.subheader("📚 Data source")

st.write("This data was collected from the **Kaggle** website. To view the original dataset, you can visit the following link:"

"https://www.kaggle.com/datasets/mokar2001/house-price-tehran-iran")

