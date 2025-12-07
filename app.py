import streamlit as st
import matplotlib.pyplot as plt
import numpy as np 
import pandas as pd 
# ---- TITLE SECTION ----
st.markdown("""
<div style="text-align:center; padding:15px 0;">
    <h1 style="color:#4A90E2;">🏡 پروژه پیش‌بینی قیمت خانه</h1>
    <h4 style="color:gray; margin-top:-10px;">(Dataset Overview)</h4>
</div>
""", unsafe_allow_html=True)
st.markdown("""
<style>
/* کل صفحه RTL و راست‌چین می‌شود */
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}
</style>
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

st.header( "📊 معرفی دیتاست" , divider=True)


st.markdown("""
این پروژه از حدود **۳۵۰۰** نمونه آگهی مسکن تشکیل شده که مربوط به **سال ۱۳۹۹** هستند.  
<strong>هر رکورد شامل اطلاعات زیر است:</strong>

- مساحت به **متر مربع** -> Area
- تعداد **اتاق‌خواب** -> Room
- وضعیت **پارکینگ** -> Parking
- وضعیت **آسانسور** -> Elevator
- وضعیت **انباری** -> Warehouse
- **منطقه** ملک -> Address
- **قیمت** -> Price


🔹 **نرخ تبدیل ارز:**  
**۱ دلار = ۳۰٬۰۰۰ تومان**

""", unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True)

st.write("")
st.write("")

# ---- FEATURES IN CARDS ----
st.subheader("🔎 ویژگی‌های موجود در دیتاست")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown('<div class="small-card"><h4>📐 مساحت</h4><p>متر مربع</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-card"><h4>🚗 پارکینگ</h4><p>دارد / ندارد</p></div>', unsafe_allow_html=True)

with col2:
    st.markdown('<div class="small-card"><h4>🛏 اتاق خواب</h4><p>تعداد</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-card"><h4>📦 انباری</h4><p>دارد / ندارد</p></div>', unsafe_allow_html=True)

with col3:
    st.markdown('<div class="small-card"><h4>🛗 آسانسور</h4><p>دارد / ندارد</p></div>', unsafe_allow_html=True)
    st.markdown('<div class="small-card"><h4>📍 منطقه</h4><p>محله / Zone</p></div>', unsafe_allow_html=True)


st.write("")
st.write("")

# ---- MODEL INFORMATION ----
st.markdown("""
<div class="big-card">
    <h3>🤖 مدل استفاده‌شده</h3>
    <p>
    این پروژه برای پیش‌بینی قیمت خانه از مدل قدرتمند 
    <strong>XGBoost Regressor</strong> استفاده می‌کند.
    <br>
    مدل بر اساس ویژگی‌های موجود آموزش دیده و قیمت ملک را تخمین می‌زند.
    <br>
    </p>
    <strong>بهترین پارامتر های پیدا شده برای مدل xgboost عبارتند از:</strong>
    <ul>
        <li><strong>learning_rate:</strong> 0.5</li>
        <li><strong>max_depth:</strong> 3</li>
        <li><strong>n_estimators:</strong> 200</li>
    </ul>
    <strong>بهترین نتایج مدل XGBRegressor:</strong>
    <ul>   
        <li>ضریب تعیین (R²) در مجموعه train : <strong>94.20٪</strong></li>
        <li>ضریب تعیین (R²) در مجموعه test : <strong>86.36٪</strong></li>
        <li>ریشه میانگین مربعات خطا (RMSE): <strong>1,074,807,672.45</strong></li>
    </ul>   
</div>
""", unsafe_allow_html=True)
st.markdown("""
<div class="big-card">
    <strong>🤖 پارامتر های تست شده برای مدل xgboost عبارتند از:</strong>
    <ul>
        <li><strong>learning_rate:</strong> [0.01, 0.1, 0.5,0.2]</li>
        <li><strong>max_depth:</strong> [3, 5, 7,9]</li>
        <li><strong>n_estimators:</strong> [50, 100, 200,250,300]</li>
    </ul>
</div>
""", unsafe_allow_html=True)
            
st.write("")
st.write("")
st.subheader("📚 منبع داده‌ها")

st.write("این داده‌ها از وب‌ سایت **Kaggle** جمع‌آوری شده‌اند. برای مشاهده دیتاست اصلی می‌توانید به لینک زیر مراجعه کنید:"
"https://www.kaggle.com/datasets/mokar2001/house-price-tehran-iran")