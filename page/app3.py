import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(BASE_DIR, "data", "cleaned_dataset.csv")
df = pd.read_csv(csv_path)
st.title(" 🔎 تحلیل داده‌ها (EDA)")

st.markdown("""
<style>
/* کل صفحه RTL و راست‌چین می‌شود */
html, body, [class*="css"] {
    direction: rtl;
    text-align: right;
}
</style>
""", unsafe_allow_html=True)
# Tabs
tab1, tab2, tab3, tab4 ,tab5 = st.tabs(["📊 Overview", "📈 Distribution Plot", "📉 result Score ", "🔥 heatmap" , "📏Technical Details"])
with tab1:
    st.subheader("📌Overview" , divider=True)
    st.markdown("""
        <div style="background-color:#ff9383; padding:18px; border-radius:12px; border:1px solid #b3d1ff;">
        <strong>✨نکته :</strong><br>
        این دیتاست نسخه‌ی تمیز شده و آماده استفاده است.  
        تمام مقادیر گمشده حذف یا اصلاح شده، داده‌های پرت بررسی شده‌اند و مجموعه اکنون برای تحلیل دقیق و مدل‌سازی آماده است.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("نمایش 5 سطر اول:")
    st.dataframe(df.head())

    st.subheader("Shape دیتاست:")
    st.info(f"{df.shape[0]} ردیف و {df.shape[1]} ستون")

    st.subheader("خلاصه آماری:")
    st.dataframe(df.describe())


with tab2:
    st.subheader("📈 Distribution Plot", divider=True)

    feature = st.selectbox("یک ویژگی را انتخاب کنید:", df.columns)

    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)
with tab3:
    st.subheader("📉 نتایج انواع مدل‌های train شده روی دیتا " , divider=True)
    data = {
    'Train Score': [0.878695, 0.949185 ,0.983538 , 0.952829 ,0.942037 ],
    'test Score': [0.838673, 0.818874 , 0.837946 ,0.867382 ,0.863601  ],
    }
    index = np.array(["Ridge","Random Forest","KNeighbors","GradientBoosting","XGBoosting"])
    data = pd.DataFrame(data , index=index)
    st.dataframe(data)
    res_df = pd.DataFrame(data=data,index=index)
    fig , ax = plt.subplots()
    x = np.arange(5)
    ax.plot(x,res_df["Train Score"],marker="o",mfc="red",color="m")
    ax.bar(x,res_df["Train Score"],label="Train")
    ax.plot(x,res_df["test Score"],marker="o",mfc="red",color="green")
    ax.bar(x,res_df["test Score"],label="test")
    ax.set(xticks=x,ylim=[0.7,1],xlabel="Models",ylabel="Accuracy",
        title="Comparing Models")
    ax.set_xticklabels(index,rotation=45)
    ax.legend()

    st.pyplot(fig)

with tab4:
    st.subheader("🔥 matrix heatmap" , divider=True)
    numeric_df = df.select_dtypes(include='number')
    fig, ax = plt.subplots(figsize=(10,6))
    sns.heatmap(numeric_df.corr(), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

with tab5:
    st.subheader("🛠️ Technical Details", divider=True)
    st.markdown("در ادامه جزئیات فنی پردازش داده و انتخاب مدل نهایی آورده شده است.")

    # Style
    st.markdown("""
    <style>
        .big-font {
            font-size:20px !important;
            font-weight:600 !important;
            margin-top:20px;
        }
    </style>
    """, unsafe_allow_html=True)

    # -------------------------
    # 📦 Data Preprocessing Section
    # -------------------------

    with st.expander("📦 Data Preprocessing", expanded=True):

        st.markdown("""
        #### 🔹 **1. Normalization / Scaling**

        برای ویژگی‌های عددی مانند **قیمت** و **متراژ** از اسکیلینگ استفاده شد تا:

        - مدل تحت تاثیر مقیاس متفاوت متراژ و قیمت قرار نگیرد  
        - آموزش مدل سریع‌تر و پایدارتر انجام شود  
        ---

        #### 🔹 **2. One-Hot Encoding برای آدرس**

        چون آدرس یک ویژگی *Categorical* است، با **One-Hot Encoding** تبدیل شد.  
        این روش باعث شد مدل بدون ایجاد ترتیب ساختگی بین محله‌ها، تفاوت ارزش‌گذاری هر محله را یاد بگیرد.

        ---

        #### 🔹 **3. تبدیل ویژگی‌های Boolean به عدد**

        ویژگی‌های True/False به **۰ و ۱** تبدیل شدند تا مدل بتواند از آن‌ها در یادگیری استفاده کند.

        این ویژگی‌ها معمولاً برای امکانات ملک (مانند آسانسور، پارکینگ و …) بسیار مهم هستند.
        """)

    # -------------------------
    # 🤖 Model Selection
    # -------------------------

    with st.expander("🤖 Final Model Selection", expanded=True):

        st.markdown("""
        #### چرا مدل نهایی XGBoost انتخاب شد؟

        ✔️ **Overfit نشد**  
        مدل روی داده‌های آموزش و تست عملکرد نزدیک و باثباتی نشان داد، بنابراین بیش‌برازش اتفاق نیفتاد.

        ✔️ **بهترین نتایج بین مدل‌ها**  
        مدل XGBoost در مقایسه با سایر مدل‌ها کمترین خطا (RMSE) و بیشترین دقت (R²) را داشت.

        ✔️ **پایداری و قدرت تعمیم بالا**  
        عملکرد مدل روی داده‌های جدید قابل اعتماد و پایدار بود.

        به همین دلیل، این مدل به‌عنوان بهترین مدل نهایی پروژه انتخاب شد.

        """)

