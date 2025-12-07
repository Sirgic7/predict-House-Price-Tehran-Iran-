import streamlit as st
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np 
import os

BASE_DIR = os.path.dirname(os.path.dirname(__file__))
csv_path = os.path.join(BASE_DIR, "data", "cleaned_dataset.csv")
df = pd.read_csv(csv_path)
st.title(" 🔎Data Analysis (EDA)")

# Tabs
tab1, tab2, tab3, tab4 ,tab5 = st.tabs(["📊 Overview", "📈 Distribution Plot", "📉 result Score ", "🔥 heatmap" , "📏Technical Details"])
with tab1:
    st.subheader("📌Overview" , divider=True)
    st.markdown("""
        <div style="background-color:#ff9383; padding:18px; border-radius:12px; border:1px solid #b3d1ff;">
        <strong>✨ Note:</strong><br>
        This dataset is the cleaned and ready-to-use version. All missing values have been removed or corrected, outliers have been examined, and the dataset is now prepared for precise analysis and modeling.
        </div>
        """, unsafe_allow_html=True)

    st.subheader("Showing the first 5 rows:")
    st.dataframe(df.head())

    st.subheader("data shape:")
    st.info(f"{df.shape[0]} Row , {df.shape[1]} Column")

    st.subheader("Statistical Summary:")
    st.dataframe(df.describe())


with tab2:
    st.subheader("📈 Distribution Plot", divider=True)

    feature = st.selectbox("Select a feature:", df.columns)

    fig, ax = plt.subplots()
    sns.histplot(df[feature], kde=True, ax=ax)
    st.pyplot(fig)
with tab3:
    st.subheader("📉 Results of various trained models on the data " , divider=True)
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
    st.markdown("The technical details of data processing and the selection of the final model are provided below.")

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
        Scaling was applied to numerical features such as price and area to:

        - Prevent the model from being influenced by the differing scales of area and price. 
        - Enable faster and more stable model training.
        ---

         #### 🔹 **2. One-Hot Encoding for Address**
        Since the address is a categorical feature, it was converted using One-Hot Encoding.
        This method allowed the model to learn the distinct valuation of each neighborhood without imposing an artificial order between them.
        
        ---

        #### **🔹 3. Conversion of Boolean Features to Numeric**

        True/False features were converted to **0 and 1** so the model could utilize them in the learning process.
        These features are typically very important for property amenities (such as elevator, parking, etc.).
        """)

    # -------------------------
    # 🤖 Model Selection
    # -------------------------

    with st.expander("🤖 Final Model Selection", expanded=True):

        st.markdown("""
        #### Why was XGBoost chosen as the final model?
        
        ✔️ **It did not overfit.**
        The model demonstrated close and stable performance on both training and test data, indicating no overfitting occurred.

        ✔️ **Best results among all models.**
        XGBoost achieved the lowest error (RMSE) and highest accuracy (R²) compared to other models.

        ✔️ **High stability and generalization power.**
        The model's performance on new data was reliable and stable.

        """)




