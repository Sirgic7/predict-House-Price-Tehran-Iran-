import streamlit as st 
pages = {
    "Project": [
        st.Page("app.py", title=" Project Overview"),
        st.Page("app3.py", title="EDA"),
        st.Page("app2.py", title="Model Inference"),
    ]
}

pg = st.navigation(pages)
pg.run()