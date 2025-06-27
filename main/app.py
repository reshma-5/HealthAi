import streamlit as st
import pandas as pd
import plotly.express as px
import os
from ibm_watsonx_ai.foundation_models import ModelInference

# 🔐 Load credentials
api_key = st.secrets.get("IBM_API_KEY")
project_id = st.secrets.get("IBM_PROJECT_ID")
base_url = "https://us-south.ml.cloud.ibm.com"

# ✅ Initialize Granite Model
model = ModelInference(
    model_id="ibm/granite-13b-instruct-v2",
    project_id=project_id,
    api_key=api_key,
    url=base_url
)

# 🔍 Model query function
def query_granite(prompt):
    response = model.generate(prompt=prompt, max_new_tokens=300)
    return response["results"][0]["generated_text"]

# 🎨 Streamlit Setup
st.set_page_config(page_title="HealthAI", page_icon="🩺", layout="centered")
st.sidebar.title("🩺 HealthAI Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🗣️ Patient Chat", "🔍 Disease Prediction", "💊 Treatment Plan", "📊 Health Analytics"])

# 🏠 Home Page
if page == "🏠 Home":
    st.title("🏠 Welcome to HealthAI")
    st.markdown("""
    🔹 Ask medical questions  
    🔹 Predict diseases  
    🔹 Get treatment plans  
    🔹 View health analytics  

    **Powered by IBM Watsonx.ai + Granite**
    """)

# 🧠 Patient Chat
elif page == "🗣️ Patient Chat":
    st.title("🧠 Patient Chat")
    q = st.text_input("Ask your medical question:")
    if q:
        with st.spinner("Thinking..."):
            prompt = f"You are a healthcare assistant. Help the patient:\n{q}"
            reply = query_granite(prompt)
            st.success(reply)

# 🔍 Disease Prediction
elif page == "🔍 Disease Prediction":
    st.title("🔍 Disease Predictor")
    symptoms = st.text_area("List your symptoms:")
    if symptoms:
        with st.spinner("Analyzing symptoms..."):
            prompt = f"A patient reports: {symptoms}. Suggest possible conditions and recommended next steps."
            st.success(query_granite(prompt))

# 💊 Treatment Plan
elif page == "💊 Treatment Plan":
    st.title("💊 Treatment Planner")
    condition = st.text_input("Enter diagnosed condition:")
    if condition:
        with st.spinner("Generating treatment plan..."):
            prompt = f"Provide a complete, personalized treatment plan for {condition}, including medication and lifestyle recommendations."
            st.success(query_granite(prompt))

# 📊 Health Analytics
elif page == "📊 Health Analytics":
    st.title("📊 Health Analytics Dashboard")
    
    uploaded_file = st.file_uploader("Upload a health CSV file (e.g., symptoms, patient stats)", type=["csv"])
    
    if uploaded_file:
        df = pd.read_csv(uploaded_file)
        st.subheader("📄 Raw Data Preview")
        st.dataframe(df.head())

        st.subheader("📈 Chart Visualization")
        chart_type = st.selectbox("Choose a chart type", ["Bar", "Pie", "Line"])

        column = st.selectbox("Select column to visualize", df.columns)

        if chart_type == "Bar":
            chart = px.bar(df, x=column, title=f"{column} Distribution")
        elif chart_type == "Pie":
            chart = px.pie(df, names=column, title=f"{column} Breakdown")
        elif chart_type == "Line":
            chart = px.line(df, y=column, title=f"{column} Trend Over Index")

        st.plotly_chart(chart)

        st.subheader("🤖 AI Health Insight")
        with st.spinner("Generating AI summary..."):
            prompt = f"Analyze the following health data and provide a brief insight:\n{df.head(10).to_string(index=False)}"
            insight = query_granite(prompt)
            st.success(insight)
    else:
        st.info("Please upload a CSV file to view analytics.")
