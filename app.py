import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Load Hugging Face token from secrets
hf_token = st.secrets["HF_TOKEN"]

# ✅ Page settings
st.set_page_config(page_title="HealthAI", page_icon="🩺", layout="centered")
st.sidebar.title("🩺 HealthAI Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🗣️ Patient Chat", "🔍 Disease Prediction", "💊 Treatment Plan", "📊 Health Analytics"])

# ✅ Load model with caching
@st.cache_resource
def load_model():
    try:
        tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct", token=hf_token)
        model = AutoModelForCausalLM.from_pretrained(
            "ibm-granite/granite-3.3-2b-instruct",
            torch_dtype=torch.float16,
            device_map="auto",
            token=hf_token
        )
        return tokenizer, model
    except Exception as e:
        st.error("❌ Failed to load model. Please check Hugging Face token or model access.")
        st.exception(e)
        st.stop()

tokenizer, model = load_model()

# ✅ Home Page
if page == "🏠 Home":
    st.title("🏠 Welcome to HealthAI")
    st.markdown("HealthAI is your intelligent healthcare assistant powered by **IBM Granite**.")
    st.markdown("Use the sidebar to explore features like Chat, Disease Prediction, Treatment Plans, and Health Analytics.")

# ✅ Patient Chat
elif page == "🗣️ Patient Chat":
    st.title("🗣️ Patient Chat Assistant")
    user_query = st.text_input("Ask a health-related question:")
    if user_query:
        with st.spinner("🧠 Thinking..."):
            prompt = f"You are a helpful medical assistant. Answer the following question:\n{user_query}"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            st.success(reply)

# ✅ Disease Prediction
elif page == "🔍 Disease Prediction":
    st.title("🔍 Disease Prediction")
    symptoms = st.text_area("📝 Describe your symptoms (e.g., fever, cough, fatigue):")
    if symptoms:
        with st.spinner("🧠 Analyzing symptoms..."):
            prompt = f"A patient reports: {symptoms}. List 3–5 possible diseases and suggest what they should do next."
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            st.success(reply)

# ✅ Treatment Plan
elif page == "💊 Treatment Plan":
    st.title("💊 Treatment Plan Generator")
    condition = st.text_input("Enter diagnosed condition (e.g., Asthma, Diabetes):")
    if condition:
        with st.spinner("📋 Generating treatment plan..."):
            prompt = f"A patient is diagnosed with {condition}. Provide a detailed treatment plan with medication, lifestyle, and precautions."
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            st.success(reply)

# ✅ Health Analytics
elif page == "📊 Health Analytics":
    st.title("📊 Health Analytics")
    file = st.file_uploader("📁 Upload health data CSV (with date/time and metrics):", type=["csv"])
    if file:
        import pandas as pd
        import plotly.express as px

        df = pd.read_csv(file)
        st.write("🧾 Preview:")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        selected_metric = st.selectbox("Select a metric to analyze", columns)

        if "date" in df.columns or "time" in df.columns:
            date_col = "date" if "date" in df.columns else "time"
            df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
            fig = px.line(df, x=date_col, y=selected_metric, title=f"{selected_metric} Over Time")
        else:
            fig = px.line(df, y=selected_metric, title=f"{selected_metric} Trend")

        st.plotly_chart(fig)

        if st.button("🧠 Generate Summary"):
            mean = df[selected_metric].mean()
            min_ = df[selected_metric].min()
            max_ = df[selected_metric].max()
            st.success(
                f"📊 **{selected_metric} Summary**\n\n"
                f"- Average: {mean:.2f}\n"
                f"- Min: {min_:.2f}\n"
                f"- Max: {max_:.2f}\n\n"
                "ℹ️ These are automated insights. For medical advice, consult a professional."
            )
