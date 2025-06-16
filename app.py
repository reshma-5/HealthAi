import streamlit as st
import os
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

# Load token securely from environment or Streamlit secrets
hf_token = os.getenv("HF_TOKEN") or st.secrets.get("HF_TOKEN")

# Set page configuration
st.set_page_config(page_title="HealthAI", page_icon="🩺", layout="centered")

# Sidebar for navigation
st.sidebar.title("🩺 HealthAI Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🗣️ Patient Chat", "🔍 Disease Prediction", "💊 Treatment Plan", "📊 Health Analytics"])

# Model load (cache to avoid reloading)
@st.cache_resource
def load_model():
    tokenizer = AutoTokenizer.from_pretrained("ibm-granite/granite-3.3-2b-instruct", token=hf_token)
    model = AutoModelForCausalLM.from_pretrained(
        "ibm-granite/granite-3.3-2b-instruct",
        torch_dtype=torch.float16,
        device_map="auto",
        token=hf_token
    )
    return tokenizer, model

tokenizer, model = load_model()

# Pages
if page == "🏠 Home":
    st.title("🏠 Welcome to HealthAI")
    st.markdown("HealthAI is your intelligent healthcare assistant powered by IBM Granite.")
    st.markdown("Use the sidebar to navigate through features like chatting, predicting diseases, generating treatment plans, and analyzing health data.")

elif page == "🗣️ Patient Chat":
    st.title("🗣️ Patient Chat Assistant")
    st.info("Type your health-related question below and get an intelligent, caring response.")
    user_query = st.text_input("Ask a question:")
    if user_query:
        with st.spinner("🧠 Thinking... please wait..."):
            prompt = f"You are a helpful medical assistant. Answer the following question:\n{user_query}\n"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True)
            st.success(reply[len(prompt):].strip())

elif page == "🔍 Disease Prediction":
    st.title("🔍 Disease Prediction")
    st.info("Describe your symptoms and get a list of possible conditions.")
    symptoms = st.text_area("📝 Enter your symptoms (e.g., fever, cough, fatigue):")
    if symptoms:
        with st.spinner("🧠 Analyzing symptoms... please wait..."):
            prompt = f"You are a medical diagnosis assistant. A patient reports the following symptoms: {symptoms}. List 3-5 possible diseases or conditions they might have and what they should do next."
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
            st.success(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip())

elif page == "💊 Treatment Plan":
    st.title("💊 Treatment Plan Generator")
    st.info("Enter your diagnosed condition to receive a personalized treatment plan.")
    condition = st.text_input("🧾 Enter your condition (e.g., Type 2 Diabetes, Hypertension):")
    if condition:
        with st.spinner("🩺 Preparing your personalized plan..."):
            prompt = (
                f"A patient has been diagnosed with {condition}. "
                "Provide a complete treatment plan including medications, diet/lifestyle changes, follow-ups, and any special precautions."
            )
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
            st.success(tokenizer.decode(outputs[0], skip_special_tokens=True)[len(prompt):].strip())

elif page == "📊 Health Analytics":
    st.title("📊 Health Analytics")
    st.info("Upload your CSV containing health metrics (e.g., heart rate, blood pressure, blood glucose).")
    file = st.file_uploader("Upload CSV with health metrics", type=["csv"])
    if file:
        import pandas as pd
        import plotly.express as px

        df = pd.read_csv(file)
        st.write("Here's a preview of your data:")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        selected_metric = st.selectbox("Select the metric to visualize", columns)

        if "time" in df.columns or "date" in df.columns:
            date_column = "time" if "time" in df.columns else "date"
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            fig = px.line(df, x=date_column, y=selected_metric, title=f"{selected_metric} over Time")
        else:
            fig = px.line(df, y=selected_metric, title=f"{selected_metric} Trend")

        st.plotly_chart(fig)

        if st.button("Get Health Insights"):
            with st.spinner("🧠 Analyzing your health data... please wait..."):
                mean_val = df[selected_metric].mean()
                min_val = df[selected_metric].min()
                max_val = df[selected_metric].max()
                st.success(
                    f"**Summary for {selected_metric}:**\n\n"
                    f"- Average: {mean_val:.2f}\n"
                    f"- Minimum: {min_val:.2f}\n"
                    f"- Maximum: {max_val:.2f}\n\n"
                    "These values provide a preliminary overview. For more detailed insights, consult a healthcare professional."
                )
