import streamlit as st
import os
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM



# ✅ Load Hugging Face token from Streamlit Secrets
hf_token = st.secrets["HF_TOKEN"]  # Must be set in Streamlit Cloud settings

# ✅ Page configuration
st.set_page_config(page_title="HealthAI", page_icon="🩺", layout="centered")

# ✅ Sidebar navigation
st.sidebar.title("🩺 HealthAI Navigation")
page = st.sidebar.radio("Go to", ["🏠 Home", "🗣️ Patient Chat", "🔍 Disease Prediction", "💊 Treatment Plan", "📊 Health Analytics"])

# ✅ Cache model loading
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
        st.error("🔐 Failed to load model. Check if Hugging Face token is correct and access to the model is granted.")
        st.stop()

# ✅ Load model
tokenizer, model = load_model()

# ✅ Home Page
if page == "🏠 Home":
    st.title("🏠 Welcome to HealthAI")
    st.markdown("HealthAI is your intelligent healthcare assistant powered by **IBM Granite**.")
    st.markdown("Use the sidebar to explore features like: Chatting, Predicting Diseases, Treatment Planning, and Health Analytics.")

# ✅ Patient Chat
elif page == "🗣️ Patient Chat":
    st.title("🗣️ Patient Chat Assistant")
    st.info("Type your health-related question below.")
    user_query = st.text_input("Ask a question:")
    if user_query:
        with st.spinner("🧠 Thinking... please wait..."):
            prompt = f"You are a helpful medical assistant. Answer the following question:\n{user_query}\n"
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            st.success(reply)

# ✅ Disease Prediction
elif page == "🔍 Disease Prediction":
    st.title("🔍 Disease Prediction")
    st.info("Describe your symptoms and get possible conditions.")
    symptoms = st.text_area("📝 Enter your symptoms (e.g., fever, cough, fatigue):")
    if symptoms:
        with st.spinner("🧠 Analyzing symptoms..."):
            prompt = (
                f"You are a medical diagnosis assistant. A patient reports the following symptoms: {symptoms}. "
                "List 3-5 possible diseases or conditions they might have and what they should do next."
            )
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=200, temperature=0.7)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            st.success(reply)

# ✅ Treatment Plan
elif page == "💊 Treatment Plan":
    st.title("💊 Treatment Plan Generator")
    st.info("Enter your diagnosed condition to get a personalized plan.")
    condition = st.text_input("🧾 Enter your condition (e.g., Diabetes, Hypertension):")
    if condition:
        with st.spinner("🩺 Preparing your treatment plan..."):
            prompt = (
                f"A patient has been diagnosed with {condition}. "
                "Provide a complete treatment plan including medications, diet/lifestyle changes, follow-ups, and any special precautions."
            )
            inputs = tokenizer(prompt, return_tensors="pt", padding=True)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            outputs = model.generate(**inputs, max_new_tokens=300, temperature=0.7)
            reply = tokenizer.decode(outputs[0], skip_special_tokens=True).replace(prompt, "").strip()
            st.success(reply)

# ✅ Health Analytics
elif page == "📊 Health Analytics":
    st.title("📊 Health Analytics")
    st.info("Upload your health metrics CSV (e.g., heart rate, blood pressure).")
    file = st.file_uploader("📁 Upload CSV file", type=["csv"])
    if file:
        import pandas as pd
        import plotly.express as px

        df = pd.read_csv(file)
        st.write("📄 Data Preview:")
        st.dataframe(df.head())

        columns = df.columns.tolist()
        selected_metric = st.selectbox("📈 Choose metric to visualize", columns)

        if "time" in df.columns or "date" in df.columns:
            date_column = "time" if "time" in df.columns else "date"
            df[date_column] = pd.to_datetime(df[date_column], errors="coerce")
            fig = px.line(df, x=date_column, y=selected_metric, title=f"{selected_metric} Over Time")
        else:
            fig = px.line(df, y=selected_metric, title=f"{selected_metric} Trend")

        st.plotly_chart(fig)

        if st.button("📊 Get Health Insights"):
            with st.spinner("🧠 Analyzing your data..."):
                mean_val = df[selected_metric].mean()
                min_val = df[selected_metric].min()
                max_val = df[selected_metric].max()
                st.success(
                    f"**Summary for {selected_metric}:**\n\n"
                    f"- Average: {mean_val:.2f}\n"
                    f"- Minimum: {min_val:.2f}\n"
                    f"- Maximum: {max_val:.2f}\n\n"
                    "These are basic insights. Please consult a doctor for medical advice."
                )
