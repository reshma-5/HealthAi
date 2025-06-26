# 🩺 HealthAI: Intelligent Healthcare Assistant

**Project by**:  
Momin Reshma (Team Leader)  
Alavalapati Alekhya  
Gangalakunta Aakarsha  
Katagouni Sreevalli  

Institution: SREC College, Nandyal  
Internship: SmartInternz – Virtual Internship (AI using IBM Granite)

---

## 📌 Project Overview

HealthAI is a smart AI-powered healthcare assistant that delivers interactive, real-time responses across four key modules:

- 💬 AI Health Chat – Ask health-related questions  
- 🦠 Disease Prediction – Predict illness based on user-entered symptoms  
- 💊 Treatment Plans – Get AI-generated treatment plans for diagnosed conditions  
- 📊 Health Analytics – Upload CSV files and view health trends with visual insights  

Built using Python and Streamlit, and integrated with the IBM Granite 13B Instruct v2 model via Hugging Face, HealthAI simplifies healthcare interactions and improves accessibility to medical insights.

---

## 👥 Team Members & Contributions

| Name                   | Role                                     | Key Contributions                                                                 |
|------------------------|------------------------------------------|------------------------------------------------------------------------------------|
| Momin Reshma       | Team Leader, AI Integration & Deployment | Integrated IBM Granite API, built full app logic, deployed app, managed GitHub     |
| Alavalapati Alekhya| Module Developer                         | Designed healthcare logic flows, refined prompts, structured module interactions   |
| Gangalakunta Aakarsha | UI/UX Design & Branding              | Designed app layout and interface, supported visuals, branding, and styling        |
| Katagouni Sreevalli| Testing & Optimization                   | Tested features, improved app responses, managed offline functionality, recording  |

---

## 📽️ Demo Video

🎥 Watch our full demo here:  
🔗 https://youtu.be/cwCj4w5zgPM?si=J06R0qJTyS1aPfW-

---

## 🖼️ Screenshots

Included in the attached project documentation (PDF).

---

## 📁 Folder Structure

```
HealthAI/
├── app.py
├── requirements.txt
├── runtime.txt
├── sample.csv
└── README.md
```

---

## 🚀 How to Run This Project

1️⃣ Clone the Repository:
```bash
git clone https://github.com/YOUR_USERNAME/HealthAI.git
cd HealthAI
```

2️⃣ (Optional) Create Virtual Environment:
```bash
python -m venv venv
source venv/bin/activate  # On Mac/Linux
venv\Scripts\activate     # On Windows
```

3️⃣ Install Dependencies:
```bash
pip install -r requirements.txt
```

4️⃣ Run the Application:
```bash
streamlit run app.py
```

---

## 🧪 Features Explained

### 💬 AI Health Chat  
Enter general health questions and get friendly, AI-generated responses.

### 🦠 Disease Prediction  
Input symptoms like "fever and rash" or "fatigue and weight loss" to get likely medical conditions and recommended actions.

### 💊 Treatment Plans  
Type a diagnosed condition like "Diabetes" or "Dengue" to receive structured, AI-suggested treatment plans.

### 📊 Health Analytics  
Upload a CSV file like:

```
Symptom,Count
Fever,30
Cough,25
Fatigue,18
Headache,12
```

Visualizations and AI-generated insights are provided based on the uploaded data.

---

## 🧠 Tech Stack

- Python  
- Streamlit  
- Plotly  
- IBM Granite 3.3B Instruct v2 (via Hugging Face API)  
- GitHub for version control  

---

## 🙏 Acknowledgements

- IBM Granite for powerful foundation model  
- SmartInternz & IBM for internship mentorship  
- SREC College, Nandyal for institutional support  

---

✨ Built with care by Momin Reshma, Alavalapati Alekhya, Gangalakunta Aakarsha & Katagouni Sreevalli ✨
