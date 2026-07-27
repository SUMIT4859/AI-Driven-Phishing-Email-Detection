import re
import joblib
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from datetime import datetime

# ===========================================
# Page Configuration
# ===========================================

st.set_page_config(
    page_title="AI Phishing Email Detection",
    page_icon="🛡️",
    layout="wide"
)

# ===========================================
# Load Model
# ===========================================

MODEL_PATH = "models/tuned_logistic_regression.pkl"

@st.cache_resource
def load_model():
    return joblib.load(MODEL_PATH)

model = load_model()

# ===========================================
# Session History
# ===========================================

if "history" not in st.session_state:
    st.session_state.history = []

# ===========================================
# Suspicious Keywords
# ===========================================

SUSPICIOUS_WORDS = [
    "urgent","verify","password","bank",
    "login","click","limited","winner",
    "claim","bonus","gift","free",
    "bitcoin","crypto","account","security",
    "confirm","payment","invoice","update"
]

# ===========================================
# Feature Extraction
# ===========================================

def extract_features(text):

    text = str(text)

    return pd.DataFrame([{

        "Clean_Text": text,

        "Email_Length": len(text),

        "Word_Count": len(text.split()),

        "URL_Count": len(re.findall(r"http[s]?://|www\.", text)),

        "Email_Count": len(re.findall(r"\S+@\S+", text)),

        "Digit_Count": sum(c.isdigit() for c in text),

        "Uppercase_Count": sum(c.isupper() for c in text),

        "Exclamation_Count": text.count("!"),

        "Question_Count": text.count("?"),

        "Special_Char_Count":
            len(re.findall(r"[!@#$%^&*()_+\-=\[\]{};':\"\\|,.<>/?]", text)),

        "Suspicious_Word_Count":
            sum(
                bool(re.search(r"\b" + re.escape(word) + r"\b", text.lower()))
                for word in SUSPICIOUS_WORDS
            )

    }])

# ===========================================
# Header
# ===========================================

st.title("🛡️ AI-Driven Phishing Email Detection")

st.markdown(
"""
### Detect phishing emails using Natural Language Processing (NLP) and Machine Learning.

This application predicts whether an email is **Safe** or **Phishing** using a **Tuned Logistic Regression** model trained on TF-IDF and engineered features.
"""
)

st.markdown("### 📊 Project Dashboard")

c1, c2, c3, c4 = st.columns(4)

c1.metric("🎯 Accuracy", "97.83%")
c2.metric("🤖 Best Model", "Logistic Regression")
c3.metric("📚 Algorithms", "4")
c4.metric("📄 Dataset", "17,537 Emails")

# ===========================================
# Sidebar
# ===========================================

st.sidebar.title("📊 Project Dashboard")

st.sidebar.metric("🎯 Model Accuracy", "97.83%")
st.sidebar.metric("🤖 Best Model", "Logistic Regression")
st.sidebar.metric("📚 Algorithms Tested", "4")
st.sidebar.metric("📄 Dataset Size", "17,537")

st.sidebar.markdown("---")

st.sidebar.success("""
### 🚀 Features

✅ NLP (TF-IDF)

✅ Feature Engineering

✅ Tuned Logistic Regression

✅ Confidence Score

✅ TXT Upload

✅ Probability Chart

✅ Risk Meter

✅ Suspicious Keyword Detection
""")

st.sidebar.markdown("---")

st.sidebar.info("""
### 👨‍💻 Developed With

• Python

• Streamlit

• Plotly

• Scikit-learn

• Pandas

• Machine Learning
""")

# ===========================================
# Input
# ===========================================

option = st.radio(
    "Choose Input Method",
    ["Paste Email", "Upload TXT File"]
)

email_text = ""

if option == "Paste Email":

    email_text = st.text_area(
        "Paste Email Content",
        height=250
    )

else:

    uploaded = st.file_uploader(
        "Upload TXT File",
        type=["txt"]
    )

    if uploaded is not None:

        email_text = uploaded.read().decode("utf-8")

        st.text_area(
            "Email Preview",
            email_text,
            height=250
        )

# ===========================================
# Prediction
# ===========================================

if st.button("🔍 Analyze Email", use_container_width=True):

    if email_text.strip() == "":

        st.warning("Please enter email text.")

    else:

        with st.spinner("Analyzing email..."):

            X = extract_features(email_text)

            prediction = model.predict(X)[0]

            probability = model.predict_proba(X)[0]

        safe = probability[0] * 100
        phishing = probability[1] * 100


        st.subheader("🚦 Risk Level")

        if phishing < 30:
            st.success("🟢 LOW RISK")

        elif phishing < 70:
            st.warning("🟡 MEDIUM RISK")

        else:
            st.error("🔴 HIGH RISK")

        st.divider()

        if prediction == 1:

            st.error("🚨 ALERT: This email appears to be a phishing attempt. Avoid clicking links or sharing personal information.")

        else:

            st.success("✅ This email appears to be safe based on the AI model.")

        st.subheader("Confidence")

        col1, col2 = st.columns(2)

        col1.metric(
            "Safe",
            f"{safe:.2f}%"
        )

        col2.metric(
            "Phishing",
            f"{phishing:.2f}%"
        )

        fig = go.Figure()

        fig.add_bar(
            x=["Safe Email", "Phishing Email"],
            y=[safe, phishing],
            text=[f"{safe:.2f}%", f"{phishing:.2f}%"],
            textposition="outside"
        )

        fig.update_layout(
            title="Prediction Confidence",
            yaxis_title="Probability (%)",
            height=420
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

        found = []

        for word in SUSPICIOUS_WORDS:
            if re.search(r"\b" + re.escape(word) + r"\b", email_text.lower()):
                found.append(word)

        st.session_state.history.append({
            "Time": datetime.now().strftime("%H:%M:%S"),
            "Prediction": "🚨 Phishing" if prediction == 1 else "✅ Safe",
            "Confidence": f"{max(safe, phishing):.2f}%"
        })

        st.subheader("Suspicious Keywords")

        if found:

            st.warning("Detected Keywords")

            cols = st.columns(min(len(found), 4))

            for i, word in enumerate(found):
                cols[i % 4].success(word)

        else:

            st.success("✅ No suspicious keywords detected.")

        report = f"""
==========================================
AI-Driven Phishing Email Detection Report
==========================================

Prediction:
{"Phishing Email" if prediction == 1 else "Safe Email"}

Safe Probability:
{safe:.2f}%

Phishing Probability:
{phishing:.2f}%

Risk Level:
{"HIGH" if phishing>=70 else "MEDIUM" if phishing>=30 else "LOW"}

Suspicious Keywords:
{", ".join(found) if found else "None"}

==========================================

Email Content

{email_text}

==========================================
"""

    st.download_button(
        "📥 Download Prediction Report",
        report,
        file_name="prediction_report.txt",
        mime="text/plain"
    )


st.markdown("---")

st.subheader("📜 Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(st.session_state.history)

    st.dataframe(
        history_df,
        use_container_width=True,
        hide_index=True
    )

else:

    st.info("No predictions yet.")




# ===========================================
# Footer
# ===========================================
st.markdown("---")

st.info("""
### 🛡️ AI-Driven Phishing Email Detection

Built using

- 🐍 Python
- 🤖 Scikit-learn
- 📚 TF-IDF
- 📊 Plotly
- 🌐 Streamlit
- 🧠 Machine Learning

Model Accuracy: **97.83%**

Developed as an NLP & Machine Learning Internship Project.
""")