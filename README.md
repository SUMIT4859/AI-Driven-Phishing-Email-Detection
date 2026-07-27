# 🛡️ AI-Driven Phishing Email Detection using NLP & Machine Learning

An AI-powered phishing email detection system that identifies whether an email is **Safe** or **Phishing** using **Natural Language Processing (NLP)** and **Machine Learning**.

The application is built with **Python**, **Scikit-learn**, and **Streamlit**, providing a user-friendly web interface for real-time email analysis.

---

## 📌 Project Overview

Phishing emails are one of the most common cyber threats used to steal sensitive information such as passwords, banking details, and personal data.

This project uses **TF-IDF**, **Feature Engineering**, and a **Tuned Logistic Regression** model to accurately classify emails as **Safe** or **Phishing**.

---

## 🚀 Features

- 🛡️ Detect Safe and Phishing Emails
- 📄 Paste Email Content
- 📂 Upload TXT Email Files
- 📊 Prediction Confidence Chart
- 🚦 Risk Level Indicator
- 🔍 Suspicious Keyword Detection
- 📜 Prediction History
- 📥 Download Prediction Report
- ⚡ Fast Prediction using Cached Model
- 🎨 Professional Streamlit Interface

---

## 🤖 Machine Learning Pipeline

- Data Collection
- Data Cleaning & Preprocessing
- Feature Engineering
- TF-IDF Feature Extraction
- Model Training
- Hyperparameter Tuning
- Model Evaluation
- Streamlit Deployment

---

## 📊 Models Evaluated

| Model | Accuracy |
|--------|---------:|
| Logistic Regression | 97.60% |
| Random Forest | 97.46% |
| Neural Network | 97.58% |
| Naive Bayes | 96.35% |
| **Tuned Logistic Regression** | **97.83%** ✅ |

---

## 📈 Final Model

**Best Model:** Tuned Logistic Regression

**Accuracy:** **97.83%**

**Feature Extraction:**

- TF-IDF
- Engineered Features

---

## 📂 Project Structure

```text
AI-Driven-Phishing-Email-Detection/
│
├── app.py
├── README.md
├── requirements.txt
│
├── data/
│   ├── raw/
│   └── processed/
│
├── models/
│
├── results/
│   ├── figures/
│   ├── metrics/
│   └── reports/
│
└── src/
    ├── preprocess.py
    ├── eda.py
    ├── feature_engineering.py
    ├── train.py
    ├── tune.py
    ├── evaluate.py
    └── predict.py
```

---

## 📊 Dataset Information

- Total Emails: **17,537**
- Safe Emails
- Phishing Emails

Dataset includes legitimate and phishing emails used for supervised machine learning.

---

## 🖥️ Installation

Clone the repository

```bash
git clone https://github.com/SUMIT4859/AI-Driven-Phishing-Email-Detection.git
```

Go to project folder

```bash
cd AI-Driven-Phishing-Email-Detection
```

Create virtual environment

```bash
python -m venv venv
```

Activate virtual environment

### Windows

```bash
venv\Scripts\activate
```

Install dependencies

```bash
pip install -r requirements.txt
```

Run the application

```bash
streamlit run app.py
```

---

## 📷 Application Features

- Home Dashboard
- Email Prediction
- Confidence Score
- Risk Meter
- Probability Graph
- Suspicious Keywords
- Prediction History
- Download Report

---

## 🛠️ Technologies Used

- Python
- Streamlit
- Scikit-learn
- Pandas
- Plotly
- Joblib
- Regular Expressions (Regex)

---

## 📌 Future Improvements

- Deep Learning Models (LSTM/BERT)
- URL Reputation Checking
- Email Attachment Analysis
- Browser Extension
- Email Client Integration
- Multi-language Email Detection

---

## 👨‍💻 Developer

**SUMITKUMAR PANDIT**

M.Sc. Information Technology

AI | Machine Learning | Full Stack Development | Web3

GitHub:

https://github.com/SUMIT4859

---

## ⭐ If you like this project, don't forget to Star the repository!
