# 📉 Customer Churn Prediction

A machine learning project to predict customer churn based on behavioral and contract-related data from a telecom company. This notebook implements a full data science pipeline using Python and scikit-learn — from data cleaning and preprocessing to model training, evaluation, and interpretation.

---

## 🧠 Problem Statement

Customer churn is one of the most important KPIs in subscription-based businesses. Identifying customers who are likely to cancel helps companies act proactively, improve retention, and reduce losses. This project applies **logistic regression** to predict churn using a real-world dataset.

---

## 📁 Dataset

**Telco Customer Churn Dataset**  
- Source: [Kaggle](https://www.kaggle.com/blastchar/telco-customer-churn)  
- Each row represents a customer, with columns describing customer profile, account information, and churn status (Yes/No).

---

## 🔧 Technologies Used

- **Python** 3.9+
- **Pandas**, **NumPy** for data handling
- **Scikit-learn** for machine learning
- **Matplotlib**, **Seaborn** for visualization
- **Logistic Regression** for modeling

---

## 📊 ML Workflow Overview

1. **Data loading** and initial exploration  
2. **Cleaning** (null values, irrelevant columns)  
3. **Encoding** categorical features (LabelEncoder, One-Hot)  
4. **Scaling** numeric features with `StandardScaler`  
5. **Splitting** into train and test sets  
6. **Model training** using logistic regression  
7. **Evaluation** using confusion matrix, precision, recall, and F1-score  
8. **Visualization** of results (confusion matrix)

---

## 🔎 Key Results

- Accuracy: ~80%  
- F1-score for churn class: ~0.68  
- Model performed better at identifying non-churners, and moderately well at detecting churn cases.

---

## 📂 Project Structure

churn-prediction/
├── data/
│ └── WA_Fn-UseC_-Telco-Customer-Churn.csv # Dados utilizados
├── notebooks/
│ └── churn_prediction.ipynb # Main notebook
├── requirements.txt # Python dependencies
├── README.md # Project documentation
└── .gitignore # Optional, ignores checkpoints, pycache, etc.


---

## 🚀 To run it

### Option 1: Run in Google Colab

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](
https://colab.research.google.com/github/LucasR2D2S/churn-prediction/blob/main/notebooks/churn_prediction.ipynb)

### Option 2: Clone locally

```bash
git clone https://github.com/LucasR2D2S/churn_prediction.git
cd churn-prediction
pip install -r requirements.txt
jupyter notebook notebooks/churn_prediction.ipynb
