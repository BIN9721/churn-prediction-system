# 🏦 End-to-End Bank Customer Churn Prediction System

![Python](https://img.shields.io/badge/Python-3.9-blue.svg)
![Framework](https://img.shields.io/badge/Framework-FastAPI%20%26%20Streamlit-red.svg)
![Deployment](https://img.shields.io/badge/Deployment-Render%20%26%20Streamlit%20Cloud-success.svg)

> **A full-stack Machine Learning solution designed to predict and prevent bank customer churn.**
> From data processing and model training to API deployment and interactive dashboard creation.

---

## 🌟 Live Demo
Experience the application directly in your browser:

| Component | Access Link | Status |
|-----------|-------------|--------|
| **📊 User Dashboard** | [👉 CLICK HERE TO OPEN APP](https://churn-prediction-system-bdhec36beuuva4lkgyvxbd.streamlit.app/) | ![Active](https://img.shields.io/badge/Status-Active-brightgreen) |
| **⚙️ Backend API** | [👉 CLICK HERE FOR API DOCS](https://churn-prediction-system-azwj.onrender.com) | ![Active](https://img.shields.io/badge/Status-Active-brightgreen) |

*(Note: Since this project is hosted on a free-tier server, please allow approx. 50 seconds for the backend to wake up during the first request.)*

---

## 🎯 Project Overview

In the highly competitive banking sector, customer retention is significantly more cost-effective than acquisition. This project addresses the critical business problem: **How can we identify at-risk customers early to intervene effectively?**

### Key Features:
1.  **Robust ML Engine:** Utilizes **XGBoost/Random Forest** optimized for performance. Implements **SMOTE** (Synthetic Minority Over-sampling Technique) to handle imbalanced datasets effectively.
2.  **Business-Driven Threshold (0.02):** Instead of the default 0.5 probability threshold, this system uses an **optimal threshold of 2%**. This "High Recall" strategy ensures we capture the maximum number of potential churners, prioritizing risk mitigation over precision.
3.  **Decoupled Architecture:** A modern microservices approach where the Backend (FastAPI) and Frontend (Streamlit) operate independently, communicating via RESTful APIs.
4.  **Actionable Insights:** The system doesn't just output probabilities; it provides specific recommendation strategies (e.g., "Send Discount Voucher", "VIP Call") based on risk levels.

---

## 💰 Business Impact Analysis

Using the optimal threshold of **0.02** (instead of the default 0.5), we shift the focus to **Risk Minimization**.

* **Scenario:** 10,000 Customers.
* **Cost of False Negative (Missed Churn):** $500 (LTV lost).
* **Cost of False Positive (Wrong Promotion):** $10 (Voucher cost).
* **Result:** By accepting a higher False Positive rate (sending vouchers to safe customers), we capture **96% of actual churners**, saving the bank approximately **$1.2M** in potential lost revenue compared to doing nothing.

---

## 📊 Model Performance

After hyperparameter tuning, the **XGBoost** model achieved exceptional results on the test set. The metrics indicate a highly robust model capable of distinguishing between loyal and churning customers with high confidence.

| Metric | Score | Business Interpretation |
|--------|:-----:|-------------------------|
| **ROC-AUC** | **0.9930** | Near-perfect separation between churners and non-churners. The model is highly reliable. |
| **Accuracy** | **97%** | Overall, the model predicts correctly in 97 out of 100 cases. |
| **Recall (Churn)** | **91%** | **CRITICAL:** The model successfully identifies **91%** of all actual churners. This minimizes the risk of losing high-value customers unnoticed. |
| **Precision (Churn)**| **89%** | When the AI predicts a customer will churn, it is correct **89%** of the time. This ensures marketing budget is not wasted on "safe" customers. |
| **F1-Score** | **0.90** | A strong harmonic mean between Precision and Recall, proving the model handles the imbalanced data effectively. |

> *Performance metrics based on the classification report of the Tuned XGBoost model on the test dataset.*

---

## 🔍 Model Interpretability (Explainable AI)

To overcome the "Black Box" nature of machine learning models, we utilized **SHAP (SHapley Additive exPlanations)** values to explain individual predictions and understand global feature importance.

### Why SHAP?
* **Global Interpretability:** Identifies which features drive customer churn the most across the entire dataset.
* **Local Interpretability:** Explains *why* a specific customer was flagged as "High Risk" (e.g., *"This customer is risky because their Transaction Count dropped by 50% vs last quarter"*).

---

## 💾 Dataset Information

* **Source:** [Credit Card Customers (Kaggle)](https://www.kaggle.com/sakshigoyal7/credit-card-customers)
* **Size:** 10,127 records, 21 features.

---

## 🛠️ Tech Stack

* **Language:** Python 3.9
* **Data Science:** Pandas, NumPy, Scikit-learn, Imbalanced-learn.
* **Machine Learning:** XGBoost, Joblib (Pipeline serialization).
* **Backend Engineering:** FastAPI, Uvicorn (High-performance ASGI server).
* **Frontend Engineering:** Streamlit, Matplotlib (Data Visualization).
* **DevOps & Deployment:** Render (Backend), Streamlit Cloud (Frontend), Git/GitHub.

---

## 📂 Project Structure

```text
├── app.py                 # Backend API (FastAPI) - Handles logic & Model serving
├── dashboard.py           # Frontend (Streamlit) - User Interface
├── requirements.txt       # Project dependencies
├── .gitignore             # Git configuration
├── models/                # Serialized trained models
│   └── churn_prediction_pipeline.pkl
├── notebooks/             # (Optional) EDA & Training notebooks
└── README.md              # Project Documentation
