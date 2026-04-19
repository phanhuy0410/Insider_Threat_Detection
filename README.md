## Reference GitHub Link
https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets?tab=readme-ov-file

# Insider Threat Detection from User Log Data (CERT r4.2)
## 📌 Overview
This project focuses on analyzing user behavior from system log data to detect anomalous activities (insider threats).  
We utilize the CERT r4.2 dataset to build a data processing and machine learning pipeline for identifying abnormal user behavior patterns.
The goal is to support system monitoring by detecting suspicious activities based on user behavior sequences.
---

## 📂 Dataset
- Source: CERT Insider Threat Dataset (r4.2)
- Type: Synthetic user activity logs
- Data includes:
  - Logon/logoff events
  - File access
  - Email activity
  - Device usage
## ⚙️ Methodology
### 1. Data Preprocessing
- Clean and normalize raw log data  
- Handle missing values  
- Convert timestamps into sequential format  
- Sort logs by user and time  
### 2. Feature Engineering
- Construct user behavior features based on session/activity:
  - Number of logins  
  - File access frequency  
  - Email activity  
  - After-hours activity  
- Transform raw logs into structured numerical features  
### 3. Model Training
We experimented with several Machine Learning and Deep Learning models:
- Machine Learning:
  - LightGBM  
- Deep Learning (sequence-based):
  - LSTM  
  - GRU  
- (Additional experiments include advanced and hybrid architectures)
### 4. Handling Imbalanced Data
- Applied techniques such as:
  - Random Undersampling  
  - Class weighting  
  - SMOTE
### 5. Evaluation
Models are evaluated using:
- F1-score  
- AUC (ROC-AUC)  
We compare model performance to identify the most suitable approach for anomaly detection.
## 🏗️ Pipeline
The workflow of the project: Raw Logs → Preprocessing → Feature Engineering → Model Training → Evaluation
## 🛠️ Technologies Used
- Python  
- Pandas, NumPy  
- Scikit-learn  
- TensorFlow / Keras  
## 📊 Results
- Successfully built a pipeline for user behavior analysis  
- Compared multiple models for anomaly detection  
- Identified suitable approaches for detecting abnormal user activities  
## 🎯 Application
This project can be applied to:
- System monitoring  
- Insider threat detection  
- User behavior analytics (UBA)  
