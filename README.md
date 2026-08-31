# Insider Threat Detection from User Log Data

## Overview

This project investigates **insider threat detection** using user activity logs from the **CERT Insider Threat Dataset**.

The framework combines behavioral feature engineering, fuzzy learning, deep learning, and machine learning approaches to identify anomalous user behavior.

---

## Dataset

Experiments are conducted on:

* **CERT r4.2**
* **CERT r5.2**

The dataset contains synthetic enterprise user activity logs, including:

* Logon / Logoff
* File activity
* Email activity
* Device usage
* HTTP / web activity

Raw logs are processed and transformed into structured behavioral features for model training.

---

## Overall Architecture

The overall architecture of the proposed framework is illustrated below.

<p align="center">
  <img src="image/Sơ đồ kiến trúc tổng quan_final.drawio.png" width="900">
</p>

The framework extracts behavioral features from CERT logs and models user behavior at both **session-level** and **user-level**. The resulting predictions are combined through **probability-level fusion** for final classification.

---

## Feature Engineering

Behavioral features are extracted from different types of user activities, including:

* Authentication behavior
* File access behavior
* Email behavior
* Device usage
* Web activity
* Temporal and after-hours activity

Features are constructed at different behavioral levels and further processed for machine learning and deep learning models.

---

## Models

Several machine learning and deep learning approaches are investigated:

### Machine Learning

* **LightGBM**

### Deep Learning

* **LSTM**
* **TCN**
* **TCN+LSTM**
* **SAINT**
* **BiLSTM**
* **GRU**
* **Transformer**
* **BiLSTM+Transformer**
* **Mamba**

### Fuzzy Learning

* **Fuzzy Input**
* **Fuzzy Learnable Layer**
* **Fuzzy Output**
* **Fuzzy Input + Output**
* **Fuzzy Learnable Layer + Output**

The experiments also investigate hybrid architectures that combine fuzzy learning, deep learning, session-level and user-level representations.

---

## Session & User Modeling

User behavior is modeled at two complementary levels:

* **Session-level:** captures short-term behavioral patterns.
* **User-level:** captures broader behavioral characteristics.

Their prediction probabilities are combined through **probability-level fusion** before the final classification stage.

---

## Class Imbalance

Since insider threat datasets are highly imbalanced, several strategies are investigated:

* Random Undersampling
* Class Weighting
* SMOTE
* Hybrid sampling strategies

---

## Evaluation

Models are evaluated using:

* Accuracy
* Macro Precision
* Macro Recall
* Macro F1-score
* Weighted F1-score
* ROC-AUC

Particular emphasis is placed on **Macro Recall, Macro F1, and ROC-AUC** due to the imbalanced nature of the dataset.

---

## Results

The project compares traditional machine learning, sequence-based deep learning, Transformer-based, fuzzy-learning, and hybrid approaches.

Detailed experimental results and comparisons are provided in the accompanying thesis and experiment files.

---

## Technologies

* Python
* NumPy
* Pandas
* Scikit-learn
* TensorFlow / Keras
* LightGBM
* imbalanced-learn
* Matplotlib

---

## Reference

This project uses the **CERT Insider Threat Dataset** and builds upon the feature extraction work from:

**Feature Extraction for CERT Insider Threat Test Datasets**

[GitHub Repository](https://github.com/lcd-dal/feature-extraction-for-CERT-insider-threat-test-datasets)

---

## Disclaimer

The CERT Insider Threat Dataset is a **synthetic dataset** designed for research and experimentation. Results should therefore be interpreted as experimental findings rather than direct evidence of performance in real-world enterprise environments.
