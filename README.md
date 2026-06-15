# 🔬 ML Model Debugger

### Advanced Machine Learning Diagnostics & Explainability Platform

🚀 **Live Demo:** https://ml-debugg.streamlit.app/

A professional-grade Streamlit application for training machine learning models and analyzing model behavior beyond traditional accuracy metrics.

---

## ✨ Overview

ML Model Debugger helps data scientists and ML practitioners understand **why** a model performs the way it does through advanced diagnostics, explainability tools, and interactive visualizations.

Instead of stopping at accuracy scores, this platform provides insights into:

* Feature importance drift
* Prediction confidence
* Misclassification patterns
* Error clustering
* SHAP explainability
* Model generalization

---

## 📸 Application Preview

> Add screenshots here after deployment

### Dashboard

![Dashboard](assets/dashboard.png)

### Feature Drift Analysis

![Feature Drift](assets/feature_drift.png)

### SHAP Explainability

![SHAP](assets/shap.png)

---

## 🎯 Key Features

### 🤖 Multiple Model Support

* Logistic Regression
* Random Forest
* XGBoost

### 📊 Advanced Diagnostics

#### Feature Importance Drift

Compare feature importance between training and validation datasets to identify unstable predictors.

#### Prediction Confidence Analysis

Visualize confidence distributions and detect high-confidence prediction errors.

#### Error Clustering

Analyze misclassified samples using PCA and t-SNE to uncover hidden patterns.

#### Interactive Confusion Matrix

Explore prediction performance with normalized and raw confusion matrices.

#### SHAP Explainability

Understand global feature importance and local prediction explanations.

---

## 📈 Evaluation Metrics

* Accuracy
* Precision
* Recall
* F1 Score
* ROC-AUC
* Classification Report
* Train vs Validation Comparison

---

## 💾 Export Options

* Trained Models (.pkl)
* Predictions with Confidence Scores
* Misclassified Samples
* Metrics Reports (JSON)

---

## 🛠️ Tech Stack

| Category             | Tools                 |
| -------------------- | --------------------- |
| Frontend             | Streamlit             |
| Machine Learning     | Scikit-Learn, XGBoost |
| Visualization        | Plotly                |
| Explainability       | SHAP                  |
| Data Processing      | Pandas, NumPy         |
| Scientific Computing | SciPy                 |

---

## 🏗️ Project Structure

```text
ML_Model_Debugger/
│
├── app.py
├── data.py
├── model.py
├── viz.py
├── requirements.txt
└── README.md
```

---

## 🚀 Getting Started

### Clone Repository

```bash
git clone https://github.com/KAyush16/ML-Model-Debugger.git
cd ML-Model-Debugger
```

### Install Dependencies

```bash
pip install -r requirements.txt
```

### Run Application

```bash
streamlit run app.py
```

Application will be available at:

```text
http://localhost:8501
```

---

## 📊 Supported Datasets

Any classification dataset in CSV format:

* Iris Dataset
* Titanic Dataset
* Wine Quality Dataset
* Credit Card Fraud Dataset
* Custom User Datasets

---

## 🔍 Why This Project?

Most ML dashboards stop at showing model accuracy.

ML Model Debugger focuses on:

* Understanding model behavior
* Diagnosing generalization issues
* Identifying unstable features
* Explaining individual predictions
* Visualizing systematic errors

This makes it useful for both learning and real-world model validation workflows.

---

## 👨‍💻 Author

**Ayush K Jha**

Computer Science Student | Machine Learning Enthusiast

GitHub: https://github.com/KAyush16

---

## ⭐ If you found this project useful

Consider starring the repository.
