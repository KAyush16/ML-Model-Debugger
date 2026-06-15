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

<img width="1906" height="1012" alt="Screenshot 2026-06-15 211458" src="https://github.com/user-attachments/assets/e561e3d7-0e2e-40b7-a39a-4eeb30f13913" />

### Dashboard

<img width="1742" height="955" alt="Screenshot 2026-06-15 211521" src="https://github.com/user-attachments/assets/077d4e17-05ea-4563-8723-8c759ae91d7d" />

<img width="1907" height="1013" alt="Screenshot 2026-06-15 211655" src="https://github.com/user-attachments/assets/c5d3d3d8-373f-44c7-bec4-970ba9b82672" />
<img width="1812" height="937" alt="Screenshot 2026-06-15 211728" src="https://github.com/user-attachments/assets/8b8b4361-7ad5-4abe-9b0d-74881f62e1d7" />


### Feature Drift Analysis
<img width="1875" height="980" alt="Screenshot 2026-06-15 211719" src="https://github.com/user-attachments/assets/17102598-4bba-455f-a712-efefebc15d64" />



### SHAP Explainability
<img width="1897" height="962" alt="Screenshot 2026-06-15 211734" src="https://github.com/user-attachments/assets/aa5c5366-594c-445f-8e84-085c633a8e0c" />



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
