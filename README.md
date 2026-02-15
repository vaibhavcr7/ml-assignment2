# Machine Learning Assignment 2 - Classification Model Deployment

## a. Problem Statement
Build and deploy a web application that demonstrates multiple classification models on a public dataset. The app allows users to upload test data, choose a model, and view evaluation metrics and confusion matrices. The goal is to learn the end-to-end ML workflow: data preparation, model training, evaluation, UI design, and deployment on Streamlit Community Cloud.

## b. Dataset Description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) from UCI Machine Learning Repository  
- **Source:** [UCI MLR](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- **Instances:** 569  
- **Features:** 30 numeric features computed from digitized images of fine needle aspirates (FNA) of breast masses. Features include mean radius, texture, perimeter, area, smoothness, compactness, concavity, concave points, symmetry, fractal dimension, and their standard errors and worst values.  
- **Target Variable:** Binary (0 = malignant, 1 = benign)  
- **Missing Values:** None  
- **Train/Test Split:** 80% training, 20% testing (stratified to preserve class distribution)

## c. Models Used & Evaluation Metrics
All six classification models were implemented using scikit-learn and XGBoost. Each model was trained on the same training set and evaluated on the held-out test set. The following metrics were computed for every model:

- **Accuracy** – Overall correct predictions.
- **AUC** – Area Under the ROC Curve.
- **Precision** – True positives / (True positives + False positives).
- **Recall** – True positives / (True positives + False negatives).
- **F1 Score** – Harmonic mean of precision and recall.
- **Matthews Correlation Coefficient (MCC)** – Balanced measure even for imbalanced classes.

### Comparison Table of Model Performance

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.9825   | 0.9978 | 0.9811   | 0.9863 | 0.9837 | 0.9648 |
| Decision Tree          | 0.9474   | 0.9512 | 0.9385   | 0.9444 | 0.9414 | 0.8932 |
| K-Nearest Neighbors    | 0.9649   | 0.9914 | 0.9627   | 0.9583 | 0.9605 | 0.9294 |
| Naive Bayes (Gaussian) | 0.9386   | 0.9810 | 0.9254   | 0.9315 | 0.9284 | 0.8741 |
| Random Forest          | 0.9825   | 0.9983 | 0.9811   | 0.9863 | 0.9837 | 0.9648 |
| XGBoost                | 0.9825   | 0.9985 | 0.9811   | 0.9863 | 0.9837 | 0.9648 |

### Observations on Model Performance

| ML Model Name          | Observation about model performance |
|------------------------|-------------------------------------|
| Logistic Regression    | Achieves excellent accuracy and AUC, with high precision and recall. It is fast to train and interpretable. The high MCC indicates strong correlation between predictions and actuals. |
| Decision Tree          | Performs well but slightly lower than ensemble methods. It is prone to overfitting, which explains the lower generalization compared to Random Forest. However, it provides clear decision rules. |
| K-Nearest Neighbors    | With feature scaling, KNN achieves very good results (accuracy ~96%). It is sensitive to the choice of `k` and distance metric, but performs robustly on this dataset. |
| Naive Bayes (Gaussian) | Despite the strong independence assumption, Gaussian Naive Bayes delivers respectable performance (accuracy ~94%). It is the fastest model but slightly lower in precision and recall. |
| Random Forest          | As an ensemble of decision trees, it matches the top accuracy and AUC. It is robust to overfitting and handles non-linear relationships well. |
| XGBoost                | Provides similarly high performance with a slightly better AUC. Gradient boosting often edges out bagging methods on structured data. |

## Deployment

- **GitHub Repository:** [https://github.com/vaibhavcr7/ml-assignment2](https://github.com/vaibhavcr7/ml-assignment2)  
- **Live Streamlit App:** [https://share.streamlit.io/vaibhavcr7/ml-assignment2/app.py](https://share.streamlit.io/vaibhavcr7/ml-assignment2/app.py)  
- **BITS Virtual Lab Execution Screenshot:** Attached in the PDF submission.

## How to Use the App

1. Open the live app link.
2. Use the sidebar to select one of the six classification models.
3. Download the sample CSV file (provided in the sidebar) or upload your own test CSV containing the same 30 features and a `target` column.
4. After uploading, the app instantly displays:
   - Evaluation metrics (Accuracy, AUC, Precision, Recall, F1, MCC)
   - Confusion matrix
   - Classification report
5. Switch between models to compare performance interactively.

## Files in the Repository

- `app.py` – Main Streamlit application.
- `train.py` – Script to train and save all six models.
- `requirements.txt` – List of Python dependencies.
- `README.md` – This file.
- `model/` – Folder containing the trained model `.pkl` files (generated by `train.py`).
- `model_metrics.csv` – Saved evaluation metrics (optional).

## Local Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/vaibhavcr7/ml-assignment2.git
   cd ml-assignment2