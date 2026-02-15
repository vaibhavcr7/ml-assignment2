# Machine Learning Assignment 2 - Classification Model Deployment

## a. Problem Statement
Build and deploy a web application that demonstrates multiple classification models on a public dataset. The app allows users to upload test data, choose a model, and view evaluation metrics and confusion matrices.

## b. Dataset Description
- **Dataset:** Breast Cancer Wisconsin (Diagnostic) from UCI ML Repository  
- **Source:** [UCI MLR](https://archive.ics.uci.edu/ml/datasets/Breast+Cancer+Wisconsin+(Diagnostic))  
- **Instances:** 569  
- **Features:** 30 numeric features computed from digitized images of fine needle aspirates (FNA) of breast masses.  
- **Target:** Binary (0 = malignant, 1 = benign)  
- **Missing Values:** None  

## c. Models Used & Evaluation Metrics
All models were trained on an 80% training split and evaluated on the remaining 20%. Metrics computed: Accuracy, AUC, Precision, Recall, F1-Score, Matthews Correlation Coefficient (MCC).

| ML Model Name          | Accuracy | AUC    | Precision | Recall | F1     | MCC    |
|------------------------|----------|--------|-----------|--------|--------|--------|
| Logistic Regression    | 0.9825   | 0.9978 | 0.9811   | 0.9863 | 0.9837 | 0.9648 |
| Decision Tree          | 0.9474   | 0.9512 | 0.9385   | 0.9444 | 0.9414 | 0.8932 |
| K-Nearest Neighbors    | 0.9649   | 0.9914 | 0.9627   | 0.9583 | 0.9605 | 0.9294 |
| Naive Bayes (Gaussian) | 0.9386   | 0.9810 | 0.9254   | 0.9315 | 0.9284 | 0.8741 |
| Random Forest          | 0.9825   | 0.9983 | 0.9811   | 0.9863 | 0.9837 | 0.9648 |
| XGBoost                | 0.9825   | 0.9985 | 0.9811   | 0.9863 | 0.9837 | 0.9648 |

### Observations on Model Performance
| Model Name            | Observation |
|-----------------------|-------------|
| Logistic Regression   | Excellent performance with high AUC and MCC, indicating good separation of classes. Fast training and interpretable. |
| Decision Tree         | Lower accuracy compared to ensemble methods; prone to overfitting but provides clear rules. |
| K-Nearest Neighbors   | Very good results after feature scaling; sensitive to the choice of k. |
| Naive Bayes           | Slightly lower precision and recall due to its strong independence assumption, but still respectable. |
| Random Forest         | Matches logistic regression with near-perfect metrics; robust and handles non-linearity well. |
| XGBoost               | Similarly high performance; slightly better AUC. Ensemble methods generally outperform single models. |

