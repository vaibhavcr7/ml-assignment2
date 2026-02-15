import pandas as pd
import numpy as np
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef
from sklearn.pipeline import make_pipeline
import joblib
import os

# Load dataset
data = load_breast_cancer()
X = pd.DataFrame(data.data, columns=data.feature_names)
y = pd.Series(data.target, name='target')

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Create model directory
os.makedirs('model', exist_ok=True)

# Define models and scaling requirement
models = {
    'logistic_regression': (LogisticRegression(max_iter=1000, random_state=42), True),
    'decision_tree': (DecisionTreeClassifier(random_state=42), False),
    'knn': (KNeighborsClassifier(), True),
    'naive_bayes': (GaussianNB(), False),
    'random_forest': (RandomForestClassifier(random_state=42), False),
    'xgboost': (XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss'), False)
}

metrics = []

for name, (model, need_scale) in models.items():
    if need_scale:
        pipeline = make_pipeline(StandardScaler(), model)
        pipeline.fit(X_train, y_train)
        joblib.dump(pipeline, f'model/{name}.pkl')
        y_pred = pipeline.predict(X_test)
        y_proba = pipeline.predict_proba(X_test)[:, 1]
    else:
        model.fit(X_train, y_train)
        joblib.dump(model, f'model/{name}.pkl')
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    acc = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    mcc = matthews_corrcoef(y_test, y_pred)

    metrics.append({
        'Model': name.replace('_', ' ').title(),
        'Accuracy': acc,
        'AUC': auc,
        'Precision': prec,
        'Recall': rec,
        'F1': f1,
        'MCC': mcc
    })

# Save metrics table
pd.DataFrame(metrics).to_csv('model_metrics.csv', index=False)
print(pd.DataFrame(metrics).to_string(index=False))