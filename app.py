import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, matthews_corrcoef, confusion_matrix, classification_report

st.set_page_config(page_title="ML Classification App", layout="wide")
st.title("Machine Learning Classification Models")
st.markdown("This app demonstrates multiple classification models trained on the **Breast Cancer Wisconsin** dataset.")

# Sidebar model selection
st.sidebar.header("Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose a model",
    ("Logistic Regression", "Decision Tree", "K-Nearest Neighbors", "Naive Bayes", "Random Forest", "XGBoost")
)

model_files = {
    "Logistic Regression": "logistic_regression.pkl",
    "Decision Tree": "decision_tree.pkl",
    "K-Nearest Neighbors": "knn.pkl",
    "Naive Bayes": "naive_bayes.pkl",
    "Random Forest": "random_forest.pkl",
    "XGBoost": "xgboost.pkl"
}

@st.cache_resource
def load_model(model_name):
    return joblib.load(f"model/{model_files[model_name]}")

model = load_model(model_choice)

# Main content
st.header("Upload Test Data")
st.markdown("Upload a CSV file with the same 30 features as the dataset. The file **must** include a column named **'target'** (0 = malignant, 1 = benign).")

uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    st.subheader("Uploaded Data Preview")
    st.write(df.head())

    if 'target' not in df.columns:
        st.error("The uploaded file must contain a 'target' column.")
    else:
        X_test = df.drop(columns=['target'])
        y_test = df['target']

        # Check number of features
        expected_features = model.n_features_in_ if hasattr(model, 'n_features_in_') else len(model.feature_names_in_)
        if X_test.shape[1] != expected_features:
            st.error(f"Expected {expected_features} features, but got {X_test.shape[1]}. Check your CSV.")
        else:
            y_pred = model.predict(X_test)
            y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

            # Compute metrics
            acc = accuracy_score(y_test, y_pred)
            auc = roc_auc_score(y_test, y_proba) if y_proba is not None else None
            prec = precision_score(y_test, y_pred)
            rec = recall_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            mcc = matthews_corrcoef(y_test, y_pred)

            st.subheader("Evaluation Metrics")
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Accuracy", f"{acc:.4f}")
                st.metric("Precision", f"{prec:.4f}")
            with col2:
                st.metric("AUC", f"{auc:.4f}" if auc else "N/A")
                st.metric("Recall", f"{rec:.4f}")
            with col3:
                st.metric("F1 Score", f"{f1:.4f}")
                st.metric("MCC", f"{mcc:.4f}")

            # Confusion Matrix
            st.subheader("Confusion Matrix")
            cm = confusion_matrix(y_test, y_pred)
            fig, ax = plt.subplots()
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel('Predicted')
            ax.set_ylabel('Actual')
            st.pyplot(fig)

            # Classification Report
            st.subheader("Classification Report")
            report = classification_report(y_test, y_pred, output_dict=True)
            report_df = pd.DataFrame(report).transpose()
            st.dataframe(report_df)

else:
    st.info("Please upload a CSV file to proceed.")

# Sample data download
st.sidebar.header("Sample Data")
if st.sidebar.button("Download Sample Test CSV"):
    data = load_breast_cancer()
    X = pd.DataFrame(data.data, columns=data.feature_names)
    y = pd.Series(data.target, name='target')
    sample = pd.concat([X, y], axis=1).sample(100, random_state=42)
    csv = sample.to_csv(index=False)
    st.sidebar.download_button(
        label="Download CSV",
        data=csv,
        file_name="sample_test.csv",
        mime="text/csv"
    )