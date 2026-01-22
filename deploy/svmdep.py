import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

st.set_page_config(page_title="SVM Loan Prediction", layout="wide")

st.title("Loan Prediction using SVM")

uploaded_file = st.file_uploader("Upload Loan Dataset CSV", type="csv")

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("Dataset Preview")
    st.dataframe(df.head())

    df = df.drop(columns=['Loan_ID'], errors='ignore')

    le = LabelEncoder()
    for col in df.select_dtypes(include='object').columns:
        df[col] = df[col].fillna(df[col].mode()[0])
        df[col] = le.fit_transform(df[col])

    for col in df.select_dtypes(include=['float64', 'int64']).columns:
        df[col] = df[col].fillna(df[col].median())

    if 'Dependents' in df.columns:
        df['Dependents'] = df['Dependents'].astype(int)

    X = df.drop(columns=['Loan_Status'])
    y = df['Loan_Status']

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    models = {
        "RBF Kernel": SVC(kernel='rbf', C=1, gamma='scale'),
        "Linear Kernel": SVC(kernel='linear', C=1),
        "Polynomial Kernel": SVC(kernel='poly', C=1)
    }

    accuracies = {}

    st.subheader("Model Performance")

    for name, model in models.items():
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        accuracies[name] = acc

        with st.expander(name):
            st.write(f"Accuracy: {acc:.4f}")
            st.text(classification_report(y_test, y_pred))
            st.write(confusion_matrix(y_test, y_pred))

    st.subheader("Accuracy Comparison")

    fig, ax = plt.subplots()
    ax.bar(accuracies.keys(), accuracies.values())
    ax.set_ylabel("Accuracy")
    ax.set_ylim(min(accuracies.values()) * 0.95, max(accuracies.values()) * 1.05)

    for i, v in enumerate(accuracies.values()):
        ax.text(i, v, f"{v:.4f}", ha='center', va='bottom')

    st.pyplot(fig)

else:
    st.info("Upload the dataset to continue")
