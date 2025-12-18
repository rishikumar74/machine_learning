import streamlit as st
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# --- Page Config ---
st.set_page_config(page_title="Linear Regression", layout="centered")

# --- Load CSS ---
def load_css(file):
    with open(file) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

# --- Title ---
st.markdown(""" 
<div class="card">
    <h1>Linear Regression</h1>
    <p>Predict <b>tip amount</b> from total bill</p>         
</div>   
""", unsafe_allow_html=True)

# --- Load Data ---
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# --- Dataset Preview ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Dataset Preview")
st.dataframe(df.head())
st.markdown('</div>', unsafe_allow_html=True)

# --- Prepare Data ---
X = df[['total_bill']]
y = df['tip']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardization
scaler = StandardScaler()
X_train_trans = scaler.fit_transform(X_train)
X_test_trans = scaler.transform(X_test)

# Model
model = LinearRegression()
model.fit(X_train_trans, y_train)
y_pred = model.predict(X_test_trans)

# --- Metrics ---
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
r2score = r2_score(y_test, y_pred)

# --- Visualization ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Total Bill vs Tip")

fig, ax = plt.subplots()
ax.scatter(df["total_bill"], df["tip"], alpha=0.6, label="Actual Data")

# Transform entire X for regression line
X_all_trans = scaler.transform(df[["total_bill"]])
ax.plot(df["total_bill"], model.predict(X_all_trans), color="red", label="Regression Line")

ax.set_xlabel("Total Bill ($)")
ax.set_ylabel("Tip ($)")
ax.legend()
st.pyplot(fig)
st.markdown('</div>', unsafe_allow_html=True)

# --- Performance Metrics ---
st.markdown('<div class="card">', unsafe_allow_html=True)
st.subheader("Model Performance Metrics")

c1, c2 = st.columns(2)
c1.metric("MAE", f"{mae:.2f}")
c2.metric("RMSE", f"{rmse:.2f}")

c3, c4 = st.columns(2)
c3.metric("R² Score", f"{r2score:.2f}")
c4.metric("MSE", f"{mse:.2f}")

st.markdown('</div>', unsafe_allow_html=True)

# --- Model Intercept and Coefficient ---
st.markdown(f"""
<div class="card">
    <h3>Model Parameters</h3>
    <p><b>Coefficient:</b> {model.coef_[0]:.3f}</p>
    <p><b>Intercept:</b> {model.intercept_:.3f}</p>
</div>
""", unsafe_allow_html=True)

# --- Prediction Input ---
st.markdown('<div class="card">', unsafe_allow_html=True)
bill = st.slider(
    "Select total bill amount ($)", 
    float(df["total_bill"].min()), 
    float(df["total_bill"].max()), 
    30.0
)
tip_pred = model.predict(scaler.transform([[bill]]))[0]
st.markdown(f'<div class="prediction-box">Predicted Tip: ${tip_pred:.2f}</div>', unsafe_allow_html=True)
st.markdown('</div>', unsafe_allow_html=True)
