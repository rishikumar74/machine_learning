# Save as app.py
import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

st.set_page_config(page_title="Telco Churn Predictor", layout="wide")
st.title(" Telco Customer Churn Predictor")
st.markdown("<h4 style='text-align:center; color:purple;'>Interactive Logistic Regression Model</h4>", unsafe_allow_html=True)

# 1️⃣ Load dataset from URL
@st.cache_data
def load_data():
    url = "https://raw.githubusercontent.com/datasciencedojo/datasets/master/Telco-Customer-Churn.csv"
    df = pd.read_csv(url)
    return df

df = load_data()

# 2️⃣ Show dataset
if st.checkbox("Show Dataset"):
    st.subheader("First 5 rows")
    st.dataframe(df.head())
    st.subheader("Dataset Info")
    st.write(df.describe())

# 3️⃣ Preprocessing
df = df.drop('customerID', axis=1)
df['TotalCharges'] = pd.to_numeric(df['TotalCharges'], errors='coerce').fillna(0)

binary_cols = ['gender','Partner','Dependents','PhoneService','PaperlessBilling']
le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

multi_cols = ['MultipleLines','InternetService','OnlineSecurity','OnlineBackup',
              'DeviceProtection','TechSupport','StreamingTV','StreamingMovies',
              'Contract','PaymentMethod']
df = pd.get_dummies(df, columns=multi_cols, drop_first=True)

X = df.drop('Churn', axis=1)
y = df['Churn']

x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

num_cols = ['MonthlyCharges', 'TotalCharges']
sc = StandardScaler()
x_train[num_cols] = sc.fit_transform(x_train[num_cols])
x_test[num_cols] = sc.transform(x_test[num_cols])

# 4️⃣ Train model
model = LogisticRegression(max_iter=1000)
model.fit(x_train, y_train)
y_pred = model.predict(x_test)

# 5️⃣ Model Evaluation
st.subheader("📊 Model Metrics")
st.markdown(f"<h3 style='text-align:center; color:green;'>Accuracy: {accuracy_score(y_test, y_pred):.2f}</h3>", unsafe_allow_html=True)
st.markdown("<h4 style='text-align:center; color:blue;'>Confusion Matrix</h4>", unsafe_allow_html=True)
st.dataframe(pd.DataFrame(confusion_matrix(y_test, y_pred),
                          columns=["Predicted No", "Predicted Yes"],
                          index=["Actual No", "Actual Yes"]))
st.markdown("<h4 style='text-align:center; color:blue;'>Classification Report</h4>", unsafe_allow_html=True)
st.text(classification_report(y_test, y_pred))

# 6️⃣ Predict new customer
st.subheader("Predict Churn for a New Customer")
st.markdown("Fill the customer information below:")

def user_input_features():
    gender = st.selectbox("Gender", ["Female", "Male"])
    SeniorCitizen = st.selectbox("SeniorCitizen", [0,1])
    Partner = st.selectbox("Partner", ["No", "Yes"])
    Dependents = st.selectbox("Dependents", ["No", "Yes"])
    tenure = st.number_input("Tenure (months)", 0, 100, 1)
    PhoneService = st.selectbox("PhoneService", ["No", "Yes"])
    PaperlessBilling = st.selectbox("PaperlessBilling", ["No", "Yes"])
    MonthlyCharges = st.number_input("MonthlyCharges", 0.0, 500.0, 70.0)
    TotalCharges = st.number_input("TotalCharges", 0.0, 10000.0, 2000.0)
    InternetService = st.selectbox("InternetService", ["DSL","Fiber optic","No"])
    Contract = st.selectbox("Contract", ["Month-to-month","One year","Two year"])
    PaymentMethod = st.selectbox("PaymentMethod", ["Electronic check","Mailed check","Bank transfer (automatic)","Credit card (automatic)"])
    
    data = {
        'gender': gender,
        'SeniorCitizen': SeniorCitizen,
        'Partner': Partner,
        'Dependents': Dependents,
        'tenure': tenure,
        'PhoneService': PhoneService,
        'PaperlessBilling': PaperlessBilling,
        'MonthlyCharges': MonthlyCharges,
        'TotalCharges': TotalCharges,
        'InternetService': InternetService,
        'Contract': Contract,
        'PaymentMethod': PaymentMethod
    }
    return pd.DataFrame([data])

input_df = user_input_features()

# Encode new input
for col in binary_cols:
    input_df[col] = le.fit_transform(input_df[col])

input_df = pd.get_dummies(input_df)
input_df = input_df.reindex(columns=X.columns, fill_value=0)
input_df[num_cols] = sc.transform(input_df[num_cols])

# 7️⃣ Centered Prediction
if st.button("Predict Churn"):
    prediction = model.predict(input_df)[0]
    prediction_prob = model.predict_proba(input_df)[0][1]
    
    st.markdown(
        f"""
        <div style='text-align: center; background-color:#FDEBD0; padding:25px; border-radius:15px;'>
        <h2>Prediction: {'🛑 Churn' if prediction==1 else '✅ No Churn'}</h2>
        <h3>Churn Probability: {prediction_prob:.2f}</h3>
        </div>
        """,
        unsafe_allow_html=True
    )
