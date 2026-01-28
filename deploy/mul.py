import kagglehub
import pandas as pd
import os
import joblib

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, StackingRegressor

# Download dataset
path = kagglehub.dataset_download("harlfoxem/housesalesprediction")
df = pd.read_csv(os.path.join(path, "kc_house_data.csv"))

# Features & target
X = df.drop(columns=['id', 'date', 'price'])
y = df['price']

# Split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Base models
base_models = [
    ('le', LinearRegression()),
    ('str', DecisionTreeRegressor(max_depth=3, random_state=42)),
    ('rfr', RandomForestRegressor(n_estimators=10, max_depth=3, random_state=42))
]

# Meta model
meta_model = LinearRegression()

# Stacking regressor
stack_model = StackingRegressor(
    estimators=base_models,
    final_estimator=meta_model,
    cv=5
)

# Train
stack_model.fit(x_train, y_train)

# Save model
joblib.dump(stack_model, "stacking_house_model.pkl")
joblib.dump(X.columns.tolist(), "feature_names.pkl")

print("✅ Model and feature names saved")
