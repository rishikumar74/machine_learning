import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import StandardScaler
import numpy as np

# Load dataset
df = sns.load_dataset("tips")
print(df.head())
print(df.info())
print(df.describe())

# Visualize relationship
plt.scatter(df["total_bill"], df["tip"])
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.title("Total Bill vs Tip")
plt.show()

# Feature & target separation
X = df[["total_bill"]]  # 2D
y = df["tip"]           # 1D

# Train-test split
x_train, x_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Feature standardization
scaler = StandardScaler()
x_train_scaled = scaler.fit_transform(x_train)
x_test_scaled = scaler.transform(x_test)

# Train Linear Regression model
model = LinearRegression()
model.fit(x_train_scaled, y_train)

print("Coefficient:", model.coef_[0])
print("Intercept:", model.intercept_)

# Prediction
y_pred = model.predict(x_test_scaled)

# Model evaluation
print("MSE:", mean_squared_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Visualize regression line (on original scale for clarity)
# Create smooth range of total_bill
x_line = np.linspace(X["total_bill"].min(), X["total_bill"].max(), 100).reshape(-1, 1)

# Scale and predict
x_line_scaled = scaler.transform(x_line)
y_line = model.predict(x_line_scaled)

# Plot
plt.scatter(x_train["total_bill"], y_train, label="Training data")
plt.plot(x_line, y_line, color="red", label="Regression line")
plt.xlabel("Total Bill")
plt.ylabel("Tip")
plt.title("Linear Regression Best Fit Line")
plt.legend()
plt.show()


#input output reaal time
bill_amt=float(input("Enter bill amt"))
bill_scaled=scaler.transform([[bill_amt]])
tip_pred=model.predict(bill_scaled)
print(f"predicted tip value is {tip_pred}")