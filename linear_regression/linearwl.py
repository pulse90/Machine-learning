import os
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt



df = pd.read_csv("linear_regression/data.csv")

# Prepare data
X = df[["x"]].values
y = df["y"].values

# ---------- Linear Regression ----------
model = LinearRegression()
model.fit(X, y)

m = model.coef_[0]
b = model.intercept_

print("Slope (m):", m)
print("Intercept (b):", b)

# Predictions
y_pred = model.predict(X)

mse = mean_squared_error(y, y_pred)
print("Mean Squared Error (MSE):", mse)

# ---------- Plotting ----------
plt.figure(figsize=(8,5))

# Scatter plot of actual data
plt.scatter(X, y, label="Actual Data")

# Regression line
plt.plot(X, y_pred, label="Regression Line")

plt.xlabel("X")
plt.ylabel("Y")
plt.title("Linear Regression Plot")
plt.legend()
plt.grid(True)

plt.show()
