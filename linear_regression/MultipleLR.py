import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split

# Load CSV
df = pd.read_csv("linear_regression/dataset.csv")

# Prepare features (all except last column) and target (last column)
X = df.iloc[:, :-1].values
Y = df.iloc[:, -1].values

print("X shape:", X.shape)   # (rows, columns)
print("Y shape:", Y.shape)   # (rows,)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X, Y, test_size=0.2, random_state=42
)

# Create and train the model
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test data
y_pred = model.predict(X_test)

# Show predictions + error
print("Predicted:", y_pred)
print("Actual:", y_test)
print("MSE:", mean_squared_error(y_test, y_pred))

# Plot (works if X has only 1 feature)
plt.scatter(X_test[:,0], y_test, color="blue", label="data points")
plt.plot(X_test[:,0], y_pred, color="red", label="prediction line")
plt.legend()
plt.show()