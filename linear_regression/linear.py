# ---------- Read Data from CSV (no external libraries) ----------

x = []
y = []

with open("linear_regression/data.csv", "r") as file:


    header = file.readline()  # skip header line: x,y
    for line in file:
        line = line.strip()
        if line == "":       # skip empty lines
            continue
        parts = line.split(",")
        # assume first column is x, second is y
        x_val = float(parts[0])
        y_val = float(parts[1])
        x.append(x_val)
        y.append(y_val)

# ---------- Helper Functions ----------

def mean(values):
    return sum(values) / len(values)

def linear_regression(x, y):
    x_mean = mean(x)
    y_mean = mean(y)

    num = 0.0
    den = 0.0
    for i in range(len(x)):
        num += (x[i] - x_mean) * (y[i] - y_mean)
        den += (x[i] - x_mean) ** 2

    m = num / den
    b = y_mean - m * x_mean
    return m, b

def predict(x_value, m, b):
    return m * x_value + b

def mean_squared_error(actual, predicted):
    error_sum = 0.0
    for i in range(len(actual)):
        error_sum += (actual[i] - predicted[i]) ** 2
    return error_sum / len(actual)

# ---------- Train Model ----------

m, b = linear_regression(x, y)

print("Slope (m):", m)
print("Intercept (b):", b)

# ---------- Predictions ----------

predicted = []
for xi in x:
    yi_pred = predict(xi, m, b)
    predicted.append(yi_pred)

# ---------- Error Calculation ----------

mse = mean_squared_error(y, predicted)
print("Mean Squared Error (MSE):", mse)

# (Optional) Show actual vs predicted
print("\nActual vs Predicted:")
for xi, yi, yi_pred in zip(x, y, predicted):
    print(f"x={xi}  actual={yi}  predicted={yi_pred}")
