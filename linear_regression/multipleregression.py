# ===============================
# Multiple Linear Regression from scratch
# - Reads dataset from CSV
# - No NumPy, no pandas, no sklearn
# ===============================

# ---------- 1. Read dataset (CSV) ----------
def read_dataset(path):
    X = []  # feature rows (without intercept)
    Y = []  # target values (last column)

    # ✅ use the path passed as argument
    with open(path, "r") as f:
        header = f.readline()  # skip header line
        for line in f:
            line = line.strip()
            if not line:
                continue  # skip empty lines
            parts = line.split(",")
            # convert all values on the row to float
            nums = [float(v) for v in parts]
            X.append(nums[:-1])   # all except last = features
            Y.append(nums[-1])    # last = target
    return X, Y


# ---------- 2. Matrix operations ----------

def transpose(M):
    # Transpose of matrix M
    return [list(row) for row in zip(*M)]

def matmul(A, B):
    # Matrix multiplication: A(m x n) * B(n x p) = C(m x p)
    m, n = len(A), len(A[0])
    n2, p = len(B), len(B[0])
    if n != n2:
        raise ValueError("Incompatible matrix sizes for multiplication")
    result = [[0.0] * p for _ in range(m)]
    for i in range(m):
        for j in range(p):
            s = 0.0
            for k in range(n):
                s += A[i][k] * B[k][j]
            result[i][j] = s
    return result

def inverse(M):
    # Inverse of matrix M using Gauss-Jordan elimination
    n = len(M)
    # Create augmented matrix [M | I]
    aug = [row[:] + [0.0] * n for row in M]
    for i in range(n):
        aug[i][n + i] = 1.0

    # Gauss-Jordan
    for col in range(n):
        # Find pivot row
        pivot_row = None
        for r in range(col, n):
            if abs(aug[r][col]) > 1e-12:
                pivot_row = r
                break
        if pivot_row is None:
            raise ValueError("Matrix is singular and cannot be inverted")

        # Swap pivot row into place
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]

        # Normalize pivot row
        pivot = aug[col][col]
        for j in range(2 * n):
            aug[col][j] /= pivot

        # Eliminate this column in other rows
        for r in range(n):
            if r != col:
                factor = aug[r][col]
                if abs(factor) > 1e-12:
                    for j in range(2 * n):
                        aug[r][j] -= factor * aug[col][j]

    # Extract inverse matrix (right half)
    inv = [row[n:] for row in aug]
    return inv


# ---------- 3. Fit linear regression (Normal Equation) ----------

def fit_linear_regression(X_raw, Y):
    # Add intercept term (column of 1s)
    X = [[1.0] + row for row in X_raw]   # shape: (m x (n+1))
    Y_col = [[y] for y in Y]            # shape: (m x 1)

    XT = transpose(X)                   # (n+1 x m)
    XT_X = matmul(XT, X)                # (n+1 x n+1)
    XT_X_inv = inverse(XT_X)            # (n+1 x n+1)
    XT_Y = matmul(XT, Y_col)            # (n+1 x 1)

    # β = (XᵀX)^(-1) XᵀY
    beta = matmul(XT_X_inv, XT_Y)       # (n+1 x 1)
    return beta  # list of lists


# ---------- 4. Prediction, MSE, R² ----------

def predict(X_raw, beta):
    # Add intercept and multiply
    X = [[1.0] + row for row in X_raw]
    preds_col = matmul(X, beta)
    return [row[0] for row in preds_col]

def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n

def r2(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    return 1 - ss_res / ss_tot


# ---------- 5. Main program ----------

def main():
    # ✅ Put the correct relative/absolute path here:
    # If dataset.csv is in SAME FOLDER as this .py file:
    file_path = "linear_regression/dataset.csv"
    # If it's in linear_regression/dataset.csv, use:
    # file_path = "linear_regression/dataset.csv"

    # Step 1: Read data
    X_raw, Y = read_dataset(file_path)
    print("Loaded", len(X_raw), "rows with", len(X_raw[0]), "features each.")

    # Step 2: Fit model
    beta = fit_linear_regression(X_raw, Y)

    print("\nCoefficients (beta):")
    print("Intercept (b0):", beta[0][0])
    for i in range(1, len(beta)):
        print(f"b{i} (feature {i}):", beta[i][0])

    # Step 3: Predict on training data (for demo)
    y_pred = predict(X_raw, beta)

    # Step 4: Evaluate
    print("\nPredictions vs Actual:")
    for yt, yp in zip(Y, y_pred):
        print(f"Actual: {yt:.3f}  |  Predicted: {yp:.3f}")

    print("\nMean Squared Error (MSE):", mse(Y, y_pred))
    print("R^2 Score:", r2(Y, y_pred))


if __name__ == "__main__":
    main()
