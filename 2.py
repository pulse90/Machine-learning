import matplotlib.pyplot as plt
import numpy as np

# Data
xpt = np.array([0, 10])
ypt = np.array([0, 120])

# Plot
plt.figure()
plt.plot(xpt, ypt)
plt.xlabel("x")
plt.ylabel("y")
plt.title("Simple Plot")

# Save to file
plt.savefig("plot.png")   # <- added line to store the file

# Optional: close the figure to free memory
plt.close()
