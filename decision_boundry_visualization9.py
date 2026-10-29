
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.linear_model import Perceptron

# Generate 2D linearly separable data
X, y = make_classification(
    n_samples=100,
    n_features=2,
    n_informative=2,
    n_redundant=0,
    n_repeated=0,
    n_classes=2,
    n_clusters_per_class=1,
    class_sep=2.0,
    flip_y=0,
    random_state=42
)

# Train the Perceptron
p = Perceptron(max_iter=1000, tol=1e-3, random_state=42)
p.fit(X, y)

# Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1

xx, yy = np.meshgrid(
    np.linspace(x_min, x_max, 300),
    np.linspace(y_min, y_max, 300)
)

# Predict classes for the mesh grid
grid_points = np.c_[xx.ravel(), yy.ravel()]
Z = p.predict(grid_points).reshape(xx.shape)

# Get weights and bias
w = p.coef_[0]
b = p.intercept_[0]

# Plot decision regions and data points
plt.figure(figsize=(10, 8))

plt.contourf(
    xx, yy, Z,
    alpha=0.3,
    cmap=plt.cm.coolwarm
)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=plt.cm.coolwarm,
    edgecolors="k",
    s=60,
    label="Data points"
)

# Plot decision boundary
if not np.isclose(w[1], 0):
    x_line = np.array([x_min, x_max])
    y_line = -(w[0] * x_line + b) / w[1]
    plt.plot(
        x_line, y_line,
        "k--",
        linewidth=2,
        label="Decision Boundary"
    )
else:
    # Vertical decision boundary
    x_boundary = -b / w[0]
    plt.axvline(
        x=x_boundary,
        color="black",
        linestyle="--",
        linewidth=2,
        label="Decision Boundary"
    )

# Print model details
print("Weights:", w)
print("Bias:", b)
print(
    f"Decision boundary: "
    f"{w[0]:.2f}*x1 + {w[1]:.2f}*x2 + {b:.2f} = 0"
)

if not np.isclose(w[1], 0):
    slope = -w[0] / w[1]
    intercept = -b / w[1]
    print(f"Line equation: x2 = {slope:.2f}*x1 + {intercept:.2f}")
else:
    print(f"Vertical line equation: x1 = {-b / w[0]:.2f}")

# Labels and formatting
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.grid(alpha=0.2)
plt.tight_layout()
plt.show()
