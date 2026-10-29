
import numpy as np
import matplotlib.pyplot as plt


# Activation functions
def step(x):
    return np.where(x >= 0, 1, 0)


def sigmoid(x):
    # Numerically stable sigmoid
    x = np.asarray(x, dtype=float)
    result = np.empty_like(x)

    positive = x >= 0
    result[positive] = 1 / (1 + np.exp(-x[positive]))

    exp_x = np.exp(x[~positive])
    result[~positive] = exp_x / (1 + exp_x)

    return result


def tanh(x):
    return np.tanh(x)


def relu(x):
    return np.maximum(0, x)


# Generate input values
x = np.linspace(-10, 10, 1000)

# Compute outputs
y_step = step(x)
y_sigmoid = sigmoid(x)
y_tanh = tanh(x)
y_relu = relu(x)

# Activation function details
functions = [
    ("Step", y_step),
    ("Sigmoid", y_sigmoid),
    ("Tanh", y_tanh),
    ("ReLU", y_relu)
]

derivatives = {
    "Step": "Not differentiable at x = 0",
    "Sigmoid": "sigmoid(x) * (1 - sigmoid(x))",
    "Tanh": "1 - tanh(x)^2",
    "ReLU": "0 if x < 0, 1 if x > 0"
}

# Plot activation functions
plt.figure(figsize=(14, 10))

for i, (name, y_vals) in enumerate(functions, 1):
    plt.subplot(2, 2, i)

    plt.plot(x, y_vals, "b-", linewidth=2)
    plt.axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    plt.axvline(x=0, color="gray", linestyle="--", alpha=0.5)

    plt.title(f"{name} Activation")
    plt.xlabel("Input (x)")
    plt.ylabel("Output")
    plt.grid(True, alpha=0.3)

    if name == "ReLU":
        plt.ylim(-1, 11)
    else:
        plt.ylim(-1.5, 1.5)

    # Print properties
    print(f"\n{name}:")
    print(
        f"  Output range: "
        f"[{y_vals.min():.2f}, {y_vals.max():.2f}]"
    )
    print(f"  Derivative: {derivatives[name]}")

plt.tight_layout()
plt.show()
