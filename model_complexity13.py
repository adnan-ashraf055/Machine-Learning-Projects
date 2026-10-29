
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# 1. Generate classification dataset
X, y = make_classification(
    n_samples=500,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# 2. Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 3. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 4. Test different hidden layer sizes
hidden_sizes = [2, 4, 8, 16, 32, 64, 128, 256]

train_errors = []
test_errors = []
accuracy_gaps = []

for size in hidden_sizes:

    model = Sequential([
        Input(shape=(10,)),
        Dense(size, activation="relu"),
        Dense(size, activation="relu"),
        Dense(1, activation="sigmoid")
    ])

    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss="binary_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        X_train,
        y_train,
        epochs=30,
        batch_size=32,
        validation_data=(X_test, y_test),
        verbose=0
    )

    # Evaluate final model
    train_loss, train_acc = model.evaluate(
        X_train, y_train, verbose=0
    )

    test_loss, test_acc = model.evaluate(
        X_test, y_test, verbose=0
    )

    train_error = 1 - train_acc
    test_error = 1 - test_acc
    gap = test_acc - train_acc

    train_errors.append(train_error)
    test_errors.append(test_error)
    accuracy_gaps.append(gap)

    print(
        f"Hidden size={size:3d}  "
        f"Train error={train_error:.4f}  "
        f"Test error={test_error:.4f}  "
        f"Accuracy gap={gap:+.4f}"
    )

# 5. Find hidden size with lowest test error
best_index = np.argmin(test_errors)
best_size = hidden_sizes[best_index]

print(f"\nBest hidden layer size: {best_size}")
print(f"Lowest test error: {test_errors[best_index]:.4f}")

# 6. Plot training and test errors
plt.figure(figsize=(10, 6))

plt.plot(
    hidden_sizes,
    train_errors,
    "bo-",
    label="Training Error",
    linewidth=2
)

plt.plot(
    hidden_sizes,
    test_errors,
    "rs--",
    label="Test Error",
    linewidth=2
)

plt.axvline(
    x=best_size,
    color="green",
    linestyle=":",
    alpha=0.7,
    label=f"Lowest test error: {best_size}"
)

plt.xlabel("Hidden Layer Size (Model Complexity)")
plt.ylabel("Error Rate")
plt.title("Model Complexity: Underfitting vs Overfitting")

plt.xscale("log", base=2)
plt.xticks(hidden_sizes, labels=hidden_sizes)

plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
