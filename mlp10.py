
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import Adam

# 1. Generate non-linear dataset
X, y = make_moons(
    n_samples=500,
    noise=0.2,
    random_state=42
)

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 2. Build MLP with one hidden layer
model = Sequential([
    Input(shape=(2,)),
    Dense(16, activation="relu"),
    Dense(1, activation="sigmoid")
])

# 3. Compile model
model.compile(
    optimizer=Adam(learning_rate=0.01),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 4. Train model
history = model.fit(
    X_train,
    y_train,
    epochs=50,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# 5. Evaluate model
train_loss, train_acc = model.evaluate(
    X_train, y_train, verbose=0
)

test_loss, test_acc = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 6. Create mesh grid for decision boundary
x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5

xx, yy = np.meshgrid(
    np.arange(x_min, x_max, 0.02),
    np.arange(y_min, y_max, 0.02)
)

# 7. Predict classes for the mesh grid
grid_points = np.c_[xx.ravel(), yy.ravel()]

probabilities = model.predict(grid_points, verbose=0)
Z = (probabilities.ravel() >= 0.5).astype(int)
Z = Z.reshape(xx.shape)

# 8. Plot decision boundary and training history
plt.figure(figsize=(12, 5))

# Decision boundary
plt.subplot(1, 2, 1)

plt.contourf(
    xx,
    yy,
    Z,
    alpha=0.3,
    cmap=plt.cm.coolwarm
)

plt.scatter(
    X[:, 0],
    X[:, 1],
    c=y,
    cmap=plt.cm.coolwarm,
    edgecolors="k",
    s=35
)

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("MLP Decision Boundary")

# Accuracy history
plt.subplot(1, 2, 2)

plt.plot(
    history.history["accuracy"],
    label="Training Accuracy"
)

plt.plot(
    history.history["val_accuracy"],
    label="Validation Accuracy"
)

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.title("Training History")

plt.tight_layout()
plt.show()
