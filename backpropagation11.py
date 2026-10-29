
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD

# 1. Load Iris dataset
iris = load_iris()
X = iris.data
y = iris.target

# 2. One-hot encode the target labels
try:
    encoder = OneHotEncoder(sparse_output=False)
except TypeError:
    # Compatibility with older scikit-learn versions
    encoder = OneHotEncoder(sparse=False)

y_encoded = encoder.fit_transform(y.reshape(-1, 1))

# 3. Split the dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y_encoded,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 4. Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 5. Build the MLP model
model = Sequential([
    Input(shape=(4,)),
    Dense(10, activation="relu"),
    Dense(10, activation="relu"),
    Dense(3, activation="softmax")
])

# 6. Compile the model
model.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 7. Train the model using backpropagation
history = model.fit(
    X_train,
    y_train,
    epochs=100,
    batch_size=8,
    validation_data=(X_test, y_test),
    verbose=0
)

print("\nTraining completed!")

# 8. Evaluate the model
train_loss, train_acc = model.evaluate(
    X_train, y_train, verbose=0
)

test_loss, test_acc = model.evaluate(
    X_test, y_test, verbose=0
)

print(f"\nTraining Accuracy: {train_acc:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")
print(f"Training Loss: {train_loss:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# 9. Plot loss and accuracy
plt.figure(figsize=(12, 4))

# Loss graph
plt.subplot(1, 2, 1)

plt.plot(history.history["loss"], label="Training Loss")
plt.plot(history.history["val_loss"], label="Validation Loss")

plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Loss (Backpropagation)")

# Accuracy graph
plt.subplot(1, 2, 2)

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Model Accuracy")

plt.tight_layout()
plt.show()

# 10. Explain backpropagation
print("\nBackpropagation Steps per Epoch:")
print("1. Forward pass: Calculate predictions.")
print("2. Compute loss: Calculate categorical cross-entropy.")
print("3. Backward pass: Calculate gradients using the chain rule.")
print("4. Update weights: Adjust weights using the optimizer.")

# 11. Predict an example flower
sample = X_test[:1]

probabilities = model.predict(sample, verbose=0)
predicted_class = np.argmax(probabilities, axis=1)[0]

print("\nExample Prediction:")
print("Predicted flower:", iris.target_names[predicted_class])
print("Class probabilities:", probabilities[0])
