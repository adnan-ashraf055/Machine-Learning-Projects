
import numpy as np
import matplotlib.pyplot as plt

from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.optimizers import SGD

# 1. Generate binary classification dataset
X, y = make_classification(
    n_samples=1000,
    n_features=10,
    n_informative=5,
    n_redundant=2,
    random_state=42
)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y
)

# 2. Scale features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 3. Build neural network
def build_model():
    model = Sequential([
        Input(shape=(10,)),
        Dense(16, activation="relu"),
        Dense(16, activation="relu"),
        Dense(1, activation="sigmoid")
    ])
    return model


# 4. SGD without momentum
model_sgd = build_model()

model_sgd.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.0),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

h_sgd = model_sgd.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0
)


# 5. SGD with momentum
model_mom = build_model()

model_mom.compile(
    optimizer=SGD(learning_rate=0.01, momentum=0.9),
    loss="binary_crossentropy",
    metrics=["accuracy"]
)

h_mom = model_mom.fit(
    X_train,
    y_train,
    epochs=30,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=0
)


# 6. Find first epoch reaching 90% training accuracy
def epochs_to_target(accuracy_history, target=0.90):
    for epoch, accuracy in enumerate(accuracy_history, start=1):
        if accuracy >= target:
            return str(epoch)
    return "Never"


sgd_epochs = epochs_to_target(h_sgd.history["accuracy"])
mom_epochs = epochs_to_target(h_mom.history["accuracy"])

# 7. Compare results
print(
    f"{'Optimizer':20} {'Train Acc':12} "
    f"{'Test Acc':12} {'Epochs to 90%':16}"
)
print("-" * 65)

print(
    f"{'SGD (momentum=0)':20} "
    f"{h_sgd.history['accuracy'][-1]:12.4f} "
    f"{h_sgd.history['val_accuracy'][-1]:12.4f} "
    f"{sgd_epochs:>16}"
)

print(
    f"{'SGD (momentum=0.9)':20} "
    f"{h_mom.history['accuracy'][-1]:12.4f} "
    f"{h_mom.history['val_accuracy'][-1]:12.4f} "
    f"{mom_epochs:>16}"
)


# 8. Plot training loss and accuracy
plt.figure(figsize=(12, 5))

# Loss comparison
plt.subplot(1, 2, 1)
plt.plot(h_sgd.history["loss"], label="SGD Loss")
plt.plot(h_mom.history["loss"], label="SGD + Momentum Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Training Loss Comparison")

# Accuracy comparison
plt.subplot(1, 2, 2)
plt.plot(h_sgd.history["accuracy"], label="SGD Accuracy")
plt.plot(h_mom.history["accuracy"], label="SGD + Momentum Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True, alpha=0.3)
plt.title("Training Accuracy Comparison")

plt.tight_layout()
plt.show()
