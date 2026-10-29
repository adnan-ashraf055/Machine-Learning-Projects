
import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Input, Conv2D, MaxPooling2D,
    Flatten, Dense, Dropout
)
from tensorflow.keras.utils import to_categorical

# 1. Load CIFAR-10 dataset
(X_train, y_train), (X_test, y_test) = cifar10.load_data()

# Use smaller subsets for faster training
X_train = X_train[:10000].astype("float32") / 255.0
X_test = X_test[:2000].astype("float32") / 255.0

y_train = y_train[:10000]
y_test = y_test[:2000]

# 2. One-hot encode labels
y_train_cat = to_categorical(y_train, num_classes=10)
y_test_cat = to_categorical(y_test, num_classes=10)

# CIFAR-10 class names
class_names = [
    "airplane", "automobile", "bird", "cat", "deer",
    "dog", "frog", "horse", "ship", "truck"
]

# 3. Build CNN model
model = Sequential([
    Input(shape=(32, 32, 3)),

    Conv2D(32, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Conv2D(64, (3, 3), activation="relu"),
    MaxPooling2D((2, 2)),

    Flatten(),
    Dense(128, activation="relu"),
    Dropout(0.5),

    Dense(10, activation="softmax")
])

# 4. Compile model
model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# 5. Train the model
history = model.fit(
    X_train,
    y_train_cat,
    epochs=5,
    batch_size=64,
    validation_data=(X_test, y_test_cat),
    verbose=1
)

# 6. Evaluate model
test_loss, test_acc = model.evaluate(
    X_test,
    y_test_cat,
    verbose=0
)

print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# 7. Predict the first 10 test images
images = X_test[:10]
true_classes = y_test[:10].flatten()

predictions = model.predict(images, verbose=0)
pred_classes = np.argmax(predictions, axis=1)

# 8. Display images and predictions
plt.figure(figsize=(14, 6))

for i in range(10):
    plt.subplot(2, 5, i + 1)

    plt.imshow(images[i])

    true_label = true_classes[i]
    predicted_label = pred_classes[i]

    color = "green" if predicted_label == true_label else "red"

    plt.title(
        f"True: {class_names[true_label]}\n"
        f"Pred: {class_names[predicted_label]}",
        color=color,
        fontsize=9
    )

    plt.axis("off")

plt.suptitle("CIFAR-10 CNN Predictions", fontsize=14)
plt.tight_layout()
plt.show()

# 9. Plot training and validation accuracy
plt.figure(figsize=(8, 5))

plt.plot(history.history["accuracy"], label="Training Accuracy")
plt.plot(history.history["val_accuracy"], label="Validation Accuracy")

plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.title("CNN Training History")
plt.legend()
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.show()
