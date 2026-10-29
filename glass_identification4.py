
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# Load Glass dataset
url = "https://archive.ics.uci.edu/ml/machine-learning-databases/glass/glass.data"

columns = [
    "id", "RI", "Na", "Mg", "Al", "Si",
    "K", "Ca", "Ba", "Fe", "type"
]

df = pd.read_csv(url, header=None, names=columns)

# Remove ID column
df = df.drop(columns=["id"])

# Separate features and target
X = df.drop(columns=["type"])
y = df["type"]

# Split dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Scale the features
scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Experiment with different K values
k_values = range(1, 21)
accuracies = []

for k in k_values:
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred = knn.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    accuracies.append(acc)
    print(f"K={k:2d}  Accuracy={acc:.4f}")

# Find the best K value
best_k = list(k_values)[np.argmax(accuracies)]

print(
    f"\nBest K: {best_k} "
    f"with accuracy {max(accuracies):.4f}"
)

# Train the final KNN model
final_knn = KNeighborsClassifier(n_neighbors=best_k)
final_knn.fit(X_train, y_train)

# Make predictions
y_pred = final_knn.predict(X_test)

# Display classification report
print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# Plot accuracy against K values
plt.figure(figsize=(10, 6))

plt.plot(
    list(k_values),
    accuracies,
    "bo-",
    linewidth=2,
    markersize=8
)

plt.axvline(
    x=best_k,
    color="red",
    linestyle="--",
    alpha=0.5,
    label=f"Best K = {best_k}"
)

plt.xlabel("K Value")
plt.ylabel("Accuracy")
plt.title("Glass Classification: Accuracy vs K")
plt.xticks(list(k_values))
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
