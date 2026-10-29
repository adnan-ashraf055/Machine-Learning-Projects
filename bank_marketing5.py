
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    classification_report
)

# Load the Bank Marketing dataset
df = pd.read_csv("bank-additional-full.csv", sep=";")

# Clean column names
df.columns = df.columns.str.strip()

# Convert target column safely
if "y" not in df.columns:
    raise ValueError("The dataset must contain a 'y' target column.")

if df["y"].dtype == object:
    df["y"] = (
        df["y"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": 1, "no": 0})
    )
else:
    df["y"] = pd.to_numeric(df["y"], errors="coerce")

# Remove rows with missing target values
df = df.dropna(subset=["y"]).copy()
df["y"] = df["y"].astype(int)

# Select numeric features, excluding the target
numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [col for col in numeric_cols if col != "y"]

if not feature_cols:
    raise ValueError("No numeric feature columns were found.")

X = df[feature_cols].replace([np.inf, -np.inf], np.nan)
y = df["y"]

# Remove missing feature values
valid_rows = X.notna().all(axis=1)
X = X.loc[valid_rows]
y = y.loc[valid_rows]

# Check that enough data and both classes remain
if len(df) < 2 or y.nunique() < 2:
    raise ValueError(
        "The dataset needs at least two valid rows and both target classes."
    )

print("Dataset loaded successfully!")
print("Rows:", len(X))
print("Features:", feature_cols)
print("\nTarget distribution:")
print(y.value_counts())

# Split into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

# Create and train the Decision Tree
dt = DecisionTreeClassifier(
    max_depth=5,
    criterion="entropy",
    random_state=42
)

dt.fit(X_train, y_train)

# Make predictions
y_pred = dt.predict(X_test)

# Evaluate the model
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, zero_division=0)

print(f"\nAccuracy: {acc:.4f}")
print(f"Precision: {prec:.4f}")

print("\nClassification Report:")
print(
    classification_report(
        y_test,
        y_pred,
        zero_division=0
    )
)

# Display top 5 important features
feat_imp = sorted(
    zip(X.columns, dt.feature_importances_),
    key=lambda item: item[1],
    reverse=True
)

print("\nTop 5 Features:")
for name, importance in feat_imp[:5]:
    print(f"{name}: {importance:.4f}")

# Visualize the Decision Tree
plt.figure(figsize=(20, 12))

plot_tree(
    dt,
    filled=True,
    feature_names=X.columns.tolist(),
    class_names=["No", "Yes"],
    rounded=True,
    max_depth=3,
    fontsize=9
)

plt.title("Bank Marketing - Decision Tree Classifier")
plt.tight_layout()
plt.show()
