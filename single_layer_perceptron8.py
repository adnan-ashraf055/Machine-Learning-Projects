
import numpy as np


class Perceptron:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = None

    # Step activation function
    def activation(self, x):
        return 1 if x >= 0 else 0

    # Train the perceptron
    def fit(self, X, y):
        n_samples, n_features = X.shape

        self.weights = np.zeros(n_features)
        self.bias = 0.0

        for epoch in range(self.epochs):
            errors = 0

            for idx, x_i in enumerate(X):
                linear = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear)

                update = self.lr * (y[idx] - y_pred)

                self.weights += update * x_i
                self.bias += update

                if update != 0:
                    errors += 1

            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                return True

        return False

    # Predict output
    def predict(self, X):
        predictions = []

        for x_i in X:
            linear = np.dot(x_i, self.weights) + self.bias
            predictions.append(self.activation(linear))

        return np.array(predictions)


# XOR dataset (linearly inseparable)
X_xor = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_xor = np.array([0, 1, 1, 0])

print("XOR Problem - Single Layer Perceptron")
print("=" * 40)
print(f"Input:\n{X_xor}")
print(f"Expected: {y_xor}")

# Create and train the perceptron
p = Perceptron(lr=0.1, epochs=200)

converged = p.fit(X_xor, y_xor)

# Make predictions
predictions = p.predict(X_xor)

# Display results
print(f"\nWeights: {p.weights}")
print(f"Bias: {p.bias}")
print(f"Predictions: {predictions}")
print(f"Correct: {np.array_equal(predictions, y_xor)}")
print(f"Converged: {converged}")

print(
    "\nA single-layer perceptron cannot solve XOR "
    "because XOR is not linearly separable."
)

