
import numpy as np


class Perceptron:
    def __init__(self, learning_rate=0.01, epochs=100):
        self.lr = learning_rate
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
                linear_output = np.dot(x_i, self.weights) + self.bias
                y_pred = self.activation(linear_output)

                update = self.lr * (y[idx] - y_pred)

                self.weights += update * x_i
                self.bias += update

                if update != 0:
                    errors += 1

            if errors == 0:
                print(f"Converged at epoch {epoch + 1}")
                break

    # Make predictions
    def predict(self, X):
        return np.array([
            self.activation(np.dot(x, self.weights) + self.bias)
            for x in X
        ])


# AND gate
X_and = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_and = np.array([0, 0, 0, 1])

p_and = Perceptron(learning_rate=0.1, epochs=50)
p_and.fit(X_and, y_and)

print("\nAND Gate:")
print("Weights:", p_and.weights)
print("Bias:", p_and.bias)
print("Predictions:", p_and.predict(X_and))
print("Actual output:", y_and)


# OR gate
X_or = np.array([
    [0, 0],
    [0, 1],
    [1, 0],
    [1, 1]
])

y_or = np.array([0, 1, 1, 1])

p_or = Perceptron(learning_rate=0.1, epochs=50)
p_or.fit(X_or, y_or)

print("\nOR Gate:")
print("Weights:", p_or.weights)
print("Bias:", p_or.bias)
print("Predictions:", p_or.predict(X_or))
print("Actual output:", y_or)
