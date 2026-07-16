import numpy as np

class Adaline:
    def __init__(self, learning_rate=1.0, max_iterations=1000):
        """
        Initialize Adaline

        learning_rate: step size for weight updates
        max_iterations: maximum training iterations
        """
        self.learning_rate = learning_rate
        self.max_iterations = max_iterations
        self.w = None
        self.b = 0

    def fit(self, X, y):
        """
        Train Adaline using batch gradient descent

        X : (n_samples , n_features)
        y : (n_samples ,)

        Returns: history dictionary
        """
        n_samples, n_features = X.shape

        self.w = np.zeros(n_features)
        self.b = 0

        mse_history = []

        for epoch in range(self.max_iterations):

            # linear output
            y_pred = np.dot(X, self.w) + self.b

            errors = y - y_pred

            # gradients
            dw = (-2/n_samples) * np.dot(X.T, errors)
            db = (-2/n_samples) * np.sum(errors)

            # update
            self.w = self.w - self.learning_rate * dw
            self.b = self.b - self.learning_rate * db

            mse = np.mean(errors ** 2)
            mse_history.append(mse)

        return {"mse": mse_history}

    def predict(self, X):
        """Return predicted outputs"""
        return np.dot(X, self.w) + self.b

    def score(self, X, y):
        """Return mean squared error"""
        y_pred = self.predict(X)
        return np.mean((y - y_pred) ** 2)