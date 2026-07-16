import numpy as np


class Sigmoid:

    def forward(self, x):
        return 1 / (1 + np.exp(-x))

    def derivative(self, x):
        s = self.forward(x)
        return s * (1 - s)


class Tanh:

    def forward(self, x):
        return np.tanh(x)

    def derivative(self, x):
        return 1 - np.tanh(x) ** 2


class ReLU:

    def forward(self, x):
        return np.maximum(0, x)

    def derivative(self, x):
        return (x > 0).astype(float)


class LeakyReLU:

    def __init__(self, alpha=0.01):
        self.alpha = alpha

    def forward(self, x):
        return np.where(x > 0, x, self.alpha * x)

    def derivative(self, x):
        return np.where(x > 0, 1, self.alpha)