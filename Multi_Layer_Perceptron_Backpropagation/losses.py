import numpy as np


class MSELoss:

    def forward(self, y_pred, y_true):

        return np.mean((y_pred - y_true) ** 2)

    def derivative(self, y_pred, y_true):

        return 2 * (y_pred - y_true) / y_true.shape[0]