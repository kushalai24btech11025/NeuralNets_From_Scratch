import numpy as np

from activations import Sigmoid, Tanh, ReLU, LeakyReLU
from losses import MSELoss
from initializers import random_init, xavier_init, he_init
from optimizers import (
    SGD, Momentum, Adam,
    Nesterov, AdaGrad, RMSProp, Muon
)


class MLP:

    def __init__(
        self,
        layer_sizes,
        activations,
        loss='mse',
        learning_rate=0.01,
        optimizer='sgd',
        batch_size=32,
        weight_init='xavier',
        regularization=None,
        lambda_reg=0.01
    ):

        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.regularization = regularization
        self.lambda_reg = lambda_reg

        # initialize activation functions
        self.activations = self._init_activations(activations)

        # initialize loss
        self.loss_fn = MSELoss()

        # optimizer
        self.optimizer = self._init_optimizer(optimizer)

        # parameters
        self.weights = []
        self.biases = []

        for i in range(len(layer_sizes) - 1):

            in_dim = layer_sizes[i]
            out_dim = layer_sizes[i + 1]

            if weight_init == 'xavier':
                W = xavier_init(in_dim, out_dim)

            elif weight_init == 'he':
                W = he_init(in_dim, out_dim)

            else:
                W = random_init(in_dim, out_dim)

            b = np.zeros((1, out_dim))

            self.weights.append(W)
            self.biases.append(b)

    # --------------------------

    def _init_activations(self, names):

        act_map = {
            'sigmoid': Sigmoid(),
            'tanh': Tanh(),
            'relu': ReLU(),
            'leaky_relu': LeakyReLU()
        }

        return [act_map[name] for name in names]

    # --------------------------

    def _init_optimizer(self, name):

        if name == 'sgd':
            return SGD()

        if name == 'momentum':
            return Momentum()

        if name == 'adam':
            return Adam()

        if name == 'nesterov':
            return Nesterov()

        if name == 'adagrad':
            return AdaGrad()

        if name == 'rmsprop':
            return RMSProp()

        if name == 'muon':
            return Muon()

        raise ValueError("Unknown optimizer")

    # --------------------------

    def forward(self, X):

        a = X

        activations = [X]
        zs = []

        for W, b, act in zip(self.weights, self.biases, self.activations):

            z = np.dot(a, W) + b
            a = act.forward(z)

            zs.append(z)
            activations.append(a)

        return activations, zs

    # --------------------------

    def backward(self, X, y):

        activations, zs = self.forward(X)

        grads_w = []
        grads_b = []

        delta = self.loss_fn.derivative(activations[-1], y)

        for l in reversed(range(len(self.weights))):

            dz = delta * self.activations[l].derivative(zs[l])

            dw = np.dot(activations[l].T, dz)
            db = np.sum(dz, axis=0, keepdims=True)

            # regularization
            if self.regularization == 'l2':
                dw += self.lambda_reg * self.weights[l]

            if self.regularization == 'l1':
                dw += self.lambda_reg * np.sign(self.weights[l])

            grads_w.insert(0, dw)
            grads_b.insert(0, db)

            delta = np.dot(dz, self.weights[l].T)

        return grads_w, grads_b

    # --------------------------

    def fit(self, X_train, y_train, X_val=None, y_val=None, epochs=100):

        history = {
            'train_loss': [],
            'val_loss': []
        }

        n = X_train.shape[0]

        for epoch in range(epochs):

            indices = np.random.permutation(n)

            X_train = X_train[indices]
            y_train = y_train[indices]

            for i in range(0, n, self.batch_size):

                X_batch = X_train[i:i+self.batch_size]
                y_batch = y_train[i:i+self.batch_size]

                grads_w, grads_b = self.backward(X_batch, y_batch)

                for j in range(len(self.weights)):
                    dw = grads_w[j]
                    db = grads_b[j]
                    self.weights[j] -= self.learning_rate * dw
                    self.biases[j] -= self.learning_rate * db

            train_pred = self.forward(X_train)[0][-1]
            train_loss = self.loss_fn.forward(train_pred, y_train)
            history['train_loss'].append(train_loss)

            if X_val is not None and y_val is not None:
                val_pred = self.forward(X_val)[0][-1]
                val_loss = self.loss_fn.forward(val_pred, y_val)
                history['val_loss'].append(val_loss)
                val_str = f" Val Loss: {val_loss:.4f}"
            else:
                val_str = ""

            print(
                f"Epoch {epoch+1}/{epochs} "
                f"Train Loss: {train_loss:.4f}"
                + val_str
            )

        if X_val is None or y_val is None:
            return history['train_loss']
        return history

    # --------------------------

    def predict(self, X):

        output = self.forward(X)[0][-1]

        return output

    # --------------------------

    def score(self, X, y):

        preds = self.predict(X)

        mse = np.mean((preds - y) ** 2)

        return mse