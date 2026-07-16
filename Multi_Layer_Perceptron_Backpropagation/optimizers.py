import numpy as np


class SGD:

    def update(self, params, grads, lr):

        for i in range(len(params)):
            params[i] -= lr * grads[i]


class Momentum:

    def __init__(self, beta=0.9):
        self.beta = beta
        self.v = None

    def update(self, params, grads, lr):

        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        for i in range(len(params)):
            self.v[i] = self.beta * self.v[i] + lr * grads[i]
            params[i] -= self.v[i]


class Nesterov:

    def __init__(self, beta=0.9):
        self.beta = beta
        self.v = None

    def update(self, params, grads, lr):

        if self.v is None:
            self.v = [np.zeros_like(p) for p in params]

        for i in range(len(params)):

            prev_v = self.v[i]
            self.v[i] = self.beta * self.v[i] - lr * grads[i]

            params[i] += -self.beta * prev_v + (1 + self.beta) * self.v[i]


class AdaGrad:

    def __init__(self, eps=1e-8):

        self.eps = eps
        self.G = None

    def update(self, params, grads, lr):

        if self.G is None:
            self.G = [np.zeros_like(p) for p in params]

        for i in range(len(params)):

            self.G[i] += grads[i] ** 2

            params[i] -= lr * grads[i] / (np.sqrt(self.G[i]) + self.eps)


class RMSProp:

    def __init__(self, beta=0.9, eps=1e-8):

        self.beta = beta
        self.eps = eps
        self.s = None

    def update(self, params, grads, lr):

        if self.s is None:
            self.s = [np.zeros_like(p) for p in params]

        for i in range(len(params)):

            self.s[i] = self.beta * self.s[i] + (1 - self.beta) * grads[i] ** 2

            params[i] -= lr * grads[i] / (np.sqrt(self.s[i]) + self.eps)


class Adam:

    def __init__(self, beta1=0.9, beta2=0.999, eps=1e-8):

        self.beta1 = beta1
        self.beta2 = beta2
        self.eps = eps

        self.m = None
        self.v = None
        self.t = 0

    def update(self, params, grads, lr):

        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]
            self.v = [np.zeros_like(p) for p in params]

        self.t += 1

        for i in range(len(params)):

            self.m[i] = self.beta1 * self.m[i] + (1 - self.beta1) * grads[i]
            self.v[i] = self.beta2 * self.v[i] + (1 - self.beta2) * (grads[i] ** 2)

            m_hat = self.m[i] / (1 - self.beta1 ** self.t)
            v_hat = self.v[i] / (1 - self.beta2 ** self.t)

            params[i] -= lr * m_hat / (np.sqrt(v_hat) + self.eps)


class Muon:
    """
    Simple experimental optimizer similar to momentum + adaptive scaling
    """

    def __init__(self, beta=0.9, eps=1e-8):

        self.beta = beta
        self.eps = eps
        self.m = None

    def update(self, params, grads, lr):

        if self.m is None:
            self.m = [np.zeros_like(p) for p in params]

        for i in range(len(params)):

            self.m[i] = self.beta * self.m[i] + grads[i]

            params[i] -= lr * self.m[i] / (np.sqrt(np.abs(self.m[i])) + self.eps)