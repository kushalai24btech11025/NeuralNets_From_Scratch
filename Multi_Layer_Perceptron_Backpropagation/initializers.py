import numpy as np


def random_init(in_dim, out_dim):

    return np.random.randn(in_dim, out_dim) * 0.01


def xavier_init(in_dim, out_dim):

    limit = np.sqrt(6 / (in_dim + out_dim))
    return np.random.uniform(-limit, limit, (in_dim, out_dim))


def he_init(in_dim, out_dim):

    std = np.sqrt(2 / in_dim)
    return np.random.randn(in_dim, out_dim) * std