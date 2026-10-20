import numpy as np


def logistic_lattice(K, D, N, kaneko=False):
    def F(u):
        f = K * u * (1.0 - u)
        if kaneko:
            return (1.0 - D) * f + 0.5 * D * (np.roll(f, 1) + np.roll(f, -1)) - u
        return f - u + D * (np.roll(u, 1) - 2.0 * u + np.roll(u, -1))
    return F


def logistic_lattice_jacobian_symbol(K, D, x, q, kaneko=False):
    fp = K * (1.0 - 2.0 * x)
    if kaneko:
        return -1.0 + fp * (1.0 - 2.0 * D * np.sin(0.5 * q) ** 2)
    return fp - 1.0 - 4.0 * D * np.sin(0.5 * q) ** 2


def cubic_lattice(a, b3, D, N, b2=0.0):
    def F(u):
        return a * u + b2 * u ** 2 + b3 * u ** 3 + D * (np.roll(u, 1) - 2.0 * u + np.roll(u, -1))
    return F


def henon_lattice(a, b, D, N):
    def F(z):
        x = z[:N]
        y = z[N:]
        xn = 1.0 - a * x ** 2 + y + D * (np.roll(x, 1) - 2.0 * x + np.roll(x, -1))
        yn = b * x
        return np.concatenate([xn - x, yn - y])
    return F


def sine_lattice(K, D, N):
    def F(u):
        return K * np.sin(np.pi * u) - u + D * (np.roll(u, 1) - 2.0 * u + np.roll(u, -1))
    return F
