import os
import numpy as np
from common import RESULTS
from flm.models import logistic_lattice, sine_lattice, henon_lattice
from flm.solvers import run_fft_hls, run_ours, run_short_memory, run_nested_mesh

T = 2 ** 14
KEEP = 2500
N = 16


def cubic_lattice(beta, D):
    def F(u):
        return (beta - 1.0) * u - u ** 3 + D * (np.roll(u, 1) - 2.0 * u + np.roll(u, -1))
    return F


MODELS = [
    ("Logistic", 0.5, lambda: logistic_lattice(3.4, 0.1, N, kaneko=True), 1),
    ("Sine", 0.8, lambda: sine_lattice(1.0, 0.05, N), 1),
    ("Cubic", 0.5, lambda: cubic_lattice(2.4, 0.05), 1),
    ("Henon", 0.9, lambda: henon_lattice(1.0, 0.3, 0.02, N), 2),
]

COLUMNS = [
    ("exact", lambda F, u0, nu, rec: run_fft_hls(F, u0, nu, T, record=rec)),
    ("WHC8", lambda F, u0, nu, rec: run_ours(F, u0, nu, T, delta=1e-8, record=rec)),
    ("WHC4", lambda F, u0, nu, rec: run_ours(F, u0, nu, T, delta=1e-4, record=rec)),
    ("SM32", lambda F, u0, nu, rec: run_short_memory(F, u0, nu, T, L=32, record=rec)),
    ("SM512", lambda F, u0, nu, rec: run_short_memory(F, u0, nu, T, L=512, record=rec)),
    ("NM512", lambda F, u0, nu, rec: run_nested_mesh(F, u0, nu, T, W=512, record=rec)),
]


def main():
    rng = np.random.default_rng(11)
    data = {}
    rec = list(range(T - KEEP, T + 1))
    for name, nu, mk, comp in MODELS:
        F = mk()
        if comp == 2:
            u0 = np.concatenate([0.1 + 0.01 * rng.standard_normal(N), np.zeros(N)])
        else:
            u0 = (0.3 if name != "Cubic" else 0.1) + 0.01 * rng.standard_normal(N)
        for col, run in COLUMNS:
            with np.errstate(all="ignore"):
                _, r = run(F, u0, nu, rec)
            X = np.array([r[k] for k in rec])
            data["%s|%s" % (name, col)] = X[:, [0, N]] if comp == 2 else X[:, [0, 1]]
            print(name, col, np.nanmin(X[:, 0]), np.nanmax(X[:, 0]), flush=True)
    np.savez_compressed(os.path.join(RESULTS, "attractors.npz"), **data)


if __name__ == "__main__":
    main()
