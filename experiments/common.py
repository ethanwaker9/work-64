import os
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
import sys
import time
import json
import tracemalloc
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from flm.kernel import compress
from flm.solvers import (run_direct, run_short_memory, run_nested_mesh, run_fft_hls,
                         run_soe_jiang, run_focq, run_ours, jiang_kernel, focq_parameters)

RESULTS = os.path.join(ROOT, "results")
FIGURES = os.path.join(ROOT, "figures")
os.makedirs(RESULTS, exist_ok=True)
os.makedirs(FIGURES, exist_ok=True)

METHODS = ["DS", "SM", "NM", "FFT", "SOE", "FOCQ", "WHC"]

LABELS = {
    "DS": "DS, direct summation",
    "SM": "SM, short memory",
    "NM": "NM, nested meshes",
    "FFT": "FFT, fast convolution",
    "SOE": "SOE, sum of exponentials",
    "FOCQ": "FOCQ, oblivious quadrature",
    "WHC": "WHC (ours)",
}

COSTS = {
    "DS": lambda T: T * T,
    "SM": lambda T: T,
    "NM": lambda T: T * np.log2(T),
    "FFT": lambda T: T * np.log2(T) ** 2,
    "SOE": lambda T: T,
    "FOCQ": lambda T: T * np.log2(T),
    "WHC": lambda T: T,
}


class Tuned:
    def __init__(self):
        self.cache = {}

    def soe(self, nu, T, delta):
        key = ("soe", nu, T, delta)
        if key not in self.cache:
            self.cache[key] = jiang_kernel(nu, T, delta)
        return self.cache[key]

    def focq(self, nu, T, delta):
        key = ("focq", nu, T, delta)
        if key not in self.cache:
            self.cache[key] = focq_parameters(nu, T, delta)
        return self.cache[key]


TUNED = Tuned()


def runner(method, nu, T, delta, L=1000, W=32):
    if method == "DS":
        return lambda F, u0, rec: run_direct(F, u0, nu, T, record=rec)
    if method == "SM":
        return lambda F, u0, rec: run_short_memory(F, u0, nu, T, L=L, record=rec)
    if method == "NM":
        return lambda F, u0, rec: run_nested_mesh(F, u0, nu, T, W=W, record=rec)
    if method == "FFT":
        return lambda F, u0, rec: run_fft_hls(F, u0, nu, T, record=rec)
    if method == "SOE":
        ker = TUNED.soe(nu, T, delta)
        return lambda F, u0, rec: run_soe_jiang(F, u0, nu, T, record=rec, kernel=ker)
    if method == "FOCQ":
        par = TUNED.focq(nu, T, delta)
        return lambda F, u0, rec: run_focq(F, u0, nu, T, record=rec, params=par)
    if method == "WHC":
        return lambda F, u0, rec: run_ours(F, u0, nu, T, delta=delta, record=rec)
    raise ValueError(method)


def timed(fn, *args):
    t0 = time.perf_counter()
    out = fn(*args)
    return out, time.perf_counter() - t0


def traced(fn, *args):
    tracemalloc.start()
    tracemalloc.reset_peak()
    out = fn(*args)
    _, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return out, peak


def save_json(name, obj):
    with open(os.path.join(RESULTS, name), "w") as f:
        json.dump(obj, f, indent=1)


def load_json(name):
    with open(os.path.join(RESULTS, name)) as f:
        return json.load(f)
