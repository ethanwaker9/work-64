import os
import sys
import numpy as np
from common import METHODS, COSTS, runner, timed, traced, save_json, load_json, RESULTS
from flm.models import logistic_lattice
from flm.solvers import run_fft_hls, run_soe_jiang, jiang_kernel

NU = 0.5
DELTA = 1e-8
BUDGET = 300.0
MEMCAP = 5.0e9


def problem(N, seed=7):
    rng = np.random.default_rng(seed)
    u0 = 0.5 + 0.01 * rng.standard_normal(N)
    return logistic_lattice(2.8, 0.01, N), u0


def reference(F, u0, T, rec):
    if T <= 2 ** 16:
        return run_fft_hls(F, u0, NU, T, record=rec)[1]
    ker = jiang_kernel(NU, T, 1e-12)
    return run_soe_jiang(F, u0, NU, T, record=rec, kernel=ker)[1]


def mem_estimate(method, T, N):
    if method == "DS":
        return 8.0 * T * N
    if method == "FFT":
        return 40.0 * T * N
    return 0.0


def sweep(name, Ts, Ns):
    path = os.path.join(RESULTS, name)
    rows = load_json(name) if os.path.exists(path) else []
    done = {(r["method"], r["T"], r["N"]) for r in rows}
    calib = {}
    for r in sorted(rows, key=lambda r: (r["N"], r["T"])):
        if not r["skipped"]:
            calib[r["method"]] = (r["time"], r["T"], r["N"])
    for N in Ns:
        for T in Ts:
            F, u0 = problem(N)
            rec = [T // 4, T // 2, T]
            ref = None
            for m in METHODS:
                if (m, T, N) in done:
                    continue
                if m in calib:
                    t0, T0, N0 = calib[m]
                    pred = t0 * COSTS[m](T) / COSTS[m](T0) * N / N0
                else:
                    pred = 0.0
                if pred > BUDGET or mem_estimate(m, T, N) > MEMCAP:
                    row = dict(method=m, T=T, N=N, time=None, mem=None, err=None, skipped=True, pred=pred)
                    rows.append(row)
                    save_json(name, rows)
                    print(row, flush=True)
                    continue
                run = runner(m, NU, T, DELTA)
                (u, out), t = timed(run, F, u0, rec)
                calib[m] = (t, T, N)
                (u2, _), peak = traced(run, F, u0, [T])
                if ref is None:
                    ref = reference(F, u0, T, rec)
                with np.errstate(invalid="ignore"):
                    err = max(float(np.max(np.abs(out[k] - ref[k]))) for k in rec)
                if not np.isfinite(err):
                    err = float("inf")
                row = dict(method=m, T=T, N=N, time=t, mem=peak, err=err, skipped=False)
                rows.append(row)
                save_json(name, rows)
                print(row, flush=True)


if __name__ == "__main__":
    which = sys.argv[1] if len(sys.argv) > 1 else "T"
    if which == "T":
        sweep("bench_T.json", [2 ** 10, 2 ** 12, 2 ** 14, 2 ** 16, 2 ** 20], [1024])
    else:
        sweep("bench_N.json", [2 ** 13], [1, 16, 256, 4096])
