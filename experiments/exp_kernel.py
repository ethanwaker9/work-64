import time
import numpy as np
from common import save_json, TUNED
from flm.kernel import compress, hankel_lower_bound, kappa, weights, max_relative_error

NUS = [0.2, 0.5, 0.8]
TS = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
DELTAS = [1e-4, 1e-6, 1e-8, 1e-10]


def main():
    rows = []
    for nu in NUS:
        for T in TS:
            _, sv = hankel_lower_bound(nu, T, 1e-4, ref_delta=1e-13)
            for d in DELTAS:
                t0 = time.perf_counter()
                out = compress(nu, T, d)
                tc = time.perf_counter() - t0
                lb = int(np.sum(sv > kappa(nu) * d))
                soe = TUNED.soe(nu, T, d)
                try:
                    fq = TUNED.focq(nu, T, d)
                    fstates = fq["states"]
                except Exception:
                    fq, fstates = None, None
                row = dict(nu=nu, T=T, delta=d, r=int(out["r"]), K0=int(out["K0"]), err=out["err"],
                           checks=out["checks"], time=tc, lb=lb, soe=int(soe["K"]),
                           soe_par=[int(soe["no"]), int(soe["ns"]), int(soe["nl"]), int(soe["Nup"])],
                           focq=fstates, focq_par=fq)
                rows.append(row)
                print(row, flush=True)
            save_json("kernel_counts.json", rows)
    nu, T = 0.5, 10 ** 6
    m = np.unique(np.round(np.geomspace(1, T, 3000)).astype(np.int64))
    curves = {}
    w = weights(nu, m)
    for d in [1e-4, 1e-8, 1e-12]:
        out = compress(nu, T, d)
        approx = np.exp(-np.outer(m.astype(float), out["s"])) @ out["c"]
        curves["WHC_%g" % d] = (np.abs(approx - w) / w).tolist()
    soe = TUNED.soe(nu, T, 1e-8)
    approx = np.exp(-np.outer(m.astype(float), soe["s"])) @ soe["c"]
    curves["SOE_1e-08"] = (np.abs(approx - w) / w).tolist()
    curves["m"] = m.tolist()
    save_json("kernel_curves.json", curves)


if __name__ == "__main__":
    main()
