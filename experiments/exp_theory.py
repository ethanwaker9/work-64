import math
import sys
import numpy as np
from scipy.special import gamma
from common import save_json
from flm.kernel import compress
from flm.solvers import soe_loop
from flm.theory import cascade_thresholds, logistic_threshold_formula

EDELMAN_HELMAN = [2.4142136, 3.0315081, 3.1288030, 3.1522546, 3.1569819, 3.1580209, 3.1582410]


def imprint(nu, Kfac=0.6, d0s=(-0.1, 0.05, 0.1), T=2 ** 20, delta=1e-10):
    K = 1.0 + Kfac * 2.0 ** nu
    xs = 1.0 - 1.0 / K
    A = 1.0 - K
    ker = compress(nu, T, delta)
    rec = sorted(set(np.unique(np.round(np.geomspace(1, T, 200)).astype(int)).tolist()))
    out = dict(nu=nu, K=K, A=A, n=rec, curves={})

    def Faug(z):
        u, J = z[0], z[1]
        return np.array([(K - 1.0) * u - K * u * u, ((K - 1.0) - 2.0 * K * u) * J])

    for d0 in d0s:
        _, r = soe_loop(Faug, np.array([xs + d0, 1.0]), ker["s"], ker["c"], T, record=rec)
        u = np.array([r[k][0] for k in rec])
        J = np.array([r[k][1] for k in rec])
        pred = -d0 / (A * gamma(1.0 - nu))
        n = np.array(rec, dtype=float)
        out["curves"][str(d0)] = dict(ratio=((n ** nu) * (u - xs) / pred).tolist(),
                                      jac=((n ** nu) * J * (-A * gamma(1.0 - nu))).tolist(),
                                      ftle=(np.log(np.abs(J)) / n).tolist())
        print(nu, d0, out["curves"][str(d0)]["ratio"][-1], out["curves"][str(d0)]["jac"][-1], flush=True)
    return out


def d2cycle(z, a, d2, nu):
    return ((1 - z) ** nu - a * z) * ((1 + z) ** nu + a * z) + d2 * z * z


def _theta_grid():
    g = list(np.linspace(0.0, 2.0 * math.pi, 6001))
    for c in (0.0, math.pi, 2.0 * math.pi):
        for k in range(1, 14):
            for s in (-1.0, 1.0):
                t = c + s * 10.0 ** (-k)
                if 0.0 < t < 2.0 * math.pi:
                    g.append(t)
    return np.unique(np.array(g))


THETA = _theta_grid()


def two_cycle_mode_zeros(a, d2, nu):
    v = d2cycle(np.exp(1j * THETA), a, d2, nu)
    w = np.unwrap(np.angle(v))
    return int(round((w[-1] - w[0]) / (2.0 * math.pi)))


def two_cycle_mode_stable(a, d2, nu):
    return two_cycle_mode_zeros(a, d2, nu) == 0


def lattice_F(Ks, D, N):
    nK = Ks.size

    def F(u):
        v = u.reshape(nK, N)
        out = Ks[:, None] * v * (1.0 - v) - v + D * (np.roll(v, 1, axis=1) - 2.0 * v + np.roll(v, -1, axis=1))
        return out.reshape(-1)
    return F


def classify(x, N, tol=2e-3):
    if not np.all(np.isfinite(x)) or np.max(np.abs(x)) > 10:
        return 9
    spatial = np.max(np.abs(x - x.mean(axis=1, keepdims=True)))
    temporal = np.max(np.abs(x[-1] - x[-2]))
    alt = np.max(np.abs(x[-1] - x[-3]))
    sign = (-1.0) ** np.arange(N)
    y = x[-1] - x[-1].mean()
    cb = np.max(np.abs(y - sign * np.mean(sign * y)))
    if spatial < tol and temporal < tol:
        return 1
    if spatial < tol and alt < tol:
        return 2
    if cb < tol and alt < tol:
        return 3
    return 4


def phase_diagram(nu=0.5, N=32, nD=26, nK=69, T=2 ** 16):
    Ds = np.linspace(0.0, 0.25, nD)
    Ks = np.linspace(1.6, 3.3, nK)
    q = 2.0 * math.pi * np.arange(N) / N
    s2 = np.unique(np.round(np.sin(0.5 * q) ** 2, 12))
    theory = np.zeros((nD, nK), dtype=int)
    for i, D in enumerate(Ds):
        for j, K in enumerate(Ks):
            lam = 1.0 - K - 4.0 * D * s2
            if np.all((lam > -2.0 ** nu) & (lam < 0)):
                theory[i, j] = 1
            elif K > 1.0 + 2.0 ** nu:
                d2 = (K - 1.0) ** 2 - 4.0 ** nu
                if all(two_cycle_mode_stable(-2.0 ** nu - 4.0 * D * s, d2, nu) for s in s2):
                    theory[i, j] = 2
    ker = compress(nu, T, 1e-8)
    rng = np.random.default_rng(3)
    sim = np.zeros((nD, nK), dtype=int)
    sim2 = np.zeros((nD, nK), dtype=int)
    rec = [T - 3, T - 2, T - 1, T]
    for i, D in enumerate(Ds):
        F = lattice_F(Ks, D, N)
        u0 = np.repeat(1.0 - 1.0 / Ks, N) + 1e-3 * rng.standard_normal(nK * N)
        _, r = soe_loop(F, u0, ker["s"], ker["c"], T, record=rec)
        X = np.array([r[k].reshape(nK, N) for k in rec])
        u1 = 0.3 + 1e-4 * rng.standard_normal(nK * N)
        _, r2 = soe_loop(F, u1, ker["s"], ker["c"], T, record=rec)
        X2 = np.array([r2[k].reshape(nK, N) for k in rec])
        for j in range(nK):
            sim[i, j] = classify(X[:, j, :], N)
            sim2[i, j] = classify(X2[:, j, :], N)
        print("D", round(D, 4), flush=True)
    return dict(nu=nu, N=N, T=T, Ds=Ds.tolist(), Ks=Ks.tolist(), theory=theory.tolist(), sim=sim.tolist(), sim2=sim2.tolist())


def zigzag_amplitude(nu=0.5, D=0.05, N=32, T=2 ** 17, nK=25):
    Kc = 1.0 + 2.0 ** nu - 4.0 * D
    Ks = Kc + np.linspace(0.002, 0.12, nK)
    ker = compress(nu, T, 1e-10)
    sign = (-1.0) ** np.arange(N)
    u0 = np.repeat(1.0 - 1.0 / Ks, N) + 1e-3 * np.tile(sign, nK)
    rec = [T - 1, T]
    _, r = soe_loop(lattice_F(Ks, D, N), u0, ker["s"], ker["c"], T, record=rec)
    X = np.array([r[k].reshape(nK, N) for k in rec])
    amp = 0.5 * np.abs(np.mean(sign[None, :] * (X[1] - X[0]), axis=1))
    lam = 1.0 - Ks - 4.0 * D
    c3 = 2.0 * Ks ** 2 / (Ks - 1.0)
    pred = np.sqrt(np.maximum(-(2.0 ** nu + lam) / c3, 0.0))
    return dict(nu=nu, D=D, N=N, T=T, K=Ks.tolist(), Kc=Kc, amp=amp.tolist(), pred=pred.tolist())


if __name__ == "__main__":
    part = sys.argv[1] if len(sys.argv) > 1 else "all"
    if part in ("cascade", "all"):
        res = {}
        for nu in [0.2, 0.5, 0.8, 1.0]:
            Ks = cascade_thresholds(nu, 7)
            res[str(nu)] = Ks
            print(nu, Ks, flush=True)
        res["edelman_helman_0.5"] = EDELMAN_HELMAN
        res["formula"] = {str(nu): logistic_threshold_formula(nu) for nu in [0.2, 0.5, 0.8, 1.0]}
        save_json("cascade.json", res)
    if part in ("imprint", "all"):
        save_json("imprint.json", [imprint(nu) for nu in [0.3, 0.5, 0.7]])
    if part in ("phase", "all"):
        save_json("phase.json", phase_diagram())
    if part in ("zigzag", "all"):
        save_json("zigzag.json", zigzag_amplitude())
