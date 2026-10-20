import math
import numpy as np
from scipy.special import roots_jacobi, roots_legendre
from .kernel import weights, partial_sum, moment_sum, compress, max_relative_error, _psi, _phi_left


def _as_state(u0):
    return np.array(u0, dtype=float).reshape(-1)


def _keys(record):
    return set(record) if record is not None else set()


def run_direct(F, u0, nu, T, record=None):
    u0 = _as_state(u0)
    W = weights(nu, np.arange(T))
    G = np.empty((T, u0.size))
    u = u0.copy()
    keys, rec = _keys(record), {}
    if 0 in keys:
        rec[0] = u.copy()
    for n in range(T):
        G[n] = F(u)
        u = u0 + W[n::-1] @ G[: n + 1]
        if n + 1 in keys:
            rec[n + 1] = u.copy()
    return u, rec


def run_short_memory(F, u0, nu, T, L=1000, record=None):
    u0 = _as_state(u0)
    L = max(1, min(L, T))
    wrev = weights(nu, np.arange(L))[::-1].copy()
    buf = np.zeros((2 * L, u0.size))
    u = u0.copy()
    keys, rec = _keys(record), {}
    if 0 in keys:
        rec[0] = u.copy()
    for n in range(T):
        f = F(u)
        p = n % L
        buf[p] = f
        buf[p + L] = f
        u = u0 + wrev @ buf[p + 1: p + L + 1]
        if n + 1 in keys:
            rec[n + 1] = u.copy()
    return u, rec


def _nested_edges(n, W, jmax):
    e = [n + 1, max(0, 2 * ((n + 1 - W) // 2))]
    for j in range(1, jmax + 1):
        h = 1 << j
        e.append(max(0, 2 * h * ((e[-1] - h * W) // (2 * h))))
    return e


def run_nested_mesh(F, u0, nu, T, W=32, record=None):
    u0 = _as_state(u0)
    jmax = max(1, int(math.ceil(math.log2(max(2.0, T / W)))) + 1)
    w0 = weights(nu, np.arange(W + 2))
    store = {}
    u = u0.copy()
    keys, rec = _keys(record), {}
    if 0 in keys:
        rec[0] = u.copy()
    for n in range(T):
        f = F(u)
        store[n] = f
        e = _nested_edges(n, W, jmax)
        idx = np.arange(e[1], n + 1)
        S = w0[n - idx] @ np.array([store[i] for i in idx])
        A, H = [], []
        for j in range(1, jmax + 1):
            h = 1 << j
            if e[j] <= e[j + 1]:
                continue
            a = np.arange(e[j + 1], e[j], h)
            A.append(a)
            H.append(np.full(a.size, h))
        if A:
            A = np.concatenate(A)
            H = np.concatenate(H)
            hi = n - A + 1
            lo = n - A - H + 1
            s0 = partial_sum(nu, hi) - partial_sum(nu, lo)
            s1 = moment_sum(nu, hi) - moment_sum(nu, lo)
            beta = ((n - A) * s0 - s1) / H
            alpha = s0 - beta
            FA = np.array([store[a] for a in A])
            FB = np.array([store[a + h] for a, h in zip(A, H)])
            S = S + alpha @ FA + beta @ FB
        u = u0 + S
        en = _nested_edges(n + 1, W, jmax)
        for i in list(store.keys()):
            if i >= en[1]:
                continue
            v = min((i & -i).bit_length() - 1 if i > 0 else jmax, jmax)
            if v < 1 or i < en[v + 1]:
                del store[i]
        if n + 1 in keys:
            rec[n + 1] = u.copy()
    return u, rec


def run_fft_hls(F, u0, nu, T, base=32, record=None):
    u0 = _as_state(u0)
    Wt = weights(nu, np.arange(T + 1))
    G = np.zeros((T, u0.size))
    acc = np.zeros((T, u0.size))
    keys, rec = _keys(record), {}
    if 0 in keys:
        rec[0] = u0.copy()

    def solve(lo, hi):
        if hi - lo <= base:
            for n in range(lo, hi):
                S = acc[n]
                if n > lo:
                    S = S + Wt[n - 1 - lo:: -1][: n - lo] @ G[lo:n]
                un = u0 + S
                if n in keys and n > 0:
                    rec[n] = un.copy()
                G[n] = F(un)
            return
        mid = (lo + hi) // 2
        solve(lo, mid)
        m, L = mid - lo, hi - mid
        nfft = 1
        while nfft < m + L:
            nfft <<= 1
        fk = np.fft.rfft(Wt[: m + L], nfft)
        fg = np.fft.rfft(G[lo:mid], nfft, axis=0)
        acc[mid:hi] += np.fft.irfft(fk[:, None] * fg, nfft, axis=0)[m - 1: m - 1 + L]
        solve(mid, hi)

    solve(0, T)
    uT = u0 + Wt[T - 1:: -1][:T] @ G
    if T in keys:
        rec[T] = uT.copy()
    return uT, rec


def soe_loop(F, u0, s, c, T, record=None):
    u0 = _as_state(u0)
    rho = np.exp(-np.asarray(s))[:, None]
    c = np.asarray(c, dtype=float)
    H = np.zeros((rho.shape[0], u0.size))
    u = u0.copy()
    keys, rec = _keys(record), {}
    if 0 in keys:
        rec[0] = u.copy()
    for n in range(T):
        f = F(u)
        u = u0 + f + c @ H
        H += f
        H *= rho
        if n + 1 in keys:
            rec[n + 1] = u.copy()
    return u, rec


def run_ours(F, u0, nu, T, delta=1e-8, record=None, kernel=None):
    if kernel is None:
        kernel = compress(nu, T, delta)
    return soe_loop(F, u0, kernel["s"], kernel["c"], T, record)


def jiang_nodes(nu, T, no, ns, nl, Nup):
    M = int(math.ceil(math.log2(max(2, T))))
    cnu = math.sin(math.pi * nu) / math.pi
    x, w = roots_jacobi(no, 0.0, -nu)
    s0 = 2.0 ** (-M)
    sg = 0.5 * s0 * (x + 1.0)
    S = [sg]
    C = [cnu * w * (0.5 * s0) ** (1 - nu) * _phi_left(nu, sg)]
    for j in range(-M, Nup + 1):
        n = ns if j < 0 else nl
        xx, ww = roots_legendre(n)
        a, b = 2.0 ** j, 2.0 ** (j + 1)
        s = 0.5 * (a + b) + 0.5 * (b - a) * xx
        S.append(s)
        C.append(cnu * 0.5 * (b - a) * ww * _psi(nu, s))
    return np.concatenate(S), np.concatenate(C)


def _sample_lags(T):
    a = np.arange(1, min(T, 2000) + 1)
    b = np.round(np.geomspace(2000, T, 3000)).astype(np.int64) if T > 2000 else np.array([], dtype=np.int64)
    return np.unique(np.concatenate([a, b]))


def _sampled_error(nu, s, c, m):
    approx = np.exp(-np.outer(m.astype(float), s)) @ c
    w = weights(nu, m)
    return float(np.max(np.abs(approx - w) / w))


def jiang_kernel(nu, T, delta, start=(24, 24, 24, 6)):
    m = _sample_lags(T)
    par = list(start)
    lows = [1, 1, 1, 0]

    def err(p):
        s, c = jiang_nodes(nu, T, p[0], p[1], p[2], p[3])
        return _sampled_error(nu, s, c, m)

    while err(par) > delta:
        par = [p + 4 for p in par[:3]] + [par[3] + 1]
    changed = True
    while changed:
        changed = False
        for i in range(4):
            lo, hi = lows[i], par[i]
            while lo < hi:
                mid = (lo + hi) // 2
                trial = list(par)
                trial[i] = mid
                if err(trial) <= delta:
                    hi = mid
                else:
                    lo = mid + 1
            if lo != par[i]:
                par[i] = lo
                changed = True
    s, c = jiang_nodes(nu, T, *par)
    while max_relative_error(nu, s, c, 1, T) > delta:
        par = [p + 1 for p in par[:3]] + [par[3]]
        s, c = jiang_nodes(nu, T, *par)
    return dict(s=s, c=c, no=par[0], ns=par[1], nl=par[2], Nup=par[3], K=s.size)


def run_soe_jiang(F, u0, nu, T, record=None, kernel=None, delta=1e-8):
    if kernel is None:
        kernel = jiang_kernel(nu, T, delta)
    return soe_loop(F, u0, kernel["s"], kernel["c"], T, record)


def talbot_nodes(K, Tl, mu_c=8.0, nuT=0.6):
    mu = mu_c / Tl
    th = np.arange(0, K + 1) * math.pi / (K + 1)
    lam = np.empty(K + 1, dtype=complex)
    dg = np.empty(K + 1, dtype=complex)
    lam[0] = mu
    dg[0] = mu * 1j * nuT
    t = th[1:]
    lam[1:] = mu * (t / np.tan(t) + 1j * nuT * t)
    dg[1:] = mu * (1.0 / np.tan(t) - t / np.sin(t) ** 2 + 1j * nuT)
    return lam, -1j * dg / (2.0 * (K + 1))


def focq_level(nu, B, K, l, mu_c=8.0, nuT=0.6):
    lam, wq = talbot_nodes(K, 2.0 * B ** l, mu_c, nuT)
    coef = wq * lam ** (-nu)
    coef[1:] *= 2.0
    return coef, 1.0 / (1.0 - lam)


def focq_levels(T, B):
    return max(2, int(math.ceil(math.log(max(T, 2) / 2.0) / math.log(B))) + 1)


def focq_weight_error(nu, B, K, Lmax, mu_c=8.0, nuT=0.6):
    err = 0.0
    for l in range(2, Lmax + 1):
        coef, rho = focq_level(nu, B, K, l, mu_c, nuT)
        lags = np.unique(np.round(np.geomspace(B ** (l - 1), 2 * B ** l - 2, 400)).astype(np.int64))
        approx = np.real(np.exp(np.outer(lags + 1, np.log(rho))) @ coef)
        w = weights(nu, lags)
        err = max(err, float(np.max(np.abs(approx - w) / w)))
    return err


def focq_parameters(nu, T, delta, Bs=(3, 4, 5, 6, 8, 10), mus=(4.0, 6.0, 8.0, 12.0, 16.0)):
    best = None
    for B in Bs:
        Lmax = focq_levels(T, B)
        for mu_c in mus:
            if mu_c / (2.0 * B ** 2) >= 0.5:
                continue
            for K in range(2, 160):
                if focq_weight_error(nu, B, K, Lmax, mu_c) <= delta:
                    states = 5 * (K + 1) * (Lmax - 1) * 2 + 4 * B
                    if best is None or states < best[0]:
                        best = (states, dict(B=B, K=K, Lmax=Lmax, mu_c=mu_c, states=states))
                    break
    return best[1]


def run_focq(F, u0, nu, T, record=None, params=None, delta=1e-8):
    if params is None:
        params = focq_parameters(nu, T, delta)
    B, K, Lmax = params["B"], params["K"], params["Lmax"]
    u0 = _as_state(u0)
    N = u0.size
    nlev = Lmax - 1
    coef = np.empty((nlev, K + 1), dtype=complex)
    rho = np.empty((nlev, K + 1), dtype=complex)
    for i, l in enumerate(range(2, Lmax + 1)):
        coef[i], rho[i] = focq_level(nu, B, K, l, params.get("mu_c", 8.0))
    crho = coef * rho
    rr = rho[None, :, :, None]
    Y = np.zeros((2, nlev, K + 1, N), dtype=complex)
    Z = np.zeros((3, nlev, K + 1, N), dtype=complex)
    D = 2 * B
    wd = weights(nu, np.arange(D))
    buf = np.zeros((2 * D, N))
    q = [0] * (Lmax + 2)
    L = 1
    blk = [B ** (l - 1) for l in range(2, Lmax + 1)]
    sup = [B ** l for l in range(2, Lmax + 1)]
    lv = np.arange(nlev)
    u = u0.copy()
    keys, rec = _keys(record), {}
    if 0 in keys:
        rec[0] = u.copy()
    for n in range(T):
        old_b = [q[k] * B ** k for k in range(Lmax + 1)]
        if n >= 1:
            if 2 * B ** L == n + 1:
                L += 1
            k = 1
            while (n + 1) % (B ** k) == 0 and k < L:
                q[k] += 1
                k += 1
        new_b = [q[k] * B ** k for k in range(Lmax + 1)]
        for i, l in enumerate(range(2, Lmax + 1)):
            if new_b[l] != old_b[l]:
                Y[0, i] = Y[1, i]
                Y[1, i] = 0.0
            if new_b[l - 1] != old_b[l - 1]:
                Y[0 if old_b[l - 1] < new_b[l] + sup[i] else 1, i] += Z[0, i]
                Z[0, i] = Z[1, i]
                Z[1, i] = Z[2, i]
                Z[2, i] = 0.0
        f = F(u)
        p = n % D
        buf[p] = f
        buf[p + D] = f
        nd = n - new_b[1] + 1
        wv = np.zeros(D)
        wv[:nd] = wd[:nd]
        S = wv[::-1] @ buf[p + 1: p + D + 1]
        S = S + np.einsum("lk,lkn->n", crho, Y[0] + Y[1]).real
        Y *= rr
        Z *= rr
        zi = np.array([(n - new_b[l - 1]) // blk[i] for i, l in enumerate(range(2, Lmax + 1))])
        Z[zi, lv] += rho[:, :, None] * f[None, None, :]
        u = u0 + S
        if n + 1 in keys:
            rec[n + 1] = u.copy()
    return u, rec
