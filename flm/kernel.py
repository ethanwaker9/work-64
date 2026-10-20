import math
import numpy as np
from scipy.special import roots_jacobi, roots_legendre, gamma, beta, gammaln

_BERN = [
    lambda a: a ** 2 - a + 1.0 / 6.0,
    lambda a: a ** 3 - 1.5 * a ** 2 + 0.5 * a,
    lambda a: a ** 4 - 2 * a ** 3 + a ** 2 - 1.0 / 30.0,
    lambda a: a ** 5 - 2.5 * a ** 4 + (5.0 / 3.0) * a ** 3 - a / 6.0,
    lambda a: a ** 6 - 3 * a ** 5 + 2.5 * a ** 4 - 0.5 * a ** 2 + 1.0 / 42.0,
    lambda a: a ** 7 - 3.5 * a ** 6 + 3.5 * a ** 5 - (7.0 / 6.0) * a ** 3 + a / 6.0,
    lambda a: a ** 8 - 4 * a ** 7 + (14.0 / 3.0) * a ** 6 - (7.0 / 3.0) * a ** 4 + (2.0 / 3.0) * a ** 2 - 1.0 / 30.0,
]

_SWITCH = 2048


def _log_gamma_ratio_large(z, a, b):
    out = (a - b) * np.log(z)
    zp = np.ones_like(z)
    for n, B in enumerate(_BERN, start=1):
        zp = zp * z
        out = out + ((-1.0) ** (n + 1)) * (B(a) - B(b)) / (n * (n + 1) * zp)
    return out


def _small_table(nu, mmax):
    w = np.empty(mmax + 1)
    w[0] = 1.0
    if mmax > 0:
        k = np.arange(mmax, dtype=float)
        w[1:] = np.cumprod((k + nu) / (k + 1.0))
    return w


_TABLES = {}


def weights(nu, m):
    m = np.asarray(m, dtype=np.int64)
    key = round(float(nu), 15)
    tab = _TABLES.get(key)
    if tab is None:
        tab = _small_table(nu, _SWITCH)
        _TABLES[key] = tab
    out = np.empty(m.shape, dtype=float)
    small = m <= _SWITCH
    out[small] = tab[m[small]]
    big = ~small
    if np.any(big):
        z = m[big].astype(float)
        out[big] = np.exp(_log_gamma_ratio_large(z, nu, 1.0) - gammaln(nu))
    return out


def partial_sum(nu, m):
    m = np.asarray(m, dtype=np.int64)
    out = np.zeros(m.shape, dtype=float)
    pos = m > 0
    out[pos] = weights(nu + 1.0, m[pos] - 1)
    return out


def moment_sum(nu, m):
    m = np.asarray(m, dtype=np.int64)
    out = np.zeros(m.shape, dtype=float)
    pos = m > 1
    out[pos] = nu * weights(nu + 2.0, m[pos] - 2)
    return out


def _phi_left(nu, s):
    return np.exp(-nu * s) * ((-np.expm1(-s)) / s) ** (-nu)


def _psi(nu, s):
    return np.exp(-nu * s) * (-np.expm1(-s)) ** (-nu)


def _gj_rule(q, nu, s0):
    x, w = roots_jacobi(q, 0.0, -nu)
    s = 0.5 * s0 * (x + 1.0)
    c = w * (0.5 * s0) ** (1.0 - nu)
    return s, c * _phi_left(nu, s)


def _gl_panels(n, xa, xb, P, nu):
    npan = max(1, int(math.ceil((xb - xa) / P)))
    edges = np.linspace(xa, xb, npan + 1)
    x, w = roots_legendre(n)
    S, C = [], []
    for a, b in zip(edges[:-1], edges[1:]):
        xx = 0.5 * (a + b) + 0.5 * (b - a) * x
        s = np.exp(xx)
        S.append(s)
        C.append(0.5 * (b - a) * w * s * _psi(nu, s))
    return np.concatenate(S), np.concatenate(C), npan


def _middle_bound(nu, h, d, n):
    t = 2.0 * d / h
    rho = t + math.sqrt(t * t + 1.0)
    eta = 0.5 * h * (0.5 * (rho + 1.0 / rho) - 1.0)
    g1 = gamma(1.0 - nu)
    pre = (32.0 / 15.0) * 2.0 ** (1.0 - nu) / (math.cos(d) * g1)
    return pre * (g1 + 2.0 * h + 2.0 * eta - math.log(math.cos(d))) * rho ** (-2 * n) / (rho * rho - 1.0)


def apriori_parameters(nu, T, m0, delta, sigma=0.5):
    target = delta / 12.0
    s_plus = 1.0
    while 2.0 * (1.0 - math.exp(-s_plus)) ** (-nu) * math.exp(-m0 * s_plus) > target:
        s_plus *= 1.02
    s_minus = sigma / T
    cphi = (math.pi / (1.0 - math.exp(-math.pi))) ** nu
    g2 = gamma(2.0 - nu)
    q = 1
    while True:
        rho = 4.0 * (2 * q - 1) / sigma
        epsL = 4.0 * (2 * sigma) ** (1 - nu) * cphi / g2 * math.exp(2 * q - 1) * rho ** (-(2 * q - 1)) / (rho - 1.0)
        if epsL <= target and (2.0 + rho) * s_minus / 4.0 <= math.pi:
            break
        q += 1
    Lam = math.log(s_plus / s_minus)
    best = None
    for P in (1.0, 1.5, 2.0, 2.5, 3.0, 4.0, 5.0, 6.0):
        npan = max(1, int(math.ceil(Lam / P)))
        h = Lam / npan
        for d in (0.25 * math.pi, math.pi / 3, 0.375 * math.pi, 5 * math.pi / 12, 0.45 * math.pi):
            n = 1
            while _middle_bound(nu, h, d, n) > target:
                n += 1
            K0 = q + n * npan
            if best is None or K0 < best["K0"]:
                best = dict(s_minus=s_minus, s_plus=s_plus, q=q, n=n, P=h, d=d, npan=npan, K0=K0)
    return best


def beta_quadrature(nu, T, m0, delta):
    p = apriori_parameters(nu, T, m0, delta)
    cnu = math.sin(math.pi * nu) / math.pi
    sL, cL = _gj_rule(p["q"], nu, p["s_minus"])
    sM, cM, _ = _gl_panels(p["n"], math.log(p["s_minus"]), math.log(p["s_plus"]), p["P"] * 1.0000001, nu)
    s = np.concatenate([sL, sM])
    c = cnu * np.concatenate([cL, cM])
    return s, c, p


def kappa(nu):
    return beta(0.5 * (1 - nu), 0.5 * (1 - nu)) / gamma(nu)


def _gram_factor(s, b, nu, M, i0=512, npan=24, ngl=12, rows=128):
    R = np.zeros((0, s.size))
    i = np.arange(0, min(M, i0) + 1, dtype=float)
    for a in range(0, i.size, rows):
        ii = i[a:a + rows]
        blk = ((ii + 0.5) ** (-0.5 * nu))[:, None] * b[None, :] * np.exp(-np.outer(ii, s))
        R = np.linalg.qr(np.vstack([R, blk]), mode="r")
    if M > i0:
        xa, xb = math.log(i0 + 1.0), math.log(M + 1.0)
        edges = np.linspace(xa, xb, npan + 1)
        x, w = roots_legendre(ngl)
        for a, bb in zip(edges[:-1], edges[1:]):
            xx = np.exp(0.5 * (a + bb) + 0.5 * (bb - a) * x)
            ww = 0.5 * (bb - a) * w * xx
            blk = (np.sqrt(ww) * xx ** (-0.5 * nu))[:, None] * b[None, :] * np.exp(-np.outer(xx - 0.5, s))
            R = np.linalg.qr(np.vstack([R, blk]), mode="r")
    return R


def _reduce(s, c, Vt, r, m0):
    rr = np.exp(-s)
    b = np.sqrt(c * rr ** m0)
    U = Vt[:r].T
    X = np.sqrt(-np.expm1(-s))[:, None] * U
    _, sx, Qt = np.linalg.svd(X, full_matrices=False)
    lam = sx ** 2
    sr = -np.log1p(-lam)
    br = Qt @ (U.T @ b)
    return sr, br ** 2 * np.exp(m0 * sr)


def max_relative_error(nu, s, c, m0, T, chunk=512):
    rho = np.exp(-s)
    E = np.exp(-np.outer(np.arange(chunk, dtype=float), s))
    P = np.exp(-m0 * s)
    step = np.exp(-chunk * s)
    err = 0.0
    for a in range(m0, T + 1, chunk):
        L = min(T, a + chunk - 1) - a + 1
        approx = E[:L] @ (c * P)
        w = weights(nu, np.arange(a, a + L))
        err = max(err, float(np.max(np.abs(approx - w) / w)))
        P = P * step
    return err


def compress(nu, T, delta, m0=1, theta=0.002):
    s0, c0, p = beta_quadrature(nu, T, m0, delta)
    rr = np.exp(-s0)
    b = np.sqrt(c0 * rr ** m0)
    M = max(1, (T - m0) // 2)
    R = _gram_factor(s0, b, nu, M)
    _, sv, Vt = np.linalg.svd(R)
    lam = sv ** 2
    K0 = len(s0)
    r = int(np.sum(lam > theta * kappa(nu) * delta))
    r = min(max(r, 1), K0)
    checks = 0
    while True:
        s, c = _reduce(s0, c0, Vt, r, m0)
        if np.all(np.isfinite(s)) and np.all(s > 0) and np.all(c > 0):
            e = max_relative_error(nu, s, c, m0, T)
            checks += 1
            if e <= delta:
                return dict(s=s, c=c, r=r, err=e, K0=K0, lam=lam, params=p, checks=checks)
        if r >= K0:
            e = max_relative_error(nu, s0, c0, m0, T)
            return dict(s=s0, c=c0, r=K0, err=e, K0=K0, lam=lam, params=p, checks=checks + 1)
        r += 1


def hankel_lower_bound(nu, T, delta, m0=1, ref_delta=None):
    if ref_delta is None:
        ref_delta = min(1e-3 * delta, 1e-12)
    s0, c0, _ = beta_quadrature(nu, T, m0, ref_delta)
    rr = np.exp(-s0)
    b = np.sqrt(c0 * rr ** m0)
    M = max(1, (T - m0) // 2)
    R = _exact_gram_factor(s0, b, nu, M)
    sv = np.linalg.svd(R, compute_uv=False) ** 2
    return int(np.sum(sv > kappa(nu) * delta)), sv


def _exact_gram_factor(s, b, nu, M, chunk=100000):
    R = None
    for a in range(0, M + 1, chunk):
        i = np.arange(a, min(M, a + chunk - 1) + 1, dtype=float)
        blk = ((i + 0.5) ** (-0.5 * nu))[:, None] * b[None, :] * np.exp(-np.outer(i, s))
        R = np.linalg.qr(blk if R is None else np.vstack([R, blk]), mode="r")
    return R
