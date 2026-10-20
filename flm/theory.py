import math
import numpy as np
from scipy.optimize import brentq
from scipy.special import gamma


def boundary_point(nu, theta):
    z = np.exp(1j * np.asarray(theta))
    return (1.0 - z) ** nu / z


def in_stability_region(nu, lam):
    lam = np.asarray(lam, dtype=complex)
    a = np.angle(lam) % (2.0 * math.pi)
    psi = a - math.pi
    ok_arg = np.abs(psi) < math.pi - 0.5 * nu * math.pi
    rad = np.where(ok_arg, (2.0 * np.cos(np.clip(psi / (2.0 - nu), -0.5 * math.pi, 0.5 * math.pi))) ** nu, 0.0)
    return ok_arg & (np.abs(lam) < rad)


def cycle_coefficients(nu, T):
    om = np.exp(2j * math.pi / T)
    k = np.arange(T)
    S = np.zeros(T)
    for l in range(1, T):
        S += np.real(om ** (-l * k) * (1.0 - om ** l) ** (1.0 - nu))
    return S / T


def cycle_residual(F, p, nu):
    T = p.shape[0]
    S = cycle_coefficients(nu, T)
    Fp = np.array([F(p[j]) for j in range(T)])
    res = [Fp.sum(axis=0)]
    for r in range(T - 1):
        acc = sum(S[k] * Fp[(r - k) % T] for k in range(T))
        res.append(p[(r + 1) % T] - p[r] - acc)
    return np.concatenate([np.atleast_1d(x) for x in res])


def solve_cycle(F, dF, p0, nu, tol=1e-14, maxit=100):
    p = np.array(p0, dtype=float)
    T = p.shape[0]
    shape = p.shape
    S = cycle_coefficients(nu, T)
    for _ in range(maxit):
        Fp = np.array([F(p[j]) for j in range(T)])
        J = [np.atleast_2d(dF(p[j])) for j in range(T)]
        M = J[0].shape[0]
        res = cycle_residual(F, p, nu)
        Jac = np.zeros((T * M, T * M))
        for j in range(T):
            Jac[0:M, j * M:(j + 1) * M] = J[j]
        for r in range(T - 1):
            row = slice((r + 1) * M, (r + 2) * M)
            Jac[row, ((r + 1) % T) * M:((r + 1) % T + 1) * M] += np.eye(M)
            Jac[row, r * M:(r + 1) * M] -= np.eye(M)
            for k in range(T):
                j = (r - k) % T
                Jac[row, j * M:(j + 1) * M] -= S[k] * J[j]
        dp = np.linalg.solve(Jac, -res)
        p = p + dp.reshape(shape)
        if np.max(np.abs(dp)) < tol:
            break
    return p


def symbol(z, Bs, nu):
    T = len(Bs)
    Bs = [np.atleast_2d(B) for B in Bs]
    M = Bs[0].shape[0]
    om = np.exp(2j * math.pi / T)
    j = np.arange(T)
    beta = [sum(Bs[jj] * om ** (-jj * k) for jj in range(T)) / T for k in range(T)]
    Smat = np.zeros((T * M, T * M), dtype=complex)
    for l in range(T):
        zl = om ** l * z
        Smat[l * M:(l + 1) * M, l * M:(l + 1) * M] += (1.0 - zl) ** nu * np.eye(M)
        for k in range(T):
            Smat[l * M:(l + 1) * M, k * M:(k + 1) * M] -= zl * beta[(k - l) % T]
    return Smat


def symbol_det(z, Bs, nu):
    return np.linalg.det(symbol(z, Bs, nu))


def zeros_inside(Bs, nu, r=0.999999, n=4096):
    th = np.linspace(0.0, 2.0 * math.pi, n + 1)
    vals = np.array([symbol_det(r * np.exp(1j * t), Bs, nu) for t in th])
    ang = np.unwrap(np.angle(vals))
    return int(round((ang[-1] - ang[0]) / (2.0 * math.pi)))


def min_det_circle(Bs, nu, n=2048):
    th = np.linspace(0.0, 2.0 * math.pi, n, endpoint=False)
    vals = np.array([abs(symbol_det(np.exp(1j * t), Bs, nu)) for t in th])
    return float(vals.min()), float(th[vals.argmin()])


def logistic_F(K):
    return (lambda x: (K - 1.0) * x - K * x * x), (lambda x: np.array([[K - 1.0 - 2.0 * K * float(np.ravel(x)[0])]]))


def logistic_two_cycle(K, nu):
    s = (K - 1.0 + 2.0 ** nu) / K
    d2 = 2.0 * (K - 1.0) * s / K - s * s
    if d2 < 0:
        return None
    d = math.sqrt(d2)
    return np.array([0.5 * (s - d), 0.5 * (s + d)])


def logistic_threshold_formula(nu):
    return 1.0 + math.sqrt(2.0 ** nu + 2.0 ** (1 + 2 * nu) - 2.0 ** (1 + 1.5 * nu) * math.sin(0.25 * math.pi * nu))


def period_doubling_threshold(nu, T, Klo, Khi, p_guess, F_of_K=logistic_F):
    state = {"p": np.array(p_guess, dtype=float)}

    def g(K):
        F, dF = F_of_K(K)
        p = solve_cycle(lambda x: np.atleast_1d(F(x)), dF, state["p"].reshape(T, -1), nu)
        state["p"] = p
        Bs = [dF(p[j]) for j in range(T)]
        return np.real(symbol_det(np.exp(1j * math.pi / T), Bs, nu))

    Kc = brentq(g, Klo, Khi, xtol=1e-13)
    return Kc, state["p"]


def flip_coefficient_scalar(nu, b2, b3):
    return b3 / 6.0 + b2 * b2 / 2.0 ** (nu + 1)


def memory_imprint_constant(nu, A, d0):
    return -np.linalg.solve(np.atleast_2d(A), np.atleast_1d(d0)) / gamma(1.0 - nu)


def cycle_jacobian(F, dF, p, nu):
    T = p.shape[0]
    S = cycle_coefficients(nu, T)
    J = [np.atleast_2d(dF(p[j])) for j in range(T)]
    M = J[0].shape[0]
    Jac = np.zeros((T * M, T * M))
    for j in range(T):
        Jac[0:M, j * M:(j + 1) * M] = J[j]
    for r in range(T - 1):
        row = slice((r + 1) * M, (r + 2) * M)
        Jac[row, ((r + 1) % T) * M:((r + 1) % T + 1) * M] += np.eye(M)
        Jac[row, r * M:(r + 1) * M] -= np.eye(M)
        for k in range(T):
            j = (r - k) % T
            Jac[row, j * M:(j + 1) * M] -= S[k] * J[j]
    return Jac


def _crossing(nu, T, F_of_K, p, Ka, Kb, steps=400):
    def g(K, state={"p": p}):
        F, dF = F_of_K(K)
        q = solve_cycle(lambda x: np.atleast_1d(F(x)), dF, state["p"], nu)
        state["p"] = q
        return np.real(symbol_det(np.exp(1j * math.pi / T), [dF(q[j]) for j in range(T)], nu)), q

    v0, q = g(Ka)
    Kprev = Ka
    for K in np.linspace(Ka, Kb, steps)[1:]:
        v, q = g(K)
        if np.sign(v) != np.sign(v0):
            state = {"p": q}

            def h(KK):
                F, dF = F_of_K(KK)
                qq = solve_cycle(lambda x: np.atleast_1d(F(x)), dF, state["p"], nu)
                state["p"] = qq
                return np.real(symbol_det(np.exp(1j * math.pi / T), [dF(qq[j]) for j in range(T)], nu))

            Kc = brentq(h, Kprev, K, xtol=1e-13)
            return Kc, state["p"]
        v0, Kprev = v, K
    return None, q


def cascade_thresholds(nu, levels, F_of_K=logistic_F, x_of_K=lambda K: 1.0 - 1.0 / K, K1=None):
    if K1 is None:
        K1 = 1.0 + 2.0 ** nu
    Ks = [K1]
    p = np.array([[x_of_K(K1)]])
    gap = 1.5
    for lev in range(1, levels):
        T = 2 ** lev
        half = T // 2
        F, dF = F_of_K(Ks[-1])
        pd = np.vstack([p, p])
        Jac = cycle_jacobian(lambda x: np.atleast_1d(F(x)), dF, pd, nu)
        _, sv, Vt = np.linalg.svd(Jac)
        phi = Vt[-1].reshape(T, -1)
        h = gap * 1e-3
        found = None
        for sgn in (1.0, -1.0):
            for amp in (1.0, 0.3, 3.0):
                F2, dF2 = F_of_K(Ks[-1] + h)
                try:
                    q = solve_cycle(lambda x: np.atleast_1d(F2(x)), dF2, pd + sgn * amp * math.sqrt(h) * phi, nu)
                except np.linalg.LinAlgError:
                    continue
                if np.all(np.isfinite(q)) and np.max(np.abs(q[:half] - q[half:])) > 1e-7 * math.sqrt(h):
                    found = q
                    break
            if found is not None:
                break
        if found is None:
            break
        Kc, q = _crossing(nu, T, F_of_K, found, Ks[-1] + h, Ks[-1] + gap)
        if Kc is None:
            break
        gap = Kc - Ks[-1]
        Ks.append(Kc)
        p = q
    return Ks
