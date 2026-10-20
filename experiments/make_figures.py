import os
import sys
import json
import math
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d import Axes3D
from scipy.special import gamma

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
RES = os.path.join(ROOT, "results")
FIG = os.path.join(ROOT, "figures")
os.makedirs(FIG, exist_ok=True)

from flm.theory import logistic_threshold_formula

plt.rcParams.update({
    "font.family": "serif",
    "mathtext.fontset": "cm",
    "font.size": 7.5,
    "axes.labelsize": 7.5,
    "axes.titlesize": 7.5,
    "legend.fontsize": 6.2,
    "xtick.labelsize": 6.5,
    "ytick.labelsize": 6.5,
    "lines.linewidth": 1.0,
    "axes.linewidth": 0.6,
    "xtick.major.width": 0.5,
    "ytick.major.width": 0.5,
    "xtick.minor.width": 0.4,
    "ytick.minor.width": 0.4,
    "legend.frameon": False,
    "ps.fonttype": 42,
})

C = {"WHC": "#2a78d6", "SOE": "#eb6834", "FOCQ": "#1baf7a", "FFT": "#eda100",
     "NM": "#e87ba4", "SM": "#008300", "DS": "#6250d6", "LB": "#52514e"}
MK = {"WHC": "o", "SOE": "s", "FOCQ": "D", "FFT": "^", "NM": "v", "SM": "P", "DS": "X"}
NUC = {0.3: "#2a78d6", 0.5: "#eb6834", 0.7: "#1baf7a", 0.2: "#2a78d6", 0.8: "#1baf7a", 1.0: "#6250d6"}


def load(name):
    with open(os.path.join(RES, name)) as f:
        return json.load(f)


def save(fig, name, tight=True):
    fig.savefig(os.path.join(FIG, name + ".eps"), format="eps", bbox_inches="tight" if tight else None, pad_inches=0.03)
    plt.close(fig)


def fig_theory():
    im = load("imprint.json")
    zz = load("zigzag.json")
    ph = load("phase.json")
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 1.85))
    for d in im:
        nu = d["nu"]
        n = np.array(d["n"], dtype=float)
        for k, ls in zip(["-0.1", "0.05", "0.1"], ["-", "--", ":"]):
            r = np.abs(np.array(d["curves"][k]["ratio"]) - 1.0)
            ax[0].loglog(n[1:], r[1:], ls, color=NUC[nu], lw=0.9)
        g = min(nu, 1.0 - nu)
        r1 = np.abs(np.array(d["curves"]["-0.1"]["ratio"]) - 1.0)
        k0 = int(np.searchsorted(n, 1e3))
        nn = np.array([1e3, 1e6])
        ax[0].loglog(nn, 0.5 * r1[k0] * (nn / 1e3) ** (-g), ":", color="0.45", lw=0.8)
    ax[0].set_xlabel(r"$n$")
    ax[0].set_ylabel(r"$|n^{\nu}(u_n-u^\ast)/\Theta-1|$")
    ax[0].set_title("(a) memory imprint", loc="left")
    hs = [Line2D([0], [0], color=NUC[v], lw=1) for v in (0.3, 0.5, 0.7)]
    ax[0].set_ylim(1e-7, 3)
    ax[0].legend(hs, [r"$\nu=0.3$", r"$\nu=0.5$", r"$\nu=0.7$"], loc="lower left", ncol=1, handlelength=1.2, labelspacing=0.2)
    for d in im:
        nu = d["nu"]
        n = np.array(d["n"], dtype=float)
        f = np.abs(np.array(d["curves"]["0.1"]["ftle"]))
        A = abs(d["A"])
        pred = np.abs(-nu * np.log(n) - np.log(A * gamma(1.0 - nu))) / n
        ax[1].loglog(n[5:], f[5:], "-", color=NUC[nu], lw=0.9)
        ax[1].loglog(n[5:], pred[5:], "--", color="0.2", lw=0.6)
    ax[1].set_xlabel(r"$n$")
    ax[1].set_ylabel(r"$|n^{-1}\log|J_n||$")
    ax[1].set_title("(b) finite-time exponents", loc="left")
    ax[1].legend([Line2D([0], [0], color="0.4", lw=0.9), Line2D([0], [0], color="0.2", ls="--", lw=0.6)],
                 ["computed", "prediction"], loc="lower left")
    Ds = np.array(ph["Ds"])
    Ks = np.array(ph["Ks"])
    th = np.array(ph["theory"])
    nu = ph["nu"]
    from matplotlib.colors import ListedColormap
    cmap = ListedColormap(["#ffffff", "#cfe0f5", "#fbd5c4"])
    ax[2].pcolormesh(Ds, Ks, th.T, cmap=cmap, vmin=0, vmax=2, shading="nearest", rasterized=True)
    sim = np.array(ph["sim"])
    sim2 = np.array(ph["sim2"])
    DD, KK = np.meshgrid(Ds, Ks, indexing="ij")
    s = 1.2
    ax[2].plot(DD[sim == 1][::2], KK[sim == 1][::2], ".", ms=s, color=C["WHC"])
    ax[2].plot(DD[sim2 == 2][::2], KK[sim2 == 2][::2], ".", ms=s, color=C["SOE"])
    ax[2].plot(DD[sim == 3][::1], KK[sim == 3][::1], "x", ms=1.6, mew=0.4, color="0.25")
    dl = np.linspace(0, Ds.max(), 60)
    ax[2].plot(dl, 1 + 2 ** nu - 4 * dl, "-", color="k", lw=0.6)
    ax[2].plot(dl, 1 + np.sqrt(4.0 ** nu + 2.0 ** (nu + 2) * dl + 16 * dl ** 2), "--", color="k", lw=0.6)
    ax[2].axhline(logistic_threshold_formula(nu), color="k", ls=":", lw=0.6)
    ax[2].set_xlim(0, Ds.max())
    ax[2].set_ylim(Ks.min(), Ks.max())
    ax[2].set_xlabel(r"$D$")
    ax[2].set_ylabel(r"$K$")
    ax[2].set_title(r"(c) lattice, $\nu=0.5$", loc="left")
    K = np.array(zz["K"])
    ax[3].plot(K - zz["Kc"], zz["pred"], "-", color="k", lw=0.8, label="prediction")
    ax[3].plot(K - zz["Kc"], zz["amp"], "o", ms=2.2, color=C["WHC"], label="WHC simulation")
    ax[3].set_xlabel(r"$K-K_c$")
    ax[3].set_ylabel(r"checkerboard amplitude")
    ax[3].set_title("(d) flip of the zigzag mode", loc="left")
    ax[3].legend(loc="lower right")
    fig.tight_layout(w_pad=0.6)
    save(fig, "theory")


def fig_attractors():
    data = np.load(os.path.join(RES, "attractors.npz"))
    rows = ["Logistic", "Sine", "Cubic", "Henon"]
    rlab = ["logistic lattice", "sine lattice", "cubic lattice", "H\u00e9non lattice"]
    cols = ["exact", "WHC8", "WHC4", "SM32", "SM512", "NM512"]
    ctit = ["exact (FFT)", r"WHC, $\delta=10^{-8}$", r"WHC, $\delta=10^{-4}$", r"SM, $L_{\rm s}=32$", r"SM, $L_{\rm s}=512$", r"NM, $W=512$"]
    fig = plt.figure(figsize=(7.1, 4.55))
    for i, rname in enumerate(rows):
        ref = data["%s|exact" % rname]
        lo = np.nanmin(ref[:-2, 0])
        hi = np.nanmax(ref[:-2, 0])
        for j, cname in enumerate(cols):
            X = data["%s|%s" % (rname, cname)]
            ax = fig.add_subplot(4, 6, 6 * i + j + 1, projection="3d")
            ax.view_init(elev=22, azim=-58)
            x = X[:, 0]
            ok = np.all(np.isfinite(x)) and np.max(np.abs(x)) < 1e3
            if rname == "Henon":
                y = X[:, 1]
                a, b, cc = x[:-1], y[:-1], x[1:]
                labs = [r"$x_n$", r"$y_n$", r"$x_{n+1}$"]
            else:
                a, b, cc = x[:-2], x[1:-1], x[2:]
                labs = [r"$u_n$", r"$u_{n+1}$", r"$u_{n+2}$"]
            col = "0.1" if cname == "exact" else C["WHC"] if cname.startswith("WHC") else C["SM"] if cname.startswith("SM") else C["NM"]
            if ok:
                ax.plot(a, b, cc, ".", ms=0.55, mew=0, color=col)
            else:
                ax.text2D(0.5, 0.55, "diverges", transform=ax.transAxes, ha="center", va="center", fontsize=7, color="0.2")
                ax.set_xlim(lo, hi)
                ax.set_ylim(lo, hi)
                ax.set_zlim(lo, hi)
            ax.tick_params(labelsize=4.2, pad=-3.5)
            for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
                axis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
                axis.pane.set_facecolor("white")
                axis.pane.set_edgecolor("0.85")
                axis._axinfo["grid"]["linewidth"] = 0.25
                axis._axinfo["grid"]["color"] = "0.88"
            ax.set_xlabel(labs[0], fontsize=5.2, labelpad=-9)
            ax.set_ylabel(labs[1], fontsize=5.2, labelpad=-9)
            ax.text2D(0.86, 0.80, labs[2], transform=ax.transAxes, fontsize=5.4, ha="left", va="bottom")
            ax.set_box_aspect(None, zoom=0.82)
            if i == 0:
                ax.set_title(ctit[j], fontsize=6.6, pad=-1)
            if j == 0:
                ax.text2D(-0.13, 0.5, rlab[i], transform=ax.transAxes, rotation=90, ha="center", va="center", fontsize=7)
    fig.subplots_adjust(left=0.025, right=0.985, top=0.965, bottom=0.025, wspace=0.10, hspace=0.04)
    save(fig, "attractors")


def fig_surface3d():
    fig = plt.figure(figsize=(3.45, 2.75))
    ax = fig.add_subplot(111, projection="3d")
    th = np.linspace(1e-4, 2 * math.pi - 1e-4, 400)
    z = np.exp(1j * th)
    cm = plt.get_cmap("Blues")
    for k, nu in enumerate(np.linspace(0.1, 1.0, 10)):
        h = (1 - z) ** nu / z
        h = np.concatenate([[0], h, [0]])
        ax.plot(h.real, h.imag, np.full(h.size, nu), color=cm(0.35 + 0.6 * k / 9), lw=0.8)
    nus = np.linspace(0.1, 1.0, 60)
    for T, col, lab in [(4, "#eb6834", r"$\zeta=\pm i$: period 4"), (3, "#008300", r"$\zeta=e^{\pm 2\pi i/3}$: period 3"),
                        (2, "#e34948", r"$\zeta=-1$: period 2")]:
        for k in range(1, T):
            zz = np.exp(2j * math.pi * k / T)
            hh = (1 - zz) ** nus / zz
            ax.plot(hh.real, hh.imag, nus, color=col, lw=1.6, label=lab if k == 1 else None)
    ax.plot(np.zeros_like(nus), np.zeros_like(nus), nus, color="k", lw=1.6, label=r"$\zeta=1$: fixed points")
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$", labelpad=-5)
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$", labelpad=-5)
    ax.zaxis.set_rotate_label(False)
    ax.set_zlabel(r"$\nu$", labelpad=-3, rotation=0)
    ax.set_zlim(0.1, 1.0)
    ax.tick_params(labelsize=5.5, pad=-2.5)
    ax.view_init(elev=22, azim=-60)
    ax.set_box_aspect(None, zoom=0.9)
    for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
        axis.pane.set_facecolor("white")
        axis.pane.set_edgecolor("0.85")
        axis._axinfo["grid"]["linewidth"] = 0.25
    fig.legend(loc="lower center", ncol=2, fontsize=5.8, handlelength=1.3, columnspacing=0.8, bbox_to_anchor=(0.5, -0.01))
    fig.subplots_adjust(left=-0.05, right=0.92, top=1.1, bottom=0.12)
    save(fig, "surface3d", tight=False)


def fig_benchmark():
    bT = load("bench_T.json")
    bN = load("bench_N.json")
    kc = load("kernel_counts.json")
    fig, ax = plt.subplots(1, 4, figsize=(7.1, 2.2))
    order = ["DS", "SM", "NM", "FFT", "SOE", "FOCQ", "WHC"]
    handles = []
    for m in order:
        rows = sorted([r for r in bT if r["method"] == m and not r["skipped"]], key=lambda r: r["T"])
        if rows:
            T = [r["T"] for r in rows]
            h, = ax[0].loglog(T, [r["time"] for r in rows], "-", marker=MK[m], ms=2.6, color=C[m], label=m)
            handles.append(h)
            ax[1].loglog(T, [r["mem"] / 2 ** 20 for r in rows], "-", marker=MK[m], ms=2.6, color=C[m])
        rows = sorted([r for r in bN if r["method"] == m and not r["skipped"]], key=lambda r: r["N"])
        if rows:
            ax[2].loglog([r["N"] for r in rows], [r["time"] for r in rows], "-", marker=MK[m], ms=2.6, color=C[m])
    ax[0].set_xlabel(r"$T$")
    ax[0].set_ylabel("time (s)")
    ax[0].set_title(r"(a) time, $N=1024$", loc="left")
    ax[1].set_xlabel(r"$T$")
    ax[1].set_ylabel("peak memory (MiB)")
    ax[1].set_title(r"(b) memory, $N=1024$", loc="left")
    ax[2].set_xlabel(r"$N$")
    ax[2].set_ylabel("time (s)")
    ax[2].set_title(r"(c) time, $T=2^{13}$", loc="left")
    rows = sorted([r for r in kc if r["nu"] == 0.5 and r["delta"] == 1e-8], key=lambda r: r["T"])
    T = [r["T"] for r in rows]
    ax[3].loglog(T, [2 * t for t in T], "-", marker="^", ms=2.6, color=C["FFT"], label="FFT")
    ax[3].loglog([r["T"] for r in rows if r["focq"]], [r["focq"] for r in rows if r["focq"]], "-", marker="D", ms=2.6, color=C["FOCQ"], label="FOCQ")
    ax[3].loglog(T, [r["soe"] for r in rows], "-", marker="s", ms=2.6, color=C["SOE"], label="SOE")
    ax[3].loglog(T, [r["r"] for r in rows], "-", marker="o", ms=2.6, color=C["WHC"], label="WHC")
    ax[3].loglog(T, [r["lb"] for r in rows], "--", color=C["LB"], label="bound")
    ax[3].set_xlabel(r"$T$")
    ax[3].set_ylabel("numbers per site")
    ax[3].set_title(r"(d) storage, $\delta=10^{-8}$", loc="left")
    ax[3].set_ylim(4, 1e12)
    ax[3].legend(loc="upper left", ncol=2, handlelength=1.2, columnspacing=0.6, fontsize=6, frameon=False)
    fig.legend(handles=handles, loc="upper center", ncol=7, frameon=False, handlelength=1.6, columnspacing=1.2,
               bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout(w_pad=0.6, rect=(0, 0, 1, 0.91))
    save(fig, "benchmark")


if __name__ == "__main__":
    which = sys.argv[1:] or ["theory", "attractors", "surface3d", "benchmark"]
    for w in which:
        globals()["fig_" + w]()
        print("done", w)
