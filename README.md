# Fractional Lattice Maps: Asymptotic Cycles and Efficient Memory Compression

This repository contains our code of the research *Asymptotic Cycles, Resonant Bifurcations, and Optimal Memory Compression for Fractional Lattice Maps*. A fractional lattice map of order 0 < ν < 1 on a lattice with state space X = R^M is
```
u(n) = u0 + sum_{j=0}^{n-1} w_{n-1-j} F(u(j)),   w_m = Γ(m+ν) / (Γ(ν) Γ(m+1)),
```
which is the solution of the Caputo difference equation of order ν with right side F.

## Files and Content

| Path | Purpose |
|------|---------|
| `flm/kernel.py` | Exact kernel `w_m` (recurrence and asymptotic series), Beta moment quadrature with the a priori rule, weighted Hankel compression (WHC), precise verification of the relative kernel error, and the weighted Hankel lower bound |
| `flm/solvers.py` | The seven solvers compared in our research: direct summation (DS), short memory (SM), nested meshes (NM), FFT convolution of Hairer, Lubich and Schlichte (FFT), sum of exponentials of Jiang et al. (SOE), fast and oblivious convolution quadrature of Schädle, López-Fernández and Lubich (FOCQ), and WHC |
| `flm/models.py` | Lattice models: logistic lattice with diffusive or Kaneko coupling, sine, cubic and Hénon lattices |
| `flm/theory.py` | Stability region, cycle equations, Newton solver for asymptotic cycles, sublattice symbol and its determinant, zero counts, period doubling thresholds by branch switching |
| `experiments/exp_kernel.py` | Numbers stored per site and kernel error curves |
| `experiments/exp_theory.py` | Cascade thresholds, memory imprint and finite time exponents, lattice phase diagram, checkerboard amplitude |
| `experiments/exp_cascade_long.py` | Cascade thresholds of order 0.5 up to the period 512 cycle |
| `experiments/exp_attractors.py` | Attractors of four lattices for six memory treatments |
| `experiments/exp_benchmark.py` | Time, peak memory and error of all methods |
| `experiments/make_tables.py`, `experiments/make_figures.py` | rows and EPS figures from the stored results |
| `results/` | All raw results as JSON and NPZ files |


## Usage

Python 3.10 or newer with the packages numpy>=1.26, scipy>=1.11, matplotlib>=3.8 are required. Run all experiments, tables and figures:
```
sh run_all.sh
```
The benchmark runs on one thread; `experiments/common.py` sets the BLAS thread variables to one. On an Apple M2 the full run takes a few hours, most of it in the benchmark and the phase diagram. A minimal use of our WHC:
```python
import numpy as np
from flm.kernel import compress
from flm.solvers import soe_loop
from flm.models import logistic_lattice

nu, T, N = 0.5, 2 ** 16, 256
ker = compress(nu, T, 1e-8)
F = logistic_lattice(2.8, 0.01, N)
u0 = 0.5 + 0.01 * np.random.default_rng(0).standard_normal(N)
uT, rec = soe_loop(F, u0, ker["s"], ker["c"], T, record=[T])
print(ker["r"], ker["err"])
```
`compress(nu, T, delta)` returns positive rates `s` and weights `c` with `max_{1<=m<=T} |sum_j c_j exp(-s_j m) - w_m| / w_m <= delta`, the order `r`, and the verified error `err`. Asymptotic cycles of the fractional logistic map:
```python
from flm.theory import cascade_thresholds
print(cascade_thresholds(0.5, 7))
```
prints the parameters at which the cycles of period 1, 2, 4, ..., 64 lose stability.

## Notes

* The benchmark uses the diffusive logistic lattice with K = 2.8, D = 0.01, order 0.5 and tolerance 1e-8, with T = 2^10, 2^12, 2^14, 2^16, 2^20 on N = 1024 sites and N = 1, 16, 256, 4096 sites at T = 2^13. A run is skipped when its time, predicted from the previous run of the same method by the cost law of `experiments/common.py`, exceeds 300 s, or when its predicted memory exceeds 5 GB.
* Times are wall clock times on one core, including kernel construction for WHC and FOCQ; the parameters of SOE and FOCQ are tuned once to the tolerance and the tuning time is excluded.
* Memory is the peak allocation traced by `tracemalloc` during a whole run.
* Errors are measured against the FFT solver for T <= 2^16 and against SOE with tolerance 1e-12 for larger T.
* Random initial data use fixed seeds.
