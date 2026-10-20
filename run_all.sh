#!/bin/sh
set -e
cd "$(dirname "$0")/experiments"
python3 exp_kernel.py
python3 exp_theory.py cascade
python3 exp_cascade_long.py
python3 exp_theory.py imprint
python3 exp_theory.py zigzag
python3 exp_theory.py phase
python3 exp_attractors.py
python3 exp_benchmark.py T
python3 exp_benchmark.py N
python3 make_tables.py
python3 make_figures.py
cd ../figures
for f in *.eps; do epstopdf "$f"; done
