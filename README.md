# Training-free computations (revision)

Reproduces the training-free results added in the revision of
"Learning Memory Operators with Physics-Informed Neural Networks".
Requirements: Python 3.12, NumPy 2.4, SciPy 1.17 (single CPU core, ~1 h total).

- `check_reproduction.py`: reproduces the condition numbers and cosines of
  Tables 10, 11 and 14 and sigma_min/kappa of Table 18.
- `run_M_study.py` -> Table 2 (controlled study of M)
- `run_second_benchmark.py` -> Table 8 (profiles g1, g2)
- `run_kappa_vs_error.py` -> Table 12 (kappa vs E_K, E_param)
- `run_ut_perturbation.py` -> Table 13 (error in the state derivative)
- `run_window.py` -> Table 15 and fixed-40 comparison (stretched exponential)
- `run_spline_knots.py` -> Table 6 (spline knots, exact state)
- `run_fixed_budget.py` -> fixed observation budget table
- `run_discretization.py` -> discretization and quadrature table (N_t, N_q)

Core modules: `core.py` (quadrature, Jacobians), `fit.py` (reference VarPro
solver, metrics), `exps.py` (experiment definitions; EPS = 1e-3, 10 draws).
Seeds are fixed inside `exps.py` and the run scripts.
