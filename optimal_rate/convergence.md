# Optimal-rate computations

The target is second order in nominal h. The Neumann width compares bounds for the same finite-domain eigenvalue. Dirichlet and whole-space gaps also include truncation error; they need not decay to zero on a fixed domain. Tables use nominal h EOCs; convergence.csv also contains actual h_max EOCs.

A separate geometry-only full-core competitor is recorded in [geometric_certificate.csv](geometric_certificate.csv); it avoids potential and Lp quadrature.

All reported bounds use floating-point quadrature and eigensolves; they are numerical evidence, not interval-certified enclosures.

## 2d_one_centre

| h | triangles | patch radius | d_h | EOC d_h | Gamma_h | EOC Gamma_h | U_N - L_opt | EOC width | U_D - L_opt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 7560 | 2.95e-07 | 0.027 | — | 0.0575 | — | 0.623 | — | 0.624 |
| 0.4 | 33440 | 1.72e-08 | 0.00683 | 1.98 | 0.016 | 1.84 | 0.216 | 1.53 | 0.217 |
| 0.2 | 149120 | 1.05e-09 | 0.0017 | 2.01 | 0.00422 | 1.92 | 0.0623 | 1.79 | 0.0641 |
| 0.1 | 660480 | 6.43e-11 | 0.000425 | 2 | 0.00109 | 1.96 | 0.0166 | 1.91 | 0.0184 |

Plot: [convergence_2d_one_centre.png](convergence_2d_one_centre.png)

## 2d_two_centre

| h | triangles | patch radius | d_h | EOC d_h | Gamma_h | EOC Gamma_h | U_N - L_opt | EOC width | U_D - L_opt |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 0.8 | 15120 | 1.83e-07 | 0.0908 | — | 0.128 | — | 1.19 | — | 1.19 |
| 0.4 | 66880 | 1.02e-08 | 0.024 | 1.92 | 0.0352 | 1.86 | 0.394 | 1.6 | 0.395 |
| 0.2 | 298240 | 5.66e-10 | 0.00626 | 1.94 | 0.00925 | 1.93 | 0.113 | 1.8 | 0.114 |
| 0.1 | 1320960 | 3.35e-11 | 0.00159 | 1.97 | 0.00238 | 1.96 | 0.0299 | 1.92 | 0.0309 |

Plot: [convergence_2d_two_centre.png](convergence_2d_two_centre.png)

