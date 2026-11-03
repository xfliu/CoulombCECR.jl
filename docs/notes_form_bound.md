# Track A — cheaper certified C_ε (radial Neumann decoupling) and per-k choice of ε

Clone: `/mnt/ssd2T-2/xfliu/CoulombCECR_trackA`, base `abf1d8d`, commits `06c77a6`, `95ec234`, `<final>` (see patch).
All arithmetic is floating point (non-rigorous stage), as in the release.

## 1. Method

**Radial Neumann ground state** (`src/radialbound.jl`, `radial_neumann_ground(dim, Z, ρ; n=40)`).
The lowest Neumann eigenvalue ν of -Δ - Z/|x| on B(0,ρ) ⊂ R^dim is radial. We use Rayleigh–Ritz over polynomials f(r) of degree n
(Legendre basis in 2r/ρ-1), weight r^(dim-1). The numerator is ∫f'² r^(d-1) - Z∫f² r^(d-2) and the denominator is ∫f² r^(d-1).
The (n+3)-point Gauss–Legendre rule is exact for these integrands, and no boundary condition is imposed. The function returns ν(n) and |ν(n)-ν(n/2)|.

**Decoupled form bound** (`decoupled_form_bound(centers, charges, ε, ρ; bdry_dist, domain_is_ball, optimise=true)`). This is the constant C_V(ε) in
(V⁻u,u) ≤ ε‖∇u‖² + C_V‖u‖² on H¹(Ω). Balls B_i = B(a_i,ρ_i) are taken disjoint and inside Ω, and the bound is split as follows:
- On B_i: V⁻ ≤ Z_i/|x-a_i| + b_i with b_i = Σ_{j≠i} Z_j/(|a_i-a_j|-ρ_i). Since u|B_i ∈ H¹(B_i), the Neumann ground state gives ∫_{B_i} Z_i u²/|x-a_i| ≤ ε∫_{B_i}|∇u|² - εν_i∫_{B_i}u², where ν_i = ν_N(B(ρ_i), Z_i/ε).
- On the remainder: V⁻ ≤ s_rem = max(max_i(Z_i/ρ_i + b_i), Σ_j Z_j/dist(a_j,∂Ω)), by the maximum principle (1/|x| is harmonic in 3D and subharmonic in 2D).
- Result: C_V = max(max_i(-εν_i + b_i), s_rem). For Ω = ball with one centre, C_V = -εν_1.

The radii are optimised as ρ_i = t·ρ_max,i: a 100-point grid in t ∈ (0,1], then golden-section refinement.

**Pipeline** (`src/pipeline.jl`):
- New `CaseSpec` fields: `bdry_dist`, `rho_max`, `domain_is_ball`, `eps_grid` (default 0.30:0.025:0.95). They are filled in `scripts/cases.jl`:
  - 2D square: [5], [5]
  - 2D box: [5,5], [2,2]
  - hydrogen: [6], [6], ball
  - H2+: [6,6], [2,2]
- New candidate `radial:direct`: C = C_V(ε-d_h) + d_h (Lemma 3). The analytic and transfer candidates and ladders are kept, and C_cert is the minimum over all candidates.
- `certify_ceps_rung`/`ceps_ladder` take `eigfun`; the pipeline passes `cecr_eigs_case`.
- `CECR_MAX_RUNGS=0` now switches the ladders off. Values ≥ 1 keep the old meaning (ladders with 0..max_rungs intermediate ε).
- `CECR_NO_TRANSFER=1` ignores contbound files. This run used it; the output directory contained none anyway.
- **Per-k choice of ε**: the grid is eps_grid ∪ {spec.eps}, filtered by ε-d_h > 0.05 and d_h/(1-ε) < 0.5. On each grid point:
  - C(ε) = C_V(ε-d_h) + d_h; at ε = spec.eps, min(radial, C_cert) is used.
  - σ = C + 1 - ε, κ = 1 - ε, A_h = Γ_h/(1-ε), ε_h = d_h/(1-ε).

  L_k(ε) is predicted from μ^σ - σ, interpolated piecewise linearly in σ through all CECR solves already done (analytic σ, σ_opt, σ_optP1, σ_optA), with linear extrapolation. Each k gets its own ε_k. One exact solve is then done per distinct ε_k (in all cases two solves were enough), and the exact L_k is reported.
- New JSON keys: `eps_best, C_best, C_best_how, sigma_best, L_best, L_best_pred, L_best_pred_minus_exact, perk_grid, perk_solves, radial_*, C_cert_radial, L1_trivial_radial (= -C_V(1)), timing`.

## 2. Checks

- Tests: 38/38 pass. These are the 23 old tests plus 15 new ones:
  - radial ν for 3D, Z=1, ρ=6: -0.30459813360 (reference -0.3045981336, difference 4e-12).
  - 2D, Z=1, ρ=20: -0.99999999999999 (reference -1).
  - scaling law; ρ=30 gives -1/4.
  - H2+ C_V(0.6; ρ=1.95) = 1.31678 (reference 1.3168); optimised radii give 1.31041 at ρ=2.
  - hydrogen C_V(0.6) = 0.421799 (reference 0.4218).
  - admissibility error; 2D ball sanity check.
  - P1 Galerkin optimal constant ≤ radial C_V on a unit square.
  - `interp_shifted`.
- ν convergence |ν(n)-ν(n/2)| ≤ 7e-14 in every run (n=40).
- Predicted vs exact L_k: |difference| ≤ 2.7e-6 in all 30 (mesh, k) pairs; the largest is hydrogen h140. Linear interpolation in σ of μ^σ-σ is very accurate, even when extrapolating from σ_opt down to σ≈0.41.
- H2+ check: L_1 goes from -0.920 to -0.786 at ε=0.6 (L_opt) and to -0.7817 with per-k ε (prototype: about -0.78).
- Hydrogen h=0.8 check: L_1 goes from -0.3646 to -0.3449 (prototype: about -0.345).

## 3. Results (k = 1; full table in form_bound_summary.csv)

| case / mesh | C_cert old → new [source] | L_opt old → new | per-k: ε_1, L_best,1 | -C_V(1) | U |
|---|---|---|---|---|---|
| 2D square h040 | 1.8458 → 1.8555 [radial] | -1.0880 → -1.0889 | 0.950, -1.0327 | -1.0021 | -0.9983 |
| 2D square h020_fixed | 1.8241 → 1.8287 [analytic ladder4; radial 1.8472] | -1.0363 → -1.0365 | 0.925, -1.0196 | -1.0021 | -0.9989 |
| 2D square h010 | 1.8196 → 1.8196 [analytic ladder4; radial 1.8443] | -1.0221 → -1.0221 | 0.900, -1.0154 | -1.0021 | -0.9991 |
| 2D box h040 | 2.2213 → 2.4050 [radial; analytic ladder 3.95] | -1.4446 → -1.4693 | 0.750, -1.4349 | -1.7925 | -1.3168 |
| 2D box h020_fixed | 2.1296 → 2.1350 [analytic ladder4; radial 2.4002] | -1.3744 → -1.3747 | 0.550, -1.3747 | -1.7925 | -1.3170 |
| 3D H h140 | 0.5687 → 0.5320 [radial] | -0.5331 → -0.5177 | 0.850, -0.4528 | -0.3046 | U_N -0.3034 |
| 3D H h120 | 0.5112 → 0.5028 | -0.4616 → -0.4589 | 0.900, -0.4104 | -0.3046 | U_N -0.3037 |
| 3D H h100 | 0.4696 → 0.4763 | -0.4077 → -0.4092 | 0.900, -0.3748 | -0.3046 | U_N -0.3041 |
| 3D H h080 | 0.4425 → 0.4541 | -0.3646 → -0.3662 | 0.925, -0.3449 | -0.3046 | U_N -0.3044 |
| 3D H2+ h100 | 1.7546 → 1.3248 [radial] | -0.9203 → -0.7860 | 0.675, -0.7817 | -1.2807 | U_D -0.5441 |

How the "old" values were obtained:
- 2D and hydrogen: release `results/{2d,3d}`. These used transfer bounds certified on finer meshes (contbound files) plus ladders.
- H2+: release `results/3d_analytic_only`, the only release H2+ run.

For k = 2, 3, the per-k gain is larger:
- hydrogen h080: L_2 = -0.1942 (old L_opt) → -0.1845 (ε=0.75)
- H2+: -0.7887/-0.7346 → -0.6400/-0.5820
- 2D square h040: -0.3501 → -0.2970

## 4. Timings (HPC, BLAS/OpenMP threads capped = Julia threads)

| run | threads | wall | peak RSS |
|---|---|---|---|
| 2D square h040 / h020_fixed / h010 (ladders max_rungs=3) | 8 | 1:04 / 2:18 / 7:11 | 1.4 / 2.2 / 5.2 GB |
| 2D box h040 / h020_fixed (ladders max_rungs=3) | 8 | 2:36 / 6:58 | 2.2 / 5.0 GB |
| 3D hydrogen h140 / h120 / h100 / h080 (CECR_MAX_RUNGS=0) | 8 | 9:06 / 11:42 / 11:20 / 13:52 | 10.8 / 12.8 / 13.2 / 15.5 GB |
| 3D H2+ box_R6_h100_opt, 4.72M DOFs (CECR_MAX_RUNGS=0) | 16 | 46:27 | 50.7 GB |

H2+ breakdown:
- CECR solve at the analytic σ: 670 s
- 3 solves (opt, optP1, optA): 1224 s, about 400 s each
- 2 per-k solves: 743 s
- P1 Dirichlet: 45 s
- d_h: 21 s

The radial bound for the whole ε grid costs a few seconds. The release H2+ run took 61 min with ladders.

## 5. Things to note / surprising

1. **One-centre k=1 is degenerate, as expected.**
   - λ_1 ≥ -C_V(1) holds directly: take ε=1 in the continuous form bound.
   - For hydrogen on B(0,6), -C_V(1) = ν_N(B6, Z=1) = -0.30460, the *exact* Neumann ground state. It lies below U_N = -0.3044 at h080, so the radial computation alone already gives λ_1 sharply.
   - The CECR-based L_best,1 (-0.3449 at h080) is far weaker than this trivial bound. Its optimum ε_1 is pushed to 0.85–0.95, limited by the A_h = Γ_h/(1-ε) and ε_h = d_h/(1-ε) penalties.
   - The same holds in 2D: -C_V(1) = -1.0021 beats L_best,1 = -1.0154 on h010.
   - For k ≥ 2, and for the two-centre H2+ case (-C_V(1) = -1.28 against L_best,1 = -0.78), the CECR bound is the meaningful one.
2. **The radial constant is not always better than the old ladder or transfer constants.**
   - 3D hydrogen h100/h080: the radial C (0.4763/0.4541) is slightly *worse* than the release transfer:ladder2 values (0.4696/0.4425). The Lemma 3 shift dominates: C_V(ε-d_h)+d_h against the exact C_V(0.6) = 0.4218. The old discrete ladder certifies the discrete constant via Thm 4.2 and avoids part of that loss.
   - L_opt at fixed ε=0.6 is therefore 0.001–0.002 worse than the release on these meshes. The per-k ε more than compensates: the gain is +0.02 at h080 and +0.03 at h100.
   - 2D two-centre: the radial C (2.40) is clearly worse than analytic:ladder4 (2.135 on h020_fixed), because b_i bounds the other centre's potential by its maximum over the ball. On h020_fixed the per-k choice therefore returns ε=0.55 with the ladder constant (L_best = L_opt).
   - On box h040 the analytic ladders failed to reach a good C (3.95). The release value 2.2213 came from transfer bounds, which are excluded here, so L_opt is worse (-1.4693 vs -1.4446). The per-k L_best = -1.4349 is better than the release.
   - 2D one-centre h020/h010: analytic:ladder4 (1.8287/1.8196) beats radial (1.847/1.844).
   - Natural follow-up (not implemented): use the radial bound as the *start* of a one-rung ladder, so the discrete certification removes the Lemma-3 loss, and allow ladders at the per-k ε values.
3. **H2+** is the case where the new bound matters. C_cert drops from 1.7546 (analytic ladder2) to 1.3248 (radial, balls of radius 2, touching). L_1 goes from -0.920 to -0.786 (fixed ε) and to -0.782 (per-k ε=0.675). The gap to U_D = -0.544 shrinks from 0.376 to 0.238.
4. **Thread oversubscription.** The first launch did not cap BLAS/OpenMP threads, and CHOLMOD/OpenBLAS used about 32 cores per job. The jobs were killed after about 17 min and relaunched with `OMP_NUM_THREADS = OPENBLAS_NUM_THREADS = -t`. No results from the killed runs were used.
5. **JSON -Inf bug.** s_rem = -Inf (ball domain) broke the JSON writer on the first hydrogen h140 run. It is now written as `null` and that run was redone.
6. **Code versions per JSON.**
   - 2D (all five) and hydrogen h100: final per-k code (`95ec234`).
   - hydrogen h140/h120/h080: `06c77a6` + the s_rem fix.
   - H2+: `06c77a6`.

   The only later change is that the per-k grid uses C_cert at ε = spec.eps when it beats the radial constant. In all 3D runs C_cert *is* the radial constant, so their numbers are unaffected; only the `C_best_how` key is missing, and the CSV fills it as radial:direct. The five 2D runs were redone with the final code; the superseded 2D JSONs are not in the tarball.
7. Item 5 of the task names 10 meshes (3 + 2 + 4 + 1), not 12. All 10 were run.
