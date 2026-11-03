# Track B — mirror-symmetry splitting of the CECR eigenproblem (notes)

**Design.** `cecr_eigs_symmetric(vm, c, σ, k, axes; shift, tol, sectors)` in `src/symmetry.jl` (3D `Mesh3D`; 2D `Mesh2D` also supported).
For each reflection x_i → −x_i: node permutation by hashing coordinates on a grid of size tol = 1e-6·(shortest edge), searching the 3^D neighbouring cells, then an exact distance check. Element and facet permutations come from dictionaries of sorted vertex tuples (VFEM `ElementList`/`FacetList`). DOF permutation: facets `1..NumF`, then cells `NumF+e` (2D: VFEM `build_ecr_dof_ordering`). Permutations of all 2^r group elements are cached per mesh.
`c` is replaced by its element-orbit average, and the maximum change is recorded. A and M are assembled once per σ. Each generator must leave them invariant, ‖A[p,p]−A‖_F/‖A‖_F < 1e-10, otherwise the function throws an error.
Sector bases P_χ: one column per DOF orbit whose stabiliser lies in ker χ, with entries χ(g)/√|O|. A_χ = P_χ'AP_χ and M_χ = P_χ'MP_χ are solved one sector at a time with `smallest_eigs` (Cholesky shift-invert; sectors of dimension ≤ 200 use dense `eigen`). The k smallest values are merged. `info` reports the sector label of each eigenvalue and, per sector: dimension, nnz, Cholesky factor nnz, timings and maxrss. Integration: `CaseSpec.symmetry` → `cecr_eigs_case`, and `scripts/run.jl` reads `ENV CECR_SYMMETRY="1,2,3"`.

**Mesh generator** `meshgen/gen_box_symmetric.py`. Builds the octant [0,8]×[0,6]² with the grading constants of `gen_box_h2plus.py`, with θ = 0.125 and r_start = 1.5625e-3 independent of h. Point set:
- quarter shells around both nuclei (2,0,0) and (−2,0,0), restricted to x ≥ 0. Each shell is a Fibonacci set kept at distance ≥ 0.4·sp from the planes y=0 and z=0, plus exact points on the half great circles in those planes (different angular offsets on the two planes), plus the poles;
- a graded polar point set on x=0 with local spacing min(c_ang·θ·r, h);
- face seeds and the jittered background, as in the original.

The octant is triangulated with scipy Delaunay; tets with vol ≤ 1e-12·h_K³ would be removed (none occur). The mesh is then reflected across x=0, y=0 and z=0, nodes on the planes are merged, and the nuclei become nodes 1 and 2.
A first version sampled x=0 only with face seeds. It produced flat tets along x=0 near the origin (h_K ≈ 0.86), which raised Γ_h from 0.0277 to 0.0693 and d_h from 0.0125 to 0.0401. That version is superseded; its results are kept under `results_trackB/v1`.

**Checks (all passed).**
- Unit tests: 23 existing + 19 new (`test/test_symmetry.jl`). Reflected Kuhn cube, with Z₂³ and with the subgroups {x2} and {x1,x3}, at σ = 5.4 and 3. 2D criss-cross square. Single-sector run. Asymmetric potential detected through the orbit change of c. A non-symmetric mesh is rejected.
- Mesh box_sym_h100: 1,647,712 tets, 264,029 nodes, 4,946,016 DOFs, h_max 0.984. 0 interior free facets, 0 over-shared facets, volume 2304. Every node and tet has an exact mirror image for all three planes.
- Full vs sector solve, k = 3, on box_sym_h100. Maximum relative difference 1.8e-13 (σ = 5.4) and 1.7e-13 (σ = 2.1546). Invariance errors: A ≤ 4.4e-12, M ≤ 1.6e-13. Orbit change of c: 7e-11 relative.

**Results** (−t 16; the host was shared with another job).

| h = 1, k = 3 | full `cecr_eigs` | 8 sectors |
|---|---|---|
| solve σ = 5.4 / 2.1546 [s] | 587 / 393 | 217 / 148 |
| process wall [s], peak RSS [GiB] | 1014, 103.4 | 389, 7.8 |
| Cholesky factor nnz | 3.76e9 | 8 × ~1.2e8 = 9.6e8 |

The sector time includes about 55 s per σ for assembly (19 s), invariance check (9 s), sector bases (3 s) and projections (8 × 2 s). The symmetry data take 23 s on the first call only.

Pipeline (`CECR_MAX_RUNGS=0`, symmetry [1,2,3]) on box_sym_h100, compared with the original box_R6_h100_opt result:

| quantity | original mesh | box_sym_h100 |
|---|---|---|
| Γ_h | 0.0277 | 0.0244 |
| C_h | 0.327 | 0.313 |
| d_h | 0.012477 | 0.012484 |
| μ−σ (σ = 5.4) | −0.56173 / −0.35931 / −0.27279 | −0.56169 / −0.35927 / −0.27275 |
| L | −2.513 / −2.434 / −2.401 | −2.404 / −2.320 / −2.285 |
| U_D | −0.54414 / −0.31572 / −0.10546 | −0.54416 / −0.31567 / −0.10531 |
| C_P1 | 0.72018 | 0.72021 |

C_cert is not comparable: the original used ladder2 (1.7546), while this run used ladder1 only (2.1987, σ_opt 2.5987, L_opt −1.047). At the original C = 1.7546 and σ = 2.1546, the symmetric mesh gives L = −0.895 / −0.759 / −0.703, against the original L_opt = −0.920 / −0.789 / −0.735. Pipeline wall time: 1096 s. Peak RSS: 104.7 GiB, which comes from the full-size ε-ladder solve (`formbound.jl`, not symmetrised).

**h = 0.5** (box_sym_h050). 2,338,576 tets, 368,773 nodes, NumF 4,687,640, 7,026,216 DOFs, h_max 0.537, conforming and exactly symmetric. Generation: 8.3 s for points and Delaunay, 40.5 s process wall time including the quality statistics and file output (h = 1: 36.3 s). Fully symmetric sector at σ = 5.4: dimension 885,713, factor nnz 1.97e8, eigensolve 40.8 s and projection 3.2 s. μ−σ = −0.56047 / −0.23379 / −0.14313 (all in sector +++). Assembly 29 s, invariance check 13 s, symmetry setup 34 s (once). Process peak RSS 10.2 GiB. direct_dh: 33 s, giving d_h = 0.011116.
Estimate for one σ with all 8 sectors: 8 × 44 s + 45 s ≈ 6.6 min. With 3–4 σ values, the CECR part of the pipeline is about 25–30 min at ~15 GiB. A full (non-split) solve at 7.0M DOFs would need a factor of an estimated ≳5e9 nonzeros, i.e. more than 100 GiB.

**Limitations / requirements.**
1. The mesh must be exactly mirror-symmetric, up to the hashing tolerance of 1e-6 × the shortest edge, and the domain must be symmetric. A Delaunay mesh of the full box is generically not symmetric, so the mesh has to be built from an octant (or quadrant).
2. Only coordinate planes through the origin (Z₂^r) are supported. Rotational symmetries are not used, e.g. the y↔z degeneracy of π_u: the sectors x2− and x3− are solved separately.
3. The A invariance error grows with sliver content (2e-11 at h = 0.5). The 1e-10 threshold may need relaxing for much finer meshes.
4. All 8 sectors must be solved to obtain the k smallest values; `sectors` is a diagnostic only.
5. The ε-ladder in `formbound.jl` still solves the full system. At h = 0.5 it would exceed the memory budget unless it calls `cecr_eigs_case` or is replaced by the track-A radial bound.
6. Mesh quality is comparable to the original but slightly worse in the sliver tail. Tets with min dihedral < 1°: 832 vs 329; < 5°: 7856 vs 5277; 0.1 % quantile 1.64° vs 2.64°.
