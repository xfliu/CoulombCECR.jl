# Reproducing the numerical results

All commands are run from the repository root. Threads: `julia -t auto` (or `-t N`).
The timings were measured on the group server (multi-core x86-64, Julia 1.12.6).

## 0. Meshes

Meshes are not stored in the repository (several hundred MB). Either point `CECR_MESH_ROOT` to a folder with the
paper's original layout (`case7_2d_hydrogen_square/meshes/square_h040_opt`, …; see `MESHES` in `scripts/cases.jl`),
or regenerate them:

```sh
# 2D, one centre on [-5,5]^2. With --paper-compat the paper meshes are reproduced bit-identically
# (requires scipy 1.8.0 Qhull). Without it, boundary seeds are clipped and the mesh is conforming
# (the "_fixed" meshes).
python3 meshgen/gen_mesh_2d.py --h 0.4 --paper-compat --out meshes/square_h040_opt
python3 meshgen/gen_mesh_2d.py --h 0.2               --out meshes/square_h020_fixed
python3 meshgen/gen_mesh_2d.py --h 0.1  --paper-compat --out meshes/square_h010_opt
python3 meshgen/gen_mesh_2d.py --h 0.05 --paper-compat --out meshes/square_h005_opt   # fine mesh for the transfer
# 2D, two centres on [-7,7]x[-5,5]
python3 meshgen/gen_mesh_2d.py --Lx 7 --Ly 5 --centers "-2,0;2,0" --h 0.4 --paper-compat --out meshes/box_h040_opt
python3 meshgen/gen_mesh_2d.py --Lx 7 --Ly 5 --centers "-2,0;2,0" --h 0.2 --out meshes/box_h020_fixed
python3 meshgen/gen_mesh_2d.py --Lx 7 --Ly 5 --centers "-2,0;2,0" --h 0.1 --out meshes/box_h010_fixed  # fine mesh
# 3D hydrogen on B(0,6): same nodes as the paper's generator, conforming prism split
for h in 1.4 1.2 1.0 0.8 0.6; do julia meshgen/gen_ball_conforming.jl --R 6 --h_target $h; done   # -> ball_R6_hXXX_conf
# 3D H2+ on [-8,8]x[-6,6]^2 (byte-identical to the paper mesh)
python3 meshgen/gen_box_h2plus.py --h 1.0 --output meshes/box_R6_h100_opt
```
Mesh checks: `meshgen/compare_meshes_2d.py` and `meshgen/compare_meshes_3d.py` report h_max and the free edges/facets
in the interior (these must be 0 for a conforming mesh).

## 1. Certified continuous form bound (needed for C_eps^cert, Sec. 4.4 + Lemma 3 transfer)

```sh
# 2D: chained ladder on the fine mesh; eps list = target eps_f values
julia -t auto --project=. scripts/certify_continuous_chain.jl 2d_one_centre meshes/square_h005_opt 0.52,0.53,0.535,0.54,0.545,0.548 results/2d
julia -t auto --project=. scripts/certify_continuous_chain.jl 2d_two_centre meshes/box_h010_fixed   0.52,0.53,0.535,0.54,0.545,0.548 results/2d
# 3D hydrogen: ladder on the finest conforming ball, analytic start eps0 = 0.3
CECR_MAX_RUNGS=1 julia -t auto --project=. scripts/certify_continuous.jl 3d_hydrogen meshes/ball_R6_h060_conf 0.45,0.5 results/3d 0.3
```
Each run writes `contbound_<case>.json` (about 1 h each in 2D).

## 2. Bounds per mesh (Tables 6.2–6.5)

`scripts/run.jl <case> <level | mesh-dir> [outdir]`. If `outdir` contains `contbound_<case>.json`, the certified
C_eps is transferred from it; otherwise only the ladders on the current mesh are used (`results/*_analytic_only`).

| Table | Command | Wall time |
|---|---|---|
| 6.2 (2D, one centre) | `scripts/run.jl 2d_one_centre {1, meshes/square_h020_fixed, 3} results/2d` | 3 / 6 / 26 min |
| 6.3 (2D, two centres) | `scripts/run.jl 2d_two_centre {1, meshes/box_h020_fixed} results/2d` | 5 / 15 min |
| 6.4 (3D hydrogen) | `CECR_MAX_RUNGS=1 scripts/run.jl 3d_hydrogen meshes/ball_R6_hXXX_conf results/3d` | 16–24 min each |
| 6.5 (3D H₂⁺) | `CECR_MAX_RUNGS=1 scripts/run.jl 3d_h2plus 1 results/3d_analytic_only` | 61 min, ~50 GB RAM |

(prefix each with `julia -t auto --project=.`). Each run writes `<case>_<mesh>.json` with d_h (patch/off-patch),
Γ_h, C_h, ε_h, C_eps^cert (and the P1 diagnostic), μ_{k,h}^σ, L_k, L_k^opt and the P1 upper bounds U_k.

## 3. Tables

The stored JSONs in `results/` are those used for the validation report. The LaTeX tables are generated from them
(see `validation_all_recomputed.csv` / `tables_recomputed.tex` in the validation material).

## Notes
* Floating point only. Singular integrals use collapsed (Duffy) Gauss rules. Eigenvalues come from sparse Cholesky
  shift-invert with ArnoldiMethod.jl (Arpack is not used).
* The paper's 3D hydrogen ball meshes and the 2D `square_h020_opt`, `box_h020_opt` and `box_h010_opt` meshes are not
  conforming. The bounds in the revised paper use the conforming replacements.
* `meshgen/extract_mat*.py` read the stored MATLAB results of the original code for comparison.
