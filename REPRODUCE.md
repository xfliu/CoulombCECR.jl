# Reproducing the numerical results

All commands are run from the repository root. Threads: `julia -t auto` (or `-t N`).
The timings were measured on the group server (96 cores, 251 GB; 16 Julia threads, Julia 1.12.6).

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

## 1. Form-bound constant

The constant C_eps^cert is computed inside `run.jl` from the Neumann-decoupling bound (`src/radialbound.jl`;
1D radial problem, seconds). The fine-mesh certification of v1.0 (`scripts/certify_continuous*.jl`, about 1 h per
case) is kept for comparison only and is not needed; pass `CECR_NO_TRANSFER=1` to ignore its output files.

## 2. Bounds per mesh (Tables 6.2–6.5)

`scripts/run.jl <case> <level | mesh-dir> [outdir]`. If `outdir` contains `contbound_<case>.json`, the certified
C_eps is transferred from it; otherwise only the ladders on the current mesh are used (`results/*_analytic_only`).

| Table | Command | Wall time |
|---|---|---|
| 6.2 (2D, one centre) | `CECR_NO_TRANSFER=1 scripts/run.jl 2d_one_centre {1, meshes/square_h020_fixed, 3} results/v1.1/2d` | 1 / 2 / 7 min |
| 6.3 (2D, two centres) | `CECR_NO_TRANSFER=1 scripts/run.jl 2d_two_centre {1, meshes/box_h020_fixed} results/v1.1/2d` | 2 / 7 min |
| 6.3 (3D hydrogen) | `CECR_MAX_RUNGS=0 CECR_NO_TRANSFER=1 scripts/run.jl 3d_hydrogen meshes/ball_R6_hXXX_conf results/v1.2/3d` | 9–14 min each |
| 6.5 (3D H2+) | `CECR_SYMMETRY=1,2,3 CECR_MAX_RUNGS=0 CECR_NO_TRANSFER=1 scripts/run.jl 3d_h2plus meshes/box_sym_h{100,050} results/v1.1/3d_sym` | 20 min, 15 GB (h=1) / 29 min, 22 GB (h=0.5) |

Symmetric H2+ meshes: `python3 meshgen/gen_box_symmetric.py --h 1.0 --output meshes/box_sym_h100` (and `--h 0.5`).
On the h=1 mesh the CECR eigensolve needs ~103 GB without `CECR_SYMMETRY` and ~8 GB with it; the complete
symmetric run peaks at ~15 GB (h=1) and ~22 GB (h=0.5).

(prefix each with `julia -t auto --project=.`). Each run writes `<case>_<mesh>.json` with d_h (patch/off-patch),
Γ_h, C_h, ε_h, C_eps^cert (and the P1 diagnostic), μ_{k,h}^σ, L_k, L_k^opt and the P1 upper bounds U_k.

## 3. Tables

`results/v1.1/` holds the JSONs behind the tables of the revised paper (`results/v1.1/tables.tex`);
`results/v1.0/` holds the first validation run (fine-mesh transfer of C_eps, original and conforming meshes).
`docs/` contains the development notes of the form-bound and symmetry extensions.

## Notes
* Floating point only. Singular integrals use collapsed (Duffy) Gauss rules. Eigenvalues come from sparse Cholesky
  shift-invert with ArnoldiMethod.jl (Arpack is not used).
* The paper's 3D hydrogen ball meshes and the 2D `square_h020_opt`, `box_h020_opt` and `box_h010_opt` meshes are not
  conforming. The bounds in the revised paper use the conforming replacements.
* `meshgen/extract_mat*.py` read the stored MATLAB results of the original code for comparison.
