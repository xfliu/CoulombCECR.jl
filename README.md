# CoulombCECR.jl

Explicit two-sided eigenvalue bounds for Schrödinger operators
$H=-\Delta+V$ with Coulomb-type singular potentials
($V=-\sum_i Z_i/|x-a_i|$, $N=2,3$) on a truncated domain, by the Composite
Enriched Crouzeix–Raviart (CECR) finite element method.

Companion code for

> X. Liu, *Explicit Two-Sided Eigenvalue Bounds for Schrödinger Operators with
> Singular Potentials* (submitted).

Finite element assembly (ECR/CECR spaces in 2D and 3D) is provided by
[VFEM.jl](https://github.com/xfliu/vfem). This package implements everything
that is specific to the singular potential:

| Paper | Source file | Quantity |
|---|---|---|
| Prop. 1 | `src/constants.jl` | $C_h=h_{\max}/\pi$ |
| Def. 1, Lemma 6 | `src/constants.jl` | elementwise reaction constant $\Gamma_h$ |
| Thm 4.2 | `src/constants.jl`, `src/cecr.jl` | shifted CECR Ritz values $\mu_{k,h}^\sigma$, lower bound $L_k^{\mu,\sigma}$ |
| Lemma 3, Sec. 4.4 | `src/formbound.jl` | certified form-bound constant $C_\epsilon^{\rm cert}$ (ladders, fine-mesh transfer) |
| Thm 5.3 | `src/perturbation.jl` | patch/off-patch estimate $d_h$ with optimised patch |
| Thm 5.4 | `src/pipeline.jl` | $L_k=(1-\varepsilon_h)(L_k^{\mu,\sigma}+\sigma)-\sigma$, $U_k$ |
| — | `src/p1.jl` | conforming $P_1$ upper bounds (Dirichlet / Neumann) |
| — | `src/potential.jl`, `src/quadrature.jl` | cell averages $c_h=\Pi_0V$, singular (Duffy) quadrature |

**Arithmetic.** This release evaluates all quantities in floating point
(quadrature is used for the singular integrals). The verified version with
interval arithmetic is a separate stage.

## Installation

```julia
julia> using Pkg
julia> Pkg.develop(url = "https://github.com/xfliu/vfem")    # VFEM.jl
julia> Pkg.activate("."); Pkg.instantiate()
julia> Pkg.test()
```

Tested with Julia 1.12.6.

## Layout

```
src/        package source (see table above)
scripts/    cases.jl (problem data), run.jl (one mesh), certify_continuous*.jl (fine-mesh form bound)
meshgen/    mesh generators (2D graded meshes, conforming 3D ball, 3D H2+ box) and mesh checks
results/    JSON output of the validation runs
test/       unit tests (`Pkg.test()`)
```

## Reproducing the tables

See `REPRODUCE.md` (mesh generation, commands per table, timings).

## Notes

* The 2D generator `meshgen/gen_mesh_2d.py --paper-compat` reproduces the paper's meshes exactly (needs scipy 1.8.0).
  Without the flag it produces conforming meshes (boundary seeds clipped).
* `meshgen/gen_ball_conforming.jl` replaces the paper's ball generator: same nodes, face-conforming tetrahedra.
* The certified form-bound constant $C_\epsilon^{\rm cert}$ is obtained from Sec. 4.4 ladders. Where available, it is
  transferred from a continuous bound certified on a fine mesh (Lemma 3). The $P_1$ Galerkin value is reported
  only as a diagnostic; it is not a bound.

## License

TBD by the author (placeholder in `LICENSE`).
