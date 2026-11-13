# Complete optimal-rate eigenvalue cases and revised manuscript

This directory (`optimal_rate/`) contains the code and results of the
shrinking-core computations used in the paper.

The complete cases replace the fixed-inner-radius experiments in the
manuscript. They compute the first three CECR eigenvalues, corrected lower
bounds, and conforming P1 Neumann and Dirichlet upper values using the true
potential. The release's existing optimally graded 3D hydrogen data are
retained; both 2D cases and the 3D two-centre case are freshly computed.

## Meshes

Both 2D problems use a Cartesian comparison core with radius
`rho=1e-6*h^4`, joined to the boundary by homothetic rectangular rings.
The nominal levels are `0.8,0.4,0.2,0.1`.

The 3D H2+ case uses a Cartesian octant core with radius
`rho=0.0015625*h^2`, homothetic cuboid shells, and consistent pulling
tetrahedralization. The nominal levels are `1,2/3,1/2`. A conforming global
longest-edge bisection caps the octant element diameter at h before exact
reflection across all three coordinate planes. It splits every tetrahedron
incident to each selected edge, preserving shared facets. This outer
refinement leaves the singular core unchanged. Refinement preserves G1:
child diameters decrease and child-to-nucleus distances cannot decrease.
All geometry, conformity, quality, core-radius and grading diagnostics are
stored with the meshes.

The expanded comparison cores are unions of cells containing every
nucleus-touching cell. They satisfy the revised G1/G2 formulation. The
optimized perturbation patch can be smaller than the full comparison core.

## Complete spectral calculation

`complete_case.jl` and `complete_case_3d.jl` freshly assemble the potential,
evaluate d_h, and assemble CECR matrices once per mesh. Local isotropic
normalization avoids cancellation in the ECR quadratic moments; potential
quadrature uses displacements relative to each nucleus.

Reference epsilon is 0.55 in 2D and 0.6 in 3D. For any selected epsilon:

```
C = C_V(epsilon-d_h) + d_h
sigma = C + 1-epsilon
kappa = 1-epsilon
```

The per-k selector interpolates shifted Ritz values over a finite epsilon
grid, then actually solves the selected shifts. Every reported lower bound
uses an actual eigensolve. The strongest actually computed bound, including
the fixed-epsilon reference, is retained. This is not an exhaustive exact
maximization over every grid point.

The 3D CECR spectrum is the union of ALL eight parity sectors, preserving
multiplicities and eigenvalue indices. Projected matrices are cached across
shifts. P1 upper values use the original Coulomb potential, rather than its
cell average.

`results_complete/` records all three modes, selected epsilon/shift/form
constants, actual shift solves, Neumann and Dirichlet widths, confinement
checks, isolation tests, residuals, source and mesh checksums, quadrature
settings and timings. Preliminary uncapped 3D results are kept separately
under `results_complete/exploratory_uncapped/` and are excluded from the
revised manuscript.

## Reproduce on hpc

From the release root:

```sh
bash optimal_rate/run_complete.sh
```

To run one case:

```sh
bash optimal_rate/run_complete.sh 3d_h2plus
```

The script defaults to 16 Julia threads, four BLAS threads and four parallel
parity sectors. Override `JULIA_BIN`, `JULIA_NUM_THREADS`,
`CECR_BLAS_THREADS`, `CECR_SECTOR_PARALLEL`, `CECR_H2D_LEVELS` or
`CECR_H3D_LEVELS` as needed. Independent tests run by default and can be
disabled for a timing-only repeat with `CECR_RUN_TESTS=0`.
The mesh generators need the Python standard library and, in 3D, NumPy;
the figure generator uses matplotlib. No SciPy/Qhull dependency is needed.

The original 3D hydrogen cases can be reproduced with the existing release
commands in `REPRODUCE.md`; their stored JSON files remain the provenance
source for the retained hydrogen tables.

## Manuscript and verification

`paper/paper_revised.tex` updates the patch definition, rate argument, form
parameter admissibility, numerical setup, tables and plots. Old fixed-patch
2D and 3D H2+ results and the stagnation discussion are removed. The
reference fixed-epsilon curves and same-domain Neumann widths provide the
convergence comparison. Whole-space intervals are reported only when the
exterior confinement test holds.

Paper tables and figures are generated directly from the computation JSON:

```sh
python3 optimal_rate/paper_assets_tools/make_assets.py
cd optimal_rate/paper
pdflatex -interaction=nonstopmode -halt-on-error paper_revised.tex
bibtex paper_revised
pdflatex -interaction=nonstopmode -halt-on-error paper_revised.tex
pdflatex -interaction=nonstopmode -halt-on-error paper_revised.tex
```

The source package includes the SIAM class/style, bibliography, generated
tables and figures. `compat.tex` supports the SIAM 2016 class with modern
LaTeX/hyperref and is tested with TeX Live 2022 and 2026.

All values are Float64 numerical results. Singular and Lp quadrature,
finite-dimensional eigensolves and radial form bounds require validated
interval evaluation before the numerical intervals can be called rigorous
certified enclosures. Outward rounding of printed decimal endpoints is
applied to the computed values and does not replace interval certification.
