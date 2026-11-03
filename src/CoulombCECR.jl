"""
    CoulombCECR

Explicit two-sided eigenvalue bounds for Schrödinger operators -Δ + V with
Coulomb-type singular potentials on truncated domains, by the CECR finite
element method (X. Liu). Finite element assembly is provided by VFEM.jl;
this package implements the treatment of the singularity:

* (S1) fixed shift from a form bound (Assumption 2, Lemma 3, Sec. 4.4)
* (S2) elementwise reaction constant Γ_h and Theorem 4.2
* (S3) patch/off-patch estimate of the cell-average perturbation (Thm 5.3)

All arithmetic in this version is floating point (non-rigorous stage).
"""
module CoulombCECR

using VFEM: Mesh2D, Mesh3D, mesh2d_load_ne, mesh_load_from_folder,
            create_matrix_cecr, create_matrix_cecr_3d
import LinearAlgebra

include("quadrature.jl")
include("geometry.jl")
include("potential.jl")
include("constants.jl")
include("perturbation.jl")
include("eigsolve.jl")
include("cecr.jl")
include("p1.jl")
include("formbound.jl")
include("pipeline.jl")

export Coulomb, SimplexMesh, cell_averages, direct_dh, interp_constant,
       reaction_constant, cecr_lower, corrected_lower, cecr_eigs, p1_bounds,
       certify_ceps_rung, ceps_ladder, ceps_p1_galerkin, smallest_eigs,
       CaseSpec, run_case, interior_free_facets, load_vfem_mesh, collapsed_rule, gauss_legendre01

end
