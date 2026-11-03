# Shifted CECR Ritz values μ_{k,h}^σ (Neumann problem: all DOFs kept),
# assembled with VFEM.jl.

cecr_matrices(vm::Mesh2D, cshift::AbstractVector) = create_matrix_cecr(vm, cshift)[1:2]
cecr_matrices(vm::Mesh3D, cshift::AbstractVector) = create_matrix_cecr_3d(vm, cshift)[1:2]

"""
    cecr_eigs(vm, c, sigma, k; shift=0.0) -> (μ^σ_{1..k}, info)

Eigenvalues of the CECR form with reaction coefficient c_h + σ on the
cell averages and the ECR mass. `shift` must lie below μ_1^σ; the default
0 also certifies (via Cholesky success) that a_h^σ is positive definite.
"""
function cecr_eigs(vm, c::AbstractVector, sigma::Float64, k::Int;
                   shift::Float64 = 0.0, tol::Float64 = 1e-11)
    A, M = cecr_matrices(vm, c .+ sigma)
    return smallest_eigs(A, M, k; shift = shift, tol = tol)
end
