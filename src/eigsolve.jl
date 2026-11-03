# Smallest eigenvalues of a symmetric pencil A x = λ M x by shift-invert
# Arnoldi with an explicit sparse Cholesky factorisation of A - sM.
# (Arpack's `sigma=` path returns wrong eigenvalues on some installations,
# so it is deliberately not used.)

using SparseArrays: SparseMatrixCSC
using LinearAlgebra: cholesky, Symmetric, mul!, ldiv!, Hermitian, issuccess
import LinearAlgebra
using ArnoldiMethod: partialschur, partialeigen

struct ShiftInvert{TF,TM}
    F::TF
    M::TM
    n::Int
    tmp::Vector{Float64}
end
Base.size(S::ShiftInvert) = (S.n, S.n)
Base.size(S::ShiftInvert, i::Int) = S.n
Base.eltype(::ShiftInvert) = Float64
function LinearAlgebra.mul!(y::AbstractVector, S::ShiftInvert, x::AbstractVector)
    mul!(S.tmp, S.M, x)
    y .= S.F \ S.tmp
    return y
end

symmetrize(A) = (A + A') / 2

"""
    smallest_eigs(A, M, k; shift, tol=1e-11) -> (λ::Vector, info)

The k smallest eigenvalues of A x = λ M x, assuming A - shift*M is SPD
(i.e. `shift` lies strictly below the smallest eigenvalue). Throws if the
Cholesky factorisation fails.
"""
function smallest_eigs(A::SparseMatrixCSC, M::SparseMatrixCSC, k::Int;
                       shift::Float64, tol::Float64 = 1e-11, nev_extra::Int = 3)
    As = symmetrize(A); Ms = symmetrize(M)
    F = cholesky(Symmetric(As - shift * Ms); check = false)
    issuccess(F) || error("A - shift*M is not positive definite (shift = $shift)")
    op = ShiftInvert(F, Ms, size(As, 1), zeros(size(As, 1)))
    nev = k + nev_extra
    decomp, hist = partialschur(op; nev = nev, tol = tol, which = :LM,
                                mindim = min(max(2nev, 20), size(As,1)-1), maxdim = min(max(4nev, 40), size(As,1)),
                                restarts = 2000)
    hist.converged || @warn "Arnoldi did not fully converge" hist
    θ, X = partialeigen(decomp)
    λ = shift .+ 1 ./ real.(θ)
    p = sortperm(λ)
    λ = λ[p]; X = real.(X[:, p])
    # residuals ‖A x - λ M x‖ / ‖M x‖
    res = [begin
               x = X[:, i]; r = As * x - λ[i] * (Ms * x)
               sqrt(sum(abs2, r)) / sqrt(sum(abs2, Ms * x))
           end for i in 1:k]
    return λ[1:k], (residuals = res, nconv = hist.nconverged, mvproducts = hist.mvproducts)
end
