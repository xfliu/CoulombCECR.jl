# Conforming P1 Galerkin matrices for -Δ + W, with W either the Coulomb
# potential V (upper bounds for the Schrödinger eigenvalues) or a
# piecewise-constant coefficient (form-bound diagnostics).

using SparseArrays: sparse

"""
    p1_matrices(m; pot=nothing, cpw=nothing, qs) -> (K, M, W)

K: stiffness, M: mass, W: potential matrix ∫ w φ_i φ_j with w = V (if `pot`
is given, collapsed quadrature with the singular vertex as collapse point)
or w = cpw (piecewise constant, exact).
"""
function p1_matrices(m::SimplexMesh{D}; pot = nothing, cpw = nothing,
                     qs::QuadSet{D} = QuadSet(D; n_sing = 8, n_near = 6, n_far = 3)) where {D}
    ne = nelem(m); nl = D + 1; nn = nl * nl
    Ir = zeros(Int, nn * ne); Jc = zeros(Int, nn * ne)
    Kv = zeros(nn * ne); Mv = zeros(nn * ne); Wv = zeros(nn * ne)
    Threads.@threads for e in 1:ne
        v = m.T[:, e]
        Jm = zeros(D, D)
        for i in 1:D, d in 1:D
            Jm[d, i] = m.X[d, v[i+1]] - m.X[d, v[1]]
        end
        Gref = hcat(-ones(D), Matrix{Float64}(LinearAlgebra.I, D, D))   # D × (D+1)
        G = (Jm') \ Gref                                                # ∇λ_i columns
        vol = m.vol[e]
        Ke = vol .* (G' * G)
        Me = fill(vol / ((D + 1) * (D + 2)), nl, nl)
        for i in 1:nl
            Me[i, i] *= 2
        end
        We = zeros(nl, nl)
        if pot !== nothing
            rule, iv = _pick_rule(qs, pot, m, e)
            P, order = _element_points(m, e, rule, iv)
            for q in eachindex(rule.w)
                Vq = Vval(pot, @view P[:, q]) * rule.w[q] * vol
                for a in 1:nl, b in 1:nl
                    We[order[a], order[b]] += Vq * rule.bary[q, a] * rule.bary[q, b]
                end
            end
        elseif cpw !== nothing
            We .= cpw[e] .* Me
        end
        base = (e - 1) * nn; k = 0
        for a in 1:nl, b in 1:nl
            k += 1
            Ir[base+k] = v[a]; Jc[base+k] = v[b]
            Kv[base+k] = Ke[a, b]; Mv[base+k] = Me[a, b]; Wv[base+k] = We[a, b]
        end
    end
    n = nnode(m)
    return sparse(Ir, Jc, Kv, n, n), sparse(Ir, Jc, Mv, n, n), sparse(Ir, Jc, Wv, n, n)
end

"""
    p1_bounds(m, pot, k; bc=:dirichlet, shift) -> (λ, info)

Conforming P1 Ritz values of -Δ + V. With `bc=:dirichlet` they are upper
bounds for λ_k^{R,D} (hence for λ_k and λ_k^{R,N}); with `bc=:neumann`
upper bounds for λ_k^{R,N}. `shift` must lie below λ_1.
"""
function p1_bounds(m::SimplexMesh, pot::Coulomb, k::Int; bc::Symbol = :dirichlet,
                   shift::Float64)
    K, M, W = p1_matrices(m; pot = pot)
    A = K + W
    if bc == :dirichlet
        idx = findall(.!m.bnodes)
        A = A[idx, idx]; M = M[idx, idx]
    end
    return smallest_eigs(A, M, k; shift = shift)
end
